import bisect
import hashlib
import itertools
from array import array
from collections import deque, defaultdict, Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class NodeSpec:
    name: str
    description: str
    features: list[Callable[[], bool]]  # runnable: env -> bool
    actions: dict[str, Optional[Callable]]  # target_name -> runnable(env) 产生从本节点到目标的动作

    def check(self) -> bool:
        for feature in self.features:
            if not feature():
                return False
        return True


# ---------- 内核索引结构 ----------
@dataclass
class Index:
    name2id: dict[str, int]
    id2name: list[str]
    topo_hash: str

    # 图
    node_volume: int
    edges: set[tuple[int, int]]
    forward_map: list[list[int]]
    reverse_map: list[list[int]]

    # 区域
    block_of: list[int]  # node -> block_id
    blocks: list[list[int]]  # block_id -> [nodes]
    block_volume: int

    # 全对下一跳与距离（m 行，n 列；即从 n 去 m 的第一跳）
    next_hop: list[list[int]]  # 仍保留原始矩阵（可选：后续也可换成 array('h') 进一步省内存）
    dist: list[list[int]]

    # 紧凑列存储（逐列择优）
    # col_kind[m]==0: 直接列，用 row_direct[m]
    # col_kind[m]==1: 区域列，用 region_default[m] + exc_nodes[m]/exc_hops[m]
    col_kind: list[int]  # 0=direct, 1=block
    row_direct: list[Optional[array]]  # array('h') or None
    region_default: list[Optional[array]]  # array('h') or None
    exc_nodes: list[Optional[array]]  # array('h') or None (sorted node ids)
    exc_hops: list[Optional[array]]  # array('h') or None (aligned with exc_nodes)

    # 行为层
    edge_handle: dict[tuple[int, int], Callable[[], None]]
    validators: list[None | Callable[[], bool]]  # per node_id

    # 扫描顺序：全表扫描的优先顺序（命中后 move-to-front）
    scan_order: list[int]

    # 增量所需
    tree_edges_by_dst: dict[int, set[tuple[int, int]]]  # m -> {(u,v)...}
    edge2dst: dict[tuple[int, int], set[int]]  # (u,v) -> {m,...}


# ---------- 工具：稳定 ID 映射 ----------
def build_stable_idmap(specs: list[NodeSpec],
                       cached_idmap: Optional[Index]) -> tuple[dict[str, int], list[str], bool]:
    if cached_idmap is None:
        names = sorted([s.name for s in specs])
        name2id = {n: i for i, n in enumerate(names)}
        id2name = deepcopy(names)
        changed = True
    else:
        name2id = dict(cached_idmap.name2id)
        id2name = list(cached_idmap.id2name)
        new_names = sorted([s.name for s in specs if s.name not in name2id])
        for n in new_names:
            name2id[n] = len(id2name)
            id2name.append(n)
        # 删除的名字保留墓碑（演示简单起见不做回收）
        changed = (len(new_names) > 0)
    return name2id, id2name, changed


# ---------- 工具：抽取拓扑 + 行为绑定 ----------
def build_graph(specs: list[NodeSpec], name2id: dict[str, int]):
    node_length = len(name2id)
    forward_map = [[] for _ in range(node_length)]
    reverse_map = [[] for _ in range(node_length)]

    edges = set()
    # works as a set of existing edges and to avoid duplicate edges
    edge_handle = dict()
    node_features = [[] for _ in range(node_length)]

    for spec in specs:
        spec_id = name2id[spec.name]
        node_features[spec_id].extend(spec.features)
        for destination_name, act in spec.actions.items():
            destination_id = name2id[destination_name]
            if (spec_id, destination_id) not in edges:
                edges.add((spec_id, destination_id))
                forward_map[spec_id].append(destination_id)
                reverse_map[destination_id].append(spec_id)
            edge_handle[(spec_id, destination_id)] = act

    # 邻接排序确保确定性
    for spec_id in range(node_length):
        forward_map[spec_id].sort()
        reverse_map[spec_id].sort()
    return node_length, edges, forward_map, reverse_map, edge_handle, node_features


def compute_topo_hash(node_volume: int, edges: set[tuple[int, int]]) -> str:
    data = bytearray()
    data.extend(node_volume.to_bytes(4, 'little'))
    for (u, v) in sorted(edges):
        data.extend(u.to_bytes(4, 'little'))
        data.extend(v.to_bytes(4, 'little'))
    return hashlib.sha256(data).hexdigest()


# ---------- 全对反向 BFS（构建下一跳、距离、树边集） ----------
INF = 10 ** 9


def reverse_bfs(node_volume: int, reverse_map: list[list[int]]):
    next_hop = [[-1] * node_volume for _ in range(node_volume)]  # m行n列
    distances = [[INF] * node_volume for _ in range(node_volume)]
    tree_edges_by_dst: dict[int, set[tuple[int, int]]] = {m: set() for m in range(node_volume)}
    # tree_edges_by_dst stores the BFS tree edges for each destination m
    # this helps build edge2dst mapping and incremental updates
    # tree_edge_by_dist: m -> set of (u,v) edges in BFS tree towards m

    # TODO: entry node validation，检测不可达节点
    for m in range(node_volume):
        dq = deque()
        dq.append(m)
        distances[m][m] = 0
        while dq:
            x = dq.popleft()
            for u in reverse_map[x]:  # 原图 u->x
                if distances[m][u] == INF:  # 第一次访问
                    distances[m][u] = distances[m][x] + 1
                    next_hop[m][u] = x  # 从 u 走向 m 的第一步是 x
                    tree_edges_by_dst[m].add((u, x))
                    dq.append(u)
        # m 自己的下一跳可定义为自身或 -1，这里设置为 m（对齐“已在目标”）
        next_hop[m][m] = m
    return next_hop, distances, tree_edges_by_dst


# ---------- 区域压缩：区域默认 + 例外 ----------
def split_blocks(node_volume: int,
                 forward_map: list[list[int]],
                 reverse_map: list[list[int]],
                 block_size: int = 32) -> tuple[list[int], list[list[int]], int, int]:
    # 生成无向邻接用于 BFS 排序
    undirected_map = [sorted(set(forward_map[u] + reverse_map[u])) for u in range(node_volume)]
    # 得到稳定的 BFS 顺序 同时校验图
    dq = deque([0])
    seen = [False] * node_volume
    seen[0] = True
    order_list = []
    while dq:
        u = dq.popleft()
        order_list.append(u)
        for w in undirected_map[u]:
            if not seen[w]:
                seen[w] = True
                dq.append(w)
    if len(order_list) != node_volume:
        raise ValueError("Graph is not a single weakly connected component")

    block_of = [-1] * node_volume
    blocks: list[list[int]] = []
    current_block_id = 0
    for i in range(0, node_volume, block_size):
        block = order_list[i:i + block_size]
        blocks.append(block)
        for u in block:
            block_of[u] = current_block_id
        current_block_id += 1
    blocks_volume = len(blocks)

    return block_of, blocks, blocks_volume, block_size


def build_compact_columns(node_volume: int,
                          block_of: list[int],
                          block_volume: int,
                          next_hop: list[list[int]]):
    # 输出结构
    col_kind: list[int] = [0] * node_volume
    row_direct: list[Optional[array]] = [None] * node_volume
    region_default: list[Optional[array]] = [None] * node_volume
    exc_nodes: list[Optional[array]] = [None] * node_volume
    exc_hops: list[Optional[array]] = [None] * node_volume

    for m in range(node_volume):
        # 统计每块众数
        counts_per_block: dict[int, Counter] = defaultdict(Counter)
        for n in range(node_volume):
            rid = block_of[n]
            hop = next_hop[m][n]
            counts_per_block[rid][hop] += 1

        # 构建区域默认数组（array('h')，支持 -1）
        default_arr = array('h', [-1] * block_volume)
        for rid in range(block_volume):
            cnt = counts_per_block.get(rid)
            if cnt and len(cnt) > 0:
                mode_val, _ = cnt.most_common(1)[0]
                default_arr[rid] = int(mode_val)
            else:
                default_arr[rid] = -1

        # 构建例外稀疏数组（节点按升序）
        ex_nodes = array('h')
        ex_hops = array('h')
        exceptions = 0
        for n in range(node_volume):
            rid = block_of[n]
            hop = next_hop[m][n]
            if hop != default_arr[rid]:
                ex_nodes.append(n)
                ex_hops.append(int(hop))
                exceptions += 1

        # 估算并择优
        bytes_direct = 2 * node_volume
        bytes_region = 2 * block_volume + 4 * exceptions

        if bytes_region < bytes_direct:
            # 采用区域列
            col_kind[m] = 1
            region_default[m] = default_arr
            exc_nodes[m] = ex_nodes
            exc_hops[m] = ex_hops
            row_direct[m] = None
        else:
            # 采用直接列
            col_kind[m] = 0
            row_direct[m] = array('h', (int(next_hop[m][n]) for n in range(node_volume)))
            region_default[m] = None
            exc_nodes[m] = None
            exc_hops[m] = None

    return col_kind, row_direct, region_default, exc_nodes, exc_hops


# ---------- 查询：通过区域压缩结构取下一跳 ----------
def lookup_next_hop(idx: Index, current_node_id: int, destination_node_id: int) -> int:
    kind = idx.col_kind[destination_node_id]
    if kind == 0:
        # 直接列：O(1) 取值
        row = idx.row_direct[destination_node_id]
        # row 不应为 None
        return row[current_node_id] if row is not None else -1
    else:
        # 区域列：先查例外（二分），未命中用默认
        nodes = idx.exc_nodes[destination_node_id]
        hops = idx.exc_hops[destination_node_id]
        if nodes is not None and len(nodes) > 0:
            pos = bisect.bisect_left(nodes, current_node_id)
            if pos < len(nodes) and nodes[pos] == current_node_id:
                return hops[pos]
        block_id = idx.block_of[current_node_id]
        defaults = idx.region_default[destination_node_id]
        return defaults[block_id] if defaults is not None else -1


# ---------- 在线 BFS 兜底（可选避开某条边） ----------
def fallback_next_hop(index: Index, current_node_id: int, destination_node_id: int,
                      avoid_edge: Optional[set[tuple[int, int]]] = None) -> None | list[int]:
    if current_node_id == destination_node_id:
        return []
    path = []
    node_volume = index.node_volume
    dq = deque([current_node_id])
    parent = [-1] * node_volume
    seen = [False] * node_volume
    seen[current_node_id] = True
    while dq:
        u = dq.popleft()
        for next_node_id in index.forward_map[u]:
            if avoid_edge is not None and (u, next_node_id) in avoid_edge:
                continue
            if not seen[next_node_id]:
                seen[next_node_id] = True
                parent[next_node_id] = u
                if next_node_id == destination_node_id:
                    # 回溯第一步
                    cur = next_node_id
                    while parent[cur] != current_node_id:
                        path.append(cur)
                        cur = parent[cur]
                    path.append(cur)
                    path.reverse()
                    return path
                dq.append(next_node_id)
    return None


# ---------- 编译（全量或增量） ----------
def compile_specs(specs: list[NodeSpec], prev: Optional[Index] = None, log_output: bool = True) -> Index:
    name2id, id2name, _ = build_stable_idmap(specs, prev)

    node_volume, edges, forward, reverse_map, edge_handle, node_features = build_graph(specs, name2id)
    topo_hash = compute_topo_hash(node_volume, edges)

    # 如果拓扑未变，直接复用结构（只更新行为层）
    if prev is not None and topo_hash == prev.topo_hash:
        print("[build] topology unchanged -> reuse index; only update callables") if log_output else None
        scan_order = list(itertools.chain.from_iterable(prev.blocks)) if prev.blocks else list(range(node_volume))
        validators: list[None | Callable[[], bool]] = [None] * node_volume
        for node in specs:
            validators[name2id[node.name]] = node.check

        return Index(
            name2id=name2id, id2name=id2name, topo_hash=topo_hash,
            node_volume=prev.node_volume, edges=prev.edges,
            forward_map=prev.forward_map, reverse_map=prev.reverse_map,
            block_of=prev.block_of, blocks=prev.blocks, block_volume=prev.block_volume,
            next_hop=prev.next_hop, dist=prev.dist,
            col_kind=prev.col_kind,
            row_direct=prev.row_direct,
            region_default=prev.region_default,
            exc_nodes=prev.exc_nodes,
            exc_hops=prev.exc_hops,
            edge_handle=edge_handle,
            validators=validators,
            scan_order=scan_order,
            tree_edges_by_dst=prev.tree_edges_by_dst, edge2dst=prev.edge2dst
        )

    # 增量：如果 prev 存在且拓扑变动小，尝试部分重建
    if prev is not None and prev.node_volume == node_volume:
        added = edges - prev.edges
        removed = prev.edges - edges
        if len(added) + len(removed) <= max(1, node_volume // 3):
            # we do an incremental rebuild when the number of changed edges is small(1/3 of node volume)
            print(f"[build] incremental rebuild: +{len(added)} -{len(removed)}") if log_output else None
            print(f"[build] Added edges: {[(id2name[u], id2name[v]) for u, v in added]}") if log_output else None
            print(f"[build] Removed edges: {[(id2name[u], id2name[v]) for u, v in removed]}") if log_output else None
            # 从 prev 拷贝
            next_hop = [row[:] for row in prev.next_hop]
            dist = [row[:] for row in prev.dist]
            tree_edges_by_dst = {m: set(edges) for m, edges in prev.tree_edges_by_dst.items()}
            edge2dst = {edge: set(destination) for edge, destination in prev.edge2dst.items()}

            # 受影响目的地集合
            affected: set[int] = set()

            # 删除边：所有使用该边作为 BFS 树边的目的地都受影响
            for edge in removed:
                for m in edge2dst.get(edge, set()):
                    affected.add(m)

            # 新增边：若 1 + dist[m][v] < dist[m][u] 则受影响
            for (u, v) in added:
                for m in range(node_volume):
                    if prev.dist[m][v] < INF and 1 + prev.dist[m][v] < prev.dist[m][u]:
                        affected.add(m)

            # 对受影响目的地重算反向 BFS 列
            for m in affected:
                # 清理旧 edge2dst 关联
                for edge in tree_edges_by_dst.get(m, set()):
                    if edge in edge2dst and m in edge2dst[edge]:
                        edge2dst[edge].remove(m)
                        if not edge2dst[edge]:
                            edge2dst.pop(edge, None)

                # 重算
                dq = deque([m])
                dist[m] = [INF] * node_volume
                dist[m][m] = 0
                next_hop[m] = [-1] * node_volume
                tree_edges_by_dst[m] = set()
                while dq:
                    x = dq.popleft()
                    for uu in reverse_map[x]:
                        if dist[m][uu] == INF:
                            dist[m][uu] = dist[m][x] + 1
                            next_hop[m][uu] = x
                            tree_edges_by_dst[m].add((uu, x))
                            edge2dst.setdefault((uu, x), set()).add(m)
                            dq.append(uu)
                next_hop[m][m] = m

            # 区域化可沿用旧划分（小改动不重分区），仅重写压缩列
            block_of, blocks, blocks_volume = prev.block_of, prev.blocks, prev.block_volume
            col_kind, row_direct, region_default, exc_nodes, exc_hops = build_compact_columns(
                node_volume, block_of, blocks_volume, next_hop
            )

            scan_order = list(itertools.chain.from_iterable(blocks)) if blocks else list(range(node_volume))
            validators: list[None | Callable[[], bool]] = [None] * node_volume
            for node in specs:
                validators[name2id[node.name]] = node.check

            return Index(
                name2id=name2id, id2name=id2name, topo_hash=topo_hash,
                node_volume=node_volume, edges=edges, forward_map=forward, reverse_map=reverse_map,
                block_of=block_of, blocks=blocks, block_volume=blocks_volume,
                next_hop=next_hop, dist=dist,
                col_kind=col_kind,
                row_direct=row_direct,
                region_default=region_default,
                exc_nodes=exc_nodes,
                exc_hops=exc_hops,
                edge_handle=edge_handle,
                validators=validators,
                scan_order=scan_order,
                tree_edges_by_dst=tree_edges_by_dst, edge2dst=edge2dst
            )

    # 全量构建
    print("[build] full build") if log_output else None

    # 1) 先跑反向 BFS
    next_hop, dist, tree_edges_by_dst = reverse_bfs(node_volume, reverse_map)

    # 2) 用采样评估选择块大小并切块
    block_of, blocks, blocks_volume, chosen_blk = split_blocks(node_volume, forward, reverse_map)
    print(f"[build] chosen block_size={chosen_blk}, blocks={blocks_volume}") if log_output else None

    # 3) 逐列自适应压缩（直接 vs 区域，择优）
    col_kind, row_direct, region_default, exc_nodes, exc_hops = build_compact_columns(
        node_volume, block_of, blocks_volume, next_hop
    )

    # 4) 建 edge2dst（与原逻辑一致）
    edge2dst: dict[tuple[int, int], set[int]] = defaultdict(set)
    for m, tree_edges in tree_edges_by_dst.items():
        for tree_edge in tree_edges:
            edge2dst[tree_edge].add(m)

    scan_order = list(itertools.chain.from_iterable(blocks)) if blocks else list(range(node_volume))
    validators: list[None | Callable[[], bool]] = [None] * node_volume
    for node in specs:
        validators[name2id[node.name]] = node.check

    return Index(
        name2id=name2id, id2name=id2name, topo_hash=topo_hash,
        node_volume=node_volume, edges=edges, forward_map=forward, reverse_map=reverse_map,
        block_of=block_of, blocks=blocks, block_volume=blocks_volume,
        next_hop=next_hop, dist=dist,
        col_kind=col_kind,
        row_direct=row_direct,
        region_default=region_default,
        exc_nodes=exc_nodes,
        exc_hops=exc_hops,
        edge_handle=edge_handle,
        validators=validators,
        scan_order=scan_order,
        tree_edges_by_dst=tree_edges_by_dst, edge2dst=edge2dst
    )
