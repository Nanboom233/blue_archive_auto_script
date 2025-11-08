import hashlib
import itertools
from collections import deque, defaultdict, Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional


class NodeSpec:
    name: str
    description: str
    features: list[Callable[[], bool]]  # runnable: env -> bool
    actions: dict[str, Optional[Callable]]  # target_name -> runnable(env) 产生从本节点到目标的动作

    def __init__(self, name, description, features, actions):
        self.name = name
        self.description = description
        self.features = features
        self.actions = actions

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
    next_hop: list[list[int]]  # next_hop[m][n] = v or -1
    dist: list[list[int]]  # dist[m][n] = steps or INF (大数)

    # 压缩（区域默认 + 例外）
    block_default_hop: list[dict[int, int]]  # per m: {node -> hop}
    exception_hop: list[dict[int, int]]  # per m: {node -> hop}

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


def split_blocks(node_volume: int, forward_map: list[list[int]], reverse_map: list[list[int]], block_size):
    # validate single weakly connected component
    undirected_map = [sorted(set(forward_map[u] + reverse_map[u])) for u in range(node_volume)]

    seen = [False] * node_volume
    dq = deque([0])
    seen[0] = True
    order_list = []

    while dq:
        u = dq.popleft()
        order_list.append(u)
        for w in undirected_map[u]:
            if not seen[w]:
                seen[w] = True
                dq.append(w)

    if len(undirected_map) != node_volume:
        raise ValueError("Graph is not a single weakly connected component")

    # 按块大小切分为 Block
    block_of = [-1] * node_volume
    blocks: list[list[int]] = []
    rid = 0
    for i in range(0, node_volume, block_size):
        block = order_list[i:i + block_size]
        blocks.append(block)
        for u in block:
            block_of[u] = rid
        rid += 1

    blocks_volume = len(blocks)
    return block_of, blocks, blocks_volume


# ---------- 全对反向 BFS（构建下一跳、距离、树边集） ----------
INF = 10 ** 9


def reverse_bfs(node_volume: int, reverse_map: list[list[int]]):
    next_hop = [[-1] * node_volume for _ in range(node_volume)]  # m行n列
    distances = [[INF] * node_volume for _ in range(node_volume)]
    tree_edges_by_dst: dict[int, set[tuple[int, int]]] = {m: set() for m in range(node_volume)}
    # tree_edges_by_dst stores the BFS tree edges for each destination m
    # this helps build edge2dst mapping and incremental updates
    # tre_edge_by_dist: m -> set of (u,v) edges in BFS tree towards m

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
def compress_by_block(node_volume: int, block_of: list[int], next_hop: list[list[int]]):
    block_default: list[dict[int, int]] = [{} for _ in range(node_volume)]
    # block_default[destination_node_id] = {destination_node_id : next_hop}
    exceptions: list[dict[int, int]] = [{} for _ in range(node_volume)]
    # exceptions[destination_node_id] = {current_node_id : next_hop}

    # 对每个目的地 m 的一列进行压缩
    for m in range(node_volume):
        # 先统计每个区域的众数（忽略 -1）
        area_values: dict[int, list[int]] = defaultdict(list)
        for n in range(node_volume):
            area_values[block_of[n]].append(next_hop[m][n])
        defaults = {}
        for rid, hops in area_values.items():
            cnt = Counter([h for h in hops if h != -1])
            if cnt:
                defaults[rid] = cnt.most_common(1)[0][0]
            else:
                defaults[rid] = -1
        block_default[m] = defaults
        # 写例外（不同于默认的）
        exc = {}
        for n in range(node_volume):
            rid = block_of[n]
            if next_hop[m][n] != defaults[rid]:
                exc[n] = next_hop[m][n]
        exceptions[m] = exc
    return block_default, exceptions


# ---------- 查询：通过区域压缩结构取下一跳 ----------
def lookup_next_hop(idx: Index, current_node_id: int, destination_node_id: int) -> int:
    # 例外优先，未命中用区域默认
    if current_node_id in idx.exception_hop[destination_node_id]:
        return idx.exception_hop[destination_node_id][current_node_id]
    block_id = idx.block_of[current_node_id]
    return idx.block_default_hop[destination_node_id].get(block_id, -1)
    # 若区域默认也无定义，返回 -1


# ---------- 在线 BFS 兜底（可选避开某条边） ----------
def fallback_next_hop(index: Index, current_node_id: int, destination_node_id: int,
                      avoid_edge: Optional[list[tuple[int, int]]] = None) -> None | list[int]:
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
def compile_specs(specs: list[NodeSpec], prev: Optional[Index] = None) -> Index:
    name2id, id2name, _ = build_stable_idmap(specs, prev)

    node_volume, edges, forward, reverse_map, edge_handle, node_features = build_graph(specs, name2id)
    topo_hash = compute_topo_hash(node_volume, edges)

    # 如果拓扑未变，直接复用结构（只更新行为层）
    if prev is not None and topo_hash == prev.topo_hash:
        print("[build] topology unchanged -> reuse index; only update callables")

        # 新增：用新的 features 生成 validators（开发期可能只改了 features/runnable）
        scan_order = list(itertools.chain.from_iterable(prev.blocks)) if prev.blocks else list(range(node_volume))
        validators: list[None | Callable[[], bool]] = [None] * node_volume
        for node in specs:
            validators[name2id[node.name]] = node.check

        return Index(
            name2id=name2id, id2name=id2name, topo_hash=topo_hash,
            node_volume=prev.node_volume, edges=prev.edges, forward_map=prev.forward_map, reverse_map=prev.reverse_map,
            block_of=prev.block_of, blocks=prev.blocks, block_volume=prev.block_volume,
            next_hop=prev.next_hop, dist=prev.dist,
            block_default_hop=prev.block_default_hop, exception_hop=prev.exception_hop,
            edge_handle=edge_handle,
            validators=validators,
            scan_order=scan_order,
            tree_edges_by_dst=prev.tree_edges_by_dst, edge2dst=prev.edge2dst
        )

    # 增量：如果 prev 存在且拓扑变动小，尝试部分重建
    if prev is not None and prev.node_volume == node_volume:
        added = edges - prev.edges
        removed = prev.edges - edges
        if len(added) + len(removed) <= max(1, node_volume // 3) or True:
            # we do an incremental rebuild when the number of changed edges is small(1/3 of node volume)
            print(f"[build] incremental rebuild: +{len(added)} -{len(removed)}")
            print(f"[build] Added edges: {[(id2name[u], id2name[v]) for u, v in added]}")
            print(f"[build] Removed edges: {[(id2name[u], id2name[v]) for u, v in removed]}")
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
            block_default_hop, exceptions_hop = compress_by_block(node_volume, block_of, next_hop)

            scan_order = list(itertools.chain.from_iterable(blocks)) if blocks else list(range(node_volume))
            validators: list[None | Callable[[], bool]] = [None] * node_volume
            for node in specs:
                validators[name2id[node.name]] = node.check
            return Index(
                name2id=name2id, id2name=id2name, topo_hash=topo_hash,
                node_volume=node_volume, edges=edges, forward_map=forward, reverse_map=reverse_map,
                block_of=block_of, blocks=blocks, block_volume=blocks_volume,
                next_hop=next_hop, dist=dist,
                block_default_hop=block_default_hop, exception_hop=exceptions_hop,
                edge_handle=edge_handle,
                validators=validators,
                scan_order=scan_order,
                tree_edges_by_dst=tree_edges_by_dst, edge2dst=edge2dst
            )

    # 全量构建
    print("[build] full build")
    block_of, blocks, blocks_volume = split_blocks(node_volume, forward, reverse_map,
                                                   block_size=max(2, node_volume // max(4, node_volume // 6)))

    next_hop, dist, tree_edges_by_dst = reverse_bfs(node_volume, reverse_map)

    # 建 edge2dst
    edge2dst: dict[tuple[int, int], set[int]] = defaultdict(set)
    # `defaultdict` offers a set for each new key
    for m, tree_edges in tree_edges_by_dst.items():
        # m: destination node
        # edges: set of (u,v) edges in BFS tree towards m
        for tree_edge in tree_edges:
            edge2dst[tree_edge].add(m)

    block_default_hop, exceptions_hop = compress_by_block(node_volume, block_of, next_hop)

    scan_order = list(itertools.chain.from_iterable(blocks)) if blocks else list(range(node_volume))
    # an initial scan order for full table scan during runtime
    # we adjust this order dynamically based on hits (move-to-front)
    validators: list[None | Callable[[], bool]] = [None] * node_volume
    for node in specs:
        validators[name2id[node.name]] = node.check

    return Index(
        name2id=name2id, id2name=id2name, topo_hash=topo_hash,
        node_volume=node_volume, edges=edges, forward_map=forward, reverse_map=reverse_map,
        block_of=block_of, blocks=blocks, block_volume=blocks_volume,
        next_hop=next_hop, dist=dist,
        block_default_hop=block_default_hop, exception_hop=exceptions_hop,
        edge_handle=edge_handle,
        validators=validators,
        scan_order=scan_order,
        tree_edges_by_dst=tree_edges_by_dst, edge2dst=edge2dst
    )
