import bisect
import gzip
import hashlib
import pickle
import time
from array import array
from collections import deque, defaultdict, Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional, Final, TYPE_CHECKING

if TYPE_CHECKING:
    from core.Baas_thread import Baas_thread

INF: Final[int] = 10 ** 9


class Navigator:
    @dataclass
    class Interface:
        name: str
        description: str
        features: list[Callable[[Baas_thread], bool]]
        actions: dict[str, Optional[Callable]]

        def validate(self, baas_thread: Baas_thread) -> bool:
            for feature in self.features:
                if not feature(baas_thread):
                    return False
            return True

    @dataclass
    class BaseMetadata:
        # ====== RUNTIME METADATA ======
        # Required to perform lookups and fallback BFS

        name2id: dict[str, int]  # interface_name -> interface_id
        id2name: list[str]  # interface_id -> interface_name

        node_volume: int  # number of interfaces
        forward_map: list[list[int]]  # interface_id -> [next_interface_ids]

        wildcard_interfaces: set[int]  # interface_ids that have "*" wildcard actions

        block_of: list[int]  # interface_id -> block_id

        block_compacted: array  # array('b') of 0/1
        # block_compressed == False : use block_uncompacted_hops to lookup
        # block_compressed == True : use block_compacted_hops + block_except_* to lookup
        block_uncompacted_hops: list[
            Optional[array]]  # destination_id -> next_hops (array('h'): current_id -> next_hop_id)
        block_compacted_hops: list[Optional[array]]  # block_id -> next_hops(array('h'):  -> next_hop_id)
        block_except_nodes: list[Optional[array]]  # destination_id -> node_ids (array('h'))
        block_except_hops: list[
            Optional[
                array]]  # destination_id -> next_hops (array('h'): self.metadata_of_(block_except_nodes) -> next_hop_id)

        @classmethod
        def _pack_arr(cls, a: Optional[array]) -> Optional[dict]:
            if a is None:
                return None
            return {'typecode': a.typecode, 'len': len(a), 'bytes': a.tobytes()}

        @classmethod
        def _unpack_arr(cls, blob: Optional[dict]) -> Optional[array]:
            if blob is None:
                return None
            a = array(blob['typecode'])
            a.frombytes(blob['bytes'])
            assert len(a) == blob['len']
            return a

        @classmethod
        def _pack_arr_list(cls, lst: list[Optional[array]]) -> list[Optional[dict]]:
            return [cls._pack_arr(x) for x in lst]

        @classmethod
        def _unpack_arr_list(cls, lst: list[Optional[dict]]) -> list[Optional[array]]:
            return [cls._unpack_arr(x) for x in lst]

        def _to_payload(self) -> dict:
            return {
                'kind': 'base',
                'version': 1,
                'name2id': self.name2id,
                'id2name': self.id2name,
                'node_volume': self.node_volume,
                'forward_map': self.forward_map,
                'wildcard_interfaces': self.wildcard_interfaces,
                'block_of': self.block_of,
                'block_compacted': {
                    'typecode': self.block_compacted.typecode,
                    'len': len(self.block_compacted),
                    'bytes': self.block_compacted.tobytes()
                },
                'block_uncompacted_hops': self._pack_arr_list(self.block_uncompacted_hops),
                'block_compacted_hops': self._pack_arr_list(self.block_compacted_hops),
                'block_except_nodes': self._pack_arr_list(self.block_except_nodes),
                'block_except_hops': self._pack_arr_list(self.block_except_hops),
            }

        def save(self, path: str) -> None:
            data = pickle.dumps(self._to_payload(), protocol=pickle.HIGHEST_PROTOCOL)
            with open(path, 'wb') as f:
                f.write(gzip.compress(data))

        @classmethod
        def from_payload(cls, payload: dict) -> Navigator.BaseMetadata:
            assert payload.get('version', -1) == 1, "unsupported BaseMetadata version"
            bc = payload['block_compacted']
            block_compacted = array(bc['typecode'])
            block_compacted.frombytes(bc['bytes'])
            assert len(block_compacted) == bc['len']
            return cls(
                name2id=payload['name2id'],
                id2name=payload['id2name'],
                node_volume=payload['node_volume'],
                forward_map=payload['forward_map'],
                wildcard_interfaces=payload['wildcard_interfaces'],
                block_of=payload['block_of'],
                block_compacted=block_compacted,
                block_uncompacted_hops=cls._unpack_arr_list(payload['block_uncompacted_hops']),
                block_compacted_hops=cls._unpack_arr_list(payload['block_compacted_hops']),
                block_except_nodes=cls._unpack_arr_list(payload['block_except_nodes']),
                block_except_hops=cls._unpack_arr_list(payload['block_except_hops']),
            )

    @dataclass
    class FullMetadata(BaseMetadata):
        # ====== FULL METADATA ======
        # Required for rebuild and incremental update

        topo_hash: str  # hash of topology structure (node_volume + edges)
        edges: set[tuple[int, int]]  # set of (u,v) edges
        reverse_map: list[list[int]]  # interface_id -> [previous_interface_ids]

        blocks: list[list[int]]  # block_id -> [node_ids]
        block_volume: int  # number of blocks

        next_hop: list[list[int]]  # destination_id -> [current_id -> next_hop_id]
        distances: list[list[int]]  # destination_id -> [current_id -> distance]

        bfs_preferred_edges: dict[int, set[tuple[int, int]]]  # destination_id -> {(u,v)...}
        edge2destination: dict[tuple[int, int], set[int]]  # (u,v) -> {destination_ids}

        def to_base(self) -> Navigator.BaseMetadata:
            return Navigator.BaseMetadata(
                name2id=self.name2id,
                id2name=self.id2name,
                node_volume=self.node_volume,
                forward_map=self.forward_map,
                wildcard_interfaces=self.wildcard_interfaces,
                block_of=self.block_of,
                block_compacted=self.block_compacted,
                block_uncompacted_hops=self.block_uncompacted_hops,
                block_compacted_hops=self.block_compacted_hops,
                block_except_nodes=self.block_except_nodes,
                block_except_hops=self.block_except_hops
            )

        def _to_payload(self) -> dict:
            base = super()._to_payload()
            base['kind'] = 'full'
            base.update({
                'topo_hash': self.topo_hash,
                'edges': self.edges,
                'reverse_map': self.reverse_map,
                'blocks': self.blocks,
                'block_volume': self.block_volume,
                'next_hop': self.next_hop,
                'distances': self.distances,
                'bfs_preferred_edges': self.bfs_preferred_edges,
                'edge2destination': self.edge2destination,
            })
            return base

        def save(self, path: str) -> None:
            data = pickle.dumps(self._to_payload(), protocol=pickle.HIGHEST_PROTOCOL)
            with open(path, 'wb') as f:
                f.write(gzip.compress(data))

        @classmethod
        def from_payload(cls, payload: dict) -> Navigator.FullMetadata:
            assert payload.get('version', 1) == 1, "unsupported FullMetadata version"
            bc = payload['block_compacted']
            block_compacted = array(bc['typecode'])
            block_compacted.frombytes(bc['bytes'])
            assert len(block_compacted) == bc['len']
            return cls(
                name2id=payload['name2id'],
                id2name=payload['id2name'],
                topo_hash=payload['topo_hash'],
                node_volume=payload['node_volume'],
                edges=payload['edges'],
                forward_map=payload['forward_map'],
                wildcard_interfaces=payload['wildcard_interfaces'],
                reverse_map=payload['reverse_map'],
                block_of=payload['block_of'],
                blocks=payload['blocks'],
                block_volume=payload['block_volume'],
                next_hop=payload['next_hop'],
                distances=payload['distances'],
                block_compacted=block_compacted,
                block_uncompacted_hops=cls._unpack_arr_list(payload['block_uncompacted_hops']),
                block_compacted_hops=cls._unpack_arr_list(payload['block_compacted_hops']),
                block_except_nodes=cls._unpack_arr_list(payload['block_except_nodes']),
                block_except_hops=cls._unpack_arr_list(payload['block_except_hops']),
                bfs_preferred_edges=payload['bfs_preferred_edges'],
                edge2destination=payload['edge2destination'],
            )

    @staticmethod
    def compile_metadata(interfaces: list[Interface], cached_metadata: Optional[FullMetadata] = None,
                         log_output: bool = True) -> FullMetadata:

        # noinspection PyShadowingNames
        def build_idmap(interfaces: list[Interface],
                        cached_metadata: Optional[Navigator.FullMetadata]) -> tuple[dict[str, int], list[str]]:
            if cached_metadata is None:
                # use sort to ensure stable id assignment
                names = sorted([s.name for s in interfaces])
                name2id = {n: i for i, n in enumerate(names)}
                id2name = deepcopy(names)
            else:
                name2id: dict[str, int] = cached_metadata.name2id
                id2name: list[str] = cached_metadata.id2name
                new_names: list[str] = sorted([s.name for s in interfaces if s.name not in name2id])
                for n in new_names:
                    name2id[n] = len(id2name)
                    id2name.append(n)
            return name2id, id2name

        # noinspection PyShadowingNames
        def build_graph(interfaces: list[Interface], name2id: dict[str, int]) -> tuple[
            int, set[tuple[int, int]], list[list[int]], list[list[int]], set[int]]:
            """
            Args:
                interfaces: list of Interface specs
                name2id: interface_name -> interface_id

            Returns:
                interface_volume, edges, forward_map, reverse_map, wildcard_interfaces

            Build the directed graph from interface specs.
            Wildcard edges (destination "*") are excluded from the static graph.
            """

            interface_volume = len(name2id)
            forward_map = [[] for _ in range(interface_volume)]
            reverse_map = [[] for _ in range(interface_volume)]

            edges = set()
            wildcard_interfaces = set()  # Track interfaces with wildcard actions
            # works as a set of existing edges and to avoid duplicate edges

            for spec in interfaces:
                spec_id = name2id[spec.name]
                for destination_name, act in spec.actions.items():
                    # Skip wildcard actions - they don't create static edges
                    if destination_name == "*":
                        wildcard_interfaces.add(spec_id)
                        continue

                    destination_id = name2id[destination_name]
                    if (spec_id, destination_id) not in edges:
                        edges.add((spec_id, destination_id))
                        forward_map[spec_id].append(destination_id)
                        reverse_map[destination_id].append(spec_id)

            for spec_id in range(interface_volume):
                # sort to ensure stable map (for hash computation)
                forward_map[spec_id].sort()
                reverse_map[spec_id].sort()
            return interface_volume, edges, forward_map, reverse_map, wildcard_interfaces

        # noinspection PyShadowingNames
        def compute_topo_hash(interface_volume: int, edges: set[tuple[int, int]]) -> str:
            """
            Args:
                interface_volume: number of interfaces
                edges: set of (u,v) edges

            Returns:
                Hashes of (interface_volume + sorted edges)
            """
            data = bytearray()
            data.extend(interface_volume.to_bytes(4, 'little'))
            for (u, v) in sorted(edges):
                data.extend(u.to_bytes(4, 'little'))
                data.extend(v.to_bytes(4, 'little'))
            return hashlib.sha256(data).hexdigest()

        # noinspection PyShadowingNames
        def reverse_bfs(node_volume: int, reverse_map: list[list[int]]) \
            -> tuple[list[list[int]], list[list[int]], dict[int, set[tuple[int, int]]]]:
            """
                Args:
                    node_volume: number of nodes
                    reverse_map: node_id -> [previous_node_ids]
                Returns:
                    Generate original next_hop, distances and bfs_preferred_edges mapping.

                Perform Reverse-BFS from each node m in the reverse graph to compute
                the next hop and distance for all nodes to reach m.
            """

            next_hop = [[-1] * node_volume for _ in range(node_volume)]
            distances = [[INF] * node_volume for _ in range(node_volume)]
            bfs_preferred_edges: dict[int, set[tuple[int, int]]] = {m: set() for m in range(node_volume)}
            # bfs_preferred_edges stores the tree edges used by BFS for each destination m
            # this helps build edge2destination mapping and incremental updates
            # tree_edge_by_dist: m -> set of (u,v) edges in BFS tree towards m

            # TODO: entry node validation, detect unreachable nodes
            for m in range(node_volume):
                dq = deque()
                dq.append(m)
                distances[m][m] = 0
                while dq:
                    x = dq.popleft()
                    for u in reverse_map[x]:  # u -> x
                        if distances[m][u] == INF:  # first time visit: closest path
                            distances[m][u] = distances[m][x] + 1
                            next_hop[m][u] = x  # from u to m, next hop is x
                            bfs_preferred_edges[m].add((u, x))
                            dq.append(u)
                next_hop[m][m] = m  # we define next_hop to self as self
            return next_hop, distances, bfs_preferred_edges

        # noinspection PyShadowingNames
        def split_blocks(node_volume: int,
                         forward_map: list[list[int]],
                         reverse_map: list[list[int]],
                         block_size: int = 32) -> tuple[list[int], list[list[int]], int, int]:
            """
                Args:
                    node_volume: number of nodes
                    forward_map: node_id -> [next_node_ids]
                    reverse_map: node_id -> [previous_node_ids]
                    block_size: desired block size
                Returns:
                    block_of, blocks, blocks_volume, block_size

                Split nodes into blocks using BFS order for further compression.
            """

            # run bfs starting from node 0 to get a traversal order
            # we use the order to build blocks, which helps compression edges
            # (the nodes in the same block are likely to be close in the graph
            # so that their paths to far places are similar)
            undirected_map = [sorted(set(forward_map[u] + reverse_map[u])) for u in range(node_volume)]
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

        # noinspection PyShadowingNames
        def compact_block(node_volume: int,
                          block_of: list[int],
                          block_volume: int,
                          next_hop: list[list[int]]):
            """
            Args:
                node_volume: number of nodes
                block_of: node_id -> block_id
                block_volume: number of blocks
                next_hop: destination_id -> [current_id -> next_hop_id]

            Returns:
                block_compacted, block_uncompacted_hops, block_compacted_hops, block_except_nodes, block_except_hops

            Compact the next_hop columns using block-wise mode + exceptions.
            For each destination, we count the mode hop per block,
            then build a default array and an exception list.
            We estimate the storage cost of both direct and compacted formats,
            and choose the better one.
            """

            block_compacted: array = array('b', [0] * node_volume)
            block_uncompacted_hops: list[Optional[array]] = [None] * node_volume
            block_compacted_hops: list[Optional[array]] = [None] * node_volume
            block_except_nodes: list[Optional[array]] = [None] * node_volume
            block_except_hops: list[Optional[array]] = [None] * node_volume

            for dest_id in range(node_volume):
                # count the mode hop per block
                hop_counts_per_block: dict[int, Counter] = defaultdict(Counter)
                for n in range(node_volume):
                    block_id = block_of[n]
                    hop_counts_per_block[block_id][next_hop[dest_id][n]] += 1

                default_arr = array('h', [-1] * block_volume)
                for block_id in range(block_volume):
                    cnt = hop_counts_per_block.get(block_id)
                    if cnt and len(cnt) > 0:
                        mode_val, _ = cnt.most_common(1)[0]
                        default_arr[block_id] = int(mode_val)
                    else:
                        default_arr[block_id] = -1
                # build exception lists
                ex_nodes = array('h')
                ex_hops = array('h')
                exception_node_volume = 0
                for n in range(node_volume):
                    block_id = block_of[n]
                    hop = next_hop[dest_id][n]
                    if hop != default_arr[block_id]:
                        ex_nodes.append(n)
                        ex_hops.append(int(hop))
                        exception_node_volume += 1

                # estimate storage cost and choose the better one
                bytes_direct = 2 * node_volume
                bytes_region = 2 * block_volume + 4 * exception_node_volume

                if bytes_region < bytes_direct:
                    # use compacted block is better
                    block_compacted[dest_id] = True
                    block_compacted_hops[dest_id] = default_arr
                    block_except_nodes[dest_id] = ex_nodes
                    block_except_hops[dest_id] = ex_hops
                    block_uncompacted_hops[dest_id] = None
                else:
                    # use direct hops (next_hops) is better
                    block_compacted[dest_id] = False
                    block_uncompacted_hops[dest_id] = array('h',
                                                            (int(next_hop[dest_id][n]) for n in range(node_volume)))
                    block_compacted_hops[dest_id] = None
                    block_except_nodes[dest_id] = None
                    block_except_hops[dest_id] = None

            return block_compacted, block_uncompacted_hops, block_compacted_hops, block_except_nodes, block_except_hops

        def debug_print(*args, **kwargs):
            if log_output:
                print(*args, **kwargs)

        name2id, id2name = build_idmap(interfaces, cached_metadata)

        node_volume, edges, forward_map, reverse_map, wildcard_interfaces = build_graph(interfaces, name2id)
        topo_hash = compute_topo_hash(node_volume, edges)

        if node_volume == 0:
            raise ValueError("Cannot build Navigator metadata with zero interfaces.")

        # if the topology is unchanged, we could use cached metadata safely
        if cached_metadata is not None and topo_hash == cached_metadata.topo_hash:
            debug_print("[build] topology unchanged, use cached metadata")
            return cached_metadata

        ################ Increment build ################
        # we do an incremental rebuild when
        # 1.the node volume is unchanged AND
        # 2.the number of changed edges is small(less than 1/3 of node volume)
        increment_available = False
        if cached_metadata is not None and cached_metadata.node_volume == node_volume:
            added_edges = edges - cached_metadata.edges
            removed_edges = cached_metadata.edges - edges
            if len(added_edges) + len(removed_edges) <= max(1, node_volume // 3):
                increment_available = True

        if increment_available:
            added_edges = edges - cached_metadata.edges
            removed_edges = cached_metadata.edges - edges
            debug_print(f"[build] incremental rebuild: +{len(added_edges)} -{len(removed_edges)}")
            debug_print(f"[build] Added edges: {[(id2name[u], id2name[v]) for u, v in added_edges]}")
            debug_print(f"[build] Removed edges: {[(id2name[u], id2name[v]) for u, v in removed_edges]}")

            # copy from cache
            next_hop = [row[:] for row in cached_metadata.next_hop]
            distances = [row[:] for row in cached_metadata.distances]
            bfs_preferred_edges = {m: set(edges) for m, edges in
                                   cached_metadata.bfs_preferred_edges.items()}
            edge2destination = {edge: set(destination) for edge, destination in
                                cached_metadata.edge2destination.items()}

            # a set of destination ids that need to recompute BFS tree
            affected_destinations: set[int] = set()

            for edge in removed_edges:
                for destination_id in edge2destination.get(edge, set()):
                    affected_destinations.add(destination_id)
            for (u, v) in added_edges:
                for destination_id in range(node_volume):
                    if cached_metadata.distances[destination_id][v] < INF and 1 + \
                        cached_metadata.distances[destination_id][v] < \
                        cached_metadata.distances[destination_id][u]:
                        affected_destinations.add(destination_id)

            debug_print(f"[build] affected destinations: {affected_destinations}")

            for destination_id in affected_destinations:
                # clean up old edge2destination cache of affected destinations
                for edge in bfs_preferred_edges.get(destination_id, set()):
                    if edge in edge2destination and destination_id in edge2destination[edge]:
                        edge2destination[edge].remove(destination_id)
                        if not edge2destination[edge]:
                            edge2destination.pop(edge, None)

                # re-run Reverse-BFS for affected destinations
                dq = deque([destination_id])
                distances[destination_id] = [INF] * node_volume
                distances[destination_id][destination_id] = 0
                next_hop[destination_id] = [-1] * node_volume
                bfs_preferred_edges[destination_id] = set()
                while dq:
                    x = dq.popleft()
                    for uu in reverse_map[x]:
                        if distances[destination_id][uu] == INF:
                            distances[destination_id][uu] = distances[destination_id][x] + 1
                            next_hop[destination_id][uu] = x
                            bfs_preferred_edges[destination_id].add((uu, x))
                            edge2destination.setdefault((uu, x), set()).add(destination_id)
                            dq.append(uu)
                next_hop[destination_id][destination_id] = destination_id

            # use cached block structure, no need to split again (no nodes added_edges/removed_edges)
            # we only need to compact the blocks again
            block_of, blocks, blocks_volume = cached_metadata.block_of, cached_metadata.blocks, cached_metadata.block_volume
            block_compacted, block_uncompacted_hops, block_compacted_hops, block_except_nodes, block_except_hops = compact_block(
                node_volume, block_of, blocks_volume, next_hop
            )
        else:
            ################ Full build ################
            debug_print("[build] full build")

            # run reverse-bfs for all nodes to build next_hop
            next_hop, distances, bfs_preferred_edges = reverse_bfs(node_volume, reverse_map)

            # split nodes into blocks to compress next_hop columns
            block_of, blocks, blocks_volume, block_size = split_blocks(node_volume, forward_map, reverse_map)
            debug_print(f"[build] chosen block_size={block_size}, blocks_volume={blocks_volume}")
            (block_compacted, block_uncompacted_hops,
             block_compacted_hops, block_except_nodes, block_except_hops) = compact_block(
                node_volume, block_of, blocks_volume, next_hop
            )

            edge2destination: dict[tuple[int, int], set[int]] = defaultdict(set)
            for destination_id, tree_edges in bfs_preferred_edges.items():
                for tree_edge in tree_edges:
                    edge2destination[tree_edge].add(destination_id)

        return Navigator.FullMetadata(
            name2id=name2id, id2name=id2name, topo_hash=topo_hash,
            node_volume=node_volume, edges=edges, forward_map=forward_map,
            wildcard_interfaces=wildcard_interfaces,
            reverse_map=reverse_map,
            block_of=block_of, blocks=blocks, block_volume=blocks_volume,
            next_hop=next_hop, distances=distances,
            block_compacted=block_compacted,
            block_uncompacted_hops=block_uncompacted_hops,
            block_compacted_hops=block_compacted_hops,
            block_except_nodes=block_except_nodes,
            block_except_hops=block_except_hops,
            bfs_preferred_edges=bfs_preferred_edges, edge2destination=edge2destination
        )

    @staticmethod
    def load_metadata(path: str) -> Navigator.BaseMetadata:
        with open(path, 'rb') as f:
            payload = pickle.loads(gzip.decompress(f.read()))
        kind = payload.get('kind', 'base')
        if kind == 'full':
            return Navigator.FullMetadata.from_payload(payload)
        elif kind == 'base':
            return Navigator.BaseMetadata.from_payload(payload)
        else:
            raise ValueError(f"unknown metadata kind: {kind}")

    baas_thread: Baas_thread
    metadata: BaseMetadata
    _edge2action: dict[tuple[int, int | str], Callable[[Baas_thread], None]]  # Allow int or "*" for wildcard
    _validators: list[None | Callable[[Baas_thread], bool]]

    # maintain a scan order to optimize resolve_current_interface (move-to-front)
    _scan_order: list[int]

    def static_next_hop(self, current_id: int, destination_id: int) -> int:
        compacted = self.metadata.block_compacted[destination_id]
        if not compacted:
            # directly returns from block_uncompacted_hops
            row = self.metadata.block_uncompacted_hops[destination_id]
            # row is array('h'): current_id -> next_hop_id
            return row[current_id] if row is not None else -1
        else:
            # use block_compacted_hops + block_except_* to lookup
            # first check exceptions
            exc_nodes = self.metadata.block_except_nodes[destination_id]
            if exc_nodes is not None and len(exc_nodes) > 0:
                pos = bisect.bisect_left(exc_nodes, current_id)
                if pos < len(exc_nodes) and exc_nodes[pos] == current_id:
                    return self.metadata.block_except_hops[destination_id][pos]
            block_id = self.metadata.block_of[current_id]
            defaults = self.metadata.block_compacted_hops[destination_id]
            return defaults[block_id] if defaults is not None else -1

    def fallback_next_hops(self, current_id: int, destination_id: int,
                           skip_edges: Optional[set[tuple[int, int]]] = None) -> None | list[int]:
        if current_id == destination_id:
            return []
        path = []
        interface_volume = self.metadata.node_volume
        dq = deque([current_id])
        parent = [-1] * interface_volume
        seen = [False] * interface_volume
        seen[current_id] = True
        while dq:
            u = dq.popleft()
            for next_node_id in self.metadata.forward_map[u]:
                if skip_edges is not None and (u, next_node_id) in skip_edges:
                    continue
                if not seen[next_node_id]:
                    seen[next_node_id] = True
                    parent[next_node_id] = u
                    if next_node_id == destination_id:
                        # arrived the destination, reconstruct the path
                        cur = next_node_id
                        while parent[cur] != current_id:
                            path.append(cur)
                            cur = parent[cur]
                        path.append(cur)
                        path.reverse()
                        return path
                    dq.append(next_node_id)
        return None

    def rebuild(self, interfaces: list[Interface]) -> None:
        if isinstance(self.metadata, Navigator.FullMetadata):
            self.metadata = Navigator.compile_metadata(interfaces, cached_metadata=self.metadata)
        else:
            self.metadata = Navigator.compile_metadata(interfaces)
        self.update_actions_features(interfaces)
        self._scan_order = list(range(self.metadata.node_volume))

    def update_actions_features(self, interfaces: list[Interface]) -> None:
        self._edge2action = {}
        self._validators = [None] * self.metadata.node_volume

        for interface in interfaces:
            interface_id = self.metadata.name2id[interface.name]
            self._validators[interface_id] = interface.validate
            for destination_name, act in interface.actions.items():
                if act is not None:
                    if destination_name == "*":
                        # Store wildcard action with special key (interface_id, "*")
                        self._edge2action[(interface_id, "*")] = act
                    else:
                        destination_id = self.metadata.name2id[destination_name]
                        self._edge2action[(interface_id, destination_id)] = act

    def update_single_action(self, interface: int | str, destination: int | str,
                             new_action: Optional[Callable[[Baas_thread], None]]) -> None:
        interface_id = self.convert_to_id(interface)
        destination_id = self.convert_to_id(destination)
        if new_action is not None:
            self._edge2action[(interface_id, destination_id)] = new_action
        else:
            self._edge2action.pop((interface_id, destination_id), None)

    def update_single_features(self, interface: int | str,
                               new_features: list[Callable[[Baas_thread], bool]]) -> None:
        interface_id = self.convert_to_id(interface)
        combined_validator = lambda baas_thread: all(feature(baas_thread) for feature in new_features)
        self._validators[interface_id] = combined_validator

    def resolve_current_interface_id(self, priority: Optional[list[int]] = None,
                                     max_retries: int = 20) -> Optional[int]:
        for _ in range(max_retries):
            self.baas_thread.update_screenshot_array()
            # 1) preferred candidates (e.g., source/destination), check in order
            seen: set[int] = set()
            if priority:
                for interface_id in priority:
                    if 0 <= interface_id < self.metadata.node_volume and interface_id not in seen:
                        seen.add(interface_id)
                        if self._validators[interface_id](self.baas_thread):
                            return interface_id

            # 2) full scan optimized by _scan_order, move-to-front on hit
            for index, interface_id in enumerate(self._scan_order):
                if interface_id in seen:
                    continue
                if self._validators[interface_id](self.baas_thread):
                    # move-to-front
                    if index > 0:
                        self._scan_order.pop(index)
                        self._scan_order.insert(0, interface_id)
                    return interface_id

            # if not resolved, wait 300ms and retry
            time.sleep(0.3)
            print("[resolve_current_interface_id] retrying...")
        return None

    def resolve_current_interface(self, priority: Optional[list[int]] = None) -> Optional[str]:
        interface_id = self.resolve_current_interface_id(priority)
        return self.metadata.id2name[interface_id] if interface_id is not None else None

    def convert_to_id(self, interface: int | str) -> int:
        """
        Args:
            interface: interface id or name

        Returns:
            int: interface id

        Raises:
            ValueError: if interface name is not defined in metadata
        """

        if isinstance(interface, str):
            node_id = self.metadata.name2id.get(interface)
            if node_id is None:
                raise ValueError(f"Unknown interface name: {interface}")
            return node_id
        else:
            return interface

    def convert_to_name(self, interface: int | str) -> str:
        """
        Args:
            interface: interface id or name

        Returns:
            str: interface name

        Raises:
            ValueError: if interface id is invalid
        """

        if isinstance(interface, int):
            if 0 <= interface < self.metadata.node_volume:
                return self.metadata.id2name[interface]
            else:
                raise ValueError(f"Interface id invalid: {interface}")
        else:
            return interface

    def goto(self, destination: int | str, current: Optional[int | str] = None,
             deprecated_path: Optional[set[tuple[int, int]]] = None,
             log_output: bool = True, max_retries: int = 10,
             tentative_click: bool = True) -> bool:
        """
        Navigate from current interface to destination interface.

        Refactored to iterative implementation (non-recursive).
        Supports wildcard interfaces that have "*" actions leading to unknown destinations.
        """

        def debug_print(*args, **kwargs):
            if log_output:
                print(*args, **kwargs)

        # noinspection PyShadowingNames
        def build_static_path(current_id: int, destination_id: int) -> Optional[list]:
            """Build path using static next-hop lookups. Returns None if no path exists."""
            static_path = []
            tmp_id = current_id
            while tmp_id != destination_id:
                next_node_id = self.static_next_hop(tmp_id, destination_id)
                if next_node_id == -1:
                    return None  # No static path exists
                static_path.append(next_node_id)
                tmp_id = next_node_id
            return static_path

        # noinspection PyShadowingNames
        def tentative_resolve_id(priority: Optional[list[int]] = None) -> int:
            """Resolve current interface ID with optional tentative click recovery."""
            resolved_id: int = self.resolve_current_interface_id(priority)
            while resolved_id is None:
                if tentative_click:
                    self.baas_thread.click(1238, 45)
                    time.sleep(2)
                    resolved_id = self.resolve_current_interface_id(priority)
                else:
                    raise RuntimeError("cannot resolve current node during path execution")
            return resolved_id

        # Initialize
        if deprecated_path is None:
            deprecated_path = set()

        # If not defined current interface, resolve it
        if current is None:
            current = tentative_resolve_id()

        # Convert current/destination to ids
        current_id = self.convert_to_id(current)
        destination_id = self.convert_to_id(destination)

        # Main iterative navigation loop
        while current_id != destination_id:
            # Build or rebuild path from current to destination
            path = build_static_path(current_id, destination_id) if len(deprecated_path) == 0 \
                else self.fallback_next_hops(current_id, destination_id, deprecated_path)

            if path is None:
                # Check if current is a wildcard interface
                if current_id in self.metadata.wildcard_interfaces:
                    wildcard_action = self._edge2action.get((current_id, "*"))
                    if wildcard_action is not None:
                        debug_print(
                            f"[goto] No path available from {self.metadata.id2name[current_id]}, using wildcard action")
                        try:
                            wildcard_action(self.baas_thread)
                            time.sleep(2)
                            current_id = tentative_resolve_id()
                            debug_print(
                                f"[goto] After wildcard action, arrived at {self.metadata.id2name[current_id]}")
                            # Clear deprecated_path and try again from new position
                            deprecated_path = set()
                            continue
                        except Exception as e:
                            debug_print(f"[goto] Wildcard action failed: {e}")
                            raise RuntimeError("No available path found and wildcard action failed.")
                raise RuntimeError("[goto][error] No available path found "
                                   + " (after deprecating edges)" if len(deprecated_path) != 0 else "")

            # Execute path
            debug_print(
                f"Running path: {[self.metadata.id2name[nid] for nid in path]}, from {self.metadata.id2name[current_id]} to {self.metadata.id2name[destination_id]}")
            if len(deprecated_path) != 0:
                debug_print(
                    f"Deprecated edges: {[(self.metadata.id2name[u], self.metadata.id2name[v]) for u, v in deprecated_path]}")

            for next_node_id in path:
                # Get action for this hop
                action = self._edge2action.get((current_id, next_node_id))
                if action is None:
                    debug_print(
                        f"[goto][warning] no action defined from {self.metadata.id2name[current_id]} to {self.metadata.id2name[next_node_id]}")
                    # Deprecate this edge and retry
                    deprecated_path.add((current_id, next_node_id))
                    break  # Break to outer loop to rebuild path

                # Execute action with retries
                action_succeeded = False
                for _ in range(max_retries):
                    try:
                        action(self.baas_thread)
                        time.sleep(2)
                        resolved_id: int = self.resolve_current_interface_id([next_node_id, current_id])
                        if resolved_id != current_id:
                            action_succeeded = True
                            break
                    except Exception as e:
                        debug_print(
                            f"error executing action from {self.metadata.id2name[current_id]} to {self.metadata.id2name[next_node_id]}: {e}")
                        break

                if not action_succeeded:
                    # Action failed - deprecate edge and retry
                    deprecated_path.add((current_id, next_node_id))
                    break  # Break to outer loop to rebuild path

                # Verify we arrived at expected destination
                resolved_id = tentative_resolve_id([next_node_id, current_id].extend(self.metadata.wildcard_interfaces))
                if resolved_id == current_id:
                    debug_print(
                        f"warning: path blocked from {self.metadata.id2name[current_id]} to {self.metadata.id2name[next_node_id]}, recalculating...")
                    deprecated_path.add((current_id, next_node_id))
                    break  # Break to outer loop to rebuild path
                elif resolved_id != next_node_id:
                    debug_print(
                        f"warning: deviated from planned path at {self.metadata.id2name[current_id]}, arrived at {self.metadata.id2name[resolved_id]}, recalculating...")
                    # Update current position and rebuild path
                    current_id = resolved_id
                    break  # Break to outer loop to rebuild path

                # Successfully moved to next node
                current_id = resolved_id

                # If we reached destination, we're done
                if current_id == destination_id:
                    return True

        # Reached destination
        return True

    def __init__(self, baas_thread: Baas_thread, interfaces: list[Interface],
                 metadata: Optional[BaseMetadata] = None):
        self.baas_thread = baas_thread
        self.metadata = metadata if metadata else Navigator.compile_metadata(interfaces, cached_metadata=None)
        self.update_actions_features(interfaces)
        self._scan_order = list(range(self.metadata.node_volume))
