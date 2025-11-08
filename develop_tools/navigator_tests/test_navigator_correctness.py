"""Navigator correctness & robustness navigator_tests.

Coverage:
1. Incremental rebuild shortcut edge reduces or maintains distance.
2. Failing edge fallback on handcrafted small graph.
3. Mixed distribution batch queries (50% no failure, 30% single broken edge, 20% multi broken edges) using goto() on a large random graph.
4. Reverse BFS invariants (next_hop consistency, distance correctness) on larger graph.
"""
from __future__ import annotations
import random
import unittest
from functools import partial
from typing import Optional

from core.navigator import NodeSpec, compile_specs, lookup_next_hop
from develop_tools.navigator_tests import generator as gen

generate_specs = gen.generate_specs
set_current_node = gen.set_current_node

# ---------------- Demo-like helpers (simplified) ----------------

def resolve_current_node(index_instance, priority: Optional[list[int]] = None) -> Optional[int]:
    node_volume = index_instance.node_volume
    seen: set[int] = set()
    if priority:
        for node_id in priority:
            if 0 <= node_id < node_volume and node_id not in seen:
                seen.add(node_id)
                validator = index_instance.validators[node_id]
                if validator and validator():
                    return node_id
    for idx, node_id in enumerate(index_instance.scan_order):
        if node_id in seen:
            continue
        validator = index_instance.validators[node_id]
        if validator and validator():
            if idx > 0:
                index_instance.scan_order.pop(idx)
                index_instance.scan_order.insert(0, node_id)
            return node_id
    return None


def fallback_next_hop_wrapper(index, current_id: int, destination_id: int, deprecated: set[tuple[int, int]]):
    from core.navigator import fallback_next_hop  # local import to avoid unused earlier
    return fallback_next_hop(index, current_id, destination_id, deprecated)


def goto(index, current_id: int, destination_id: int, path=None, deprecated=None):
    if not path:
        if deprecated:
            fb = fallback_next_hop_wrapper(index, current_id, destination_id, deprecated)
            if fb is None:
                raise RuntimeError("No path after deprecating edges")
            return goto(index, current_id, destination_id, fb, deprecated)
        static = []
        tmp = current_id
        guard = 0
        while tmp != destination_id and guard <= index.node_volume:
            nxt = lookup_next_hop(index, tmp, destination_id)
            if nxt == -1:
                raise RuntimeError("No static path hop")
            static.append(nxt)
            tmp = nxt
            guard += 1
        if tmp != destination_id:
            raise RuntimeError("Static chain did not reach destination")
        return goto(index, current_id, destination_id, static, deprecated)
    for nxt in path:
        action = index.edge_handle.get((current_id, nxt))
        if action is None:
            return False
        try:
            action()
        except Exception:
            if deprecated is None:
                deprecated = set()
            deprecated.add((current_id, nxt))
            return goto(index, current_id, destination_id, deprecated=deprecated)
        resolved = resolve_current_node(index, [nxt, current_id])
        if resolved is None:
            raise RuntimeError("Cannot resolve current node")
        elif resolved == current_id:  # feature validation failure
            if deprecated is None:
                deprecated = set()
            deprecated.add((current_id, nxt))
            return goto(index, current_id, destination_id, deprecated=deprecated)
        elif resolved != nxt:  # mis-detection; deprecate and retry
            if deprecated is None:
                deprecated = set()
            deprecated.add((current_id, nxt))
            return goto(index, current_id, destination_id, deprecated=deprecated)
        current_id = resolved
    return True

# ---------------- Tests ----------------

class TestNavigatorCorrectness(unittest.TestCase):
    def test_incremental_rebuild_shortcut(self):
        # Randomize src/dst, add shortcut, and verify compiled distance matches BFS on modified specs
        from collections import deque
        INF = 10 ** 9
        for i in range(20):
            specs = generate_specs(num_nodes=120, edge_probability=0.02, seed=100 + i, failing_edge_ratio=0.0)
            index = compile_specs(specs, prev=None)

            rng = random.Random(1000 + i)
            src = rng.randrange(index.node_volume)
            dst = rng.randrange(index.node_volume)
            while src == dst:
                dst = rng.randrange(index.node_volume)

            original_dist = index.dist[dst][src]

            # Add direct shortcut edge (no-op action) using names from index mapping
            src_name = index.id2name[src]
            dst_name = index.id2name[dst]
            for s in specs:
                if s.name == src_name:
                    s.actions[dst_name] = (lambda d=dst_name: None)

            # Build adjacency from modified specs and compute expected hops via BFS
            neighbors = {s.name: list(s.actions.keys()) for s in specs}
            def bfs_hops(s_name: str, t_name: str) -> Optional[int]:
                if s_name == t_name:
                    return 0
                q = deque([(s_name, 0)])
                seen = {s_name}
                while q:
                    cur, d = q.popleft()
                    for nb in neighbors.get(cur, ()):
                        if nb == t_name:
                            return d + 1
                        if nb not in seen:
                            seen.add(nb)
                            q.append((nb, d + 1))
                return None

            expected_hops = bfs_hops(src_name, dst_name)
            expected_dist = expected_hops if expected_hops is not None else INF

            # Incremental rebuild
            new_index = compile_specs(specs, prev=index)
            new_dist = new_index.dist[dst][src]

            # Basic invariants: distance must not increase
            self.assertLessEqual(new_dist, original_dist, "Distance should not increase after adding shortcut")
            # Exact match with BFS expectation on the modified graph
            self.assertEqual(new_dist, expected_dist, f"Compiled distance {new_dist} != expected BFS distance {expected_dist} (seed {i}, src {src}, dst {dst})")
            # And the improvement equals expected improvement when both paths are finite
            if original_dist < INF and expected_dist < INF:
                self.assertEqual(original_dist - new_dist, original_dist - expected_dist,
                                 "Observed distance reduction mismatch the BFS-expected reduction")

    def test_failing_edge_fallback_small_graph(self):
        from develop_tools.navigator_tests.generator import generate_small_fallback_raise
        for seed in range(20):
            specs, start_name, dest_name = generate_small_fallback_raise(seed)
            index = compile_specs(specs, prev=None)
            # set start
            set_current_node(start_name)
            success = goto(index, index.name2id[start_name], index.name2id[dest_name])
            self.assertTrue(success)
            self.assertEqual(gen.CURRENT_NODE, dest_name)

    def test_mixed_failure_distribution_goto(self):
        successes_total = 0
        single_ok_total = 0
        multi_ok_total = 0
        for i in range(20):
            # Large-ish graph per seed
            specs = generate_specs(num_nodes=260, edge_probability=0.015, seed=200 + i, failing_edge_ratio=0.0)
            index = compile_specs(specs, prev=None)
            spec_map = {s.name: s for s in specs}

            total_queries = 50  # moderate per-seed to keep runtime reasonable
            single_fail_queries = int(total_queries * 0.30)
            multi_fail_queries = int(total_queries * 0.20)
            no_fail_queries = total_queries - single_fail_queries - multi_fail_queries

            rng = random.Random(99 + i)
            successes = 0
            single_fail_success = 0
            multi_fail_success = 0

            def build_static_path(idx, src_id, dst_id):
                path = []
                cur = src_id
                guard = 0
                while cur != dst_id and guard <= idx.node_volume:
                    nxt = lookup_next_hop(idx, cur, dst_id)
                    if nxt == -1:
                        return None
                    path.append(nxt)
                    cur = nxt
                    guard += 1
                return path if cur == dst_id else None

            # Helper to replace edge action with failing
            def patch_fail_edge(u_id, v_id):
                # Patch edge in-place on index.edge_handle to force a failure without recompiling
                u_name = index.id2name[u_id]
                v_name = index.id2name[v_id]
                key = (u_id, v_id)
                original = index.edge_handle.get(key)
                def _fail():
                    raise RuntimeError(f"forced failure {u_name}->{v_name}")
                index.edge_handle[key] = _fail
                return key, original

            queries_done = 0
            # Process no-failure queries
            while queries_done < no_fail_queries:
                src = rng.randrange(index.node_volume)
                dst = rng.randrange(index.node_volume)
                if src == dst:
                    continue
                set_current_node(index.id2name[src])
                try:
                    ok = goto(index, src, dst)
                except Exception:
                    ok = False
                if ok and gen.CURRENT_NODE == index.id2name[dst]:
                    successes += 1
                queries_done += 1

            # Single-failure queries
            sf_done = 0
            while sf_done < single_fail_queries:
                src = rng.randrange(index.node_volume)
                dst = rng.randrange(index.node_volume)
                if src == dst:
                    continue
                static_path = build_static_path(index, src, dst)
                if not static_path or len(static_path) < 2:
                    continue  # need at least one edge to break
                # reconstruct edges list
                path_edges = []
                cur = src
                for hop in static_path:
                    path_edges.append((cur, hop))
                    cur = hop
                chosen = None
                for (u,v) in path_edges:
                    if len(index.forward_map[u]) > 1:
                        chosen = (u,v)
                        break
                if chosen is None:
                    chosen = path_edges[0]
                (u,v) = chosen
                key, original = patch_fail_edge(u,v)
                set_current_node(index.id2name[src])
                try:
                    ok = goto(index, src, dst)
                except Exception:
                    ok = False
                # restore
                index.edge_handle[key] = original
                if ok and gen.CURRENT_NODE == index.id2name[dst]:
                    successes += 1
                    single_fail_success += 1
                sf_done += 1

            # Multi-failure queries
            mf_done = 0
            while mf_done < multi_fail_queries:
                src = rng.randrange(index.node_volume)
                dst = rng.randrange(index.node_volume)
                if src == dst:
                    continue
                static_path = build_static_path(index, src, dst)
                if not static_path or len(static_path) < 3:
                    continue  # need enough edges to break two
                # edges list
                path_edges = []
                cur = src
                for hop in static_path:
                    path_edges.append((cur, hop))
                    cur = hop
                # choose up to 2 edges with branching
                candidate_edges = [e for e in path_edges if len(index.forward_map[e[0]]) > 1]
                if len(candidate_edges) < 2:
                    candidate_edges = path_edges[:2]
                to_break = candidate_edges[:2]
                records = [patch_fail_edge(u,v) for (u,v) in to_break]
                set_current_node(index.id2name[src])
                try:
                    ok = goto(index, src, dst)
                except Exception:
                    ok = False
                # restore
                for (key, orig) in records:
                    index.edge_handle[key] = orig
                if ok and gen.CURRENT_NODE == index.id2name[dst]:
                    successes += 1
                    multi_fail_success += 1
                mf_done += 1

            self.assertGreaterEqual(successes, int(total_queries * 0.85), f"Overall success rate too low: {successes}/{total_queries} (seed {i})")
            self.assertGreaterEqual(single_fail_success, int(single_fail_queries * 0.7), f"Single-failure fallback success rate too low: {single_fail_success}/{single_fail_queries} (seed {i})")
            self.assertGreaterEqual(multi_fail_success, int(multi_fail_queries * 0.5), f"Multi-failure fallback success rate too low: {multi_fail_success}/{multi_fail_queries} (seed {i})")
            successes_total += successes
            single_ok_total += single_fail_success
            multi_ok_total += multi_fail_success

    def test_reverse_bfs_invariants(self):
        for i in range(20):
            specs = generate_specs(num_nodes=120, edge_probability=0.02, seed=300 + i, failing_edge_ratio=0.0)
            index = compile_specs(specs, prev=None)
            INF = 10 ** 9
            for m in range(index.node_volume):
                self.assertEqual(index.next_hop[m][m], m, "next_hop[m][m] must equal m")
                for u in range(index.node_volume):
                    nh = index.next_hop[m][u]
                    dist = index.dist[m][u]
                    if nh == -1:
                        self.assertEqual(dist, INF, "-1 hop implies INF distance")
                    else:
                        self.assertLess(dist, INF, "Defined hop implies finite distance")
                        # verify path chaining reaches destination in dist steps
                        cur = u
                        steps = 0
                        while cur != m and steps <= dist:
                            cur = index.next_hop[m][cur]
                            steps += 1
                        self.assertEqual(cur, m, "Chained next_hop did not reach destination")
                        self.assertEqual(steps, dist, "Chained path length must equal recorded distance")

    def test_no_move_failure_fallback_small_graph(self):
        """An edge fails silently (no move), requiring fallback to alternative route."""
        from develop_tools.navigator_tests.generator import generate_small_fallback_stay
        for seed in range(20):
            specs, start_name, dest_name = generate_small_fallback_stay(seed)
            index = compile_specs(specs, prev=None)
            set_current_node(start_name)
            success = goto(index, index.name2id[start_name], index.name2id[dest_name])
            self.assertTrue(success)
            self.assertEqual(gen.CURRENT_NODE, dest_name)

    def test_single_failure_optimal_path(self):
        """Inject a single failing edge on static path; verify total steps equals optimal BFS under avoid set."""
        for seed in range(20):
            rng = random.Random(1000 + seed)
            specs = generate_specs(num_nodes=200, edge_probability=0.02, seed=seed)
            index = compile_specs(specs, prev=None)
            # pick a src,dst with path length >=2
            def build_static(idx, s, t):
                path = []
                cur = s
                guard = 0
                while cur != t and guard <= idx.node_volume:
                    hop = lookup_next_hop(idx, cur, t)
                    if hop == -1:
                        return None
                    path.append(hop)
                    cur = hop
                    guard += 1
                return path if cur == t else None
            trials = 0
            found = False
            while trials < 200 and not found:
                src = rng.randrange(index.node_volume)
                dst = rng.randrange(index.node_volume)
                if src == dst:
                    trials += 1
                    continue
                sp = build_static(index, src, dst)
                if not sp or len(sp) < 2:
                    trials += 1
                    continue
                # choose a branching edge to fail if possible
                edges = []
                cur = src
                for h in sp:
                    edges.append((cur, h))
                    cur = h
                chosen = None
                for (u, v) in edges:
                    if len(index.forward_map[u]) > 1:
                        chosen = (u, v)
                        break
                if chosen is None:
                    trials += 1
                    continue
                found = True
            self.assertTrue(found, "Could not find suitable src/dst with branching edge")
            u, v = chosen
            # Patch failing edge
            key = (u, v)
            original = index.edge_handle[key]
            def _fail():
                raise RuntimeError("fail")
            index.edge_handle[key] = _fail
            # Trace execution
            executed = []
            # wrap other edges to record
            wrapped = {}
            for ekey, act in list(index.edge_handle.items()):
                if ekey == key:
                    continue
                def make_wrap(src_id=ekey[0], dst_id=ekey[1], fn=act):
                    def _w():
                        fn()
                        executed.append((src_id, dst_id))
                    return _w
                wrapped[ekey] = make_wrap()
            index.edge_handle.update(wrapped)
            # Run
            set_current_node(index.id2name[src])
            from core.navigator import fallback_next_hop
            try:
                ok = goto(index, src, dst)
            finally:
                # restore
                index.edge_handle[key] = original
                for ekey, act in wrapped.items():
                    # we can't restore original easily here; safe enough for this isolated test instance
                    pass
            self.assertTrue(ok)
            # executed steps before failure equals count until failure point
            # We can reconstruct the 'first failure' step count by simulating static until (u,v)
            static_prefix_steps = 0
            cur = src
            for h in sp:
                if (cur, h) == (u, v):
                    break
                static_prefix_steps += 1
                cur = h
            # expected rest length via BFS avoid set from u to dst
            avoid = {(u, v)}
            rest = fallback_next_hop(index, u, dst, avoid)
            self.assertIsNotNone(rest)
            expected_total = static_prefix_steps + len(rest)
            self.assertEqual(len(executed), expected_total, "Executed steps must equal optimal BFS under avoid set")

    def test_unreachable_after_failure_raises(self):
        """Static path exists A->B->D; B->D fails and no alternative -> should raise on fallback."""
        for seed in range(20):
            # Build minimal unreachable-after-failure graph
            CURRENT_NODE_global = {"value": ""}
            def set_current(name: str):
                CURRENT_NODE_global["value"] = name
            def feature(name: str):
                return CURRENT_NODE_global["value"] == name
            def act(dest: str):
                def _a():
                    set_current(dest)
                return _a
            def failing(dest: str):
                def _f():
                    raise RuntimeError("fail")
                return _f
            # Ensure weak connectivity by linking C->A (does not create alternative A->D path)
            specs = [
                NodeSpec("A", "A", [partial(feature, "A")], {"B": act("B")} ),
                NodeSpec("B", "B", [partial(feature, "B")], {"D": failing("D")} ),
                NodeSpec("C", "C", [partial(feature, "C")], {"A": act("A")} ),
                NodeSpec("D", "D", [partial(feature, "D")], {}),
            ]
            index = compile_specs(specs, prev=None)
            CURRENT_NODE_global["value"] = "A"
            with self.assertRaises(RuntimeError):
                goto(index, index.name2id["A"], index.name2id["D"])  # should raise on fallback None

if __name__ == "__main__":
    unittest.main()
