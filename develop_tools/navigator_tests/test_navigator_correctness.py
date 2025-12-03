"""Stability-focused correctness tests for Navigator class.

This replaces older tests. It stresses correctness under dynamic failures across
many random large graphs using Navigator's API directly.
"""
from __future__ import annotations
import random
import unittest
from typing import Optional
from pathlib import Path
import tempfile
import gzip
import pickle
from functools import partial

from core.navigator import Navigator
from develop_tools.navigator_tests import generator as gen

# Aliases
generate_specs = gen.generate_specs
set_current_node = gen.set_current_node

def _specs_from_metadata(meta: Navigator.BaseMetadata) -> list[Navigator.Interface]:
    """Build interfaces whose actions match metadata.forward_map so Navigator.goto can execute hops.
    Features read and write the test generator's CURRENT_NODE.
    """
    def _feature(name: str) -> bool:
        return gen.CURRENT_NODE == name
    def _make_action(dest_name: str):
        def _act():
            set_current_node(dest_name)
        return _act
    specs: list[Navigator.Interface] = []
    for uid, name in enumerate(meta.id2name):
        actions = {}
        for vid in meta.forward_map[uid]:
            actions[meta.id2name[vid]] = _make_action(meta.id2name[vid])
        specs.append(Navigator.Interface(
            name=name,
            description=name,
            features=[partial(_feature, name)],
            actions=actions,
        ))
    return specs

class TestNavigatorStability(unittest.TestCase):
    def test_large_graph_stability_correctness(self):
        """Repeated experiments over large graphs validating fallback correctness.

        For each iteration we:
        1) Generate random interfaces (~avg out-degree 6-7 via generator backbone + probability).
        2) Create Navigator(metadata) via full compile.
        3) Wrap edge actions once to consult dynamic failure sets (raise vs stay); no rebuild during queries.
        4) Execute shuffled queries (50% none, 30% single broken, 20% multi broken edges) using Navigator.goto.
        5) Expected outcome computed from metadata distances + fallback_next_hops under broken set.
        """
        ITERATIONS = 25
        NUM_NODES = 400
        TARGET_DEG = 6.5
        edge_probability = TARGET_DEG / (NUM_NODES - 1)
        INF = 10 ** 9
        BASE_SEED = 915731
        QUERIES_PER_ITER = 90

        for it in range(ITERATIONS):
            # Build connected metadata; retry if weak connectivity assertion fails
            navigator: Optional[Navigator] = None
            for attempt in range(10):
                specs = generate_specs(num_nodes=NUM_NODES, edge_probability=edge_probability,
                                       seed=BASE_SEED + it * 997 + attempt, failing_edge_ratio=0.0)
                try:
                    navigator = Navigator(specs, metadata=None)
                    break
                except AssertionError:  # not single weakly connected component
                    navigator = None
                    continue
            self.assertIsNotNone(navigator, f"Failed to compile connected graph iter={it}")
            meta = navigator.metadata

            broken_raise: set[tuple[int, int]] = set()
            broken_stay: set[tuple[int, int]] = set()

            # Wrap actions; keep original for restoration? (Not needed across queries since we only mutate wrappers referencing sets.)
            for (u, v), act in list(navigator._edge2action.items()):
                def make_wrapper(src=u, dst=v, fn=act):
                    def _wrapped():
                        edge = (src, dst)
                        if edge in broken_raise:
                            raise RuntimeError(f"forced failure {meta.id2name[src]}->{meta.id2name[dst]}")
                        if edge in broken_stay:
                            return  # stay (no movement)
                        fn()
                    return _wrapped
                navigator._edge2action[(u, v)] = make_wrapper()

            def static_path(src_id: int, dst_id: int) -> Optional[list[int]]:
                if src_id == dst_id:
                    return []
                path = []
                cur = src_id
                guard = 0
                while cur != dst_id and guard <= meta.node_volume:
                    hop = navigator.static_next_hop(cur, dst_id)
                    if hop == -1:
                        return None
                    path.append(hop)
                    cur = hop
                    guard += 1
                return path if cur == dst_id else None

            def expected_reachable(src_id: int, dst_id: int, broken: set[tuple[int, int]]) -> bool:
                if meta.distances[dst_id][src_id] == INF:
                    return False
                if not broken:
                    return True
                fb = navigator.fallback_next_hops(src_id, dst_id, broken)
                return fb is not None

            rng = random.Random(BASE_SEED * 13 + it * 271828)
            q_none = int(QUERIES_PER_ITER * 0.50)
            q_single = int(QUERIES_PER_ITER * 0.30)
            q_multi = QUERIES_PER_ITER - q_none - q_single
            plan = ["none"] * q_none + ["single"] * q_single + ["multi"] * q_multi
            rng.shuffle(plan)

            for q_index, category in enumerate(plan):
                broken_raise.clear(); broken_stay.clear()

                src = rng.randrange(meta.node_volume)
                dst = rng.randrange(meta.node_volume)
                retries = 0
                while dst == src and retries < 8:
                    dst = rng.randrange(meta.node_volume)
                    retries += 1

                broken_edges: list[tuple[int, int]] = []
                sp = static_path(src, dst)
                if category != "none" and sp:
                    # Build edges on static path
                    edge_list = []
                    cur = src
                    for h in sp:
                        edge_list.append((cur, h))
                        cur = h
                    branching = [e for e in edge_list if len(meta.forward_map[e[0]]) > 1]
                    candidates = branching if branching else edge_list
                    if category == "single" and candidates:
                        broken_edges.append(candidates[0])
                    elif category == "multi" and candidates:
                        broken_edges.extend(candidates[:2])

                for e in broken_edges:
                    (broken_raise if rng.random() < 0.5 else broken_stay).add(e)

                exp_ok = expected_reachable(src, dst, set(broken_edges))
                set_current_node(meta.id2name[src])

                if exp_ok:
                    ok = navigator.goto(src, dst, log_output=False)
                    self.assertTrue(ok, f"Expected success iter={it} q={q_index} cat={category}")
                    self.assertEqual(gen.CURRENT_NODE, meta.id2name[dst], f"Did not arrive iter={it} q={q_index}")
                else:
                    with self.assertRaises(RuntimeError, msg=f"Expected failure iter={it} q={q_index} cat={category}"):
                        navigator.goto(src, dst, log_output=False)

class TestNavigatorMetadataPersistence(unittest.TestCase):
    def _build_connected_navigator(self, num_nodes=200, seed=12345) -> Navigator:
        TARGET_DEG = 6.5
        edge_probability = TARGET_DEG / max(1, (num_nodes - 1))
        specs = generate_specs(num_nodes=num_nodes, edge_probability=edge_probability, seed=seed, failing_edge_ratio=0.0)
        nav = Navigator(specs)
        return nav

    def _sample_pairs(self, meta, k=30, seed=7):
        rng = random.Random(seed)
        pairs = []
        while len(pairs) < k:
            s = rng.randrange(meta.node_volume)
            t = rng.randrange(meta.node_volume)
            if s != t:
                pairs.append((s, t))
        return pairs

    def test_full_metadata_save_load_roundtrip(self):
        nav1 = self._build_connected_navigator(num_nodes=240, seed=101)
        meta1 = nav1.metadata
        pairs = self._sample_pairs(meta1, k=40)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "full.metadata"
            assert isinstance(meta1, Navigator.FullMetadata)
            meta1.save(str(p))
            self.assertTrue(p.exists() and p.stat().st_size > 0)
            loaded = Navigator.load_metadata(str(p))
            self.assertIsInstance(loaded, Navigator.FullMetadata)
            specs = _specs_from_metadata(loaded)
            nav2 = Navigator(specs, metadata=loaded)
            # Static next_hop equality for subset
            for dst in range(0, meta1.node_volume, max(1, meta1.node_volume // 10)):
                for src in range(0, meta1.node_volume, max(1, meta1.node_volume // 10)):
                    self.assertEqual(nav1.static_next_hop(src, dst), nav2.static_next_hop(src, dst))
            # Reachability parity (ignore original failing actions semantics)
            INF = 10 ** 9
            for src, dst in pairs:
                expected_reachable = (meta1.distances[dst][src] < INF)
                set_current_node(meta1.id2name[src])
                try:
                    ok2 = nav2.goto(src, dst, log_output=False)
                    reached2 = gen.CURRENT_NODE == meta1.id2name[dst]
                except RuntimeError:
                    ok2 = False; reached2 = False
                self.assertEqual(ok2 and reached2, expected_reachable,
                                 f"Mismatch reachable expectation after load src={src} dst={dst}")

    def test_base_runtime_metadata_save_load_and_fallback(self):
        nav = self._build_connected_navigator(num_nodes=220, seed=202)
        # Convert to runtime-only metadata and save
        base_meta = nav.metadata.to_base() if isinstance(nav.metadata, Navigator.FullMetadata) else nav.metadata
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "runtime.metadata"
            base_meta.save(str(p))
            self.assertTrue(p.exists() and p.stat().st_size > 0)
            # Load back base metadata
            loaded = Navigator.load_metadata(str(p))
            self.assertIsInstance(loaded, Navigator.BaseMetadata)
            # Rebuild navigator with base metadata
            specs = _specs_from_metadata(loaded)
            nav_rt = Navigator(specs, metadata=loaded)

        # Check static navigation on some pairs
        pairs = self._sample_pairs(nav_rt.metadata, k=35)
        for src, dst in pairs:
            set_current_node(nav_rt.metadata.id2name[src])
            try:
                ok = nav_rt.goto(src, dst, log_output=False)
                if ok:
                    self.assertEqual(gen.CURRENT_NODE, nav_rt.metadata.id2name[dst])
            except RuntimeError:
                # Allow unreachable cases (should raise)
                pass

        # Inject a single failing edge on a static path and validate fallback using runtime BFS
        # Find a pair with a non-trivial static path
        def build_static_path(nv: Navigator, s: int, t: int):
            cur = s
            path = []
            guard = 0
            while cur != t and guard <= nv.metadata.node_volume:
                nh = nv.static_next_hop(cur, t)
                if nh == -1:
                    return None
                path.append(nh)
                cur = nh
                guard += 1
            return path if cur == t else None

        rng = random.Random(555)
        tried = 0
        while tried < 200:
            s = rng.randrange(nav_rt.metadata.node_volume)
            t = rng.randrange(nav_rt.metadata.node_volume)
            if s == t:
                tried += 1; continue
            sp = build_static_path(nav_rt, s, t)
            if sp and len(sp) >= 2:
                # pick first edge
                u = s; v = sp[0]
                original = nav_rt._edge2action.get((u, v))
                if original is None:
                    tried += 1; continue
                def _fail():
                    raise RuntimeError("simulated fail")
                nav_rt._edge2action[(u, v)] = _fail
                set_current_node(nav_rt.metadata.id2name[s])
                try:
                    ok = nav_rt.goto(s, t, log_output=False)
                    # success only if fallback path exists
                    fb = nav_rt.fallback_next_hops(s, t, {(u, v)})
                    self.assertEqual(ok, fb is not None)
                    if fb is not None:
                        self.assertEqual(gen.CURRENT_NODE, nav_rt.metadata.id2name[t])
                except RuntimeError:
                    fb = nav_rt.fallback_next_hops(s, t, {(u, v)})
                    self.assertIsNone(fb)
                finally:
                    nav_rt._edge2action[(u, v)] = original
                break
            tried += 1
        else:
            self.fail("Could not find a suitable pair with a static path of length >=2 for fallback test")

    def test_load_corrupted_or_invalid_files(self):
        with tempfile.TemporaryDirectory() as td:
            # 1) Not a gzip file
            p1 = Path(td) / "bad1.metadata"
            p1.write_bytes(b"this-is-not-gzip")
            with self.assertRaises(Exception):
                Navigator.load_metadata(str(p1))
            # 2) Gzip but invalid pickle
            p2 = Path(td) / "bad2.metadata"
            p2.write_bytes(gzip.compress(b"not-a-pickle-payload"))
            with self.assertRaises(Exception):
                Navigator.load_metadata(str(p2))
            # 3) Valid pickle but unknown kind
            p3 = Path(td) / "bad3.metadata"
            payload = {'kind': 'unknown_kind', 'version': 1}
            p3.write_bytes(gzip.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)))
            with self.assertRaises(ValueError):
                Navigator.load_metadata(str(p3))
            # 4) Truncated gzip
            p4 = Path(td) / "bad4.metadata"
            data = gzip.compress(pickle.dumps({'kind': 'base', 'version': 1}))
            p4.write_bytes(data[:10])  # truncate
            with self.assertRaises(Exception):
                Navigator.load_metadata(str(p4))

if __name__ == '__main__':
    unittest.main()
