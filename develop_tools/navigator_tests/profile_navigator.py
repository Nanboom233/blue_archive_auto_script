"""Navigator profiling script (rebuilt for new Navigator class).

Features:
1. Build FullMetadata from generated specs; time the build.
2. Derive runtime BaseMetadata (to_base); time the conversion.
3. Memory comparison: direct next_hop vs compressed block representation, plus
   FullMetadata object vs BaseMetadata object, and serialized file sizes.
4. Save & load timings for both full and runtime metadata.
5. Baseline query execution (compressed static_next_hop) with timing.
6. Failing query execution (single + multi edge failures with raise/stay modes) with timing.
7. Coarse timing prints using time.perf_counter; external profiler can attach for deeper info.

Removed: baseline_direct query run (per requirement), but memory comparison between
uncompressed (next_hop) and compressed representation retained.
"""
from __future__ import annotations
import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import gzip
import pickle
import tempfile
import os
import math
import sys as _sys

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.navigator import Navigator
from develop_tools.navigator_tests import generator as gen

# ---------------- Format helpers ----------------

def format_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    idx = 0
    while x >= 1024 and idx < len(units) - 1:
        x /= 1024.0
        idx += 1
    return f"{x:.2f} {units[idx]}"

def safe_ratio(a: int, b: int) -> float:
    return a / b if b else float('inf')

def pct_reduction(original: int, reduced: int) -> float:
    if original <= 0:
        return 0.0
    return (original - reduced) * 100.0 / original

# ---------------- Spec generation wrapper ----------------

def generate_interfaces(num_nodes: int, edge_probability: Optional[float], seed: int, avg_out_degree: float) -> list[Navigator.Interface]:
    specs = gen.generate_specs(num_nodes=num_nodes, edge_probability=edge_probability, seed=seed, failing_edge_ratio=0.0, avg_out_degree=avg_out_degree)
    return specs

# ---------------- Dynamic failure injection ----------------

FailureMode = Dict[Tuple[int, int], str]  # (u_id,v_id) -> 'raise' | 'stay'

def wrap_actions_dynamic(nav: Navigator, failure_map: FailureMode) -> None:
    id2name = nav.metadata.id2name
    for (u, v), orig in list(nav._edge2action.items()):
        def make_wrapper(src=u, dst=v, fn=orig):
            def _wrapped():
                mode = failure_map.get((src, dst))
                if mode == 'raise':
                    raise RuntimeError(f"forced failure {id2name[src]}->{id2name[dst]}")
                elif mode == 'stay':
                    return  # stay at src
                else:
                    fn()  # perform movement
            return _wrapped
        nav._edge2action[(u, v)] = make_wrapper()

# ---------------- Query helpers ----------------

def static_path(nav: Navigator, src: int, dst: int) -> Optional[List[int]]:
    if src == dst:
        return []
    path = []
    cur = src
    guard = 0
    while cur != dst and guard <= nav.metadata.node_volume:
        hop = nav.static_next_hop(cur, dst)
        if hop == -1:
            return None
        path.append(hop)
        cur = hop
        guard += 1
    return path if cur == dst else None

# ---------------- Memory measurement ----------------

def _sizeof(obj) -> int:
    try:
        return _sys.getsizeof(obj)
    except Exception:
        return 0

def memory_breakdown_full(meta: Navigator.FullMetadata) -> Dict[str, int]:
    size = 0
    size += _sizeof(meta.name2id)
    size += _sizeof(meta.id2name)
    size += _sizeof(meta.forward_map) + sum(_sizeof(row) for row in meta.forward_map)
    size += _sizeof(meta.reverse_map) + sum(_sizeof(row) for row in meta.reverse_map)
    size += _sizeof(meta.edges)
    size += _sizeof(meta.block_of)
    size += _sizeof(meta.blocks) + sum(_sizeof(b) for b in meta.blocks)
    size += _sizeof(meta.block_compacted)
    size += _sizeof(meta.block_uncompacted_hops) + sum(_sizeof(a) for a in meta.block_uncompacted_hops if a is not None)
    size += _sizeof(meta.block_compacted_hops) + sum(_sizeof(a) for a in meta.block_compacted_hops if a is not None)
    size += _sizeof(meta.block_except_nodes) + sum(_sizeof(a) for a in meta.block_except_nodes if a is not None)
    size += _sizeof(meta.block_except_hops) + sum(_sizeof(a) for a in meta.block_except_hops if a is not None)
    size += _sizeof(meta.next_hop) + sum(_sizeof(row) for row in meta.next_hop)
    size += _sizeof(meta.distances) + sum(_sizeof(row) for row in meta.distances)
    size += _sizeof(meta.bfs_preferred_edges) + sum(_sizeof(v) for v in meta.bfs_preferred_edges.values())
    size += _sizeof(meta.edge2destination) + sum(_sizeof(v) for v in meta.edge2destination.values())
    return {"full_object_bytes": size}

def memory_breakdown_runtime(meta: Navigator.BaseMetadata) -> Dict[str, int]:
    size = 0
    size += _sizeof(meta.name2id)
    size += _sizeof(meta.id2name)
    size += _sizeof(meta.forward_map) + sum(_sizeof(row) for row in meta.forward_map)
    size += _sizeof(meta.block_of)
    size += _sizeof(meta.block_compacted)
    size += _sizeof(meta.block_uncompacted_hops) + sum(_sizeof(a) for a in meta.block_uncompacted_hops if a is not None)
    size += _sizeof(meta.block_compacted_hops) + sum(_sizeof(a) for a in meta.block_compacted_hops if a is not None)
    size += _sizeof(meta.block_except_nodes) + sum(_sizeof(a) for a in meta.block_except_nodes if a is not None)
    size += _sizeof(meta.block_except_hops) + sum(_sizeof(a) for a in meta.block_except_hops if a is not None)
    return {"runtime_object_bytes": size}

def memory_compression(meta: Navigator.FullMetadata) -> Dict[str, float]:
    direct_list_overhead = _sizeof(meta.next_hop) + sum(_sizeof(row) for row in meta.next_hop)
    compressed_overhead = (
        _sizeof(meta.block_compacted) +
        _sizeof(meta.block_uncompacted_hops) + sum(_sizeof(a) for a in meta.block_uncompacted_hops if a is not None) +
        _sizeof(meta.block_compacted_hops) + sum(_sizeof(a) for a in meta.block_compacted_hops if a is not None) +
        _sizeof(meta.block_except_nodes) + sum(_sizeof(a) for a in meta.block_except_nodes if a is not None) +
        _sizeof(meta.block_except_hops) + sum(_sizeof(a) for a in meta.block_except_hops if a is not None)
    )
    ratio = direct_list_overhead / max(1, compressed_overhead)
    return {
        "direct_overhead_bytes": direct_list_overhead,
        "compressed_overhead_bytes": compressed_overhead,
        "direct_vs_compressed_ratio": ratio,
    }

# ---------------- Query runners ----------------

def run_baseline(nav: Navigator, queries: int, seed: int) -> Tuple[int, int]:
    rng = random.Random(seed)
    successes = 0
    failures = 0
    for _ in range(queries):
        src = rng.randrange(nav.metadata.node_volume)
        dst = rng.randrange(nav.metadata.node_volume)
        if src == dst:
            continue
        gen.set_current_node(nav.metadata.id2name[src])
        try:
            ok = nav.goto(src, dst, log_output=False)
        except RuntimeError:
            ok = False
        if ok and gen.CURRENT_NODE == nav.metadata.id2name[dst]:
            successes += 1
        else:
            failures += 1
    return successes, failures

def run_failing(nav: Navigator, queries: int, single_ratio: float, multi_ratio: float, seed: int, mode: str) -> Tuple[int, int]:
    rng = random.Random(seed)
    failures_map: FailureMode = {}
    wrap_actions_dynamic(nav, failures_map)  # ensure wrappers installed
    single = int(queries * single_ratio)
    multi = int(queries * multi_ratio)
    none = queries - single - multi
    successes = 0
    failures = 0

    def select_mode() -> str:
        if mode == 'mixed':
            return 'stay' if rng.random() < 0.5 else 'raise'
        return mode

    # none
    for _ in range(none):
        src = rng.randrange(nav.metadata.node_volume)
        dst = rng.randrange(nav.metadata.node_volume)
        if src == dst:
            continue
        failures_map.clear()
        gen.set_current_node(nav.metadata.id2name[src])
        try:
            ok = nav.goto(src, dst, log_output=False)
        except RuntimeError:
            ok = False
        if ok and gen.CURRENT_NODE == nav.metadata.id2name[dst]:
            successes += 1
        else:
            failures += 1

    # single
    sdone = 0
    while sdone < single:
        src = rng.randrange(nav.metadata.node_volume)
        dst = rng.randrange(nav.metadata.node_volume)
        if src == dst:
            continue
        path = static_path(nav, src, dst)
        if not path:
            failures += 1
            sdone += 1
            continue
        edge_list = []
        cur = src
        for hop in path:
            edge_list.append((cur, hop))
            cur = hop
        chosen = None
        for uv in edge_list:
            if len(nav.metadata.forward_map[uv[0]]) > 1:
                chosen = uv
                break
        if chosen is None:
            chosen = edge_list[0]
        failures_map.clear()
        failures_map[chosen] = select_mode()
        gen.set_current_node(nav.metadata.id2name[src])
        try:
            ok = nav.goto(src, dst, log_output=False)
        except RuntimeError:
            ok = False
        if ok and gen.CURRENT_NODE == nav.metadata.id2name[dst]:
            successes += 1
        else:
            failures += 1
        sdone += 1

    # multi (up to 2)
    mdone = 0
    while mdone < multi:
        src = rng.randrange(nav.metadata.node_volume)
        dst = rng.randrange(nav.metadata.node_volume)
        if src == dst:
            continue
        path = static_path(nav, src, dst)
        if not path or len(path) < 2:
            failures += 1
            mdone += 1
            continue
        edge_list = []
        cur = src
        for hop in path:
            edge_list.append((cur, hop))
            cur = hop
        branching = [uv for uv in edge_list if len(nav.metadata.forward_map[uv[0]]) > 1]
        candidates = branching if branching else edge_list
        to_fail = candidates[:2]
        failures_map.clear()
        for uv in to_fail:
            failures_map[uv] = select_mode()
        gen.set_current_node(nav.metadata.id2name[src])
        try:
            ok = nav.goto(src, dst, log_output=False)
        except RuntimeError:
            ok = False
        if ok and gen.CURRENT_NODE == nav.metadata.id2name[dst]:
            successes += 1
        else:
            failures += 1
        mdone += 1

    return successes, failures

# ---------------- Serialization timing ----------------

def save_metadata(meta: Navigator.BaseMetadata | Navigator.FullMetadata, path: Path) -> float:
    t0 = time.perf_counter()
    meta.save(str(path))
    return time.perf_counter() - t0

def load_metadata(path: Path) -> Tuple[Navigator.BaseMetadata | Navigator.FullMetadata, float]:
    t0 = time.perf_counter()
    loaded = Navigator.load_metadata(str(path))
    return loaded, time.perf_counter() - t0

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', type=int, default=2000)
    ap.add_argument('--queries', type=int, default=2000000)
    ap.add_argument('--edge-prob', type=float, default=None)
    ap.add_argument('--avg-out-degree', type=float, default=6.5)
    ap.add_argument('--seed', type=int, default=time.time())
    ap.add_argument('--single-fail', type=float, default=0.05)
    ap.add_argument('--multi-fail', type=float, default=0.02)
    ap.add_argument('--fail-mode', choices=['raise', 'stay', 'mixed'], default='mixed')
    args = ap.parse_args()

    # Phase: generate specs
    t0 = time.perf_counter()
    specs = generate_interfaces(args.nodes, args.edge_prob, args.seed, args.avg_out_degree)
    gen_avg = sum(len(s.actions) for s in specs) / max(1, len(specs))
    t_gen = time.perf_counter() - t0
    print(f"[time] generate_specs={t_gen:.4f}s realized_avg_out_degree={gen_avg:.2f}")

    # Phase: build full metadata (Navigator construction)
    t0 = time.perf_counter()
    metadata = Navigator.compile_metadata(specs, cached_metadata=None, log_output=False)
    navigator = Navigator(specs, metadata=metadata)
    t_build_full = time.perf_counter() - t0
    print(f"[time] build_full_metadata={t_build_full:.4f}s nodes={navigator.metadata.node_volume}")

    # Derive runtime metadata (BaseMetadata)
    t0 = time.perf_counter()
    base_meta = navigator.metadata.to_base() if isinstance(navigator.metadata, Navigator.FullMetadata) else navigator.metadata
    t_to_base = time.perf_counter() - t0
    print(f"[time] derive_runtime_metadata={t_to_base:.4f}s")

    # --- Serialization (save & load) first, so we can include file sizes in [memory] ---
    with tempfile.TemporaryDirectory() as td:
        full_path = Path(td) / 'navigator-full.metadata'
        runtime_path = Path(td) / 'navigator-runtime.metadata'
        t_save_full = save_metadata(navigator.metadata, full_path)
        size_full = full_path.stat().st_size
        t_save_runtime = save_metadata(base_meta, runtime_path)
        size_runtime = runtime_path.stat().st_size
        loaded_full, t_load_full = load_metadata(full_path)
        loaded_runtime, t_load_runtime = load_metadata(runtime_path)

    # Memory comparisons + serialized sizes (user-friendly, layered)
    full_mem = memory_breakdown_full(navigator.metadata)
    runtime_mem = memory_breakdown_runtime(base_meta)
    compression = memory_compression(navigator.metadata)

    full_bytes = full_mem['full_object_bytes']
    runtime_bytes = runtime_mem['runtime_object_bytes']
    direct_bytes = compression['direct_overhead_bytes']
    compact_bytes = compression['compressed_overhead_bytes']
    direct_vs_compact_ratio = compression['direct_vs_compressed_ratio']

    runtime_reduction_pct = pct_reduction(full_bytes, runtime_bytes)
    compression_saving_pct = pct_reduction(direct_bytes, compact_bytes)
    file_reduction_pct = pct_reduction(size_full, size_runtime)

    print("[memory]\n"
          "  objects:\n"
          f"    full_object:       {full_bytes} bytes ({format_bytes(full_bytes)})\n"
          f"    runtime_object:    {runtime_bytes} bytes ({format_bytes(runtime_bytes)})\n"
          f"    runtime_reduction: {runtime_reduction_pct:.2f}% vs full\n"
          "  next_hop:\n"
          f"    direct:            {direct_bytes} bytes ({format_bytes(direct_bytes)})\n"
          f"    compact:           {compact_bytes} bytes ({format_bytes(compact_bytes)})\n"
          f"    compact_ratio:     {direct_vs_compact_ratio:.2f}x (direct/compact)\n"
          f"    compact_saving:    {compression_saving_pct:.2f}% vs direct\n"
          "  serialized:\n"
          f"    full_file:         {size_full} bytes ({format_bytes(size_full)})\n"
          f"    runtime_file:      {size_runtime} bytes ({format_bytes(size_runtime)})\n"
          f"    file_reduction:    {file_reduction_pct:.2f}% vs full")

    # Save/Load timings (sizes moved to [memory])
    print(f"[time] save_full={t_save_full:.4f}s save_runtime={t_save_runtime:.4f}s")
    print(f"[time] load_full={t_load_full:.4f}s load_runtime={t_load_runtime:.4f}s")

    # Baseline queries (compressed static)
    t0 = time.perf_counter()
    base_success, base_fail = run_baseline(navigator, args.queries, args.seed)
    t_baseline = time.perf_counter() - t0
    print(f"[baseline] success={base_success} fail={base_fail} time={t_baseline:.4f}s")

    # Failing queries
    t0 = time.perf_counter()
    fail_success, fail_fail = run_failing(navigator, args.queries, args.single_fail, args.multi_fail, args.seed, args.fail_mode)
    t_failing = time.perf_counter() - t0
    print(f"[failing] success={fail_success} fail={fail_fail} time={t_failing:.4f}s single_ratio={args.single_fail} multi_ratio={args.multi_fail} mode={args.fail_mode}")

    print(f"[summary] baseline_success={base_success} baseline_fail={base_fail} failing_success={fail_success} failing_fail={fail_fail}")

if __name__ == '__main__':
    main()
