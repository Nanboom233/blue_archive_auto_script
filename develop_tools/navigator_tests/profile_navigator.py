"""Navigator lightweight profiling & correctness harness.

Two phases:
1. Baseline phase: all edges succeed (no failures) -> measure success count and basic path execution.
2. Failing phase: per-query injection of dynamic failures (single / multi) without recompiling the index.

Design changes:
- Actions are wrapped to consult a global FAILURE_MODE_MAP to decide success, raise, or stay.
- No compile_specs inside query loop; topology remains stable to simulate realistic runtime failure.
- Each major step split into functions for clearer profiler call graph.
- Minimal output for integration with external profiler tools.
"""
from __future__ import annotations
import random
import argparse
import sys
from pathlib import Path
from typing import Optional, Iterable

# Ensure project root on sys.path when running as a standalone script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.navigator import compile_specs, lookup_next_hop
from develop_tools import navigator_tests as gen

# ---------------- Failure control globals ----------------
FAILURE_MODE_MAP: dict[tuple[str, str], str] = {}  # (src_name, dst_name) -> 'raise' | 'stay'

# ---------------- Helpers ----------------

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
    from core.navigator import fallback_next_hop
    return fallback_next_hop(index, current_id, destination_id, deprecated)


def goto(index, current_id: int, destination_id: int, path=None, deprecated=None, hop_func=None):
    if hop_func is None:
        hop_func = lambda idx, u, d: lookup_next_hop(idx, u, d)
    if not path:
        if deprecated:
            fb = fallback_next_hop_wrapper(index, current_id, destination_id, deprecated)
            if fb is None:
                return False
            return goto(index, current_id, destination_id, fb, deprecated, hop_func)
        static = []
        tmp = current_id
        guard = 0
        while tmp != destination_id and guard <= index.node_volume:
            nxt = hop_func(index, tmp, destination_id)
            if nxt == -1:
                return False
            static.append(nxt)
            tmp = nxt
            guard += 1
        if tmp != destination_id:
            return False
        return goto(index, current_id, destination_id, static, deprecated, hop_func)
    for nxt in path:
        action = index.edge_handle.get((current_id, nxt))
        if action is None:
            return False
        try:
            action()
        except Exception as e:
            # print(f"[debug] action from {index.id2name[current_id]} to {index.id2name[nxt]} raised exception: {e}")
            if deprecated is None:
                deprecated = set()
            deprecated.add((current_id, nxt))
            return goto(index, current_id, destination_id, deprecated=deprecated, hop_func=hop_func)
        resolved = resolve_current_node(index, [nxt, current_id])
        if resolved is None or resolved == current_id:
            # print(f"[debug] after action from {index.id2name[current_id]} to {index.id2name[nxt]}, stuck at {index.id2name[current_id]}")
            if deprecated is None:
                deprecated = set()
            deprecated.add((current_id, nxt))
            return goto(index, current_id, destination_id, deprecated=deprecated, hop_func=hop_func)
        elif resolved != nxt:
            # print(f"[debug] after action from {index.id2name[current_id]} to {index.id2name[nxt]}, diverted to {index.id2name[resolved]}")
            if deprecated is None:
                deprecated = set()
            deprecated.add((current_id, nxt))
            return goto(index, current_id, destination_id, deprecated=deprecated, hop_func=hop_func)
        current_id = resolved
    return True


def build_static_path(index, src_id: int, dst_id: int, hop_func) -> Optional[list[int]]:
    path: list[int] = []
    cur = src_id
    guard = 0
    while cur != dst_id and guard <= index.node_volume:
        nxt = hop_func(index, cur, dst_id)
        if nxt == -1:
            return None
        path.append(nxt)
        cur = nxt
        guard += 1
    return path if cur == dst_id else None

# ---------------- Dynamic action wrapping ----------------

def wrap_actions_dynamic(specs):
    """Replace each action with a dynamic wrapper consulting FAILURE_MODE_MAP."""
    for spec in specs:
        for dst_name in list(spec.actions.keys()):
            def make_wrapper(src=spec.name, dst=dst_name):
                def _wrapped():
                    mode = FAILURE_MODE_MAP.get((src, dst))
                    if mode == 'raise':
                        raise RuntimeError(f"forced failure {src}->{dst}")
                    elif mode == 'stay':
                        # do nothing (stay at src)
                        return None
                    else:
                        # success: move CURRENT_NODE
                        gen.CURRENT_NODE = dst
                        return None
                return _wrapped
            spec.actions[dst_name] = make_wrapper()
    return specs

# ---------------- Failure toggling ----------------

def clear_failures():
    FAILURE_MODE_MAP.clear()


def set_failures(edge_pairs: Iterable[tuple[str, str]], mode_selector, rng: random.Random):
    for (src, dst) in edge_pairs:
        FAILURE_MODE_MAP[(src, dst)] = mode_selector(rng)

# ---------------- Query execution variants ----------------

# Old unified baseline retained (unused in main), keep for compatibility

def run_baseline(index, specs, total_queries: int, seed: int, use_direct_hop: bool):
    rng = random.Random(seed)
    hop_func = (lambda idx, u, d: idx.next_hop[d][u]) if use_direct_hop else (lambda idx, u, d: lookup_next_hop(idx, u, d))
    successes = 0
    failures = 0
    for _ in range(total_queries):
        src = rng.randrange(index.node_volume)
        dst = rng.randrange(index.node_volume)
        if src == dst:
            continue
        gen.set_current_node(index.id2name[src])
        clear_failures()
        ok = goto(index, src, dst, hop_func=hop_func)
        if ok and gen.CURRENT_NODE == index.id2name[dst]:
            successes += 1
        else:
            failures += 1
    return successes, failures


def run_baseline_compressed(index, specs, total_queries: int, seed: int):
    """Baseline using compressed lookup (lookup_next_hop)."""
    rng = random.Random(seed)
    successes = 0
    failures = 0
    def hop_func(idx, u, d):
        return lookup_next_hop(idx, u, d)
    for _ in range(total_queries):
        src = rng.randrange(index.node_volume)
        dst = rng.randrange(index.node_volume)
        if src == dst:
            continue
        gen.set_current_node(index.id2name[src])
        ok = goto(index, src, dst, hop_func=hop_func)
        if ok and gen.CURRENT_NODE == index.id2name[dst]:
            successes += 1
        else:
            failures += 1
    return successes, failures


def run_baseline_direct(index, specs, total_queries: int, seed: int):
    """Baseline using direct next_hop table access (uncompressed)."""
    rng = random.Random(seed)
    successes = 0
    failures = 0
    def hop_func(idx, u, d):
        return idx.next_hop[d][u]
    for _ in range(total_queries):
        src = rng.randrange(index.node_volume)
        dst = rng.randrange(index.node_volume)
        if src == dst:
            continue
        gen.set_current_node(index.id2name[src])
        ok = goto(index, src, dst, hop_func=hop_func)
        if ok and gen.CURRENT_NODE == index.id2name[dst]:
            successes += 1
        else:
            failures += 1
    return successes, failures


# Existing failing runner unchanged

def run_failing(index, specs, total_queries: int, single_fail_ratio: float, multi_fail_ratio: float,
                seed: int, use_direct_hop: bool, fail_mode: str):
    rng = random.Random(seed)
    hop_func = (lambda idx, u, d: idx.next_hop[d][u]) if use_direct_hop else (lambda idx, u, d: lookup_next_hop(idx, u, d))
    single_fail = int(total_queries * single_fail_ratio)
    multi_fail = int(total_queries * multi_fail_ratio)
    no_fail = total_queries - single_fail - multi_fail

    successes = 0
    failures = 0

    def mode_selector(local_rng: random.Random):
        if fail_mode == 'mixed':
            return 'stay' if local_rng.random() < 0.5 else 'raise'
        return fail_mode  # 'raise' or 'stay'

    # No-failure queries (should mirror baseline behavior)
    for _ in range(no_fail):
        src = rng.randrange(index.node_volume)
        dst = rng.randrange(index.node_volume)
        if src == dst:
            continue
        clear_failures()
        gen.set_current_node(index.id2name[src])
        ok = goto(index, src, dst, hop_func=hop_func)
        if ok and gen.CURRENT_NODE == index.id2name[dst]:
            successes += 1
        else:
            failures += 1

    # Single failure queries
    sf_done = 0
    while sf_done < single_fail:
        src = rng.randrange(index.node_volume)
        dst = rng.randrange(index.node_volume)
        if src == dst:
            continue
        static_path = build_static_path(index, src, dst, hop_func)
        if not static_path or len(static_path) < 1:
            failures += 1
            sf_done += 1
            continue
        # Build edges on static path
        path_edges = []
        cur = src
        for hop in static_path:
            path_edges.append((cur, hop))
            cur = hop
        # choose edge with branching if possible
        chosen = None
        for (u, v) in path_edges:
            if len(index.forward_map[u]) > 1:
                chosen = (u, v)
                break
        if chosen is None:
            chosen = path_edges[0]
        clear_failures()
        set_failures([(index.id2name[chosen[0]], index.id2name[chosen[1]])], mode_selector, rng)
        gen.set_current_node(index.id2name[src])
        ok = goto(index, src, dst, hop_func=hop_func)
        clear_failures()  # reset after query
        if ok and gen.CURRENT_NODE == index.id2name[dst]:
            successes += 1
        else:
            failures += 1
        sf_done += 1

    # Multi failure queries (inject up to 2 failing edges along static path)
    mf_done = 0
    while mf_done < multi_fail:
        src = rng.randrange(index.node_volume)
        dst = rng.randrange(index.node_volume)
        if src == dst:
            continue
        static_path = build_static_path(index, src, dst, hop_func)
        if not static_path or len(static_path) < 2:
            failures += 1
            mf_done += 1
            continue
        path_edges = []
        cur = src
        for hop in static_path:
            path_edges.append((cur, hop))
            cur = hop
        candidates = [e for e in path_edges if len(index.forward_map[e[0]]) > 1]
        if len(candidates) < 2:
            candidates = path_edges[:2]
        to_fail = candidates[:2]
        fail_pairs = [(index.id2name[u], index.id2name[v]) for (u, v) in to_fail]
        clear_failures()
        set_failures(fail_pairs, mode_selector, rng)
        gen.set_current_node(index.id2name[src])
        ok = goto(index, src, dst, hop_func=hop_func)
        clear_failures()
        if ok and gen.CURRENT_NODE == index.id2name[dst]:
            successes += 1
        else:
            failures += 1
        mf_done += 1

    return successes, failures

# ---------------- Memory estimation ----------------

def memory_breakdown(index):
    """Return detailed breakdown for compressed vs direct container memory (shallow).
    Compressed-specific structures:
      - block_of (list[int])
      - col_kind (list[int])
      - row_direct (list[Optional[list[int]]]) including inner row lists
      - region_default (list[Optional[dict[int,int]]]) including dict containers
      - exc_nodes (list[Optional[list[int]]]) including inner lists
      - exc_hops (list[Optional[list[int]]]) including inner lists
    Direct-specific structure:
      - next_hop (list[list[int]]) including inner row lists
    Note: We only count container shells with sys.getsizeof to avoid O(N^2) deep traversal cost.
    """
    # Compressed
    comp_block_of = sys.getsizeof(index.block_of)
    comp_col_kind = sys.getsizeof(index.col_kind)
    comp_row_direct_list = sys.getsizeof(index.row_direct)
    comp_row_direct_rows = sum(sys.getsizeof(row) for row in index.row_direct if row is not None)
    comp_region_default_list = sys.getsizeof(index.region_default)
    comp_region_default_dicts = sum(sys.getsizeof(d) for d in index.region_default if d is not None)
    comp_exc_nodes_list = sys.getsizeof(index.exc_nodes)
    comp_exc_nodes_rows = sum(sys.getsizeof(lst) for lst in index.exc_nodes if lst is not None)
    comp_exc_hops_list = sys.getsizeof(index.exc_hops)
    comp_exc_hops_rows = sum(sys.getsizeof(lst) for lst in index.exc_hops if lst is not None)

    comp_total = (
        comp_block_of + comp_col_kind +
        comp_row_direct_list + comp_row_direct_rows +
        comp_region_default_list + comp_region_default_dicts +
        comp_exc_nodes_list + comp_exc_nodes_rows +
        comp_exc_hops_list + comp_exc_hops_rows
    )

    direct_list = sys.getsizeof(index.next_hop)
    direct_rows = sum(sys.getsizeof(row) for row in index.next_hop)
    direct_total = direct_list + direct_rows

    # Column stats
    total_cols = index.node_volume
    direct_cols = sum(1 for k in index.col_kind if k == 0)
    region_cols = total_cols - direct_cols
    avg_region_default_size = (sum(len(d) for d in index.region_default if d is not None) / max(1, region_cols)) if region_cols else 0.0
    avg_exc_nodes = (sum(len(lst) for lst in index.exc_nodes if lst is not None) / max(1, region_cols)) if region_cols else 0.0

    return {
        "compressed": {
            "block_of": comp_block_of,
            "col_kind": comp_col_kind,
            "row_direct_list": comp_row_direct_list,
            "row_direct_rows": comp_row_direct_rows,
            "region_default_list": comp_region_default_list,
            "region_default_dicts": comp_region_default_dicts,
            "exc_nodes_list": comp_exc_nodes_list,
            "exc_nodes_rows": comp_exc_nodes_rows,
            "exc_hops_list": comp_exc_hops_list,
            "exc_hops_rows": comp_exc_hops_rows,
            "total": comp_total,
        },
        "direct": {
            "next_hop_list": direct_list,
            "row_lists": direct_rows,
            "total": direct_total,
        },
        "stats": {
            "direct_cols": direct_cols,
            "region_cols": region_cols,
            "avg_region_default_size": avg_region_default_size,
            "avg_exc_nodes": avg_exc_nodes,
        }
    }

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=2000000)
    ap.add_argument("--edge-prob", type=float, default=None)
    ap.add_argument("--avg-out-degree", type=float, default=6.5, help="Target average out-degree per node (including backbone). Ignored if --edge-prob is provided.")
    ap.add_argument("--seed", type=int, default=1145141919810)
    ap.add_argument("--single-fail", type=float, default=0.1)
    ap.add_argument("--multi-fail", type=float, default=0.05)
    ap.add_argument("--fail-mode", choices=["raise", "stay", "mixed"], default="mixed")
    args = ap.parse_args()

    print(f"[phase] generate specs n={args.nodes} target_avg_out_degree={args.avg_out_degree} edge_prob={args.edge_prob}")
    specs = gen.generate_specs(args.nodes, args.edge_prob, seed=args.seed, failing_edge_ratio=0.0, avg_out_degree=args.avg_out_degree)
    wrap_actions_dynamic(specs)
    realized = sum(len(s.actions) for s in specs) / len(specs)
    print(f"[phase] realized_avg_out_degree={realized:.2f}")

    print(f"[phase] compile index (silent)")
    index = compile_specs(specs, prev=None, log_output=False)

    # Memory usage estimates
    br = memory_breakdown(index)
    comp_total = br["compressed"]["total"]
    direct_total = br["direct"]["total"]
    st = br["stats"]
    print(
        "[memory] compressed_total={c_total} (block_of={bo}, col_kind={ck}, row_direct_list={rdl}, row_direct_rows={rdr}, region_default_list={rdfl}, region_default_dicts={rdfd}, exc_nodes_list={enl}, exc_nodes_rows={enr}, exc_hops_list={ehl}, exc_hops_rows={ehr}); "
        "direct_total={d_total} (next_hop_list={nl}, row_lists={rl}); ratio_direct_over_compressed={ratio:.2f}; cols direct={dc} region={rc} avg_region_default_size={ards:.2f} avg_exc_nodes_per_region={aen:.2f}".format(
            c_total=comp_total,
            bo=br["compressed"]["block_of"],
            ck=br["compressed"]["col_kind"],
            rdl=br["compressed"]["row_direct_list"],
            rdr=br["compressed"]["row_direct_rows"],
            rdfl=br["compressed"]["region_default_list"],
            rdfd=br["compressed"]["region_default_dicts"],
            enl=br["compressed"]["exc_nodes_list"],
            enr=br["compressed"]["exc_nodes_rows"],
            ehl=br["compressed"]["exc_hops_list"],
            ehr=br["compressed"]["exc_hops_rows"],
            d_total=direct_total,
            nl=br["direct"]["next_hop_list"],
            rl=br["direct"]["row_lists"],
            ratio=(direct_total / max(1, comp_total)),
            dc=st["direct_cols"],
            rc=st["region_cols"],
            ards=st["avg_region_default_size"],
            aen=st["avg_exc_nodes"],
        )
    )

    # Baseline compressed
    print(f"[baseline-compressed] queries={args.queries}")
    base_comp_success, base_comp_fail = run_baseline_compressed(index, specs, args.queries, seed=args.seed)
    print(f"[baseline-compressed] success={base_comp_success} fail={base_comp_fail}")

    # Baseline direct next_hop (uncompressed table access)
    print(f"[baseline-direct] queries={args.queries}")
    base_direct_success, base_direct_fail = run_baseline_direct(index, specs, args.queries, seed=args.seed)
    print(f"[baseline-direct] success={base_direct_success} fail={base_direct_fail}")

    # Failing phase uses compressed lookup (representative realistic path selection)
    print(f"[failing-compressed] single_ratio={args.single_fail} multi_ratio={args.multi_fail} mode={args.fail_mode}")
    fail_success, fail_fail = run_failing(index, specs, args.queries, args.single_fail, args.multi_fail,
                                          seed=args.seed, use_direct_hop=False, fail_mode=args.fail_mode)
    print(f"[failing-compressed] success={fail_success} fail={fail_fail}")

    print("[summary] baseline_compressed_success={} baseline_compressed_fail={} baseline_direct_success={} baseline_direct_fail={} failing_success={} failing_fail={}".format(
        base_comp_success, base_comp_fail, base_direct_success, base_direct_fail, fail_success, fail_fail))


if __name__ == "__main__":
    main()
