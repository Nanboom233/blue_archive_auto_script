"""Utility for generating large random navigator specs for testing.

We ensure weak connectivity by first building a linear backbone chain N0->N1->...->N{n-1} then add random edges.
"""
from __future__ import annotations
from functools import partial
from typing import Callable, Optional
import random

from core.navigator import Navigator

# Global mutable to emulate environment for feature validators
CURRENT_NODE: str = ""


def _feature_validator(name: str) -> bool:
    return CURRENT_NODE == name


def _make_action(destination: str) -> Callable[[], None]:
    def _act():
        global CURRENT_NODE
        CURRENT_NODE = destination

    return _act


def _make_fail_raise(src: str, dest: str) -> Callable[[], None]:
    def _f():
        raise RuntimeError(f"Simulated failure from {src} to {dest}")

    return _f


def _make_fail_stay() -> Callable[[], None]:
    # Do nothing, keep CURRENT_NODE unchanged
    def _noop():
        return None

    return _noop


def generate_specs(
    num_nodes: int,
    edge_probability: float | None = None,
    seed: Optional[int] = None,
    failing_edge_ratio: float = 0.0,
    failing_raise_ratio: float | None = None,
    failing_stay_ratio: float | None = None,
    avg_out_degree: float | None = None,
) -> list[Navigator.Interface]:
    """Generate a list of NodeSpec objects.

    Parameters
    ----------
    num_nodes: number of nodes to generate (>=1)
    edge_probability: probability for each potential extra directed edge. If None, computed from avg_out_degree.
    seed: optional random seed for reproducibility
    failing_edge_ratio: legacy param: total fraction of edges to fail by raising
    failing_raise_ratio: fraction of edges to fail by raising (if provided, overrides failing_edge_ratio)
    failing_stay_ratio: fraction of edges to fail by not moving (action does nothing)
    avg_out_degree: target average total out-degree per node (including the backbone). If provided and
                    edge_probability is None, use it to compute effective probability for random edges.
    """
    assert num_nodes >= 1
    if seed is not None:
        random.seed(seed)

    # Normalize failure ratios
    if failing_raise_ratio is None and failing_stay_ratio is None:
        failing_raise_ratio = failing_edge_ratio
        failing_stay_ratio = 0.0
    if failing_raise_ratio is None:
        failing_raise_ratio = 0.0
    if failing_stay_ratio is None:
        failing_stay_ratio = 0.0
    assert 0.0 <= failing_raise_ratio <= 1.0
    assert 0.0 <= failing_stay_ratio <= 1.0
    assert failing_raise_ratio + failing_stay_ratio <= 1.0

    names = [f"N{i}" for i in range(num_nodes)]
    specs: list[Navigator.Interface] = []

    # Build backbone chain for connectivity
    edges: set[tuple[int, int]] = set()
    for i in range(num_nodes - 1):
        edges.add((i, i + 1))

    # Compute effective probability to approach desired average out-degree
    effective_p: float
    if edge_probability is not None:
        effective_p = max(0.0, min(1.0, edge_probability))
    else:
        # Target average total out-degree ≈ avg_out_degree (default ~6.5)
        target = 6.5 if avg_out_degree is None else max(0.0, avg_out_degree)
        backbone_avg = (num_nodes - 1) / num_nodes if num_nodes > 0 else 0.0  # average out-degree added by backbone
        target_extra = max(0.0, target - backbone_avg)
        if num_nodes > 1:
            effective_p = min(1.0, target_extra / (num_nodes - 1))
        else:
            effective_p = 0.0

    # Add random edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            if random.random() < effective_p:
                edges.add((i, j))

    # Split failing edges into two types
    all_edges = list(edges)
    random.shuffle(all_edges)
    raise_count = int(len(all_edges) * failing_raise_ratio)
    stay_count = int(len(all_edges) * failing_stay_ratio)
    failing_raise_edges = set(all_edges[:raise_count]) if raise_count > 0 else set()
    failing_stay_edges = set(all_edges[raise_count:raise_count + stay_count]) if stay_count > 0 else set()

    for i, name in enumerate(names):
        actions: dict[str, Callable] = {}
        for (u, v) in edges:
            if u == i:
                dest_name = names[v]
                if (u, v) in failing_raise_edges:
                    actions[dest_name] = _make_fail_raise(name, dest_name)
                elif (u, v) in failing_stay_edges:
                    actions[dest_name] = _make_fail_stay()
                else:
                    actions[dest_name] = _make_action(dest_name)
        spec = Navigator.Interface(
            name=name,
            description=f"Auto generated node {name}",
            features=[partial(_feature_validator, name)],
            actions=actions,
        )
        specs.append(spec)
    return specs


def set_current_node(name: str) -> None:
    global CURRENT_NODE
    CURRENT_NODE = name


def generate_small_fallback_raise(seed: int | None = None):
    """Generate a tiny graph ensuring static path uses B then fails on B->D (raise), with fallback via C.

    Nodes: A, B, C, D. Edges:
      - A->B, A->C
      - B->D (failing raise)
      - B->C (to allow B->C->D after failure)
      - C->D (ok)
    Optionally add one noisy node E with edges that don't create shorter than 2 path.
    """
    if seed is not None:
        random.seed(seed)
    names = ["A", "B", "C", "D"]
    actions: dict[str, dict[str, Callable]] = {n: {} for n in names}
    # base edges
    actions["A"]["B"] = _make_action("B")
    actions["A"]["C"] = _make_action("C")
    actions["B"]["D"] = _make_fail_raise("B", "D")  # failing edge on shortest path
    actions["B"]["C"] = _make_action("C")
    actions["C"]["D"] = _make_action("D")
    # optional noise node
    if random.random() < 0.7:
        names.append("E")
        actions["E"] = {}
        # edges that won't beat length 2
        # connect E arbitrarily but avoid A->D or A->E->D shortcuts
        actions["A"]["E"] = _make_action("E")
        actions["E"]["B"] = _make_action("B")
        actions["E"]["C"] = _make_action("C")
    specs: list[Navigator.Interface] = []
    for n in names:
        specs.append(Navigator.Interface(
            name=n,
            description=n,
            features=[partial(_feature_validator, n)],
            actions=actions.get(n, {}),
        ))
    return specs, "A", "D"


def generate_small_fallback_stay(seed: int | None = None):
    """Generate a tiny graph where A->B is no-move fail, forcing fallback via A->C->D.

    Nodes: A, B, C, D. Edges:
      - A->B (stay fail)
      - A->C (ok)
      - C->D (ok)
      - B->D (ok)
    Static next-hop prefers B (lower id) so we first try A->B and then fallback.
    """
    if seed is not None:
        random.seed(seed)
    names = ["A", "B", "C", "D"]
    actions: dict[str, dict[str, Callable]] = {n: {} for n in names}
    actions["A"]["B"] = _make_fail_stay()
    actions["A"]["C"] = _make_action("C")
    actions["C"]["D"] = _make_action("D")
    actions["B"]["D"] = _make_action("D")  # exists but won't be used due to stay at A->B
    # optional noise
    if random.random() < 0.7:
        names.append("E")
        actions["E"] = {}
        actions["A"]["E"] = _make_action("E")
        actions["E"]["C"] = _make_action("C")
    specs: list[Navigator.Interface] = []
    for n in names:
        specs.append(Navigator.Interface(
            name=n,
            description=n,
            features=[partial(_feature_validator, n)],
            actions=actions.get(n, {}),
        ))
    return specs, "A", "D"


__all__ = [
    "generate_specs",
    "CURRENT_NODE",
    "set_current_node",
    "generate_small_fallback_raise",
    "generate_small_fallback_stay",
]
