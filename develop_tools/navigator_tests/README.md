# Navigator Tests & Profiling

This directory contains:

- `generator.py`: builds large random graphs of `NodeSpec` with guaranteed weak connectivity and optional failing edges for fallback path testing. It also exposes `set_current_node()` to emulate environment changes.
- `test_navigator_correctness.py`: unit tests for
  - incremental rebuild behavior after adding shortcut edges,
  - fallback with broken edges using goto()-style execution like the demo,
  - mixed query batches with a failure distribution (50% none, 30% single broken edge, 20% multiple), and
  - reverse BFS consistency invariants on larger graphs.
- `profile_navigator.py`: benchmark script that runs goto()-style queries under a configurable failure distribution, measures full vs incremental build time, and supports cProfile. Includes an ablation switch to bypass block-compressed lookup and use the raw `next_hop` table.

## Run Unit Tests
```shell
python -m unittest discover -s navigator_tests -v
```

## Run Profiling
```bat
python tests\profile_navigator.py --nodes 300 --queries 2000 --edge-prob 0.02 --single-fail 0.3 --multi-fail 0.2
```

To compare compressed lookup vs direct raw table access (ablation):
```bat
python tests\profile_navigator.py --nodes 300 --queries 2000 --edge-prob 0.02 --single-fail 0.3 --multi-fail 0.2 --ablate-compression
```

## What Is Measured
- Full build time (reverse BFS + compression)
- Incremental rebuild time after adding shortcut edges
- Goto() latency distribution under mixed failures (mean/median/p95)
- Hotspots in call graph (cProfile)

## Notes
- Fallback BFS only runs when at least one edge on the static path fails during execution; tests and benchmarks reflect that behavior.
- You can scale `--nodes` and `--queries` to stress-test correctness and performance.
