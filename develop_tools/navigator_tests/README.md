# Navigator Tests & Profiling (Updated)

This directory (`develop_tools/navigator_tests/`) contains tooling to stress, validate, and profile the new `Navigator`
class implementation.

## Contents

- `generator.py`: Generates large weakly-connected random interface graphs. Ensures average out-degree (≈6–7 by
  default). Provides `set_current_node()` and helpers used by tests & profiler.
- `test_navigator_correctness.py`: Stability + persistence tests (fully rebuilt). Each test performs 20+ randomized
  large graph trials, verifying:
    - Reachability (arrive if a valid path exists; raise if unreachable).
    - Fallback BFS correctness under injected failing edges (raise or stay-in-place semantics).
    - Consistency after saving & loading Full and Runtime metadata (round-trip and corrupted file handling).
    - Failing edge variants: action raises vs action no-op (stays at original node).
- `profile_navigator.py`: Profiler script using the new `Navigator` APIs. Measures build + runtime behavior under mixed
  failure distributions with all internal logs disabled by default.
    - Full metadata build time (`compile_metadata`, reverse BFS + block compression).
    - Runtime metadata derivation time (`to_base`).
    - Layered memory breakdown:
        - Objects: Full vs Runtime metadata in bytes + human readable units.
        - Next-hop representation: direct table vs block-compacted structure (ratio and percent savings).
        - Serialized files: saved Full vs Runtime metadata file sizes + reduction percentage.
    - Save & load timings (sizes reported inside the memory section).
    - Baseline query execution (static compressed lookup).
    - Failing queries (single + multi edge failures with configurable modes: raise / stay / mixed).
    - Summary of success / failure counts.
    - All `goto()` & build logs suppressed (`log_output=False`) for clean profiler integration.

## Running Unit Tests

Use Python's unittest discovery from project root:

```bat
python -m unittest develop_tools.navigator_tests.test_navigator_correctness -v
```

Or discover the whole directory:

```bat
python -m unittest discover -s develop_tools\navigator_tests -v
```

## Running the Profiler

Basic run (Windows cmd):
```bat
python develop_tools\navigator_tests\profile_navigator.py --nodes 300 --queries 20000 --avg-out-degree 6.5 --single-fail 0.1 --multi-fail 0.05 --fail-mode mixed
```

Smaller quick sanity run:

```bat
python develop_tools\navigator_tests\profile_navigator.py --nodes 120 --queries 2000 --fail-mode mixed
```

Key CLI arguments:

- `--nodes` (int): Number of interfaces (graph size).
- `--queries` (int): Number of navigation queries in each phase.
- `--avg-out-degree` (float): Target average out-degree (adjusts edge probability during generation).
- `--edge-prob` (float or None): Override edge probability directly (if set, supersedes derived average).
- `--single-fail` (float): Fraction of queries with one failing edge injected.
- `--multi-fail` (float): Fraction of queries with 2 failing edges injected.
- `--fail-mode` (raise | stay | mixed): Failure behavior for broken edges.
- `--seed` (int): Random seed (defaults to current time if omitted).

## Sample Profiler Output (Excerpt)

```
[time] generate_specs=0.0160s realized_avg_out_degree=6.58
[time] build_full_metadata=0.0526s nodes=120
[time] derive_runtime_metadata=0.0000s
[memory]
  objects:
    full_object:       2459704 bytes (2.35 MiB)
    runtime_object:    67072 bytes (65.50 KiB)
    runtime_reduction: 97.27% vs full
  next_hop:
    direct:            123000 bytes (120.12 KiB)
    compact:           45304 bytes (44.24 KiB)
    compact_ratio:     2.71x (direct/compact)
    compact_saving:    63.17% vs direct
  serialized:
    full_file:         72006 bytes (70.33 KiB)
    runtime_file:      14480 bytes (14.14 KiB)
    file_reduction:    79.89% vs full
[time] save_full=0.0183s save_runtime=0.0024s
[time] load_full=0.0061s load_runtime=0.0007s
[baseline] success=1983 fail=0 time=0.0313s
[failing] success=1978 fail=6 time=0.0496s single_ratio=0.1 multi_ratio=0.05 mode=mixed
[summary] baseline_success=1983 baseline_fail=0 failing_success=1978 failing_fail=6
```

## Metadata Persistence Usage

Example: saving full metadata and reloading into a fresh `Navigator`.

```python
from core.navigator import Navigator
from develop_tools.navigator_tests import generator as gen

specs = gen.generate_specs(num_nodes=200, edge_probability=None, seed=123, failing_edge_ratio=0.0)
navigator = Navigator(specs)  # build full metadata
navigator.metadata.save('navigator-full.metadata')

loaded_meta = Navigator.load_metadata('navigator-full.metadata')  # returns FullMetadata
# Rebuild action spec list from metadata (see tests for helper pattern)
interfaces = [Navigator.Interface(name=n, description=n, features=[], actions={}) for n in loaded_meta.id2name]
reloaded = Navigator(interfaces, metadata=loaded_meta)
```

Refer to the test helper `_specs_from_metadata` inside `test_navigator_correctness.py` for a reusable reconstruction
pattern that attaches actions/features.

## Design Notes

- Static navigation uses block-compacted next-hop storage when smaller than direct per-destination arrays; otherwise
  falls back to raw hops.
- Fallback BFS triggers only when one or more static path edges fail (raise or stay-in-place), per injected failure
  maps.
- Memory reporting uses Python `getsizeof` shallow counts; relative ratios remain consistent though absolute totals may
  exclude deep object internals. For precise accounting you can implement a recursive deep-size traversal.
- Logs are suppressed by passing `log_output=False` to both `compile_metadata` and `goto()`; internal recursive calls
  propagate this flag.

## Future Extensions (Optional)

- JSON / CSV output (`--output-format`) for automated dashboards.
- Deep memory scanner for accurate nested container sizes.
- Per-block compression diagnostics (e.g. exception density, compact hit rate).
- Incremental rebuild benchmark scenario (add small edge batch and time recompile).

---
Feel free to open issues or extend profiler options for more experiment types.
