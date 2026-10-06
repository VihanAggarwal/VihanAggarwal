# NanoGPT speedrun: optimization work in progress

Work toward beating the current track-1 record of [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt):
record #92 (ANVIL2, 39.9 s on 8xH100). Base: upstream commit `4ea6b93`.

**Status: not yet a record.** No change here has been timed on 8xH100. Every claim below comes
from CPU tests and from the record's own published logs.

## Apply

```bash
git clone https://github.com/KellerJordan/modded-nanogpt && cd modded-nanogpt
git checkout 4ea6b93
git am /path/to/nanogpt-speedrun/patches/*.patch
```

## Patches

| # | change | kind | evidence so far |
|---|---|---|---|
| 0001 | `tools/speedrun_ab/`: interleaved ABBA A/B runner for one 8xH100 node, and rule-2/rule-4 statistics | tooling | reproduces ANVIL2's published baseline stats; 7 CPU tests |
| 0002 | Fork the canonical-mask builder before CUDA init. It now waits on a pipe and starts its on-clock build at t0, instead of forking the warmed-up trainer at t0 | systems-only, mask byte-identical | 4 CPU tests; mock-trainer benchmark: first 25 steps 1.2-6.1 s with the late fork vs 0.15-0.21 s with the early fork, on this VM |

## Where the record's time goes (from its 17 published run logs)

| part of the run | mean | sd |
|---|---|---|
| steps 0-25 | 645 ms | **122 ms** (min 556, max 1028) |
| every later 25-step interval | 421-1201 ms | 0.4-3 ms |
| final validation, on the clock | 221 ms | 6 ms |
| total | 39.910 s | 0.123 s |

The steady state of stage 0 is ~427 ms per 25 steps. So 130-600 ms of startup overhead lands in
the first 25 steps, and it accounts for nearly all of the run-to-run wall variance.

## What a record needs (rules)

1. Token streams unchanged. 2. Mean val <= 3.28 at p < 0.01, waived for systems-only changes.
3. No extra compile flags. 4. Faster than the prior record on the same hardware: run
`tools/speedrun_ab/ab_bench.py` interleaved against `4ea6b93` on one 8xH100 node.
