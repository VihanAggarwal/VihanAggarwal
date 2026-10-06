# NanoGPT speedrun: optimization work in progress

Work toward beating the current track-1 record of [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt):
record #92 (ANVIL2, 39.9 s on 8xH100). Base: upstream commit `4ea6b93`.

**Status: not yet a record.** No change here has been timed on 8xH100. Every claim below comes
from CPU tests and from the record's own published logs.

**Fork:** [VihanAggarwal/modded-nanogpt @ claude/nanogpt-optimization-n49ur2](https://github.com/VihanAggarwal/modded-nanogpt/tree/claude/nanogpt-optimization-n49ur2)
holds the same commits; the patches below mirror it.

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
| 0002 | Spawn the canonical-mask builder (fresh interpreter, vfork+exec) at t0 instead of `os.fork()`-ing the warmed-up 8-GPU trainer there. All of the child's work stays on the clock | systems-only, mask byte-identical | 5 CPU tests; `start()` costs 0.5 ms vs 8.7 ms for fork with 2 GB touched, before any copy-on-write faults. A mock trainer's first 25 steps took 1.2-6.1 s after a late fork vs 0.15-0.21 s without one |
| 0003 | Data loader: numpy BOS index (partial index 12.8 -> 1.7 ms on the step-0 critical path); `ScheduledBatches.close()` | systems-only, token stream byte-identical | the whole 1194-step schedule plus validation replayed through upstream's and this loader: identical batches on 2 ranks |
| 0004 | Loader thread: step 0's first-shard read overlaps the prefix-table build at t0; the final validation's reads overlap the GPU drain; the training loader is closed first, so the val shard reuses a cached 256 MB pinned block instead of a fresh `cudaHostAlloc` | systems-only, same batches | CPU tests: loader close frees both shards with GC off; threaded val batches identical |
| 0005 | Read only the first shard's 12 MB partial-index span before batch 0; the index thread reads the other 188 MB on the clock. Also: wait for the full index instead of silently skipping the rest of a shard if batches ever outrun the scan | systems-only, token stream byte-identical | first batch 34-50 ms -> 12.5-12.8 ms (CPU); the full-schedule replay passes with the wait path exercised |

All tests: `TIKTOKEN_CACHE_DIR=... python -m pytest tools -q` (17 pass). They need FineWeb-format shards
(`SPEEDRUN_TEST_DATA`); synthetic shards in the same format work.

## Where the record's time goes (from its 17 published run logs)

| part of the run | mean | sd |
|---|---|---|
| steps 0-25 | 645 ms | **122 ms** (min 556, max 1028) |
| every later 25-step interval | 421-1201 ms | 0.4-3 ms |
| final validation, on the clock | 221 ms | 6 ms |
| total | 39.910 s | 0.123 s |

The steady state of stage 0 is ~427 ms per 25 steps. So 130-600 ms of startup overhead lands in
the first 25 steps, and it accounts for nearly all of the run-to-run wall variance.

Those logs come from #360's original single-file trainer. The repo's current trainer (the
`track_1_short` refactor, f380c1f) added canonical masking and its t0 fork. Its author measured
it at 40.90 s against 40.60 s for #360 on the same nodes. The baseline for rule 4 is the current trainer.

## Upstream frontier (open PRs, not merged)

| PR | claim |
|---|---|
| #379 CPLM | 36.0 s |
| #380/#381 exact-match retrieval | 21.5 s |

These are ML changes. The systems patches here are orthogonal: #380 still forks at t0.

## Testing on one GPU (Colab)

Colab gives one GPU, and the speedrun needs 8xH100 in one node, so Colab cannot run or time a record.
`tools/gpu_smoke/smoke_1gpu.py` checks the patches under real CUDA and times what they remove; see
`tools/gpu_smoke/README.md` for the notebook cell.

## What a record needs (rules)

1. Token streams unchanged. 2. Mean val <= 3.28 at p < 0.01, waived for systems-only changes.
3. No extra compile flags. 4. Faster than the prior record on the same hardware: run
`tools/speedrun_ab/ab_bench.py` interleaved against `4ea6b93` on one 8xH100 node.
