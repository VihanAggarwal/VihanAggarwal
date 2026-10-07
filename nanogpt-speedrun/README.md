# NanoGPT speedrun: optimization work in progress

Work toward beating the current track-1 record of [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt):
record #92 (ANVIL2, 39.9 s on 8xH100). Base: upstream commit `4ea6b93`.

**Status: not yet a record.** No change here has been timed on 8xH100. Every claim below comes
from CPU tests and from the record's own published logs.

**Fork:** [VihanAggarwal/modded-nanogpt @ claude/nanogpt-optimization-n49ur2](https://github.com/VihanAggarwal/modded-nanogpt/tree/claude/nanogpt-optimization-n49ur2)
holds two layers:

- **Systems layer**: the patches in `patches/` (apply with `git am` on upstream `4ea6b93`). These speedups change
  no ML (rule-2 waiver): the canonical-mask builder is spawned instead of forked at the clock's start, the loader
  overlaps the first-shard read and the final validation's reads, pinned blocks are reused, the BOS index uses
  numpy, every rank starts its clock at a barrier, plus the A/B harness and tests. Token streams are
  byte-identical (tested over the record's and the stack's schedules). Estimated 0.15-0.35 s, not yet measured.
- **ML stack** (on the fork only): open PRs #375 (token-normalized n-gram hashes, Daniel Monroe) and #379 (CPLM
  copy-sink pointer, NathanGodey) merged on top. #379 measured -4.6 s on its own; the combination is unmeasured
  and needs its own p < 0.01 run pool.

- **Candidate: Canon layers** (Allen-Zhu 2025; on the fork behind `CANON_LAYERS`, off by default): a causal
  4-tap per-channel conv with a residual on each attention and MLP input. No modded-nanogpt PR has tried it. On the
  one-GPU proxy it was the biggest new win (-207 millinats alone, -48 / -92 on a stack of record techniques; 4 seeds,
  20M tokens). The record already has the partial key offset (#169), smear and the n-gram table, which overlap it, so
  the expected gain at record scale is ~5-20 millinats, worth ~0.5-3 s net if the conv is cheap. Round 3 of the proxy
  (`tools/proxy/nanogpt_canon_screen.ipynb`) re-tests it on a record-like base; `tools/speedrun_ab/sweep_canon.sh`
  measures it on 8xH100.

Tools on the fork:
- `tools/speedrun_ab/`: interleaved 8xH100 A/B and sweeps (`sweep_stack.sh`, `sweep_canon.sh`).
- `tools/retrieval_gate/`: the go/no-go test for stream-only retrieval, the only route the research found to -10 s.
- `tools/proxy/`: one-GPU (Colab) screens of architecture and optimizer ideas; self-contained notebooks.
- `tools/gpu_smoke/`: a one-GPU check of the systems layer under real CUDA.
- `tools/RULES_CHECK.md`: each layer against each rule.

## Apply the systems layer

```bash
git clone https://github.com/KellerJordan/modded-nanogpt && cd modded-nanogpt
git checkout 4ea6b93
git am /path/to/nanogpt-speedrun/patches/*.patch
```

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

| PR | claim | note |
|---|---|---|
| #379 CPLM | 36.0 s (n=8, p=0.0005) | in the fork's ML stack |
| #367 exact-match retrieval | 21.6 s | its validation index covers all 103 train shards, though a run trains on ~2 |
| #380 exact-count chain (on #367) | 9.65 s | also counts over train shards the loader never reads |
| #381 | no new measurement | |

If maintainers accept the all-shard retrieval PRs, the record falls to ~9.65 s. `tools/retrieval_gate` measures
how much of retrieval survives with a memory of only the trained-on tokens (the rule-safe version). The research
puts a legitimate -10 s from 39.9 s at ~25% overall; it hinges on that gate.

## Testing on one GPU (Colab)

Colab gives one GPU, and the speedrun needs 8xH100 in one node, so Colab cannot run or time a record.
`tools/gpu_smoke/smoke_1gpu.py` checks the patches under real CUDA and times what they remove; see
`tools/gpu_smoke/README.md` for the notebook cell.

## What a record needs (rules)

1. Token streams unchanged. 2. Mean val <= 3.28 at p < 0.01, waived for systems-only changes.
3. No extra compile flags. 4. Faster than the prior record on the same hardware: run
`tools/speedrun_ab/ab_bench.py` interleaved against `4ea6b93` on one 8xH100 node.
