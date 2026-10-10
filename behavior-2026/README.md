# b1k26: a BEHAVIOR-1K 2026 Challenge toolkit

`b1k26` serves, controls and evaluates policies for the
[2026 BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/) (100 household tasks, R1Pro, OmniGibson /
Isaac Sim). It wraps public checkpoints behind one hardened policy server and adds model-independent
inference-time control. It also covers everything around the run: planning the 1000-rollout evaluation,
scoring it the way the organizers do, choosing checkpoints without overfitting, and packaging a compliant
submission.

- **[PLAN.md](PLAN.md)**: the six-day plan, candidates, compute budget and gates. Start here.
- **[docs/RESEARCH.md](docs/RESEARCH.md)**: verified facts about the rules, evaluator, leaderboard and 2025 winners.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: module contracts and the worker protocol.
- **[docs/BACKENDS.md](docs/BACKENDS.md)**: supported model families, checkpoints, environment setup.
- **[docs/SUBMIT_CHECKLIST.md](docs/SUBMIT_CHECKLIST.md)**: the one-time final submission procedure.

## Components

| piece | what it does |
|---|---|
| `b1k26-serve` | Front server speaking the evaluator's websocket protocol on many ports at once. Handles `/healthz` gating, the metadata frame, reset-without-reply, batched/unbatched observations, exact `action_chunk` replies, and reconnect replay. Never closes a socket on error; falls back to a hold-pose action (never zeros). |
| control layer | Receding-horizon chunk execution, RLC-style action compression, gripper reopen rule for all 100 tasks, stage voting, inpainting prefix, base-velocity masking for 2025 checkpoints, micro-batched inference. |
| `b1k26-worker` + backends | One process per model family, each in its own env: openpi-comet (Comet pt50), openpi `pi05_b1k` (organizer baseline, Hoshipu 100-task), PiBehavior (RLC 2025, JackLiu 100-task + per-task fine-tunes), GR00T N1.7. |
| `b1k26-plan` | Packs 1000 rollouts onto N GPUs (longest-first; each GPU runs its share shortest-first). The resumable runner retries only rollouts that produced no result because of an infrastructure crash (never a policy failure) and logs every attempt with its command. Also shows status/ETA and merges node outputs, refusing mixed runs. |
| `b1k26-score` | Leaderboard and official Q, per-task tables, tie-breakers, submission validation. |
| `b1k26-select` | Per-task routing from held-out rollouts with empirical-Bayes shrinkage and split-half cross-validation. Refuses to select on the reported instances. |
| `b1k26-package` | `metrics.zip` (only rollout JSONs), `package.zip` (JSONs + wrapper + robot config + README + checksums), video manifest. Refuses to state anything the run did not do (wrapper, chunk size, placeholders). |
| `scripts/` | Cloud node bootstrap (driver/RT-core checks, BEHAVIOR-1K install, smoke rollout), per-family env setup, checkpoint downloads, per-node runner. |
| `docker/` | The final policy image: one 24 GB GPU, Turing-safe, enroot-friendly, weights and every file fetched at load baked in (checked offline at build time). |

## Quick start (on an RT-core GPU node)

```bash
bash scripts/setup_eval_node.sh                               # BEHAVIOR-1K v3.9.3-post2 + assets + smoke rollout
source /workspace/b1k_env.sh                                  # the behavior env: b1k26 CLIs + huggingface_hub
bash scripts/envs/openpi_comet.sh --prefix /opt/envs/openpi_comet
python scripts/download_checkpoints.py --dest /ckpt comet_pt50   # -> /ckpt/comet_pt50/pi05-b1kpt50-cs32 (the config's path)

# one held-out rollout with the Comet pt50 profile
b1k26-plan plan --tasks turning_on_radio --instances 10 --workers 1 --out jobs/smoke
bash scripts/run_node.sh --config configs/comet_pt50.example.yaml --jobs jobs/smoke/worker_00.jsonl --out runs/smoke
b1k26-score runs/smoke --per-task
```
`scripts/download_checkpoints.py --list` shows every candidate; each lands in `/ckpt/<name>/...`, the layout the
example configs and `docker/build.sh --ckpt /ckpt/<name>` use.

## Tests

```bash
pip install -e '.[test]' && python -m pytest      # torch (CPU) is needed for the evaluator-client tests
bash scripts/smoke_local.sh                        # front server + fake worker + probe, full-res, chunked and not
```
The tests run on CPU. They drive the server with **verbatim copies of the real evaluator clients** (v3.9.3-post2
and the 2026/eval multi-port client), so protocol compatibility is checked against the code the organizers run.
See "Testing" in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for what is covered and the measured latencies.

## License
MIT. The vendored evaluator client code in `tests/vendor/` is from
[BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) (MIT).
