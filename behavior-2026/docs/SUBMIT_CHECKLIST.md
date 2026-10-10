# Final submission checklist (do this once, at the end)

The latest valid submission replaces every earlier one. Never submit a test, partial, or exploratory run after the
real one.

## Before the final run
- [ ] Routing table and serving config are frozen (`configs/final.yaml`, committed). The routes were chosen on
      held-out instances 311-320 / train mode only, and `route_selection.md` was written.
- [ ] The Docker image is built from exactly that config and those checkpoints (`docker/build.sh`; its offline
      check passed), and pushed. Record the digest: `docker inspect --format '{{index .RepoDigests 0}}' <tag>`.
- [ ] **The organizers can pull it.** GHCR packages start private: make the package public (with a GR00T image,
      first check the Cosmos-Reason2 license), or put pull credentials in the portal comments. Then, from a machine
      that is logged out (`docker logout ghcr.io`), run `docker pull <repo>@sha256:<digest>`, start the pulled image
      and run `scripts/smoke_local.sh --no-start --ports 8000-8002` against it.
- [ ] The image passes `b1k26-probe --res full --chunk 20 --steps 500`, and `/healthz` turns 200 within 10 min.
      Fuller check against the running container: `scripts/smoke_local.sh --no-start --ports 8000-8002 --chunk K`.
- [ ] It also starts **without network**: the `offline test` command printed by `docker/build.sh`
      (`docker run --network none ...` with the server and the probe inside the container) reaches a healthy
      server and completes the probe.
- [ ] `/status` shows `engine.config_problems: []` (or every entry is understood) and the server log has no
      `config check:` ERROR lines.
- [ ] Decide the evaluator flags once and use them everywhere: the wrapper (default
      `omnigibson.eval.wrappers.RGBDFullResWrapper`; `DefaultWrapper` only if PLAN's wrapper switch was made) and the
      replay chunk size K (none = one query per step, always safe). K must divide `execution.execute_steps` of every
      routed profile (K > execute_steps pads with holds; a non-divisor drops planned actions and the inpainting tail
      every plan). `b1k26-package --config configs/final.yaml --replay-chunk-size K` checks it.
- [ ] The image was tested once on an Ampere GPU in fp32 or with the XLA bf16 upcast path (Turing-safe settings),
      and on 1 real rollout.
- [ ] The eval node image uses BEHAVIOR-1K `v3.9.3-post2` with the chosen wrapper and the default robot config.

## The run
- [ ] `b1k26-plan plan --instances 0-9 --final --workers N --out jobs/final`
- [ ] `scripts/run_node.sh --config configs/final.yaml --jobs jobs/final/worker_XX.jsonl --out runs/final_nodeXX
      --wrapper <W> [--extra "--replay-action-chunk-size K"]` on every node (one copy per GPU: different `--gpu`,
      `--port`, `--out`).
- [ ] No setting changes mid-run: same config, same image digest, same evaluator flags on every node.
- [ ] `b1k26-plan status runs/final_node* --jobs-dir jobs/final` reaches 1000/1000 (or you stop at the deadline
      buffer and submit what exists). Read its `policy_failures` (not re-run: they count as zero) and
      `rollouts_over_time_budget` (probably slower than the 2026 budget: investigate the node).

## Package
- [ ] `b1k26-plan collect runs/final_node* --into runs/final_all` exits 0 (no conflicting duplicates). It also
      merges every node's `status.jsonl` into `runs/final_all/status.jsonl`.
- [ ] `b1k26-score runs/final_all/json --validate --videos runs/final_all/videos --per-task`
- [ ] Locate the wrapper source used (b1k26-package also finds the official ones by itself):
      `python -c "import omnigibson.eval.wrappers.rgbd_full_res_wrapper as m; print(m.__file__)"`
- [ ] `b1k26-package --metrics runs/final_all/json --videos runs/final_all/videos --out submission --team ... --method ...
      --docker-image <repo>@sha256:<digest> --video-url <link> --config configs/final.yaml --wrapper <W>
      --wrapper-file <path> --replay-chunk-size <K or 0> --status-log runs/final_all/status.jsonl
      --routing-log route_selection.md`. It refuses placeholders, a missing wrapper, a K that does not divide every
      routed profile's execute_steps, and a README whose wrapper or K differs from the attempts in the status log.
- [ ] Upload `submission/metrics.zip` as the self-evaluation results: a **fresh HF dataset** whose only zip with
      "metric" in its path is `metrics.zip` (the leaderboard extractor takes the first such zip it finds), or a
      Google Drive **file** link to `metrics.zip`. Not a Drive folder: the extractor ignores zips in folders and
      leaves the score blank. Upload `submission/package.zip` too.
- [ ] Upload the videos (Drive folder or HF dataset). Check that every link (videos, metrics, README, image) opens in
      a private browser window / logged-out session.

## Portal (https://behavior-1k-2026-challenge-leaderboard.hf.space/submit)
- [ ] Team, affiliation, members (submitter first), contact email.
- [ ] Method (<= 25 characters).
- [ ] Submission type: Docker image submission, plus the image URI with digest (pullable without our credentials).
- [ ] Evaluation README URL, self-evaluation results URL (metrics.zip), video recordings URL.
- [ ] Additional data: "Not applicable" unless we trained on our own rollouts.
- [ ] Open-source release URL + consent (for the $1k open-source prize).
- [ ] Rules acknowledgement checked. Submit once, then wait 5 minutes: **do not resubmit**.
- [ ] Before Fri 2026-10-16 23:59 AoE (= Sat 11:59 UTC). Aim for Fri 18:00 UTC.
