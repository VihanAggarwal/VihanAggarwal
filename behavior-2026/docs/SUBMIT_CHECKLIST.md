# Final submission checklist (do this once, at the end)

The latest valid submission replaces every earlier one. Never submit a test, partial, or exploratory run after the
real one.

## Before the final run
- [ ] Routing table and serving config are frozen (`configs/final.yaml`, committed). The routes were chosen on
      held-out instances 311-320 / train mode only, and `route_selection.md` was written.
- [ ] The Docker image is built from exactly that config and those checkpoints, and pushed. Record the digest:
      `docker inspect --format '{{index .RepoDigests 0}}' <tag>`.
- [ ] The image passes `b1k26-probe --res full --chunk 20 --steps 500`, and `/healthz` turns 200 within 10 min.
- [ ] The image was tested once on an Ampere GPU in fp32 or with the XLA bf16 upcast path (Turing-safe settings),
      and on 1 real rollout.
- [ ] The eval node image uses BEHAVIOR-1K `v3.9.3-post2` with the official wrapper
      `omnigibson.eval.wrappers.RGBDFullResWrapper` and the default robot config.

## The run
- [ ] `b1k26-plan plan --instances 0-9 --workers N --out jobs/final`
- [ ] `scripts/run_node.sh --config configs/final.yaml --jobs jobs/final/worker_XX.jsonl --out runs/final_nodeXX` on every node
- [ ] No setting changes mid-run: same config, same image digest, same evaluator flags on every node.
- [ ] `b1k26-plan status runs/final_node* --jobs-dir jobs/final` reaches 1000/1000 (or you stop at the deadline
      buffer and submit what exists).

## Package
- [ ] `b1k26-plan collect runs/final_node* --into runs/final_all` exits 0 (no conflicting duplicates).
- [ ] `b1k26-score runs/final_all/json --validate --videos runs/final_all/videos --per-task`
- [ ] Copy the official wrapper source for the package:
      `python -c "import omnigibson.eval.wrappers.rgbd_full_res_wrapper as m; print(m.__file__)"`
- [ ] `b1k26-package --metrics runs/final_all/json --videos runs/final_all/videos --out submission --team ... --method ... --docker-image <tag@digest> --wrapper-file <path> --status-log <merged status.jsonl> --routing-log route_selection.md`
- [ ] Upload `submission/metrics.zip` (e.g. HF dataset, file name contains "metric") and `submission/package.zip`.
- [ ] Upload the videos (Drive folder or HF dataset). Check that the links open in a private browser window.

## Portal (https://behavior-1k-2026-challenge-leaderboard.hf.space/submit)
- [ ] Team, affiliation, members (submitter first), contact email.
- [ ] Method (<= 25 characters).
- [ ] Submission type: Docker image submission, plus the image URI with digest.
- [ ] Evaluation README URL, self-evaluation results URL (metrics.zip), video recordings URL.
- [ ] Additional data: "Not applicable" unless we trained on our own rollouts.
- [ ] Open-source release URL + consent (for the $1k open-source prize).
- [ ] Rules acknowledgement checked. Submit once, then wait 5 minutes: **do not resubmit**.
- [ ] Before Fri 2026-10-16 23:59 AoE (= Sat 11:59 UTC). Aim for Fri 18:00 UTC.
