# BEHAVIOR-1K 2026: six-day plan

Written 2026-10-10 (Sat). Deadline **Fri 2026-10-16 23:59 AoE**, which is Sat 2026-10-17 11:59 UTC / 04:59 PDT.
The research behind every number here is summarized in [`docs/RESEARCH.md`](docs/RESEARCH.md).

## 1. Where we stand

| Team (self-reported, 100 tasks) | Q | full SR | Notes |
|---|---|---|---|
| IF "VLA WAM" (10-08) | **0.485** | 0.269 | Unknown method (VLA + world-action model). Possibly worst-of-3 runs. |
| Mirua / Xiaomi "XR1" (10-08) | 0.373 | 0.176 | Best of 10 checkpoints per task on the reported instances. Best single checkpoint scored **0.270**, so expect about 0.26-0.29 on hidden. |
| Autolab / Anyverse "rondo" (09-21) | 0.322 | 0.148 | In-house model. Could resubmit. |
| Oct "WAM" (10-09) | not scored yet | | Possibly the 2025 winners (RLC) with a world-action model. |
| Zero-Shot Butlers (10-09) | ~0.185 | 0.043 | Public Comet pt50 (tasks 0-49) + private new-task model. |

**What wins:** the top 5 are re-run on hidden instances, and the hidden scores *replace* the public ones. In 2025,
entries that had selected on public instances lost 15-30% on hidden. A single honest policy that generalizes beats
a selected one.

**Honest odds.** We have a few GPUs and six days, and we start from zero. Training a model competitive with IF
(>0.5) is not possible in that time. Our path is to combine the best **public** checkpoints (strongest known:
JackLiu's 100-task PiBehavior model, Comet pt50, the GR00T N1.7 100-task model). On top of them we add the
inference-time techniques that won 2025, and we run a flawless, compliant evaluation.
- First place needs about 0.50 or more on hidden instances. That is a long shot unless a public checkpoint turns
  out much stronger than its scattered evidence suggests.
- **A podium (top 3) is realistic** at about 0.30-0.38: Mirua's honest level is about 0.27, and Autolab is 0.32.
- **Top 5** (hidden-test re-evaluation) needs about 0.2.
- The **$1k open-source prize** is very attainable: this repo, documented and released, is the entry.

## 2. Rules that shape the plan (all verified in the docs/code, see docs/RESEARCH.md)
1. One run of the final policy, one rollout per instance, on public instances **301-310** (indices 0-9).
   - No best-of-runs.
   - Re-running a rollout is allowed **only** after an infrastructure crash (no result and no sign of a policy
     failure). A rollout that failed because of the policy (lost connection, bad reply, query timeout) counts as a
     failure under the 10/09 rules and is **never** re-run (`b1k26-plan run` enforces both). We log every attempt.
2. Choose checkpoints and routes using held-out instances **311-320** (explicitly allowed "test set") or train-mode
   instances. **Never** choose using 301-310.
3. Only RGB, depth and proprioception go into the policy (plus the evaluator's `task_id`).
   - The wrapper must be submitted; we use the official `RGBDFullResWrapper` unchanged.
   - Any number of checkpoints routed by `task_id` counts as one entry.
4. Each rollout must average at least 1 step/s (timeout = max_steps seconds). Each query must take under 600 s.
5. The final eval runs our Docker image on **one 24 GB GPU (RTX 3090 / A5000 / TITAN RTX = Turing, no bf16
   tensor cores)**. IP hosting would need 50+ ports through November, which is not realistic for us.
6. **The latest valid submission replaces earlier ones.** Never submit anything after the final run.

## 3. Candidates to screen (no training needed)

| id | checkpoint | family / backend | why | status |
|---|---|---|---|---|
| A | `JackLiu0406/meta-SFT-checkpoints` `meta100-1epoch/step139999` (and `step69999`) | PiBehavior (`pibehavior`) | RLC 2025 architecture, 1 epoch over all 20k 2026 demos on 8xB300. Third-party measurement of **`step69999`** (not 139999): Q 0.65 on task 0 and 0.53 on task 1 (20 instances), similar to Mirua's best. | **Gated: request access now** |
| A' | same repo, `single-task-finetune/no-da3/<task>-140k` | PiBehavior | Per-task fine-tunes for all 100 tasks. Route candidates. | gated |
| B | `sunshk/openpi_comet` `pi05-b1kpt50-cs32` | openpi-comet (`openpi_comet`) | Measured **0.199 on tasks 0-49** under the 2026 evaluator, zero-shot. Language-conditioned, so some transfer to new tasks. | public |
| C | `kmy17518/gr00t-n1.7-b1k-multitask` `checkpoint-238000` | GR00T N1.7 (`gr00t`) | Trained on all 100 tasks, ~487M samples (2.3 epochs). Never evaluated. | public (accept Cosmos-Reason2-2B terms) |
| D | `Hoshipu/pi05-b1k100t-2026-lr2.5e5` `ckpt-4000000` | openpi `pi05_b1k` (`openpi_b1k`) | pi0.5 trained on 100 tasks. Never evaluated. | public |
| E | `IliaLarchenko/behavior_submission` ckpt1-4 | PiBehavior, tasks 0-49 only | 2025 winner. Needs base-velocity masking; transfer uncertain (FengGuo: on par with Comet on 6 tasks). | public |

## 4. Schedule with gates

### Sat 10/10 (today): access, machines, environment
1. **Request access**:
   - HF `JackLiu0406/meta-SFT-checkpoints` (manual approval).
   - Accept terms for `nvidia/Cosmos-Reason2-2B`.
   - Optionally the Inference Speed Form: https://forms.gle/5GNqc2eNhFTx4UhN8.
2. **Rent eval GPUs.** Requirements (the setup script enforces the starred ones):
   - **RT-core GPUs only**\*: 4090 / 5090 / L40S / RTX 6000 Ada / A6000. A100/H100 cannot render the simulator.
   - NVIDIA driver at least 580.65.06\*, and **not 595.x** (the script only warns on 595.x).
   - Rent at least 12 vCPU, 48 GB RAM and 250 GB disk per GPU. The script refuses below 30 GB RAM or 150 GB free
     disk\* and only warns below 8 vCPU, so check the rest yourself.
   - Vast 4090 is about $0.40-0.60/h; RunPod L40S about $0.80-1.10/h.
   - Rent elastically: 4-8 GPUs Sun-Tue (bring-up, screening, confirmation), 16-20 GPUs for the Wed-Thu final run
     (~30 h). That matches section 5's ~630-830 GPU-h, about $300-500 on Vast 4090s. Keeping 16-20 GPUs from Sun to
     Fri would be ~2000-2400 GPU-h (~$800-1400).
3. On one node, run `scripts/setup_eval_node.sh` (installs BEHAVIOR-1K `v3.9.3-post2`, assets and task instances, and runs a zero-action smoke rollout). Snapshot it as a template or volume, then clone it to the other nodes.
4. On the same node: `scripts/envs/<backend>.sh` for B, C, D (and A when access arrives), plus
   `python scripts/download_checkpoints.py --dest /ckpt comet_pt50 gr00t_multitask hoshipu_100t` (and
   `jackliu_meta100` with `HF_TOKEN`). Checkpoints land in `/ckpt/<name>/...`, the paths the example configs use.

### Sun 10/11: bring-up (gate G1: every candidate drives the robot without errors)
- `scripts/smoke_local.sh` (front server + fake worker + evaluator-faithful probe) passes on the node.
- One real rollout per candidate, `turning_on_radio` index 10 (id 311), with `--write-video`. Watch the video: the
  arms and gripper must move sensibly, the base must not spin, and there must be no NaN.
- **Reproduce the reference:** candidate B on 3 old tasks should land near ZSB's numbers (radio about 0.2-1.0, trash about 0.6). If it does not, the adapter is wrong; fix it before screening anything.

### Mon 10/12: screening (gate G2: pick the policy family)
- Probe = `configs/probe_tasks.txt` (24 tasks, 12 old + 12 new) x indices 10-11 (ids 311-312) for each candidate:
  ~19 GPU-h per candidate.
  `b1k26-plan plan --tasks @configs/probe_tasks.txt --instances 10-11 --workers N --out jobs/probe_<cand>`, then
  `scripts/run_node.sh --config configs/<cand>.yaml --jobs jobs/probe_<cand>/worker_XX.jsonl --out runs/probe_<cand>_XX`.
- For the best two candidates, also try execution variants (compression on/off, chunk length).
- `b1k26-score runs/probe_<cand> --per-task`. Compare old-task and new-task means separately.
- **Decision rule:**
  - Pick the default = best mean over all 24 probe tasks.
  - If another candidate wins the *old* half by more than 0.05 and is restricted to 0-49 (B or E), route tasks
    0-49 to it ("group routing": coarse, low-noise).
  - No per-task routing yet.
- **Office hours Mon 5-6 pm PT (Zoom):** ask about Docker GPU sharing and driver version, whether the hidden
  test uses the 2026/eval multi-port code, and confirm that routes picked on 311-320 are fine.

### Tue 10/13: confirm on all 100 tasks (gate G3: freeze)
- Run the chosen policy (and runner-up) on **all 100 tasks x index 10** (id 311): ~55-60 GPU-h each (the planner's
  worst case is 59.4 GPU-h; early successes save 5-10%).
  `b1k26-plan plan --instances 10 --workers N --out jobs/confirm_<cand>`.
- Optional per-task routing: `b1k26-select` with shrinkage.
  - Accept it only if the **split-half cross-validated gain** is positive. Otherwise keep group routing.
  - The CV scores up to 20 distinct splits of the instance ids. With only ids 311-312 there are just two (each id
    trains once, tests once), so a small positive gain is weak evidence.
- **Freeze** the routing table, configs and checkpoints by Tue night.
- Build and test the Docker image (`docker/build.sh`, then `scripts/smoke_local.sh` against the container), and
  run the Turing-safe path (fp32 or XLA upcast) once.

### Wed 10/14 - Thu 10/15: the final run (one run, never repeated)
- `b1k26-plan plan --instances 0-9 --final --workers N --out jobs/final` gives 1000 jobs packed longest-first
  (each worker then runs its own jobs shortest-first). `--final` is required for the reported instances 301-310.
- `scripts/run_node.sh --config configs/final.yaml --jobs jobs/final/worker_XX.jsonl --out runs/final_nodeXX` on
  every node, with identical `--wrapper` / `--extra` everywhere. It keeps going on its own: it re-runs only jobs that
  produced no JSON because of an infrastructure crash, and stops if the policy server cannot be brought back.
- Expected cost is about 400-600 GPU-h (RGB-D full-res plus policy): ~25-30 h on 20 GPUs.
  - Start by Wed noon UTC at the latest. The longest single rollout (gift baskets, 39k steps) takes ~1.5 h.
- `b1k26-plan status runs/final_node* --jobs-dir jobs/final` gives the ETA, policy failures and rollouts
  probably over the 2026 time budget.

### Fri 10/16: package and submit (by Fri 18:00 UTC; hard stop Sat 11:59 UTC)
1. `b1k26-plan collect` (refuses mixed runs).
2. `b1k26-score --validate`.
3. `b1k26-package` with the exact command in `docs/SUBMIT_CHECKLIST.md` (it refuses placeholders, a missing
   wrapper and a README whose wrapper or chunk size differs from the run's `status.jsonl`).
4. Upload `metrics.zip` + `package.zip` and the videos. Self-evaluation results URL: a **fresh HF dataset**
   whose only zip with "metric" in its path is `metrics.zip` (the leaderboard extractor takes the first such zip),
   or a Google Drive **file** link to `metrics.zip`. Never a Drive folder holding the zip: the extractor reads only
   loose JSONs from folders, so the score would be left blank.
5. Make sure the organizers can pull the Docker image: public registry package (or pull credentials in the
   portal comments), checked with `docker logout; docker pull <repo>@sha256:<digest>` (SUBMIT_CHECKLIST).
6. Fill the portal at https://behavior-1k-2026-challenge-leaderboard.hf.space/submit. The method field allows at most 25 characters.
- Release the repo (open-source prize) and give its URL in the portal's release field.

**If the final run is not complete by Fri 12:00 UTC:** submit what is done. Missing rollouts count as zero, and
partial submissions are allowed. A finished 900 beats an unfinished 1000. Each worker runs its jobs shortest-first,
so what is missing is each worker's last, longest jobs (instances of the longest tasks): the fewest rollouts per lost
hour.

### Contingency: no public model is decent on the new tasks 50-99
Tasks 50-99 are half the score, and only the 100-task models (A, C, D) were trained on them. Comet pt50 has seen
none of them.
- **If A is not granted and C and D both score below ~0.10 on the new half of the probe:** start per-task
  fine-tunes on your own 80 GB GPUs.
  - Pipeline: [JackLiu's 2026 trainer](https://github.com/JackLiu0406/behaviour-1k-2026-meta) (LeRobot v3 loader,
    100-task tables), initialized from the public RLC 50-task checkpoint.
  - Data: the 224-px re-encode `JackLiu0406/b1k-224x224-gop8-fixed`. Apply the seek fix noted in
    `docs/RESEARCH.md`.
  - Rena-Tian's public recipe is 10k steps at batch 16, about 9 h on one A100 per task. It gave 0.04-0.57 on new
    tasks.
- Spend the GPUs on the new tasks with the most predicates and the shortest demos, which have the cheapest partial
  credit (e.g. 77 modem, 63 smoke detectors, 89 fax, 92 scanner, 90 composting).
- Fold the results in through `routing.per_task`, at most a handful of tasks (image size).
- Expected gain is about +0.02-0.04 overall. Worth it only if the GPUs would otherwise idle.

**Wrapper variant to test on Monday.** Rendering native 224 RGB (`DefaultWrapper`) instead of full-res RGB-D
makes the simulator about 1.8x faster, which saves about 40% of the final run's GPU-hours.
- 2025 evidence is mixed: Comet lost on one task, RLC saw no difference. Zero-Shot Butlers got Comet's full 0.199
  with what appears (from their videos) to be the default 224 wrapper.
- Run the chosen policy on the probe with both wrappers. Switch only if the 224 Q is within noise (about 0.02)
  **and** compute is the binding constraint.
- To switch: `scripts/run_node.sh --wrapper omnigibson.eval.wrappers.DefaultWrapper` on every node, and
  `b1k26-package --wrapper omnigibson.eval.wrappers.DefaultWrapper` (it refuses a README whose wrapper differs from
  the one in `status.jsonl`, and copies that wrapper's source).

## 5. Compute budget (RT-core GPU-hours)

| stage | rollouts | GPU-h (approx.) |
|---|---|---|
| bring-up | ~10 | 5 |
| screening, 4 candidates x 24 tasks x 2 | 192 | 80 |
| variants on best two | 96 | 40 |
| confirmation, 2 x 100 tasks x 1 | 200 | 120 |
| **final run**, 100 x 10 | 1000 | 400-600 |
| **total** | | **~630-830** |

That is about **$300-500** on Vast 4090s, or about $600-900 on RunPod L40S. With fewer GPUs, cut the confirmation
stage first, then the variants. Never cut the final run's start date.

**Your own few GPUs:**
- If they are RT-capable (RTX/L40S), add them as eval workers.
- If they are A100/H100, they cannot render. Use them for the image build and for checkpoint conversion, and
  optionally for a small targeted fine-tune of the chosen model on its weakest new tasks (only if screening ends
  early; expected gain is small, +0.01-0.02).

## 6. What the toolkit adds on top of a checkpoint (all configurable per profile)
- **Receding-horizon chunk execution** (no temporal ensembling; 2025 ablations: 0 vs 0.25-0.30).
- **RLC action compression:** 26 predicted actions are executed in 20 steps with cubic resampling and base
  velocity x1.3, and compression is disabled while the gripper is moving.
  - The 2026 timeout is 1.5x the human mean (was 2x), so speed is worth partial credit.
- **Gripper reopen rule** for "closed on nothing" (2.2x Q on 13 tasks in 2025), generalized to all 100 tasks from
  demo statistics.
- **Stage tracking with voting** for PiBehavior models; soft inpainting across chunk boundaries.
- **Base-velocity masking** for 2025 checkpoints. 2026 proprio reports a robot-frame velocity they never saw.
- **Never crash, never zero:**
  - Errors return a hold-pose action, never zeros (a zero action makes the torso stand up).
  - Reconnects replay the cached response.
  - `/healthz` only turns green after warmup.
  - Micro-batching keeps many concurrent rollouts above 1 step/s on one GPU.

## 7. Risks

| risk | mitigation |
|---|---|
| JackLiu access not granted | Candidates B, C and D are public; the plan does not depend on A. |
| A candidate's adapter is subtly wrong (state order, norm stats, prompts) | Gate G1: reproduce ZSB's Comet numbers first; watch videos; the backends' input builders are unit-tested against the reference code. |
| Eval nodes crash or are slow | Per-rollout processes; resume; JSON-presence check (Isaac exits 0 on crashes); LPT packing; 20% spare capacity. |
| Organizers' Docker GPU is a TITAN RTX (no bf16) or has an old driver | Turing-safe settings are tested; ask at office hours / connection test. |
| Hidden eval runs many rollouts against one container | Micro-batching scheduler; capacity stated in the README. |
| Selection overfitting | Group routing by default; per-task routes only with a positive split-half CV gain; never select on 301-310. |
| Submitting a worse run last | Only `docs/SUBMIT_CHECKLIST.md` submits, and only once. |
| Too many checkpoints for one 24 GB GPU / a huge image | Final policy uses at most 2 checkpoints (both resident: ~2 x 7.5 GB bf16). Per-task fine-tunes (12.6 GB each) are routed only for a few tasks with large held-out gains, and only if office hours confirm rollouts are not interleaved across tasks per container. |
