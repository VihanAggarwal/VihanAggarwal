# Research notes: BEHAVIOR-1K 2026 Challenge (as of 2026-10-10)

These are the facts this toolkit and `PLAN.md` rely on. Each one was checked against the evaluator source, the challenge
pages, or the leaderboard Space files. **UNVERIFIED** marks anything that could not be confirmed.

## Challenge mechanics
- **Scope:** 100 household tasks (0-49 = the 2025 set, 50-99 new) across 7 scenes (4 new: restaurant_diner, Rs_int,
  hotel_suite_large, office_cubicles_right). 20,000 teleop demos (200 per task), LeRobot v3,
  `behavior-1k/2026-challenge-demos`.
- **Deadline:** Oct 16, 2026 23:59 AoE. The Space hard-codes `SUBMISSION_DEADLINE_UTC = 2026-10-17 11:59`. Winners are
  announced 11/04; the event is CoRL 2026 (Austin). Prizes: $5k, $3k, $2k, plus $1k for outstanding open source.
- **Version:** no plain `v3.9.3` tag exists. Use `v3.9.3-post1`/`-post2`. The organizers will use `v3.9.3-post1` for
  hidden tests; `-post2` is needed only for custom robots.
  - The 10/09 rules (reconnects, 600 s per query, per-rollout time budget, `--policy-endpoints` multi-port)
    exist only on the unmerged branch `2026/eval` (PR #2366, head 020ca52). Our server works with both.
- **Observations:** robot-onboard RGB, depth and proprioception only.
  - Keys: `robot_r1::proprio` (N,61); `robot_r1::robot_r1:{zed_link,left_realsense_link,right_realsense_link}:Camera:0::rgb` (N,H,W,4) uint8;
    optional `::depth_linear`; `robot_r1::cam_rel_poses` (N,21); `task_id` (N,1).
  - Observations are batched since v3.9.3; v3.9.2 is unbatched.
- **Action:** 23-D absolute.
  - base 0:3 = normalized velocity in [-1, 1], x0.75 m/s, x0.75 m/s, x1 rad/s, robot frame.
  - torso 3:7, left arm 7:14, right arm 15:22 = joint positions (rad).
  - grippers 14 and 22 in [-1 closed, +1 open].
  - **A zero action stands the torso upright**; it is not a "hold".
- **Protocol:** `GET /healthz` must return 200 first. Then the websocket server sends a msgpack metadata map before
  anything else. Observations and actions are msgpack with `__ndarray__` maps.
  - `{"reset": True}` gets **no reply**.
  - Optional chunk replay: the request carries `__action_chunk_size__: K`; the reply needs `action_chunk` (N,K,23)
    with `[...,0,:]` exactly equal to `action`.
- **Episode end:** full success (all BDDL goals) or `max_steps = int(1.5 x human mean length)`.
  - The total over 100 tasks x 10 instances is 15,818,280 steps worst case.
  - The 10/09 rule adds a wall-clock timeout of max_steps seconds per rollout (>= 1 step/s on average) and at most
    600 s per action query.
- **Score:** per rollout Q = 1 if success, else max over grounded goal options of (#literals that were **false at
  start and true at the end**) / (#literals).
  - Literals true at the start count in the denominator but can never earn credit.
  - Only the final state counts.
  - Ranking = mean over 100 tasks of per-task mean Q. Ties are broken by efficiency (time, base distance, EEF
    displacement vs human).
- **Instances:** public_test indices 0-9 = ids 301-310 are reported, one rollout each. Indices 10-19 = ids 311-320
  are "a test set before evaluating your final policy". Hidden: 321-340.
  - The top ~5 are re-evaluated on hidden instances, and **hidden scores replace public ones**.
  - Partial submissions are allowed; missing rollouts count as zero.
- **No cherry-picking:** one evaluation run of the final policy, one rollout per instance; do not assemble best
  results across runs, instances or tasks. Multiple checkpoints routed by task are one entry.
- **Delivery:** Docker image on one 24 GB GPU (RTX 3090 / A5000 / TITAN RTX), or an IP endpoint with at least 50
  ports. **The latest valid submission counts** (a later, worse one replaces an earlier, better one).
  - Package: metrics JSONs, wrapper `.py`, robot config, README with the exact commands, plus videos.
- **Data and models:** privileged info is allowed for training only. Extra data (teleop, RL, scripted) is allowed and
  must be declared. Pretrained models are allowed. External APIs are allowed if you supply the credentials.

## Competitive landscape (Space file `data/self_reported_results.jsonl`, UI scores hidden since 10/09)
- **IF "VLA WAM" 0.4846** (SR 0.269, 10-08). Method and team are unknown (VLA + world-action model). The artifact name
  suggests worst-of-3 runs (UNVERIFIED).
- **Mirua / Xiaomi "XR1" 0.3726.** This is the best of 10 checkpoints per task on the same reported instances. Their
  best single checkpoint scores **0.2699**, and shrinkage puts their true level at about 0.26.
- **Autolab / Anyverse "rondo" 0.3221** (09-21).
- **Zero-Shot Butlers ~0.185:** public Comet pt50 on tasks 0-49 (**0.199 per task, zero-shot**) plus a private 30k-step
  model for tasks 50-99 (0.17).
- **Oct "WAM"** (10-09) is not scored yet. It may be the 2025 winners (RLC member Akash Karnatak has private
  Fast-WAM / LingBot-VA B1K artifacts) (UNVERIFIED).
- **2025 for reference:** RLC 0.2599 hidden (0.2605 public); Comet 0.2514. Teams that selected on public instances
  lost 15-30% on hidden.

## What worked in 2025 (ranked by measured gain)
1. **Multi-task training for enough epochs.** A single pi0.5 trained 4 epochs on all 50 tasks = 0.2626, the same as the
   winner (G0.5 paper, arXiv 2608.11739). RLC said undertraining was their main limit.
2. **Rejection-sampling fine-tuning on perturbed starts** (Comet): validation 0.192 to 0.345 post-challenge
   (contaminated by validation instances).
3. **Gripper "reopen if closed on nothing" rule** (RLC): 2.2x Q on 13 tasks.
4. **Receding-horizon execution of most of a ~30-step chunk:** temporal ensembling 0.00 vs receding 0.25-0.30.
   Horizon 8 gives 0; 32 gives 0.30.
5. **RLC action compression** (26 to 20 steps, base x1.3, off while the gripper moves): "no loss". It matters more
   with the 1.5x timeout.
6. **No gain from:** depth input, manipulation/navigation reweighting, or duration-based advantage weighting.
   Delta vs absolute actions is contested.
- RLC architecture (PiBehavior): task embeddings instead of text; a stage head with 3-vote tracking; correlated
  flow noise; KV transform; FAST auxiliary loss; inpainting across chunks; 20 flow steps.

## Public checkpoints that matter
| checkpoint | trained on | evidence | notes |
|---|---|---|---|
| `JackLiu0406/meta-SFT-checkpoints` meta100 step69999/139999 | all 100 tasks, 1 epoch (8xB300) | Q 0.65 (task 0), 0.53 (task 1) on 20 instances | gated (manual). Per-task fine-tunes for all 100 tasks are in `single-task-finetune/no-da3/` |
| `sunshk/openpi_comet` pt50 | 2025 50 tasks | **0.199/task zero-shot on 2026 tasks 0-49** | language-conditioned; quantile norm; horizon 32 |
| `kmy17518/gr00t-n1.7-b1k-multitask` ckpt-238000 | all 100 tasks, ~487M samples | none published | LR not annealed; serve receding-horizon, not temporal ensemble |
| `Hoshipu/pi05-b1k100t-2026-*` | 100 tasks (per name) | none | standard openpi format |
| `IliaLarchenko/behavior_submission` | 2025 50 tasks | 2025 winner; 2026 transfer unclear | mask base velocity (2025 data had ~0) |

**2025 checkpoints on the 2026 evaluator:**
- 2026 proprio is 61-D, not 256-D.
- base velocity is robot-frame and real-valued; in the 2025 data it was about 0.
- Comet orders the state with grippers last, as raw widths. RLC uses action order, with grippers mapped to [-1, 1].

## Compute facts
- **Rendering needs RT cores.** A100/H100 cannot render BEHAVIOR scenes. Isaac Sim 5.1 needs driver >= 580.65.06;
  595.x has been reported to crash.
- **Speed (RTX 4090 + 7950X, random actions):**

  | wrapper | FPS |
  |---|---|
  | RGB 224 | 24.6 |
  | RGB-D full-res | 13.5 |

  - Measured all-in with a policy: 0.064-0.2 s/step.
  - Scene load: 90-300 s per process.
- **Full 1000-rollout run:** about 91-100% of the worst-case steps get used, i.e. ~400-700 GPU-h on 4090/L40S-class
  hosts.
- **Policy VRAM:** pi0.5 inference about 7.5-9 GB (bf16). The simulator takes 8-16 GB. Both fit on one 24 GB GPU.
- **Turing (TITAN RTX):** no bf16 tensor cores.
  - JAX/XLA upcasts bf16 matmuls to f32 (works, slower).
  - fp16 is unsafe for Gemma.
  - flash-attn 2 refuses sm_75 (GR00T must use SDPA).

## Sources
- Challenge pages: https://behavior.stanford.edu/challenge/ (index, evaluation, submission, updates, baselines, dataset)
- Code: https://github.com/StanfordVL/BEHAVIOR-1K (tags v3.9.3-post1/post2, branch 2026/eval, PRs #2364-#2366)
- Leaderboard Space: https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard (files `app.py`,
  `scripts/extract_self_reported_scores.py`, `data/self_reported_results.jsonl`)
- 2025 reports: RLC arXiv 2512.06951; Comet arXiv 2512.10071; G0.5 arXiv 2608.11739; "How VLAs (Really) Work" arXiv 2604.21192
- Competitor artifacts: HF datasets `li-qing/behavior-challenge-2026-submission` (Mirua), `sriharsha4444/b1k-2026-comet-pt50` (ZSB)
- Checkpoint repos linked in the table above; JackLiu port: https://github.com/JackLiu0406/behaviour-1k-2026-meta
