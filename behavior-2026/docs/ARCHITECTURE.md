# b1k26 architecture

`b1k26` is the serving and evaluation toolkit for our BEHAVIOR-1K 2026 Challenge entry. It runs the policy
behind the evaluator's websocket protocol, adds inference-time control logic that is independent of the model,
and provides the offline tools for screening checkpoints, scoring and packaging the submission.

```
 OmniGibson evaluator (v3.9.3-post1/post2, or 2026/eval with --policy-endpoints)
        |  websocket, msgpack+ndarray, one request per step (or per K steps with chunk replay)
        v
 +-------------------- front server  (b1k26.server, any Python >= 3.10, numpy/msgpack/websockets/scipy) ---------+
 |  multi-port listener -> per-rollout RolloutSession (queue, stage, last action, reconnect cache)               |
 |  PolicyEngine: route task_id -> Profile -> plan a chunk when the queue runs dry                              |
 |    obs.split_batch -> obs.EnvObs (squeezed, RGB uint8, resized to the profile's input size)                  |
 |    InferenceScheduler: micro-batches plan requests per worker on one thread                                  |
 |    control: corrections -> compression -> sanitize -> queue                                                  |
 +------------------------------------------|-------------------------------------------------------------------+
                                            |  local websocket, same msgpack codec ("worker protocol" below)
               +----------------------------+-----------------------------+
               v                            v                             v
     worker (openpi-comet env)     worker (RLC/PiBehavior env)     worker (GR00T env)
     b1k26.worker + backends/...   one checkpoint (or LRU set)     each in its own Python env
```

Why two processes: the candidate model families pin incompatible stacks. openpi forks all install a module
named `openpi`; GR00T needs torch/transformers 4.57 and Python 3.10. The front server holds all the logic we
can test on CPU. Each worker is a thin adapter around one family's own `policy.infer`.

## Package layout (`behavior-2026/src/b1k26`)

| module | owner | purpose |
|---|---|---|
| `constants.py` | done | evaluator constants, key names, 61-D proprio slices, 23-D action slices, task table |
| `protocol.py` | done | msgpack ndarray codec, byte-compatible with `omnigibson.eval.utils.network_utils` |
| `obs.py` | core | batch splitting, image prep, proprio-to-state helpers, hold action, fingerprints |
| `control.py` | core | chunk post-processing: compression, sanitizing, gripper-variation check |
| `corrections.py` | core | generalized RLC gripper-reopen rule for 100 tasks, driven by `data/gripper_rules.json` |
| `stage.py` | core | RLC stage voting tracker |
| `config.py` | runtime | YAML config schema, `Profile`, `RoutingTable`, loading and validation |
| `session.py` | runtime | `RolloutSession` per rollout slot |
| `engine.py` | runtime | `PolicyEngine`, `InferenceScheduler`, `WorkerClient` |
| `server.py` | runtime | multi-port websocket front server + CLI `b1k26-serve` |
| `client.py` | runtime | evaluator-faithful test client + CLI `b1k26-probe` |
| `worker.py` | runtime | worker websocket server + CLI `b1k26-worker` |
| `backends/base.py`, `backends/fake.py` | runtime | backend interface and deterministic fake backends |
| `backends/openpi_comet.py`, `backends/openpi_b1k.py` | openpi | Comet pt50/pt12, wensi-ai `pi05_b1k`, Hoshipu pi05-b1k100t |
| `backends/pibehavior.py` | rlc | RLC 2025 checkpoints and the JackLiu meta100 / per-task fine-tunes |
| `backends/gr00t.py` | groot | GR00T N1.7 (kmy17518 multitask, organizer baseline) |
| `scoring.py`, `selection.py`, `orchestrate.py`, `package.py` | offline | score, choose routes, plan eval jobs, build the submission |

## Contracts

These are the interfaces as built (the "owner" column above records who wrote each module). Extras beyond the
original plan are marked *(added)*.

### obs.py
```python
@dataclass
class EnvObs:
    task_id: int                       # from obs["task_id"] (shape (N,1) int64 on v3.9.3+, may be (1,) or scalar)
    proprio: np.ndarray                # (61,) float32, always a private copy
    rgb: dict[str, np.ndarray]         # role -> (H, W, 3) uint8 for the roles present ("head" always)
    depth: dict[str, np.ndarray] = {}  # role -> (H, W) float32, empty if not sent
    cam_rel_poses: np.ndarray | None = None   # (21,) float32
    fingerprint: bytes = b""           # 16-byte blake2b of proprio + task_id + strided image subsample (~0.05 ms)

def split_batch(msg: dict, default_task_id: int | None = None, copy_images: bool = True) -> list[EnvObs]
    # Accepts the evaluator's flattened obs dict. Every value has a leading batch dim N on v3.9.3+ and on
    # 2026/eval multi-port (N=1). v3.9.2 sends unbatched values: detected via proprio.ndim == 1.
    # RGB arrives as (N, H, W, 4) uint8 RGBA; alpha is dropped. Float images in [0,1] are scaled to uint8.
    # Never mutates the (read-only) input arrays. Raises ValueError on missing/malformed proprio, head image or
    # task_id (task_id outside 0-99 included); the server catches it and still replies.
    # copy_images=False (added; the front server uses it): uint8 images and depth stay read-only views of the
    # message, so a step that does not plan copies nothing (full-res split: 0.05 ms instead of 1.2-3.8 ms).
def resize_with_pad(img, height, width, method="bilinear") -> np.ndarray
    # Bit-exact port of openpi_client.image_tools.resize_with_pad (PIL; zero padding, centered); also accepts
    # "bicubic", a "_pad" suffix ("bilinear_pad") and PIL constants. Returns the input object if already sized.
def prepare_images(env, size, method="bilinear", roles=("head", "left_wrist", "right_wrist")) -> dict[str, np.ndarray]
    # (size, size, 3) uint8, C-contiguous, never aliasing env.rgb; a missing role is filled with zeros.
def state23_action_order(proprio) -> np.ndarray
    # RLC order [base_qvel3, trunk4, L arm7, L grip, R arm7, R grip]; grip = 2*(finger_sum/0.1) - 1 (unclipped)
def hold_action(proprio) -> np.ndarray
    # (23,) float32 that holds the current pose: base 0, torso/arms = current qpos, grippers = normalized width
    # clipped to [-1, 1]; non-finite entries -> 0. NEVER zeros (a zero torso stands the robot upright).
# also: hold_from_state23, gripper_width_normalized, compute_fingerprint
```

### control.py
```python
@dataclass
class ExecutionConfig:
    execute_steps: int = 20          # actions sent to the robot per plan (the replan period)
    predicted_steps_to_use: int = 26 # predicted actions consumed per plan; > execute_steps means compression
    keep_for_inpaint: int = 4        # predicted actions after the consumed ones, kept for the next plan
    base_velocity_scale_with_compression: bool = True
    disable_compression_gripper_range: float | None = 0.2   # RLC GRIPPER_VARIATION_THRESHOLD; None = never
    clip_base: float = 1.0
    # validated in __post_init__; property `compresses`

@dataclass
class PlannedChunk:
    actions: np.ndarray              # (execute_steps or fewer, 23) float32, ready to send
    inpaint_tail: np.ndarray | None  # (keep_for_inpaint, 23) absolute actions for the next plan, or None
    compressed: bool

def plan_execution(raw: np.ndarray, cfg: ExecutionConfig) -> PlannedChunk
    # raw: (T, >=23) (or (1, T, D)) absolute actions, already corrected. Matches RLC's wrapper exactly: cubic
    # interp (scipy interp1d kind="cubic"; linear if scipy is missing or < 4 points) of raw[:predicted] onto
    # execute_steps samples, base dims scaled by predicted/execute; no compression when either gripper's range over
    # raw[:predicted] exceeds the threshold (then raw[:execute_steps]). The tail starts right after the consumed
    # predicted actions. Non-finite segments are never compressed; a non-finite tail becomes None.
def sanitize(actions: np.ndarray, proprio: np.ndarray, clip_base: float = 1.0) -> np.ndarray
    # (n, 23) or (23,): non-finite rows -> hold_action(proprio); base clipped to [-clip, clip], grippers to [-1, 1].
```

### corrections.py
```python
class GripperRules:
    @classmethod
    def load(cls, path: str | None = None) -> "GripperRules"   # default: package data/gripper_rules.json
    def apply(self, task_id, stage, state23, actions, progress=None) -> tuple[np.ndarray, bool]
    def apply_with_stage(self, task_id, stage, state23, actions, progress=None
                         ) -> tuple[np.ndarray, bool, int | None]   # (added) also returns RLC's task-0 stage reset
    # If a gripper is fully closed (state23 grip < -0.98, i.e. closed on nothing) in a task/stage/phase where the
    # human demos never close it fully, replace the chunk with "hold pose" (from state23 positions, base 0) with that
    # gripper set to +1 (open). Stage rules apply only when stage is not None; progress rules (min_progress, with
    # progress = task_progress(task_id, step) = step / human_mean_len) only when no stage rule applies.
```
`data/gripper_rules.json`: `{"closed_threshold": -0.98, "tasks": {"<id>": {"left": {"always_open": bool,
"min_stage": int|null, "min_progress": float|null}, "right": {...}, "exempt_right": bool, "rlc_task0_rule": bool}}}`.
Tasks 0-49 encode RLC's 2025 tables verbatim (plus a demo-derived progress gate used only without a stage); tasks
50-99 come from the 2026 demo statistics (`scripts/compute_gripper_rules.py`). The engine applies rules in RLC's
order: corrections (with `tracker.stage`, possibly forcing a new stage with `set_stage`) -> `plan_execution` ->
`sanitize` -> `tracker.update(logits)`.

### stage.py
```python
class StageTracker:     # exact port of RLC update_current_stage: history 3; once the history is full, promote to
                        # s+1 on >=2 votes for s+1, also move to s+1 (one stage only, like RLC's code) on 3 votes
                        # for s+2, roll back on 3 votes for s-1; nothing changes at the last stage; history cleared
                        # on every transition. None/empty/non-finite logits are ignored (backends must map -inf for
                        # masked stages to a large negative number, as pibehavior does).
    def __init__(self, num_stages: int, history: int = 3, votes_to_promote: int = 2)
    stage: int
    def update(self, logits: np.ndarray | None) -> int
    def set_stage(self, stage: int) -> None   # (added) forced stage; clears the history
    def reset(self) -> None
```

### config.py (YAML)
```yaml
server:
  host: 0.0.0.0
  ports: "8000-8049"          # int, "8000", "8000-8049", "8000,8002" or a list; CLI --ports / env B1K26_PORTS override
  health_requires_warm: true  # /healthz returns 503 until the default profile's worker is ready
engine:                       # (added) every key optional
  plan_timeout_s: 120         # one plan (queueing + inference); on timeout the step gets hold actions
  max_batch: 8                # micro-batch size per worker request
  batch_wait_ms: 5            # an idle worker waits this long for concurrent plan requests (multi-port batching)
  gripper_rules: null         # path to a gripper_rules.json (default: packaged)
  fallback_to_default: true   # route to the default profile when a routed worker failed or does not serve the task
workers:
  comet_pt50:
    endpoint: ws://127.0.0.1:9101          # and/or `launch:` (argv) + `port:`; the front starts and supervises it
    launch: ["/opt/envs/openpi_comet/venv/bin/python", "-m", "b1k26.worker", "--backend", "openpi_comet", "--checkpoint", "/ckpt/pt50", "--port", "9101"]
    startup_timeout_s: 900
    env: {}                   # (added) extra environment for the launched process
    cwd: null                 # (added)
    max_restarts: 3           # (added) restarts allowed within any restart_window_s (start-up relaunches count)
    restart_window_s: 3600    # (added) sliding window: a crash loop gives up, one crash a day never exhausts it
    max_batch: null           # (added) per-worker cap on engine.max_batch
profiles:
  comet:
    worker: comet_pt50
    image_size: 224
    resize: bilinear_pad
    mask_base_qvel: true            # zero proprio[0:3] before sending (2025-trained checkpoints)
    prompt: comet2025               # comet2025 | instruction | snake_case
    execution: {execute_steps: 32, predicted_steps_to_use: 32, keep_for_inpaint: 0}
    corrections: true
    use_stage: false
    cameras: [head, left_wrist, right_wrist]   # (added) roles sent to the worker
routing:
  default: comet
  per_task: {}                      # task name or id -> profile
```
In `launch`, `{python}` expands to the front server's interpreter and `{port}` to the worker port. Unknown keys at
any level are errors (`ConfigError`). `load_config(path)`, `parse_config(doc)`, `parse_ports(spec)`,
`resolve_prompt(style, task_id)`.

### session.py
```python
class RolloutSession:
    key: tuple                       # (port, slot group, batch index)  (changed: groups, see server.py)
    task_id: int | None
    profile_name: str | None
    queue: collections.deque         # pending (23,) actions
    inpaint_tail: np.ndarray | None
    stage: StageTracker | None
    step: int                        # actions handed to the evaluator in this rollout (= executed steps)
    last_action: np.ndarray | None
    last_fingerprint: bytes | None
    last_response: tuple | None      # (action, chunk) returned for last_fingerprint
    fallback_note: str | None        # why the routed profile was replaced
    stats: RolloutStats              # counters + latency arrays for the per-rollout log line
    def reset(self) -> None          # new rollout in this slot
    def reset_plan_state(self) -> None   # drop queue, tail and stage (task changed without a reset)
```

### engine.py
```python
class PolicyEngine:
    def __init__(self, config: Config, worker_clients: dict[str, WorkerClient] | None = None,
                 rules: GripperRules | None = None)
    async def start(self) -> None                # launch/connect workers concurrently, then check_profiles();
                                                 # raises WorkerError if the default profile's worker cannot start
    @property
    def warm(self) -> bool                       # default worker ready (False again if it fails for good)
    async def step(self, sessions: list[RolloutSession], envs: list[EnvObs], chunk_k: int
                   ) -> tuple[np.ndarray, np.ndarray | None]
    def check_profiles(self) -> list[str]        # (added) profile vs worker info mismatches, logged as ERROR
    def status(self) -> dict                     # workers, batches, config_problems
```
`step` returns action (B,23) float32 and, if chunk_k > 1, action_chunk (B,chunk_k,23) whose [:,0] equals action
exactly. A session plans when its queue has fewer than max(chunk_k, 1) actions (a partial leftover queue is
discarded with its inpainting tail, so a returned chunk is always one contiguous open-loop plan). If a plan is
shorter than chunk_k (chunk_k > execute_steps), the rest is padded with the last pose and base 0. Consumes the
returned actions from the queue. Never raises: on any error it logs and returns hold actions (never zeros).
**So the evaluator's `--replay-action-chunk-size K` must be <= `execute_steps` of every routed profile, and should
divide it** (otherwise leftovers are dropped every plan).

Routing per session: `routing.profile_for(task_id)`, replaced by the default profile (with
`fallback_to_default`) when the routed worker failed permanently or does not serve the task (`info.supported_tasks`,
or `num_stages[t] == 0`). A task is never sent to a worker that does not serve it (the worker would reject the whole
micro-batch). `WorkerClient` launches, health-polls, connects (`proxy=None`), fetches `info`, warms up and
supervises a worker; a launched worker that exits is restarted (`max_restarts` per `restart_window_s`), one that
times out 3 times in a row is killed and restarted, and a worker gets `PR_SET_PDEATHSIG` plus a parent watchdog.

Worker protocol (front -> worker, msgpack over websocket, one request in flight per connection):
- `{"op": "info"}` -> `{"flavor", "action_horizon", "image_size", "num_stages": [100 ints] | None, "supports_inpaint",
  "supports_stage"}` plus `"warm"`, `"backend"`, `"pid"` (added) and optionally `"supported_tasks"` (pibehavior)
- `{"op": "warmup"}` -> `{"ok": True, "ms": float}`
- `{"op": "infer", "items": [{"task_id", "prompt", "proprio" (61,), "images": {role: (S,S,3) uint8}, "stage": int|None, "initial_actions": (k,23)|None}]}`
  -> `{"chunks": [{"actions": (T,23) float32 absolute, "subtask_logits": (S,)|None}], "ms": float}`
- A request may carry `"id"`; the reply echoes it, so the front drops late replies to requests that timed out.
- An error answer is `{"error": str}`. It is never a text frame. `/healthz` on the worker port is 503 while loading.

### server.py
- Listens on every configured port with `websockets.asyncio.server.serve(..., compression=None, max_size=None,
  ping_interval=None, process_request=health)`.
- `/healthz`: 200 "OK\n" when warm, 503 otherwise. `/status`: JSON (workers, counters, slot groups,
  `engine.config_problems`). Any other HTTP path upgrades to a websocket.
- On connect, sends `packb({"server": "b1k26", ...})` first (the client blocks on this metadata frame).
- `{"reset": True}` resets every session of that connection's slot group and gets **no reply**.
- Otherwise: pop `__action_chunk_size__`, `split_batch(copy_images=False)`, map batch index b -> session
  `(port, group, b)`, `await engine.step`, reply `{"action": (B,23) float32, "action_chunk"?: (B,K,23),
  "server_timing": {...}}`. Return `(B,23)` even for B=1: 2026/eval multi-port accepts `(1,A)`, and v3.9.2/post1
  single-port accept it too.
- Reconnect handling: sessions belong to a slot group of the port, not to the connection. A new connection takes
  over the free group whose cached observation fingerprints match its first observation, else the most recently
  released free group, else a fresh group; a group whose live connection is half-open is taken over when the same
  observation is resent. If the first observation on a new connection equals the group's last one (2026/eval resends
  the same obs after a reconnect), the cached reply is re-sent instead of stepping. A second *concurrent* connection on
  the same port gets a fresh slot group.
- Never closes a connection on error and never sends text frames; a malformed observation gets hold actions from
  its raw proprio (or the slot's last actions).
- Logs one line per rollout (at reset or shutdown): steps, plans, failures, holds, corrections, compressed plans,
  replays, query and plan latency.
- CLI: `b1k26-serve --config X [--ports P] [--host H] [--log-level L] [--check]`; exit 0 on SIGTERM/SIGINT, 1 if the
  default worker cannot start, 2 on a bad config. `--check` validates the config and the launched workers'
  interpreters without starting anything (the Docker build runs it).

### Offline tools
- `scoring.py`: load rollout JSONs, compute the leaderboard Q (mean of per-task means over submitted episodes,
  divided by 100), the official rule (missing instances count as zero), SR, per-task tables and efficiency
  tie-breakers; `validate_submission(dir)` checks names `{task}_{301..310}_0.json`, fields, a 1:1 match of
  videos to JSONs, no duplicates, no rollout_id != 0.
- `selection.py`: per-task route selection from held-out results (IDs 311-320 or train-mode instances).
  Empirical-Bayes shrinkage toward each candidate's global mean. A per-task route switches away from the
  globally best candidate only if the posterior gain exceeds a margin. It refuses results on reported IDs
  301-310 unless `--allow-reported` is given, and writes the decision log into the README.
- `orchestrate.py`: job planning (tasks x instances -> workers, longest-processing-time-first by `max_steps`),
  per-worker job files, a resume that skips jobs with an existing JSON, and crash detection (no JSON after the
  process exits = infrastructure failure, re-queued once and logged).
- `package.py`: the submission zip (metrics JSONs, wrapper `.py`, robot config, README with the exact
  commands, SHA256 manifest), a video manifest, and a pre-submit check with `scoring.validate_submission`.

## Testing

```bash
pip install -e '.[test]'            # + torch (CPU) for the evaluator-client tests, pyarrow for one gripper-rules test
python -m pytest                     # ~480 tests, ~40 s on 4 CPU cores; skipped tests print their reason (-rs)
bash scripts/smoke_local.sh          # end-to-end: b1k26-serve (configs/fake.yaml, ports 18000-18003) + b1k26-probe
```
Everything runs on CPU with no checkpoints. The suite also passes on Python 3.10 with websockets 14.1, the oldest
versions the worker envs use (without torch the evaluator-client tests skip). The model-family backends are tested
against stand-ins for their forks' policies (the forks themselves were exercised by their authors in throwaway CPU envs; see BACKENDS.md
"Verification status"). Nothing here exercises a GPU, a real checkpoint or the simulator: before a real run, use
`scripts/smoke_local.sh --no-start --ports ...` against the real server/container and one real rollout.

| area | files | what is covered |
|---|---|---|
| observations | `test_obs.py` | batched / multi-port N=1 / unbatched v3.9.2 splitting, RGBA->RGB, float and gray images, task_id shapes and validation, read-only inputs never mutated, `copy_images=False` views equal to copies (values, fingerprints, prepared images), bit-exact `resize_with_pad` vs openpi's code, fingerprint cost at full resolution, state23 and hold action (never zeros, finite) |
| control | `test_control.py`, `test_stage.py`, `test_corrections.py` | compression / gripper-variation / inpainting tail identical to RLC's wrapper (copied reference code), sanitizing; stage voting identical to RLC on random sequences; gripper rules identical to RLC's tables on tasks 0-49, demo-derived rules for 50-99, the rules script |
| config | `test_config.py`, `test_integration.py` | schema and error messages, every `configs/*.yaml` loads; each shipped config's launched backend exists in the registry, its kwargs are accepted by the backend constructor, its interpreter follows the Docker env layout (`/opt/envs/<name>/venv/bin/python`) and its profiles fit the model's chunk horizon |
| engine | `test_engine.py` | chunk queue semantics (`action_chunk[:,0] == action` bit for bit, leftover discard, padding), corrections -> compression -> sanitize order, stage tracking and inpainting passed to the worker, micro-batching (one message and concurrent ports), timeouts and worker failures give holds, routing and fallback (failed worker, task not served), profile-vs-worker checks, sliding restart window |
| worker | `test_worker.py`, `test_fake_backends.py` | worker protocol ops and errors (never text frames), health while loading, request ids and stale replies, CLI exit codes, deterministic fake backends |
| front server | `test_server_protocol.py`, `test_server_unit.py`, `test_client_probe.py` | driven by **verbatim copies of the evaluator clients** (`tests/vendor/`, hash-checked): v3.9.3-post2 at N=1/3 with chunk sizes 0/8/20 (chunk replay bit-identical to per-step serving), 2026/eval reconnects (drop after/before stepping and mid-plan: no double step), MultiWebsocketPolicy over 3 ports, health gating, worker errors/timeouts, malformed observations, unbatched v3.9.2, metadata/reset-without-reply, half-open takeover, idle group eviction; the probe's own violation detection |
| processes | `test_launch.py`, `test_integration.py` | `b1k26-serve` with `configs/fake.yaml`: launched worker restart after SIGKILL (holds meanwhile), sliding restart budget then permanent failure (/healthz 503), SIGTERM exit 0 stops the worker, SIGKILL of the front takes the worker down, exit 1/2 on worker/config failure, `--check`; `scripts/smoke_local.sh` end to end on free ports; a worker that serves only tasks 0-49 behind the real worker protocol (unserved routes fall back, never reach it); every console script resolves and prints `--help`; the backend registry matches its classes |
| backends | `test_backends_openpi.py`, `test_backends_rlc_groot.py` | input dicts identical to each fork's wrapper, info contracts, prompt/gripper/state conventions, batching, error propagation, env scripts' arguments, example configs |
| offline | `test_offline_tools.py` | scoring formulas, submission validation, route selection, job planning/resume, packaging |

Performance (CPU, 4 cores, `configs/fake.yaml`, measured with the probe and `server_timing`): a full-resolution
RGBD observation is 7.8 MB. The transport floor (client packing + websocket, against a trivial echo server) is
~9 ms per request. A step that does not plan costs the front server ~1.1 ms (p50, receipt to reply) and ~12.3 ms
round trip; a planning step (resize 3 cameras to 224, worker round trip, compression) ~14 ms server-side and
~24-28 ms round trip, of which the fake worker's plan is ~5 ms (3 ms of it is `batch_wait_ms`). With
`--replay-action-chunk-size 20` that is ~1.2 ms per executed step; without it ~13 ms per step.
