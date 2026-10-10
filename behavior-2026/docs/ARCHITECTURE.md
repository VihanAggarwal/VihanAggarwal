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

### obs.py
```python
@dataclass
class EnvObs:
    task_id: int                       # from obs["task_id"] (shape (N,1) int64 on v3.9.3+, may be (1,) or scalar)
    proprio: np.ndarray                # (61,) float32
    rgb: dict[str, np.ndarray]         # role -> (H, W, 3) uint8, roles "head", "left_wrist", "right_wrist"
    depth: dict[str, np.ndarray]       # role -> (H, W) float32, empty if not sent
    cam_rel_poses: np.ndarray | None   # (21,) float32
    fingerprint: bytes                 # cheap hash of proprio + task_id + strided image bytes

def split_batch(msg: dict) -> list[EnvObs]
    # Accepts the evaluator's flattened obs dict. Every value has a leading batch dim N on v3.9.3+ and on
    # 2026/eval multi-port (N=1). v3.9.2 sends unbatched values: detect via proprio.ndim == 1.
    # RGB arrives as (N, H, W, 4) uint8 RGBA; drop alpha. Float images in [0,1] are scaled to uint8.
    # Never mutate the (read-only) input arrays.
def resize_with_pad(img: np.ndarray, height: int, width: int, method: str = "bilinear") -> np.ndarray
    # Bit-exact port of openpi_client.image_tools.resize_with_pad (PIL; zero padding, centered).
def prepare_images(env: EnvObs, size: int, method: str = "bilinear") -> dict[str, np.ndarray]
def state23_action_order(proprio: np.ndarray) -> np.ndarray
    # RLC order [base_qvel3, trunk4, L arm7, L grip, R arm7, R grip]; grip = 2*(finger_sum/0.1) - 1
def hold_action(proprio: np.ndarray) -> np.ndarray
    # (23,) float32 that holds the current pose: base 0, torso/arms = current qpos,
    # grippers = 2*(finger_sum/0.1) - 1 clipped to [-1, 1]. NEVER return zeros (zero torso = stand upright).
```

### control.py
```python
@dataclass
class ExecutionConfig:
    execute_steps: int = 20          # actions sent to the robot per plan (the replan period)
    predicted_steps_to_use: int = 26 # predicted actions consumed per plan; > execute_steps means compression
    keep_for_inpaint: int = 4        # predicted actions after the consumed ones, kept for the next plan
    base_velocity_scale_with_compression: bool = True
    disable_compression_gripper_range: float = 0.2   # RLC GRIPPER_VARIATION_THRESHOLD
    clip_base: float = 1.0

@dataclass
class PlannedChunk:
    actions: np.ndarray              # (execute_steps or fewer, 23) float32, ready to send
    inpaint_tail: np.ndarray | None  # (keep_for_inpaint, 23) absolute actions for the next plan, or None
    compressed: bool

def plan_execution(raw: np.ndarray, cfg: ExecutionConfig) -> PlannedChunk
    # raw: (T, 23) absolute actions from the model, already corrected. If compression applies, cubic
    # interp (scipy interp1d kind="cubic"; linear fallback if scipy is missing) of raw[:predicted_steps_to_use]
    # onto execute_steps samples, base dims scaled by predicted/execute. Compression is disabled when the
    # gripper range over raw[:predicted_steps_to_use] exceeds the threshold (then execute raw[:execute_steps]).
    # The inpaint tail starts right after the consumed predicted actions.
def sanitize(actions: np.ndarray, proprio: np.ndarray) -> np.ndarray
    # Replace non-finite rows with hold_action(proprio); clip base to [-clip, clip] and grippers to [-1, 1].
```

### corrections.py
```python
class GripperRules:
    @classmethod
    def load(cls, path: str | None = None) -> "GripperRules"   # default: package data/gripper_rules.json
    def apply(self, task_id: int, stage: int | None, state23: np.ndarray, actions: np.ndarray
              ) -> tuple[np.ndarray, bool]
    # If a gripper is fully closed (state23 grip < -0.98, i.e. closed on nothing) in a task/stage where the
    # human demos never close it fully, replace the chunk with "hold pose" (from state23 positions, base 0)
    # with that gripper set to +1 (open). Stage-based rules apply only when stage is not None.
```
`data/gripper_rules.json`: `{"closed_threshold": -0.98, "tasks": {"<id>": {"left": {"always_open": bool,
"min_stage": int|null, "min_progress": float|null}, "right": {...}, "exempt_right": bool}}}`. Tasks 0-49 start
from RLC's 2025 tables; tasks 50-99 come from the 2026 demo statistics (`scripts/compute_gripper_rules.py`).

### stage.py
```python
class StageTracker:     # RLC voting: history 3, promote on >=2 votes for s+1, skip on 3 votes for s+2,
    def __init__(self, num_stages: int, history: int = 3, votes_to_promote: int = 2)   # roll back on 3 votes for s-1
    stage: int
    def update(self, logits: np.ndarray) -> int
    def reset(self) -> None
```

### config.py (YAML)
```yaml
server:
  host: 0.0.0.0
  ports: "8000-8049"          # single port or range; every port behaves identically
  health_requires_warm: true  # /healthz returns 503 until default profile workers are warmed up
workers:
  comet_pt50:
    endpoint: ws://127.0.0.1:9101          # or `launch:` (argv list) + `port:`; the front starts and supervises it
    launch: ["/opt/envs/comet/bin/python", "-m", "b1k26.worker", "--backend", "openpi_comet", "--checkpoint", "/ckpt/pt50", "--port", "9101"]
    startup_timeout_s: 900
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
routing:
  default: comet
  per_task: {}                      # task name or id -> profile
```

### session.py
```python
class RolloutSession:
    key: tuple                       # (port, slot)
    task_id: int | None
    profile_name: str | None
    queue: collections.deque         # pending (23,) actions
    inpaint_tail: np.ndarray | None
    stage: StageTracker | None
    step: int
    last_action: np.ndarray | None
    last_fingerprint: bytes | None
    last_response: tuple | None      # (action, chunk) returned for last_fingerprint (reconnect replay)
    def reset(self) -> None
```

### engine.py
```python
class PolicyEngine:
    def __init__(self, config: Config, worker_clients: dict[str, WorkerClient])
    async def start(self) -> None                # connect/launch workers, warm up default profile
    @property
    def warm(self) -> bool
    async def step(self, sessions: list[RolloutSession], envs: list[EnvObs], chunk_k: int
                   ) -> tuple[np.ndarray, np.ndarray | None]
    # Returns action (B,23) float32 and, if chunk_k > 1, action_chunk (B,chunk_k,23) whose [:,0] equals action
    # exactly. Plans per session when its queue has fewer than max(chunk_k, 1) actions (a partial leftover
    # queue is discarded, so a returned chunk is always one contiguous open-loop plan). Consumes the returned
    # actions from the queue. Never raises: on any error it logs and returns hold actions.
```
Worker protocol (front -> worker, msgpack over websocket, one request in flight per connection):
- `{"op": "info"}` -> `{"flavor", "action_horizon", "image_size", "num_stages": [100 ints] | None, "supports_inpaint", "supports_stage"}`
- `{"op": "warmup"}` -> `{"ok": True, "ms": float}`
- `{"op": "infer", "items": [{"task_id", "prompt", "proprio" (61,), "images": {role: (S,S,3) uint8}, "stage": int|None, "initial_actions": (k,23)|None}]}`
  -> `{"chunks": [{"actions": (T,23) float32 absolute, "subtask_logits": (S,)|None}], "ms": float}`
- An error answer is `{"error": str}`. It is never a text frame.

### server.py
- Listens on every configured port with `websockets.asyncio.server.serve(..., compression=None, max_size=None,
  ping_interval=None, process_request=health)`.
- `/healthz`: 200 "OK\n" when warm, 503 otherwise. Any other HTTP path upgrades to a websocket.
- On connect, sends `packb({"server": "b1k26", ...})` first (the client blocks on this metadata frame).
- `{"reset": True}` resets every session of that connection's slot group and gets **no reply**.
- Otherwise: pop `__action_chunk_size__`, `split_batch`, map batch index b -> session `(port, b)`, `await engine.step`,
  reply `{"action": (B,23) float32, "action_chunk"?: (B,K,23), "server_timing": {...}}`. Return `(B,23)` even for B=1:
  2026/eval multi-port accepts `(1,A)`, and v3.9.2/post1 single-port accept it too.
- Reconnect handling: sessions belong to the port (slot group), not to the connection. A new connection on a port
  whose previous connection is closed inherits its sessions. If its first obs fingerprint equals the session's
  `last_fingerprint`, the cached response is re-sent instead of stepping (2026/eval resends the same obs after a
  reconnect). A second *concurrent* connection on the same port gets a fresh slot group.
- Never closes a connection on error and never sends text frames.
- Logs per-rollout timing: steps, plans, mean and max query latency.

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
