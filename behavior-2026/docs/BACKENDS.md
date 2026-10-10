# Worker backends

A backend wraps one model family's own inference code inside a `b1k26.worker` process, in that family's own
Python env (the families pin incompatible stacks; see `docs/ARCHITECTURE.md`). The worker receives items that the
front server has already prepared, and the backend returns absolute 23-D action chunks:

- **Input** (`b1k26.backends.base.InferItem`):
  - `images`: `head` / `left_wrist` / `right_wrist`, each `(S, S, 3)` uint8, already `resize_with_pad` to `info()["image_size"]`.
  - `proprio`: raw 61-D proprio. `proprio[0:3]` is already zeroed if the profile sets `mask_base_qvel`.
  - `prompt`: already resolved by the profile (`comet2025 | instruction | snake_case`).
  - `task_id`, and optionally `stage` and `initial_actions`.
- **Output** (`ChunkOut`): `actions` `(T, 23)` float32 in the evaluator layout (base 0:3 normalized velocity, torso
  3:7, left arm 7:14, left grip 14, right arm 15:22, right grip 22; all absolute).

| registry name | class | env script | model family |
|---|---|---|---|
| `openpi_comet` | `b1k26.backends.openpi_comet:CometBackend` | `scripts/envs/openpi_comet.sh` | Comet pi0.5 pt50 / pt12 (`mli0603/openpi-comet`) |
| `openpi_b1k` | `b1k26.backends.openpi_b1k:OpenPIB1KBackend` | `scripts/envs/openpi_b1k.sh` | organizers' pi0.5 `pi05_b1k` (`wensi-ai/openpi@behavior`), radio baseline, Hoshipu 100t |
| `pibehavior` | `b1k26.backends.pibehavior:PiBehaviorBackend` | `scripts/envs/pibehavior.sh --fork rlc2025\|jackliu2026` | RLC 2025 / JackLiu meta100 (see [pibehavior](#pibehavior)) |
| `gr00t` | `b1k26.backends.gr00t:Gr00tBackend` | `scripts/envs/gr00t.sh` | GR00T N1.7 (see [gr00t](#gr00t)) |
| `fake_hold`, `fake_sine`, `fake_replay` | `b1k26.backends.fake` | none (front-server env) | deterministic test backends (`configs/fake.yaml`) |

Env layout: a worker env lives in `PREFIX/venv` (its Python is `PREFIX/venv/bin/python`). The Docker image creates
envs with `docker/install_envs.sh NAME ...` as `/opt/envs/NAME`, with NAME one of `openpi_comet`, `openpi_b1k`,
`gr00t`, `pibehavior-2025` (`--fork rlc2025`) and `pibehavior-2026` (`--fork jackliu2026`), so a config launches
workers as `/opt/envs/NAME/venv/bin/python -m b1k26.worker ...` (the example configs do).

The front server checks every profile against its worker's `info()` at start-up (image size, stage support, chunk
horizon vs `execution`, routed tasks the worker does not serve) and logs each mismatch as an ERROR line
`config check: ...`; the list is also in `/status` under `engine.config_problems`. A task routed to a worker that
does not serve it (`supported_tasks`, or `num_stages[t] == 0`) is never sent to that worker: it falls back to the
default profile when `engine.fallback_to_default` is on, and holds otherwise.

---

## openpi family: common behaviour (`OpenPIBackendBase`)

Both openpi backends share `b1k26.backends.openpi_b1k.OpenPIBackendBase`.

- **Lazy import.** openpi and JAX are imported in `__init__`, never at module import. Each backend builds the
  per-item dict exactly as its fork's eval wrapper does, and then calls the fork's own `Policy`. All transforms are
  the fork's: state extraction, normalization, prompt + discretized-state tokenization, un-normalization,
  delta-to-absolute conversion, and truncation to 23 dims. The backend copies the output, slices it to `[:, :23]` and
  casts it to float32 (`postprocess_actions`).
- **Policy creation.** The backend uses the fork's `policy_config.create_trained_policy(train_config, ckpt,
  sample_kwargs=..., default_prompt=None, norm_stats=...)`. The norm stats are loaded by the backend first, so it can
  validate them and give a clear error.
- **Config validation.** `config_name` is checked against the fork's `_CONFIGS_DICT`. Comet's `get_config`
  silently falls back to `pi05_b1k-base` for an unknown name; the backend raises instead.
- **Fail-fast self-checks** at load:
  - the fork's `extract_state_from_proprio` is run on a probe vector and compared with b1k26's reference layout;
  - the robot config's cameras must map onto `head` / `left_wrist` / `right_wrist`;
  - the norm stats must have `state` and `actions` entries of at least 23 dims.
- **`info()`**: `{"flavor": "openpi_comet" | "openpi_b1k", "action_horizon": train_config.model.action_horizon (32),
  "image_size": 224, "num_stages": None, "supports_inpaint": False, "supports_stage": False}`. Any `stage` or
  `initial_actions` sent with an item are ignored.
- **Errors propagate.** A malformed item or a policy exception raises, and the worker answers `{"error": ...}`;
  the front server then holds the pose. Non-finite proprio values are zeroed with a warning, because NaN would
  corrupt the discretized-state tokens. Non-finite *actions* are left in place for `control.sanitize`.

### Constructor arguments (worker `--backend-kwargs` JSON)

| kwarg | default | meaning |
|---|---|---|
| `checkpoint` | required | checkpoint dir with `params/` (orbax) and `assets/`; `gs://` works via openpi `maybe_download`; a dir with `model.safetensors` uses openpi's PyTorch path |
| `config_name` | `pi05_b1k-base` (comet) / `pi05_b1k` (b1k) | the fork's TrainConfig |
| `asset_id` | config's | sub-dir of `<ckpt>/assets/` holding `norm_stats.json` |
| `norm_stats_dir` | none | explicit dir holding `norm_stats.json`; relative to the checkpoint (`"assets"` for Hoshipu); overrides `asset_id` |
| `num_steps` | model default 10 | flow-matching steps, passed as `sample_kwargs={"num_steps": n}` (`Pi0.sample_actions(..., num_steps=...)` in both forks) |
| `dtype` | `bfloat16` | `bfloat16` = upstream path. `float32` = same pipeline but params restored as f32 and `Pi0Config.dtype="float32"`. Ignored (with a warning) for PyTorch checkpoints |
| `action_horizon` | config's (32) | override `model.action_horizon`. pi0.5 has no horizon-shaped params, so only use this to match a checkpoint's training horizon |
| `use_quantile_norm` | config's | force quantile (`true`) or z-score (`false`) normalization |
| `batched` | `false` | one padded `sample_actions` call per micro-batch instead of a loop (see below) |
| `max_batch` | 8 | largest compiled batch; buckets `1, 2, 4, ..., max_batch` |
| `default_prompt_mode` | `comet2025` (comet) / `snake_case` (b1k) | used only when an item arrives without a prompt |
| `mem_fraction` | unset | sets `XLA_PYTHON_CLIENT_MEM_FRACTION` before JAX is imported |
| `warmup_task_id` | 0 | task used for the warmup dummy inference |
| `repo_id` (b1k only) | config's | like `serve_b1k.py --repo-id`; the asset id defaults to it |
| `gripper_state` (b1k only) | `auto` | `width`, `pm1`, or `auto` (decided from the norm stats); see the openpi_b1k section |

### Batched inference

- openpi's `Policy.infer` handles one example and adds the batch dim itself, so by default `infer()` loops over
  the items.
- With `batched: true`, `_infer_batched` re-implements `Policy.infer` (identical in both forks) for B > 1, using the
  policy's own private pieces (`_input_transform`, `_sample_actions`, `_rng`, `_sample_kwargs`, `_output_transform`):
  - run the input transforms per item;
  - stack the items (instead of `[np.newaxis]`);
  - split the policy RNG and sample once;
  - run the output transforms per item on `{"state", "actions"}`.
- Groups are padded to a power-of-two bucket by repeating the last item, so XLA compiles at most
  `len(bucket_sizes(max_batch))` shapes. `warmup()` compiles all of them. Measured on CPU with a dummy model:
  5-10 s each. Not measured on a GPU with the full model (estimate: tens of seconds each).
- A single item always takes the unmodified `Policy.infer` path.
- If the policy lacks those attributes, or is a PyTorch model, the backend falls back to the loop and logs a warning.
- If a batched call raises at runtime (e.g. XLA out of memory at the largest bucket), that request is re-served by the
  loop; after `MAX_BATCHED_FAILURES` (3) consecutive failures the batched path is switched off for the process.
- **Verified on CPU against both real forks** (tiny random-weight pi0.5, real transforms and tokenizer): batched
  outputs are bit-identical to `Policy.infer` when given the same noise rows. Noise is drawn per batch, so samples
  differ from the loop path for the same seed but follow the same distribution.

### Turing (TITAN RTX, sm_75) and `dtype`

- The backend logs the GPU compute capability at load. It reads it from the JAX device, falling back to
  `nvidia-smi --query-gpu=compute_cap`.
- Below sm_80 there are no bf16 tensor cores. XLA (jax 0.5.3) rewrites bf16 matmuls and convolutions to f32
  (`FloatNormalization`), so the default `bfloat16` runs correctly, at f32 SIMT speed. No code change is required.
- `dtype: float32` restores the params as f32 and computes in f32: no bf16 rounding, about 2x the parameter
  memory (~13-14 GB for pi0.5), similar speed on Turing.
- fp16 is not offered: it is unsafe for Gemma (attention overflow).
- Use `float32` only if the bf16 path misbehaves on the target GPU. Check that `mem_fraction` leaves room for the
  other workers.
- Memory: by default JAX preallocates 75% of the GPU per process. Set `mem_fraction` per worker (0.85 for a single
  worker, about 0.45 each for two pi0.5 workers on 24 GB), or set `XLA_PYTHON_CLIENT_PREALLOCATE=false` in the
  launch environment.

### Environments

`scripts/envs/<backend>.sh --prefix DIR [--repo DIR] [--commit SHA] [--install-uv] [--skip-check] [--gpu-check]`:

1. Checks out the fork at the pinned commit into `DIR/src/<fork>`.
2. Runs `GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev` from the fork's `uv.lock` into `DIR/venv` (Python 3.11).
3. Runs `uv pip install -e <repo>/behavior-2026`, constrained to the already-installed versions. All b1k26
   dependencies are satisfied by the locks: numpy 1.26.4, websockets 15.0.1, msgpack 1.1.0, pyyaml 6.0.2,
   pillow 11.2.1.
4. Prints the versions, writes `DIR/ENV_INFO.txt`, and imports the fork on CPU to run the state-extraction
   self-check.

BEHAVIOR-1K/OmniGibson is not installed in these envs.

Smoke test on the GPU node:

```bash
DIR/venv/bin/python - <<'EOF'
from b1k26.backends.base import create_backend
b = create_backend("openpi_comet", checkpoint="/ckpt/openpi_comet/pi05-b1kpt50-cs32", num_steps=10)
print(b.info(), "warmup ms", b.warmup())
EOF
```

---

## openpi_comet

**Fork:** https://github.com/mli0603/openpi-comet @ `4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5` (main head).
Its pins: jax[cuda12] 0.5.3, flax 0.10.2, orbax 0.11.13, torch 2.7.1, transformers 4.53.2, lerobot from
`huggingface/lerobot@577cd10`.

### Checkpoints (HF `sunshk/openpi_comet`, JAX orbax, ~12 GB each)

```bash
hf download sunshk/openpi_comet --include "pi05-b1kpt50-cs32/*" --local-dir /ckpt/openpi_comet   # tasks 0-49
hf download sunshk/openpi_comet --include "pi05-b1kpt12-cs32/*" --local-dir /ckpt/openpi_comet   # 12 tasks
# checkpoint dir = /ckpt/openpi_comet/pi05-b1kpt50-cs32  (params/, assets/, _CHECKPOINT_METADATA)
```

- Use the new `hf` CLI. The legacy `huggingface-cli download` keeps only the last of several repeated
  `--include` flags.
- The same names are also published as `sunshk/openpi_comet_pt50` and `sunshk/openpi_comet_pt12`, one per repo
  (not checked to be byte-identical; the layout above was read from the `sunshk/openpi_comet` tree).
- PyTorch conversions: `sunshk/openpi_comet_pytorch` and `RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32` contain
  `model.safetensors`. openpi's `create_trained_policy` would take its PyTorch path for them (untested here; the
  batched path and `dtype` do not apply).

### Configuration

- **Config:** `pi05_b1k-base` for both pt50 and pt12. It is `Pi0Config(pi05=True, action_horizon=32)` with
  `LeRobotB1KDataConfig(repo_id="behavior-1k/2025-challenge-demos")`: absolute actions, no delta transform,
  `use_quantile_norm=True`.
  - The pretraining config names (`pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k`, `...pt12...`) only differ in
    training fields and would serve identically.
  - The README's `..._gpu40` name does not exist.
  - `LeRobotB1KRGBDDataConfig` (depth / point clouds) configs are rejected, because no depth is sent.
- **Norm stats:** `<ckpt>/assets/behavior-1k/2025-challenge-demos/norm_stats.json` (quantile; 32-wide). This is
  the default asset id, so no override is needed.
- **Input dict** (the fork's `shared/eval_b1k_wrapper.py`, ZSB `serve_comet.py`):
  `observation/egocentric_camera` (head), `observation/wrist_image_left`, `observation/wrist_image_right` (224² uint8),
  `observation/state` (raw 61-D float32), `prompt`.
- **State:** the fork's `B1kInputs` builds the 23-D state itself: `[base_qvel 3, trunk 4, L arm 7, R arm 7, L width,
  R width]`. The grippers come last as finger-width sums in [0, 0.1]. This is not the action order. The norm stats
  confirm it: `state[21:23]` has q01 0 and q99 0.1.
- **`omnigibson` stub:** `b1k_policy.py` does `from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES`,
  a 2025 module path that 2026 OmniGibson does not have.
  - `install_eval_utils_stub()` registers that module with `b1k26.constants.PROPRIO_INDICES_2026` before openpi
    is imported, as ZSB does.
  - The leaf module is always replaced, so a 2025 OmniGibson install (256-D layout) cannot leak in.
  - After import, `check_comet_state_extraction` verifies the bound indices and the extracted state.
- **Prompt:** `comet2025` = the fork's `scripts/task_mapping.json` "task" texts, stored in `tasks.json` as
  `instruction_comet2025`.
  - They match the 2026 instructions except for picking_up_trash ("three can of soda … tash can").
  - Tasks 50-99 fall back to the 2026 instruction.
- **Base velocity:** `mask_base_qvel: true`. The 2025 demos have base_qvel ≈ 0 (norm-stat std ≈ 0.01), while the
  2026 evaluator sends robot-frame velocity.
  - Zero normalizes in-distribution (quantile: ≈ +0.03).
  - ZSB's 0.199 was measured *unmasked*. The effect of masking is not yet measured (A/B it in screening).
- **Execution:** receding horizon over the full 32-step chunk (`execute_steps: 32`), as Comet and ZSB did.
  `configs/comet_pt50.example.yaml` is a complete example.

### Pitfalls

- **Long 2025 instructions plus 23 state tokens can exceed `max_token_len` 200.** The tokenizer then truncates and
  logs "Token length … exceeds max length". That happened in training too, so it is not fatal, but watch for it.
- **The tokenizer replaces `_` with spaces**, so snake_case prompts become "turning on radio".
- **Weights load from the fork's `params/`.** The README's `_gpu40` config does not exist.
- **The fork's `get_config` fallback is silent** for unknown config names (the backend guards against it).

---

## openpi_b1k

**Fork:** https://github.com/wensi-ai/openpi branch `behavior` @ `0cc8e355f7bac0976db1cc3139b1ff0379feea60`.
Its pins: jax[cuda12] 0.5.3, flax 0.10.2, orbax 0.11.13, torch 2.7.1 (cu128), transformers 5.5.4, lerobot from
`wensi-ai/lerobot@release/b1k`.

### Checkpoints

**Provided turning_on_radio baseline** (Google Drive, 16 GB zip; EMA params ~12.4 GB plus train_state):

```bash
pip install gdown
gdown 1KojwNUz0HVwU3Ww2SVh3NKt-4asuI3y2 -O pi05TurningOnRadio.zip
unzip pi05TurningOnRadio.zip 'pi05_turn_on_the_radio/params/*' 'pi05_turn_on_the_radio/assets/*' -d /ckpt/radio
# checkpoint dir = /ckpt/radio/pi05_turn_on_the_radio ; norm stats: assets/turning_on_radio/norm_stats.json
```

- `pi05_b1k` already has `repo_id="turning_on_radio"`, so its asset id finds these stats.
- Do not pass `repo_id behavior-1k/2026-challenge-demos` as the baselines page does: there are no stats there.

**Hoshipu 100-task pi0.5** (public, no card, never evaluated; JAX orbax fp32, 12.4 GB per step):

```bash
hf download Hoshipu/pi05-b1k100t-2026-lr2.5e5 --include "ckpt-4000000/*" --local-dir /ckpt/hoshipu-100t
# also: ckpt-950000, ckpt-2000000, ckpt-3000000; continuation repo Hoshipu/pi05-b1k100t-2026-4mto5m-lr2.5e5
# (ckpt-27000, ckpt-38000, ckpt-169000). checkpoint dir = /ckpt/hoshipu-100t/ckpt-4000000
```

- The norm stats are at `<ckpt>/assets/norm_stats.json`, directly under `assets/`, so use
  `norm_stats_dir: "assets"`.
- Params were written FSDP-sharded (orbax `write_shape` = 1/8 of the full shape). Restoring them on one device
  through openpi's `restore_params` was checked on CPU with an FSDP-8-saved tiny model: values are exact in f32
  and bf16-rounded in bf16.

### Configuration

- **Config:** `pi05_b1k` = `Pi0Config(action_horizon=32, pi05=True)` with
  `LeRobotB1KDataConfig(repo_id="turning_on_radio", robot_config_name="b1k/R1Pro", extra_delta_transform=True)`:
  - z-score normalization (`use_quantile_norm=False`);
  - torso and arm actions are deltas against the current state (`MappedDeltaActions` /
    `MappedAbsoluteActions`); base and grippers are absolute;
  - the output transforms return absolute actions.
- **Input dict** (exactly `B1KPolicyWrapper.process_input`): `observation/image_0` (head),
  `observation/image_1` (left wrist), `observation/image_2` (right wrist), `observation/state` (raw 61-D),
  `prompt`.
  - The camera keys come from the robot config's `observations` (`ObservationConfig.name` = our role names) and are
    verified at load.
- **Robot name bug:** the fork's robot config has `name="robot"` and obs keys `robot::robot:...` (the evaluator
  sends `robot_r1::...`). This breaks only the fork's own wrapper, which we do not use.
- **State:** `[base_qvel 3, trunk 4, L arm 7, L grip, R arm 7, R grip]` (action order). The gripper is the
  finger-width sum.
- **`gripper_state`:**
  - The radio baseline's stats have `state[14]` and `state[22]` in [0, 0.1] (q99 0.1), so it uses `width`, the
    fork's own extraction.
  - The Hoshipu 100t stats have q01 ≈ -0.82 and q99 = 1.0 for the grippers. That is `2*width/0.1 - 1`, the
    RLC/JackLiu convention, so it uses `pm1`.
  - With `pm1`, the backend rewrites a copy of the finger qpos: `p[24] = 2*w/0.1-1`, `p[25] = 0` (and the same for
    49/50), so the fork's finger sum yields the [-1, 1] value. Nothing else reads those entries.
  - `auto` (the default) picks the mode from the norm stats. It raises if they are ambiguous, and warns if an
    explicit setting disagrees with them.
- **Prompts:**
  - 2026 training used `prompt_from_task=True`, which yields the snake_case task name from
    `meta/tasks.parquet`.
  - The fork's server instead sends `TASK_REGISTRY` text, which exists for `turning_on_radio` only. That is a
    train/serve mismatch upstream.
  - The example profiles use `snake_case`. A/B `instruction` for the radio baseline.
- **Base velocity:** not masked; these models were trained on 2026 robot-frame velocity.
- **Execution:** the baseline page serves `--action_horizon 16` of the 32-step chunk (`execute_steps: 16`).
- `configs/openpi_b1k.example.yaml` is a complete example (Hoshipu default plus radio for task 0, two workers on
  one GPU).

### Hoshipu 100t: unknowns (VERIFY before screening)

Hoshipu's training config is not published. These settings are inferred:

| setting | evidence | example value |
|---|---|---|
| state layout | norm stats: action-order state, gripper in [-1, 1] | `gripper_state: auto` → `pm1` |
| delta actions | action mean ≈ 0 for torso and arms; base and grippers absolute | `pi05_b1k` default (`extra_delta_transform`) |
| normalization | both mean/std and q01/q99 present; cannot tell | `use_quantile_norm: null` (z-score). A/B `true` |
| action horizon | not in params; the 10-task card used 50 | `action_horizon: 32`. A/B 50 |
| prompt | unknown (LeRobot default is snake_case) | `snake_case`. A/B `instruction` |

A wrong guess does not crash; it degrades actions. On one instance, watch the video for arms drifting (wrong
normalization or deltas) and grippers that never open or close (wrong gripper state).

### Pitfalls

- **Wrong norm-stats dir.** `serve_b1k.py --repo-id` replaces the dataset repo id and therefore the asset id. The
  baselines-page value `behavior-1k/2026-challenge-demos` does not match the radio checkpoint. A missing
  `norm_stats.json` makes the backend list the files it did find.
- **Config import needs lerobot.** `openpi.training.config` imports `lerobot` at module level (`lerobot_compat`),
  so the full `uv sync` env is required even though serving never reads a dataset.
- **z-score with std ≈ 0.** `state[6]` (torso joint 4) has std < 0.0005, so any drift from 0 normalizes to a large
  value, which falls outside the 256 discretization bins of the pi0.5 state tokens. This is inherent to the
  checkpoints (training saw the same simulator).

---

## Verification status

| check | how | result |
|---|---|---|
| input dicts vs the forks' real `B1KInputs` / `B1kInputs` | fork `b1k_policy.py` loaded with stubbed openpi base modules | state, image slots and prompt match the b1k26 reference layouts (both forks; `width` and `pm1`) |
| end-to-end through the real forks | CPU JAX 0.5.3, tiny random pi0.5 (dummy Gemma, real SigLIP), real norm stats, real tokenizer | config lookup, overrides, `create_trained_policy`, warmup, loop, batched (bit-identical to `Policy.infer` with the same noise), float32 restore, Hoshipu layout (`assets/`, pm1, quantile override, horizon 50), FSDP-sharded restore: all OK |
| unit tests | `tests/test_backends_openpi.py` (fake policy and fake fork, no JAX) | dict construction, prompt mapping, masking expectations, output post-processing, loading logic, batched path, env scripts, example configs |
| real checkpoints on GPU | not run (no GPU in the authoring environment) | VERIFY at bring-up: latency, memory, and the Hoshipu unknowns above |

Code marked `# VERIFY:` in the backends: jaxlib's `Device.compute_capability` attribute on CUDA devices (used
only for the Turing log line; falls back to `nvidia-smi --query-gpu=compute_cap`).

---

## pibehavior

`b1k26.backends.pibehavior:PiBehaviorBackend`, env script `scripts/envs/pibehavior.sh`. It serves the RLC
architecture: pi0.5 conditioned on `[task_id, stage]` embeddings instead of text, with stage prediction and rolling
inpainting.

### Forks, envs and checkpoints

Two forks ship a package named `b1k` with different stage tables. They need separate envs (one `--prefix` each):

| `--fork` | source | stage table | checkpoints |
|---|---|---|---|
| `rlc2025` | [IliaLarchenko/behavior-1k-solution](https://github.com/IliaLarchenko/behavior-1k-solution) @ `ca556f7`, openpi submodule `wensi-ai/openpi` @ `01177e0` | 50 tasks, 596 stage rows | `IliaLarchenko/behavior_submission` `checkpoint_1..4`, `IliaLarchenko/behavior_50t_checkpoint` |
| `jackliu2026` | [JackLiu0406/behaviour-1k-2026-meta](https://github.com/JackLiu0406/behaviour-1k-2026-meta) @ `7146d7b` (vendored, patched openpi) | 100 tasks, 1120 stage rows | `JackLiu0406/meta-SFT-checkpoints` (gated): `meta100-1epoch/step*`, `single-task-finetune/no-da3/<task>*` |

```bash
scripts/envs/pibehavior.sh --prefix /opt/envs/pibehavior-2025 --fork rlc2025
scripts/envs/pibehavior.sh --prefix /opt/envs/pibehavior-2026 --fork jackliu2026
hf download IliaLarchenko/behavior_submission --local-dir /ckpt/rlc     # checkpoint_{1..4}/{params,assets}, ~51 GB
hf download JackLiu0406/meta-SFT-checkpoints --include "meta100-1epoch/step139999/params/*" \
    "meta100-1epoch/step139999/assets/*" --local-dir /ckpt/meta-sft     # request access on HF first
```

- **`rlc2025` env.** The repo has no lockfile. The env is synced from its openpi submodule's `uv.lock` (jax 0.5.3,
  flax 0.10.2, orbax 0.11.13, torch 2.7.1, transformers 4.53.2, lerobot @ 577cd10). Then `b1k` is installed with
  `--no-deps`; every import it makes is covered by that lock.
- **`jackliu2026` env.** Synced from the repo's own `uv.lock` (same jax/flax/orbax). That lock points at the
  Tsinghua mirror, so the script rewrites its URLs to pypi.org (same paths and hashes) unless `--keep-mirror` is
  given. JackLiu trained with JAX >= 0.10 on Blackwell, but the lock pins 0.5.3. A Blackwell (RTX 50xx) serving GPU
  needs a newer jax (VERIFY).
- Both plans were checked with `uv sync --frozen --dry-run` on 2026-10-10.
- Checkpoint layout (both): `<ckpt>/params/` (orbax) and `<ckpt>/assets/IliaLarchenko/behavior_224_rgb/norm_stats.json`
  (18 MB: per-timestamp action stats and the 960x960 action-correlation Cholesky). Some also have `fast_tokenizer/`,
  which is not used at inference.
- A 2025 checkpoint in the 2026 env, or the other way round, fails at construction with a clear error. The check
  compares the checkpoint's stage-embedding row count with the fork's table.
- **Do not serve DA3 checkpoints** (`single-task-finetune/da3*`). They need DA3 features, which this backend never
  computes. DA3 configs are rejected. A DA3 checkpoint under a plain config would be silently stripped by the fork's
  loader, so the parameter check rejects it.

### Inputs and outputs

- **Input dict.** Exactly what the forks' `B1KPolicyWrapper.process_obs` + `prepare_batch_for_pi_behavior` hand to
  `PiBehaviorPolicy.infer`:
  - `observation/egocentric_camera`, `observation/wrist_image_left`, `observation/wrist_image_right`: 224² uint8;
  - `observation/state`: raw 61-D proprio;
  - `tokenized_prompt = int32 [task_id, stage]`, `tokenized_prompt_mask = [True, True]`, `subtask_state = int32 stage`;
  - no `prompt`.
  - With an inpainting prefix, the dict also carries `initial_actions` (k, 23) float32 absolute, and the backend calls
    `infer(obs, initial_actions=...)` the same way the wrapper does.
- **State.** The fork's `B1kInputs` builds it with `PROPRIOCEPTION_INDICES` from
  `omnigibson.learning.utils.eval_utils`. b1k26 installs a stub of that module with the 61-D layout before the fork is
  imported, then a self-check compares the fork's state with `b1k26.obs.state23_action_order`:
  `[base_qvel 3, trunk 4, L arm 7, L grip, R arm 7, R grip]`, with grip = 2·width/0.1 − 1.
- **Task ids.** The 2026 `task_id` is the model's task id in both forks:
  - The 2025 `task_data.json` order equals the 2026 order for ids 0-49 (checked against `tasks.json`).
  - JackLiu appends the 50 new activities at 50-99 in 2026 order (`build_task_index_maps`).
- **Prompt.** Unused; the profile's `prompt` setting is ignored.
- **Output.**
  - `actions`: (30, 23) float32 absolute (the fork un-normalizes, adds the state back to the delta dims, and keeps 23
    dims).
  - `subtask_logits`: (15,). The model sets stages that a task does not have to −inf. `StageTracker.update` ignores
    non-finite logits, which would freeze the stage, so the backend replaces −inf with −1e9. NaN or +inf gives
    `None`.
- **`info()`**:
  - `action_horizon` 30, `image_size` 224, `supports_inpaint` and `supports_stage` true;
  - `num_stages`: 100 ints, the fork's count for tasks this worker serves and **0 for tasks it does not serve**
    (tasks 50-99 under `rlc2025`, unmapped tasks). The engine skips the stage tracker when the count is 0.
  - Two extra keys: `supported_tasks` (list of ids) and `fork` (`rlc2025` / `jackliu2026`).
  - A task the worker does not serve raises a clear error, so the front server's routing must send those tasks
    elsewhere.
- **Stage.** `item.stage` (from the front server's `StageTracker`) is clamped to `[0, num_stages-1]`; `None` means 0.
  An out-of-range stage would otherwise index another task's rows of the stage table.

### Inpainting and flow steps

- `num_steps` (default 20, as RLC) is passed as `sample_kwargs={"num_steps": n}`.
- `PiBehaviorPolicy.infer` runs the prefix through the full input transform:
  - DeltaActions on torso/arms;
  - per-timestamp normalization of rows 0..k-1;
  - padding to 32 dims.
- `sample_actions` then pins those k steps while flow time > `PiBehaviorConfig.time_threshold_inpaint` (0.3), and
  propagates the correction through the action covariance.
- RLC's `--time-threshold-inpaint` / `B1KWrapperConfig.time_threshold_inpaint` is never passed to the model upstream
  (a no-op). Here `time_threshold_inpaint` overrides the model config field.
- Each prefix length k compiles the sampler once more. `warmup()` compiles the plain path and the k =
  `warmup_inpaint_steps` (default 4, = `execution.keep_for_inpaint`) path.
- Non-finite prefixes are dropped with a warning.

### Several checkpoints (RLC's CheckpointSwitcher, ZSB's router)

- `task_checkpoint_mapping` takes RLC's JSON format, as a path or an inline dict:
  `{"checkpoints": {name: {"path", "tasks": [ids or names], "norm_stats_dir"?}}}`.
  - Relative paths are relative to the JSON file.
  - A task may appear only once.
  - Unmapped tasks are not served, unless `checkpoint` is also given as the fallback.
- `max_resident` (default 1) checkpoints stay in memory (LRU).
- Unloading drops the policy's references, runs `gc` and `jax.clear_caches()`. Surviving models are simply
  re-traced; this was checked on CPU.
- A mixed micro-batch serves the resident checkpoint first and swaps each other one in at most once.
- After a swap, the first inference recompiles: tens of seconds on GPU (estimate, VERIFY). Under the 2026 rules this
  is fine (600 s per query), and it happens only when the task changes.
- `warmup_all` loads and warms every checkpoint at startup, which fails fast on a broken download. The warmup task's
  checkpoint is warmed last, so it stays resident.
- **Thrashing.** With concurrent rollouts (`--num-envs` > 1, or 2026/eval multi-port) whose tasks belong to
  different checkpoints, a worker with `max_resident` smaller than the number of checkpoints in flight reloads and
  recompiles on every plan. The backend logs `checkpoint thrashing` (4 loads within 10 min). Fixes:
  - raise `max_resident` (about 7.6 GB each);
  - give each checkpoint group its own worker / profile;
  - order the evaluation jobs by checkpoint group.

### Norm stats and base velocity

- The norm stats come from the checkpoint's own assets (as upstream). Overrides: `asset_id`, `norm_stats_dir`, or
  `norm_stats_dir` per mapping entry.
- At load the backend logs `state std[0:3]` and classifies it:
  - **2025** (~0.01): world-frame joint velocity, ~0 in the 2025 demos. The backend warns: serve with
    `mask_base_qvel: true` unless the checkpoint was trained on 2026 data with those stats.
  - **2026** (~0.06-0.19): robot-frame velocity; do not mask.
- RLC ckpt1-4 / 50t: 2025 stats, mask.
- JackLiu meta100: its README insists on base-frame "qvel-fixed" stats, and the checkpoint ships the stats it was
  trained with. Do not mask; confirm with the log line (VERIFY on the downloaded checkpoint).
- JackLiu's separate `norm-stats-fixed/` file is only for a model trained with it (`norm_stats_dir`).

### Execution profile

`configs/pibehavior.example.yaml` reproduces RLC:

- `execution: {execute_steps: 20, predicted_steps_to_use: 26, keep_for_inpaint: 4}`: 26 predictions are compressed
  into 20 steps (base ×1.3), and compression is off when the gripper moves;
- `use_stage: true`, `corrections: true`;
- `mask_base_qvel`: true for 2025 checkpoints, false for 2026.

With replay chunks, use K dividing 20 (10 or 20). Otherwise the engine drops leftovers and the inpainting tail before
each plan.

### Constructor arguments (worker `--backend-kwargs` JSON)

| kwarg | default | meaning |
|---|---|---|
| `checkpoint` | — | checkpoint dir; with a mapping it is the fallback for unmapped tasks |
| `task_checkpoint_mapping` | none | RLC mapping JSON path or inline dict |
| `tasks` | all | restrict a single checkpoint to these ids/names (e.g. JackLiu meta5: only 0, 5, 40, 76, 77 are trained) |
| `config_name` | `pi_behavior_b1k_fast` | fork TrainConfig; JackLiu also has `pi_behavior_b1k_stage_only`; validated against `_CONFIGS_DICT` |
| `num_tasks` | from checkpoint | task-embedding rows; the fork's named configs say 50, so JackLiu's 100 comes from the checkpoint metadata |
| `num_steps` | 20 | flow-matching steps |
| `time_threshold_inpaint` | config (0.3) | inpainting cutoff |
| `asset_id` / `norm_stats_dir` | `IliaLarchenko/behavior_224_rgb` / none | norm-stats location |
| `max_resident` | 1 | checkpoints in memory |
| `clear_jax_caches` | true | `jax.clear_caches()` after an unload |
| `strict_params` / `allow_extra_params` | true / false | parameter-tree check before loading (extra DA3/spatial params always rejected) |
| `warmup_task_id` / `warmup_inpaint_steps` / `warmup_all` | lowest served / 4 / false | warmup |
| `mem_fraction` / `xla_allocator` | unset | `XLA_PYTHON_CLIENT_MEM_FRACTION` / `XLA_PYTHON_CLIENT_ALLOCATOR` (RLC used 0.5 / platform) |

- **Batching.** Items are served one by one with the fork's `PiBehaviorPolicy.infer`. The inpainting path and
  per-item prefixes make a padded batch path not worth it here.
- **Turing.** As for openpi: XLA upcasts bf16 below sm_80, so the run is correct but slower.
- **Memory.** About 7.6 GB of bf16 params per resident checkpoint.

### Verification status

| check | how | result |
|---|---|---|
| through the real forks | CPU, lane_b JAX 0.5.3 / flax 0.10.2 / orbax 0.11.13 venv; tiny random PiBehavior ("dummy" Gemma, real SigLIP, zero-init leaves randomized so outputs depend on images/task/stage); real RLC norm stats; RLC ca556f7 + openpi 01177e0 and JackLiu 7146d7b | stub + self-check, config lookup, num_tasks from metadata (JackLiu 100 under a 50-task named config), strict param check, warmup (plain + inpaint), masked logits; input dict identical to the fork's own `B1KPolicyWrapper.process_obs`/`prepare_batch_for_pi_behavior`; actions and logits bit-identical to the fork's `PiBehaviorPolicy` with the same RNG (plain and inpainting); LRU swaps (1 and 2 resident), re-trace after `clear_caches`; DA3-like extra params rejected; each fork rejects the other's checkpoint |
| unit tests | `tests/test_backends_rlc_groot.py` (fake policies / fake fork, no JAX) | dict vs the RLC wrapper code, inpainting kwargs, stage clamp, logits vs `StageTracker`, mapping parser, LRU, param-tree helpers, loading logic, env scripts, example config |
| real checkpoints on GPU | not run | VERIFY: latency, memory, compile time per swap, the meta100 base-velocity convention, cuSolver in the inpainting correction (a Blackwell user reported a GPU cuSolver failure there) |

No `# VERIFY:` remains in `pibehavior.py`. If reading the orbax parameter metadata ever fails (checked with orbax
0.11.13), the backend logs a warning and relies on the fork's own loader. The open GPU items are listed above.

---

## gr00t

`b1k26.backends.gr00t:Gr00tBackend`, env script `scripts/envs/gr00t.sh`. It drives the fork's `Gr00tPolicy`
directly; the fork's websocket server and its temporal-ensemble wrapper are not used.

### Fork, env and checkpoints

- **Fork:** [wensi-ai/Isaac-GR00T](https://github.com/wensi-ai/Isaac-GR00T) branch `behavior` @ `ace36d9`. Its pins:
  Python 3.10, torch 2.7.1 (cu128), transformers 4.57.3, flash-attn 2.7.4.post1. The lock plan was checked with
  `uv sync --frozen --dry-run`.

```bash
HF_TOKEN=... scripts/envs/gr00t.sh --prefix /opt/envs/gr00t --download-backbone   # add --no-flash-attn on Turing-only nodes
hf download kmy17518/gr00t-n1.7-b1k-multitask --include "checkpoint-238000/*" --local-dir /ckpt/gr00t-multitask
gdown 1OXNm3SPLvWOSJR1e8In6xHMHxOYDp789 -O radio.zip && unzip radio.zip -d /ckpt/gr00t-radio
# radio checkpoint dir = /ckpt/gr00t-radio/turning_on_radio_GR00T-checkpoint-150000
```

- **Gated backbone.** `Gr00tN1d7` builds its backbone and processor from `nvidia/Cosmos-Reason2-2B` at **every**
  load. That repo is gated: accept the license and set `HF_TOKEN` before `--download-backbone`. Then serve with
  `HF_HUB_OFFLINE=1`; the example config sets it.
- **Checkpoints.** Both use embodiment `NEW_EMBODIMENT` and carry the R1Pro modality config in
  `processor_config.json`; the backend validates it at load.
  - `kmy17518/gr00t-n1.7-b1k-multitask`: all 100 tasks. Its LR was never annealed, so screen several late snapshots.
  - The organizers' `turning_on_radio` baseline.

### Inputs and outputs

- **Observation**, as `gr00t/eval/eval_b1k_wrapper.py::process_input` builds it for one env (batched as in the
  fork's `andi/vector` fix):
  - `video[head|left_wrist|right_wrist]`: (B, 1, 224, 224, 3) uint8;
  - `state`: (B, 1, D) float32, raw `r1pro.json` slices `base_qvel[0:3]`, `torso[53:57]`, `left_arm[3:10]`,
    `left_gripper[24:26]` (two finger positions, not summed), `right_arm[28:35]`, `right_gripper[49:51]`;
  - `language["annotation.human.task_description"] = [[prompt], ...]`.
  - The processor resizes 224 to 256 itself (letterbox, shortest edge 256, centre crop 0.95). `info()["image_size"]`
    is 224, the wrapper's `obs_size`.
  - With the default 224 wrapper, the front server does not resize at all. With a full-res wrapper (720/480), the
    front server's PIL resize to 224 differs slightly from the wrapper's cv2 `INTER_LINEAR`. Training saw full-res
    frames that the processor reduced to 256 with `INTER_AREA`, so `image_size: 256` (profile and worker kwarg) is
    worth an A/B there.
- **Prompt.** `prompt_style: snake_case` (default) derives the dataset task string from `task_id`, e.g.
  `turning_on_radio`; the multitask card says to serve with the matching string. The upstream server sends its
  default "pick up the object and place it on the table" (an upstream bug). Other values: `instruction`, or `item`
  for the profile prompt verbatim.
- **Output.** `get_action` returns `base (B,16,3), torso (B,16,4), left_arm (B,16,7), left_gripper (B,16,1),
  right_arm (B,16,7), right_gripper (B,16,1)`.
  - These are filled into the 23-D layout by name (the group names equal `constants.ACTION_SLICES`).
  - `decode_action` already converts the RELATIVE torso/arm groups back to absolute (state + delta).
  - Base and grippers are absolute.
  - The result is (16, 23) float32; there are no subtask logits.
- **`info()`**: `action_horizon` 16, `image_size` 224, `num_stages` None, no inpainting, no stages.
- **Batching.** One `get_action` per micro-batch (`max_batch`, default 8). The policy itself processes items one by
  one and collates them. A failing batched call (e.g. OOM) is re-served item by item; after 3 failures in a row,
  batching is switched off.

### Turing guard (TITAN RTX, sm_75) and dtype

- **Attention.** flash-attn 2.7.4 refuses GPUs below sm_80 at runtime. The fork's `Qwen3Backbone` only falls back to
  SDPA when `import flash_attn` fails.
  - `attn_implementation: auto` (default) reads the device capability with torch. Below sm_80, or on CPU, or without
    flash-attn, it blocks the `flash_attn` import (`sys.modules["flash_attn"] = None`) before the model is built.
    The backbone then requests `attn_implementation="sdpa"`; the effect is the same as
    `use_flash_attention: false` in the checkpoint's `config.json`.
  - After loading, the backend checks the backbone's `config._attn_implementation` and refuses `flash_attention_2`
    below sm_80.
  - `flash_attention_2` on a device that cannot run it is a construction error.
- **dtype.** `bfloat16` is upstream: `Gr00tPolicy` loads in bf16 and casts its inputs to bf16. `float32` (or `auto`,
  which picks float32 below sm_80 and on CPU):
  - upcasts the model after loading (lossless; the checkpoint itself is bf16);
  - redirects the module-level `_rec_to_dtype` cast in `gr00t/policy/gr00t_policy.py` to float32, so pixels and
    normalized states are not rounded to bf16.
- On sm_75, bf16 GEMMs have no tensor cores and bf16 SDPA uses the math kernel. Memory: ~6.9 GB bf16 or ~13 GB fp32.
  Latency is not measured (VERIFY).

### Execution profile

`configs/gr00t.example.yaml`:

- receding horizon over the 16-step chunk (`execute_steps: 16, predicted_steps_to_use: 16, keep_for_inpaint: 0`);
  A/B `execute_steps: 8`;
- `use_stage: false`, `mask_base_qvel: false` (trained on 2026 robot-frame velocity).

The upstream default, temporal ensembling (an inference every step, only the first 5 of 16 actions used, nearly
uniform weights), is not reproduced.

### Constructor arguments (worker `--backend-kwargs` JSON)

| kwarg | default | meaning |
|---|---|---|
| `checkpoint` | required | dir with `config.json`, `*.safetensors`, `processor_config.json` (or `processor/`) |
| `embodiment_tag` | `NEW_EMBODIMENT` | as trained |
| `device` | `cuda` | torch device |
| `prompt_style` | `snake_case` | `snake_case` / `instruction` / `item` |
| `attn_implementation` | `auto` | `auto` / `sdpa` / `flash_attention_2` |
| `dtype` | `bfloat16` | `bfloat16` / `float32` / `auto` |
| `num_steps` | checkpoint (4) | `action_head.num_inference_timesteps` |
| `image_size` | 224 | must equal the profile's `image_size` (a mismatch is logged) |
| `batched` / `max_batch` | true / 8 | batched `get_action` |
| `strict` | true | `Gr00tPolicy` input/output validation |
| `seed` | none | `torch.manual_seed` at load |
| `modality_json` | none | cross-check a fork `r1pro.json` against the hard-coded slices |

### Verification status

| check | how | result |
|---|---|---|
| against the real fork code | CPU, torch 2.7.1 + transformers 5.5.4; a `Gr00tPolicy` without `__init__` (no weights) with the real `get_action` / `check_observation` / `_unbatch_observation` / `_to_vla_step_data` / `check_action`, the real `decode_action` logic over a real `StateActionProcessor` built from the kmy checkpoint's `processor_config.json` + statistics | observation passes strict validation (B = 3); states reach the processor unchanged; shifting the torso/arm state by d shifts exactly those 18 action dims by d (relative to absolute), base/grippers unchanged; the float32 cast redirection reaches the model; blocking flash_attn makes `import flash_attn` raise |
| unit tests | `tests/test_backends_rlc_groot.py` | observation layout vs `r1pro.json`, group to 23-D mapping, prompts, batching/fallback, attention/dtype decisions, load path with fake torch/gr00t (Turing: SDPA + float32; Ampere: untouched), env script, example config |
| real checkpoints on GPU | not run | VERIFY: latency (3 cameras), memory, no "weights not used/initialized" warnings when loading kmy's checkpoint (its `config.json` carries extra keys from the `multi-task` branch) |

`# VERIFY:` in the code: the backbone's attention attribute path (`model.backbone.model.config._attn_implementation`,
transformers 4.57.3). If it is missing, the post-load check only logs `None`; the flash-attn block still applies. The
`_rec_to_dtype` cast and `action_head.num_inference_timesteps` (read at every call, no `torch.compile`) were checked in
the fork source at `ace36d9`.
