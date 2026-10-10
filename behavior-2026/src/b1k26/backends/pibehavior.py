"""PiBehavior worker backend: RLC-architecture checkpoints (task/stage-conditioned pi0.5 with inpainting).

Two forks ship the same ``b1k`` package and the same inference path:

- **2025, RLC (1st place)**: ``IliaLarchenko/behavior-1k-solution`` @ ``ca556f7`` (openpi submodule
  ``wensi-ai/openpi`` @ ``01177e0``). Checkpoints ``IliaLarchenko/behavior_submission`` ckpt 1-4 and
  ``IliaLarchenko/behavior_50t_checkpoint``. ``TASK_NUM_STAGES`` has 50 entries (596 stage rows), the task table 50
  rows. Trained on 2025 data whose base velocity is ~0, so profiles should mask ``proprio[0:3]``.
- **2026, JackLiu meta**: ``JackLiu0406/behaviour-1k-2026-meta`` @ ``7146d7b`` (vendored, patched openpi).
  Checkpoints ``JackLiu0406/meta-SFT-checkpoints`` (``meta100-1epoch/step*``, ``single-task-finetune/no-da3/*``).
  ``TASK_NUM_STAGES`` has 100 entries (1120 stage rows); the task table has 100 rows (``num_tasks=100``, which the
  fork's named configs do NOT set: it comes from the checkpoint here). Trained on 2026 robot-frame base velocity.

The 2026 task ids are the model's task ids for both forks: the 2025 ``task_data.json`` order equals the 2026
``B100_task_misc.csv`` order for ids 0-49, and the meta pipeline appends the 50 new activities at 50-99 in ascending
2026 index (``b1k/training/b1k_2026.py::build_task_index_maps``), i.e. the identity map (checked against
``tasks.json``; JackLiu's model card states the same).

Per-item input dict, exactly what the forks' ``B1KPolicyWrapper.process_obs`` + ``prepare_batch_for_pi_behavior``
(``src/b1k/shared/eval_b1k_wrapper.py``) hand to ``PiBehaviorPolicy.infer``::

    {"observation/egocentric_camera": head (224,224,3) uint8,
     "observation/wrist_image_left": left wrist, "observation/wrist_image_right": right wrist,
     "observation/state": raw 61-D proprio float32,
     "tokenized_prompt": int32 [task_id, stage], "tokenized_prompt_mask": bool [True, True],
     "subtask_state": int32 stage}                      # "prompt" is deleted by the wrapper
    (+ "initial_actions": (k, 23) float32 absolute, also passed as ``infer(..., initial_actions=...)``)

``B1kInputs`` builds the 23-D state with ``extract_state_from_proprio`` using ``PROPRIOCEPTION_INDICES`` imported
from ``omnigibson.learning.utils.eval_utils`` (absent in 2026 OmniGibson; our worker env has no OmniGibson at all).
``install_eval_utils_stub`` registers that module with the 61-D slices from ``b1k26.constants`` before the fork is
imported; the 2026 field names equal the 2025 ones used there, so the fork's extraction yields RLC's action-order
state ``[base_qvel 3, trunk 4, L arm 7, L grip, R arm 7, R grip]`` (grip = 2*width/0.1 - 1). A self-check compares it
with ``b1k26.obs.state23_action_order`` at load.

Inpainting: ``PiBehaviorPolicy.infer(obs, initial_actions=a)`` runs ``a`` through the full input transform
(DeltaActions on torso/arms, per-timestamp normalization of rows 0..k-1, padding to 32 dims) and passes it to
``PiBehavior.sample_actions(initial_actions=...)``, which pins those k steps (all 32 dims) while the flow time is above
``PiBehaviorConfig.time_threshold_inpaint`` (0.3) and propagates the correction through the action covariance. The
wrapper's ``B1KWrapperConfig.time_threshold_inpaint`` / ``serve_b1k.py --time-threshold-inpaint`` is never
plumbed into the model (a no-op upstream); here ``time_threshold_inpaint`` overrides the model config field.

Outputs: ``{"actions": (30, 23) absolute, "subtask_logits": (15,) with -inf at stages the task does not have}``.
``StageTracker.update`` ignores non-finite logits, so masked entries are replaced by a large finite negative value.

Nothing here imports JAX or the fork at module import time.
"""

from __future__ import annotations

import collections
import dataclasses
import gc
import inspect
import json
import logging
import os
import pathlib
import sys
import time
import types
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from b1k26 import constants as C
from b1k26.backends.base import Backend, ChunkOut, InferItem
from b1k26.backends.openpi_b1k import postprocess_actions, reset_pose_proprio, validate_item
from b1k26.backends.openpi_comet import install_eval_utils_stub
from b1k26.obs import hold_action, state23_action_order

logger = logging.getLogger(__name__)

# Pinned upstream sources (scripts/envs/pibehavior.sh installs exactly these commits).
RLC_REPO = "https://github.com/IliaLarchenko/behavior-1k-solution"
RLC_COMMIT = "ca556f74a455cef7987a2be4537b5ac85cc56dd7"  # main head (2026-01-24)
RLC_OPENPI_COMMIT = "01177e0242a1c7e8fad2547caa0e987def614cda"  # its openpi submodule (wensi-ai/openpi)
JACKLIU_REPO = "https://github.com/JackLiu0406/behaviour-1k-2026-meta"
JACKLIU_COMMIT = "7146d7b179d2391db963f564868f6ca313185bce"  # main head (checked 2026-10-10)

# Train config names registered in b1k/training/config.py (_CONFIGS_DICT):
#   RLC 2025:     pi_behavior_b1k_fast (the only one; README: serve_b1k.py --policy.config pi_behavior_b1k_fast)
#   JackLiu 2026: pi_behavior_b1k_fast (same entry; the meta run replaces num_tasks=100 at train time) and
#                 pi_behavior_b1k_stage_only (no FAST / kv-transform / correlated noise ablation)
DEFAULT_CONFIG_NAME = "pi_behavior_b1k_fast"
DEFAULT_NUM_STEPS = 20  # RLC serve_b1k.py --num-steps default (flow-matching Euler steps)
IMAGE_SIZE = 224  # RLC wrapper RESIZE_SIZE; the model's ResizeImages(224, 224) is then a no-op
MAX_NUM_STAGES = 15  # pi_behavior_config.MAX_NUM_STAGES: length of subtask_logits
MASKED_LOGIT = -1.0e9  # replaces the model's -inf for stages a task does not have
FORK_STAGE_TABLES = {50: "rlc2025", 100: "jackliu2026"}
THRASH_LOADS, THRASH_WINDOW_S = 4, 600.0  # warn when this many checkpoint loads happen within the window

# Example keys of the RLC wrapper -> our camera roles.
CAMERA_KEYS = {
    "observation/egocentric_camera": "head",
    "observation/wrist_image_left": "left_wrist",
    "observation/wrist_image_right": "right_wrist",
}

# Extra parameter subtrees that mean "this checkpoint needs inputs we never send" (JackLiu DA3 / spatial branch).
_SPATIAL_PARAM_HINTS = ("da3", "spatial", "perceiver", "bank")


# ------------------------------------------------------------------------------------------------------------
# Pure helpers (numpy only; unit-tested on CPU)
# ------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class CheckpointSpec:
    """One checkpoint of a task -> checkpoint mapping (RLC ``task_checkpoint_mapping.json`` entry)."""

    name: str
    path: str
    tasks: tuple[int, ...] | None  # None = every task the checkpoint's task table covers
    norm_stats_dir: str | None = None  # dir holding norm_stats.json (default: <path>/assets/<asset_id>)
    num_tasks: int | None = None  # task-embedding rows, read from the checkpoint metadata when possible
    shapes: dict[tuple[str, ...], tuple[int, ...]] | None = dataclasses.field(default=None, repr=False)


def _task_id(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{where}: invalid task {value!r}")
    if isinstance(value, (int, np.integer)) or (isinstance(value, str) and value.strip().isdigit()):
        tid = int(value)
    elif isinstance(value, str) and value in C.task_by_name():
        tid = C.task_by_name()[value].task_id
    else:
        raise ValueError(f"{where}: unknown task {value!r} (use an id 0-99 or a task name)")
    if not 0 <= tid < C.NUM_TASKS:
        raise ValueError(f"{where}: task id {tid} outside [0, {C.NUM_TASKS})")
    return tid


def _expand_path(path: str, base_dir: pathlib.Path | None) -> str:
    p = pathlib.Path(os.path.expandvars(os.path.expanduser(str(path))))
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    return str(p)


def parse_task_checkpoint_mapping(
    spec: str | os.PathLike[str] | Mapping[str, Any], base_dir: str | os.PathLike[str] | None = None
) -> dict[str, CheckpointSpec]:
    """Parse an RLC-style mapping: ``{"checkpoints": {name: {"path": str, "tasks": [ids or names]}}}``.

    Accepts a JSON file path (relative checkpoint paths are then relative to the file's directory; ``~`` and
    ``$VARS`` are expanded) or an already-parsed dict. Optional per-entry ``"norm_stats_dir"`` (relative to the
    checkpoint dir unless absolute). Every task may appear at most once. RLC's own switcher also demands that all
    50 tasks are mapped; here unmapped tasks are simply not served by this worker (the front server must route
    them elsewhere), unless a fallback ``checkpoint`` is given to the backend.
    """
    if isinstance(spec, Mapping):
        doc, root = spec, (pathlib.Path(base_dir) if base_dir is not None else None)
    else:
        path = pathlib.Path(os.path.expanduser(os.fspath(spec)))
        with open(path) as f:
            doc = json.load(f)
        root = path.resolve().parent
    if not isinstance(doc, Mapping) or not isinstance(doc.get("checkpoints"), Mapping) or not doc["checkpoints"]:
        raise ValueError("task_checkpoint_mapping must contain a non-empty 'checkpoints' mapping")
    unknown_top = set(doc) - {"checkpoints"}
    if unknown_top:
        raise ValueError(f"task_checkpoint_mapping: unknown top-level keys {sorted(unknown_top)}")
    out: dict[str, CheckpointSpec] = {}
    seen: dict[int, str] = {}
    for name, entry in doc["checkpoints"].items():
        where = f"checkpoints.{name}"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where}: expected a mapping with 'path' and 'tasks'")
        unknown = set(entry) - {"path", "tasks", "norm_stats_dir"}
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        if "path" not in entry or "tasks" not in entry:
            raise ValueError(f"{where}: 'path' and 'tasks' are required")
        tasks_raw = entry["tasks"]
        if not isinstance(tasks_raw, (list, tuple)) or not tasks_raw:
            raise ValueError(f"{where}.tasks: expected a non-empty list")
        tasks = tuple(_task_id(t, f"{where}.tasks") for t in tasks_raw)
        for t in tasks:
            if t in seen:
                raise ValueError(f"task {t} is assigned to both {seen[t]!r} and {name!r}")
            seen[t] = str(name)
        ckpt = _expand_path(str(entry["path"]), root)
        nsd = entry.get("norm_stats_dir")
        if nsd is not None:
            nsd = _expand_path(str(nsd), pathlib.Path(ckpt))
        out[str(name)] = CheckpointSpec(name=str(name), path=ckpt, tasks=tasks, norm_stats_dir=nsd)
    return out


def build_example(item: InferItem, stage: int, proprio: np.ndarray, images: Mapping[str, np.ndarray],
                  initial_actions: np.ndarray | None = None) -> dict[str, Any]:
    """The dict the RLC/JackLiu wrapper passes to ``PiBehaviorPolicy.infer`` for one item (see module doc)."""
    example: dict[str, Any] = {key: images[role] for key, role in CAMERA_KEYS.items()}
    example["observation/state"] = proprio
    example["tokenized_prompt"] = np.array([int(item.task_id), int(stage)], dtype=np.int32)
    example["tokenized_prompt_mask"] = np.array([True, True], dtype=bool)
    example["subtask_state"] = np.array(int(stage), dtype=np.int32)
    if initial_actions is not None:
        example["initial_actions"] = initial_actions
    return example


def clamp_stage(stage: int | None, num_stages: int) -> int:
    """Stage index for the model: None -> 0, clamped to [0, num_stages - 1].

    An out-of-range stage would silently index another task's rows of ``task_stage_embeddings``.
    """
    if stage is None:
        return 0
    return int(min(max(int(stage), 0), max(int(num_stages) - 1, 0)))


def sanitize_subtask_logits(logits: Any) -> np.ndarray | None:
    """(15,) float32 logits with the model's -inf (stages the task lacks) replaced by ``MASKED_LOGIT``.

    ``b1k26.stage.StageTracker.update`` ignores logits with any non-finite value, so passing the raw -inf through
    would freeze the stage. NaN or +inf anywhere means a broken prediction: return None (the tracker skips it).
    """
    if logits is None:
        return None
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    x = np.array(logits, dtype=np.float32, copy=True).reshape(-1)
    if x.size == 0 or np.isnan(x).any() or np.isposinf(x).any():
        return None
    x[np.isneginf(x)] = MASKED_LOGIT
    if np.all(x <= MASKED_LOGIT):
        return None
    return x


def validate_initial_actions(initial_actions: Any, max_steps: int) -> np.ndarray | None:
    """``(k, 23)`` float32 copy of the inpainting prefix, or None if absent/empty/unusable.

    Non-finite values would corrupt the whole sampled chunk (they are pinned into the flow), so such a prefix is
    dropped with a warning instead. ``k`` is capped at the model horizon.
    """
    if initial_actions is None:
        return None
    a = np.asarray(initial_actions)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2 or a.shape[0] == 0:
        if a.size:
            logger.warning("ignoring initial_actions with shape %s (expected (k, 23))", a.shape)
        return None
    if a.shape[1] < C.ACTION_DIM:
        raise ValueError(f"initial_actions must be (k, {C.ACTION_DIM}), got {a.shape}")
    a = np.array(a[: int(max_steps), : C.ACTION_DIM], dtype=np.float32, copy=True)
    if not np.all(np.isfinite(a)):
        logger.warning("non-finite initial_actions; inpainting disabled for this plan")
        return None
    return a


def num_stages_table(task_num_stages: tuple[int, ...] | list[int], served: set[int] | frozenset[int]) -> list[int]:
    """``info()["num_stages"]``: 100 ints, the fork's stage count for served tasks and 0 for tasks this worker
    does not serve (beyond the fork's table, outside the checkpoint's task rows, or unmapped)."""
    out = []
    for t in range(C.NUM_TASKS):
        out.append(int(task_num_stages[t]) if t in served and t < len(task_num_stages) else 0)
    return out


def base_qvel_stats_convention(std: Any) -> str:
    """Classify the state norm stats of ``base_qvel`` (dims 0:3): "2025" (world-frame joint velocity, ~0 in the
    2025 demos: std ~0.01), "2026" (robot-frame velocity: std ~0.06-0.19), or "unknown"."""
    try:
        s = np.asarray(std, dtype=np.float64).reshape(-1)[:3]
    except (TypeError, ValueError):
        return "unknown"
    if s.size < 3 or not np.all(np.isfinite(s)):
        return "unknown"
    if np.all(s < 0.03):
        return "2025"
    if np.all(s > 0.04):
        return "2026"
    return "unknown"


# ---- parameter-tree comparison (orbax metadata vs nnx abstract model) ------------------------------------------
def _is_mapping_like(tree: Any) -> bool:
    if isinstance(tree, Mapping):
        return True
    return hasattr(tree, "items") and hasattr(tree, "keys") and not hasattr(tree, "shape")


def flatten_param_shapes(tree: Any) -> dict[tuple[str, ...], tuple[int, ...]]:
    """Flatten a nested params tree (dicts of arrays, ShapeDtypeStructs or orbax ArrayMetadata) to
    ``{path: shape}``. A trailing ``"value"`` key on every path (``nnx.State`` saved by openpi training) is dropped,
    exactly like ``openpi.models.model.restore_params``."""
    flat: dict[tuple[str, ...], tuple[int, ...]] = {}

    def walk(node: Any, prefix: tuple[str, ...]) -> None:
        if _is_mapping_like(node):
            for k, v in node.items():
                walk(v, prefix + (str(k),))
        elif isinstance(node, (list, tuple)) and not hasattr(node, "shape"):
            for i, v in enumerate(node):
                walk(v, prefix + (str(i),))
        else:
            shape = getattr(node, "shape", None)
            if shape is not None:
                flat[prefix] = tuple(int(s) for s in shape)

    walk(tree, ())
    if flat and all(k and k[-1] == "value" for k in flat):
        flat = {k[:-1]: v for k, v in flat.items()}
    return flat


def _find_embedding_rows(shapes: Mapping[tuple[str, ...], tuple[int, ...]], table: str) -> int | None:
    for path, shape in shapes.items():
        if table in path and path[-1] == "embedding" and len(shape) == 2:
            return int(shape[0])
    return None


def detect_num_tasks(shapes: Mapping[tuple[str, ...], tuple[int, ...]]) -> int | None:
    """Rows of ``task_embeddings.embedding`` (50 for 2025 checkpoints, 100 for the 2026 meta checkpoints)."""
    return _find_embedding_rows(shapes, "task_embeddings")


def detect_stage_rows(shapes: Mapping[tuple[str, ...], tuple[int, ...]]) -> int | None:
    """Rows of ``task_stage_embeddings.embedding`` (= sum(TASK_NUM_STAGES): 596 for RLC 2025, 1120 for JackLiu)."""
    return _find_embedding_rows(shapes, "task_stage_embeddings")


@dataclasses.dataclass
class ParamDiff:
    missing: list[tuple[str, ...]]
    extra: list[tuple[str, ...]]
    mismatched: list[tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]]  # (path, expected, got)
    common: int

    @property
    def ok(self) -> bool:
        return not (self.missing or self.extra or self.mismatched)


def compare_param_shapes(expected: Mapping[tuple[str, ...], tuple[int, ...]],
                         got: Mapping[tuple[str, ...], tuple[int, ...]]) -> ParamDiff:
    exp_keys, got_keys = set(expected), set(got)
    common = exp_keys & got_keys
    mismatched = sorted((k, tuple(expected[k]), tuple(got[k])) for k in common if tuple(expected[k]) != tuple(got[k]))
    return ParamDiff(missing=sorted(exp_keys - got_keys), extra=sorted(got_keys - exp_keys),
                     mismatched=mismatched, common=len(common))


def _fmt_paths(paths: list[Any], limit: int = 8) -> str:
    shown = ["/".join(p) if isinstance(p, tuple) and all(isinstance(x, str) for x in p) else str(p)
             for p in paths[:limit]]
    more = f" (+{len(paths) - limit} more)" if len(paths) > limit else ""
    return ", ".join(shown) + more


# ---- real-fork helpers (used only inside the worker env) -------------------------------------------------------
def orbax_param_shapes(ocp: Any, params_dir: str | os.PathLike[str]) -> dict[tuple[str, ...], tuple[int, ...]]:
    """Shapes of the arrays stored under ``<ckpt>/params`` from orbax metadata (no array is read).

    Mirrors ``openpi.models.model.restore_params``: ``PyTreeCheckpointer().metadata(path)["params"]`` (orbax >= 0.12
    wraps it in ``item_metadata``, as JackLiu's patched copy handles).
    """
    with ocp.PyTreeCheckpointer() as ckptr:
        meta = ckptr.metadata(pathlib.Path(params_dir).resolve())
    meta = getattr(meta, "item_metadata", None) or meta
    # Checked on CPU with orbax 0.11.13 (both forks): the leaves are ArrayMetadata with `.shape`, the tree is a dict.
    return flatten_param_shapes(meta["params"])


def nnx_model_param_shapes(jax: Any, nnx: Any, model_config: Any) -> dict[tuple[str, ...], tuple[int, ...]]:
    """Shapes of every parameter of ``model_config.create`` without allocating (same as ``BaseModelConfig.load``)."""
    model = nnx.eval_shape(model_config.create, jax.random.key(0))
    _, state = nnx.split(model)
    return flatten_param_shapes(state.to_pure_dict())


def check_rlc_state_extraction(b1k_policy_module: Any) -> None:
    """Fail fast unless the fork's ``extract_state_from_proprio`` runs on the 61-D layout in RLC action order."""
    indices = getattr(b1k_policy_module, "PROPRIOCEPTION_INDICES", None)
    r1 = None if indices is None else indices.get("R1Pro")
    if r1 is None or dict(r1) != dict(C.PROPRIO_INDICES_2026):
        raise RuntimeError(
            "b1k.policies.b1k_policy bound a PROPRIOCEPTION_INDICES that is not the 2026 61-D layout; "
            "install_eval_utils_stub() must run before the fork is imported"
        )
    probe = (np.arange(C.PROPRIO_DIM, dtype=np.float32) + 1.0) * 0.013
    got = np.asarray(b1k_policy_module.extract_state_from_proprio(probe), dtype=np.float32)
    want = state23_action_order(probe)
    if got.shape != want.shape or not np.allclose(got, want, atol=1e-5):
        raise RuntimeError(f"PiBehavior state extraction mismatch:\n got  {got}\n want {want}")


def _set_xla_env(mem_fraction: float | None, allocator: str | None) -> None:
    if mem_fraction is not None or os.environ.get("B1K26_MEM_FRACTION"):
        from b1k26.backends.openpi_b1k import node_mem_fraction

        mem_fraction = node_mem_fraction(float(mem_fraction) if mem_fraction is not None else 0.85)
        if not 0.05 <= float(mem_fraction) <= 1.0:
            raise ValueError("mem_fraction must be in [0.05, 1.0]")
        if "jax" in sys.modules:
            logger.warning("jax already imported; mem_fraction=%s may have no effect", mem_fraction)
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{float(mem_fraction):.3f}"
    if allocator is not None:
        if allocator not in ("platform", "bfc", "default"):
            raise ValueError("xla_allocator must be 'platform', 'bfc' or 'default'")
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = allocator


# ------------------------------------------------------------------------------------------------------------
# Backend
# ------------------------------------------------------------------------------------------------------------
class PiBehaviorBackend(Backend):
    """RLC-architecture (PiBehavior) checkpoints, one or several (task -> checkpoint mapping with an LRU).

    Constructor arguments (keyword, JSON-serializable so ``b1k26.worker --backend-kwargs`` can pass them):
      checkpoint               checkpoint dir (``params/`` + ``assets/``). Serves every task its task table covers
                               (or ``tasks``). With ``task_checkpoint_mapping`` it is the fallback for unmapped tasks.
      task_checkpoint_mapping  RLC ``task_checkpoint_mapping.json`` path (or the parsed dict), see
                               ``parse_task_checkpoint_mapping``.
      tasks                    restrict a single ``checkpoint`` to these task ids/names (e.g. JackLiu's meta5 model,
                               whose rows 50-99 except 76/77 are untrained).
      config_name              the fork's TrainConfig (default ``pi_behavior_b1k_fast``; JackLiu also has
                               ``pi_behavior_b1k_stage_only``). Validated against ``_CONFIGS_DICT``.
      num_tasks                task-embedding rows. None: read from each checkpoint's metadata (50 / 100), falling
                               back to ``len(TASK_NUM_STAGES)`` of the fork.
      num_steps                flow-matching steps, ``sample_kwargs={"num_steps": n}`` (RLC: 20).
      time_threshold_inpaint   override ``PiBehaviorConfig.time_threshold_inpaint`` (0.3); None keeps it.
      asset_id                 norm-stats sub-dir of ``<ckpt>/assets`` (default ``IliaLarchenko/behavior_224_rgb``).
      norm_stats_dir           explicit dir holding ``norm_stats.json`` for all checkpoints (relative = to each
                               checkpoint). Per-checkpoint ``norm_stats_dir`` in the mapping wins.
      max_resident             checkpoints kept in GPU memory at once (LRU; default 1 = RLC CheckpointSwitcher).
      clear_jax_caches         call ``jax.clear_caches()`` after unloading a checkpoint (ZSB/RLC do; frees old
                               executables).
      strict_params            compare the checkpoint's parameter tree with the model before loading: shape
                               mismatches and unexpected extra parameters (e.g. a DA3 checkpoint, which the fork's
                               ``create_trained_policy`` would silently drop) raise.
      allow_extra_params       accept extra checkpoint parameters (logged) when ``strict_params``.
      warmup_task_id           task for the warmup inference (default: the lowest served task).
      warmup_inpaint_steps     also compile the inpainting path with this many prefix steps (= the profile's
                               ``execution.keep_for_inpaint``; 0 skips it).
      warmup_all               load and warm every mapped checkpoint at warmup (fails fast on a broken one).
      mem_fraction, xla_allocator  set ``XLA_PYTHON_CLIENT_MEM_FRACTION`` / ``XLA_PYTHON_CLIENT_ALLOCATOR`` before
                               JAX is imported (RLC's serve_b1k.py uses 0.5 / platform).
      policy / policy_factory  test injection: a ready policy object (``infer(obs, initial_actions=None)``) for every
                               task, or ``callable(CheckpointSpec) -> policy``. Need ``task_num_stages`` and
                               ``action_horizon``; the fork is not imported.
    """

    flavor = "pibehavior"

    def __init__(
        self,
        checkpoint: str | os.PathLike[str] | None = None,
        *,
        task_checkpoint_mapping: str | os.PathLike[str] | Mapping[str, Any] | None = None,
        tasks: list[int | str] | None = None,
        config_name: str = DEFAULT_CONFIG_NAME,
        num_tasks: int | None = None,
        num_steps: int = DEFAULT_NUM_STEPS,
        time_threshold_inpaint: float | None = None,
        asset_id: str | None = None,
        norm_stats_dir: str | None = None,
        max_resident: int = 1,
        clear_jax_caches: bool = True,
        strict_params: bool = True,
        allow_extra_params: bool = False,
        warmup_task_id: int | None = None,
        warmup_inpaint_steps: int = 4,
        warmup_all: bool = False,
        mem_fraction: float | None = None,
        xla_allocator: str | None = None,
        policy: Any = None,
        policy_factory: Callable[[CheckpointSpec], Any] | None = None,
        task_num_stages: list[int] | tuple[int, ...] | None = None,
        action_horizon: int | None = None,
    ) -> None:
        if int(num_steps) < 1:
            raise ValueError("num_steps must be >= 1")
        if int(max_resident) < 1:
            raise ValueError("max_resident must be >= 1")
        if int(warmup_inpaint_steps) < 0:
            raise ValueError("warmup_inpaint_steps must be >= 0")
        if num_tasks is not None and not 1 <= int(num_tasks) <= C.NUM_TASKS:
            raise ValueError(f"num_tasks must be in [1, {C.NUM_TASKS}]")
        if time_threshold_inpaint is not None and not 0.0 <= float(time_threshold_inpaint) <= 1.0:
            raise ValueError("time_threshold_inpaint must be in [0, 1]")
        if checkpoint is None and task_checkpoint_mapping is None and policy is None:
            raise ValueError("checkpoint or task_checkpoint_mapping is required")
        if tasks is not None and task_checkpoint_mapping is not None and checkpoint is None:
            raise ValueError("tasks restricts a single checkpoint; put the task lists in task_checkpoint_mapping")
        if policy is not None and (checkpoint is not None or task_checkpoint_mapping is not None or policy_factory):
            raise ValueError("an injected policy replaces checkpoint / task_checkpoint_mapping / policy_factory")

        self.config_name = str(config_name)
        self.num_tasks = None if num_tasks is None else int(num_tasks)
        self.num_steps = int(num_steps)
        self.time_threshold_inpaint = None if time_threshold_inpaint is None else float(time_threshold_inpaint)
        self.asset_id = asset_id
        self.norm_stats_dir = norm_stats_dir
        self.max_resident = int(max_resident)
        self.clear_jax_caches = bool(clear_jax_caches)
        self.strict_params = bool(strict_params)
        self.allow_extra_params = bool(allow_extra_params)
        self.warmup_inpaint_steps = int(warmup_inpaint_steps)
        self.warmup_all = bool(warmup_all)
        self._fork: types.SimpleNamespace | None = None
        self._policy_factory: Callable[[CheckpointSpec], Any] | None = policy_factory
        self._resident: collections.OrderedDict[str, Any] = collections.OrderedDict()
        self._load_count = 0
        self._load_times: collections.deque[float] = collections.deque(maxlen=THRASH_LOADS * 4)
        self._last_thrash_warning = -float("inf")
        self._warned: set[Any] = set()
        self._seen_inpaint_k: set[int] = set()
        self.base_qvel_convention: dict[str, str] = {}
        self.fork_variant = "injected"

        # ---- checkpoint specs: mapping entries plus an optional fallback ------------------------------------
        specs: dict[str, CheckpointSpec] = {}
        if task_checkpoint_mapping is not None:
            specs.update(parse_task_checkpoint_mapping(task_checkpoint_mapping))
        self.fallback: str | None = None
        only = None if tasks is None else tuple(sorted({_task_id(t, "tasks") for t in tasks}))
        if checkpoint is not None:
            name = "default"
            while name in specs:
                name = "_" + name
            specs[name] = CheckpointSpec(name=name, path=_expand_path(str(checkpoint), None), tasks=only)
            self.fallback = name
        elif policy is not None:
            specs["injected"] = CheckpointSpec(name="injected", path="<injected>", tasks=only)
            self.fallback = "injected"
        self.specs = specs

        if policy is not None or policy_factory is not None:
            # Dependency injection (tests, or a caller that built the policies itself): no fork import.
            if task_num_stages is None or action_horizon is None:
                raise ValueError("task_num_stages and action_horizon are required when a policy is injected")
            self.task_num_stages = tuple(int(n) for n in task_num_stages)
            self.action_horizon = int(action_horizon)
            for s in self.specs.values():
                s.num_tasks = self.num_tasks or len(self.task_num_stages)
            if policy is not None:
                self._resident["injected"] = policy
                self._policy_factory = lambda spec: policy  # noqa: E731
        else:
            _set_xla_env(mem_fraction, xla_allocator)
            for s in self.specs.values():
                self._check_checkpoint_dir(s)
            t0 = time.monotonic()
            self._fork = self._import_fork()
            self.task_num_stages = tuple(int(n) for n in self._fork.pi_behavior_config.TASK_NUM_STAGES)
            self.fork_variant = FORK_STAGE_TABLES.get(len(self.task_num_stages), "unknown")
            self._base_train_config = self._get_train_config()
            self.action_horizon = int(self._base_train_config.model.action_horizon)
            for s in self.specs.values():
                self._inspect_checkpoint(s)
            logger.info("pibehavior fork %s (%d-task stage table) imported in %.1fs; config=%s horizon=%d",
                        self.fork_variant, len(self.task_num_stages), time.monotonic() - t0, self.config_name,
                        self.action_horizon)

        # ---- task routing --------------------------------------------------------------------------------------
        self.task_to_spec: dict[int, str] = {}
        for s in self.specs.values():
            if s.name == self.fallback:
                continue
            for t in s.tasks or ():
                self._check_task_servable(t, s)
                self.task_to_spec[t] = s.name
        if self.fallback is not None:
            s = self.specs[self.fallback]
            limit = min(s.num_tasks or len(self.task_num_stages), len(self.task_num_stages))
            for t in (s.tasks if s.tasks is not None else range(limit)):
                self._check_task_servable(t, s)
                self.task_to_spec.setdefault(int(t), s.name)
        if not self.task_to_spec:
            raise ValueError("this pibehavior worker serves no task")
        self.served_tasks = frozenset(self.task_to_spec)
        self.warmup_task_id = min(self.served_tasks) if warmup_task_id is None else int(warmup_task_id)
        if self.warmup_task_id not in self.served_tasks:
            raise ValueError(f"warmup_task_id {self.warmup_task_id} is not served by this worker")
        logger.info("pibehavior serves %d tasks from %d checkpoint(s): %s", len(self.served_tasks), len(self.specs),
                    {n: sorted(t for t, m in self.task_to_spec.items() if m == n) for n in self.specs})

    # ---- construction helpers -------------------------------------------------------------------------------------
    def _check_task_servable(self, task_id: int, spec: CheckpointSpec) -> None:
        if not 0 <= int(task_id) < len(self.task_num_stages):
            raise ValueError(
                f"task {task_id} is outside this fork's {len(self.task_num_stages)}-task stage table "
                f"({self.fork_variant}); checkpoint {spec.name!r} cannot serve it (2025 RLC checkpoints know tasks "
                f"0-49 only)")
        rows = spec.num_tasks or len(self.task_num_stages)
        if int(task_id) >= rows:
            raise ValueError(f"task {task_id} is outside checkpoint {spec.name!r}'s {rows}-row task table")

    @staticmethod
    def _check_checkpoint_dir(spec: CheckpointSpec) -> None:
        path = pathlib.Path(spec.path)
        if not (path / "params").is_dir():
            raise FileNotFoundError(f"checkpoint {spec.name!r}: {path} has no params/ directory")

    def _import_fork(self) -> types.SimpleNamespace:
        """Import the installed fork (RLC 2025 or JackLiu 2026 ``b1k`` package) lazily."""
        install_eval_utils_stub()  # before b1k.policies.b1k_policy binds PROPRIOCEPTION_INDICES
        import flax.nnx as nnx
        import jax
        import orbax.checkpoint as ocp

        # Module paths checked at RLC ca556f7 and JackLiu 7146d7b (identical inference modules).
        from b1k.models import pi_behavior_config
        from b1k.policies import b1k_policy, policy_config
        from b1k.shared import normalize
        from b1k.training import config

        check_rlc_state_extraction(b1k_policy)
        return types.SimpleNamespace(
            jax=jax, nnx=nnx, ocp=ocp, pi_behavior_config=pi_behavior_config, b1k_policy=b1k_policy,
            policy_config=policy_config, normalize=normalize, config=config,
            checkpoint_param_shapes=lambda path: orbax_param_shapes(ocp, path),
            model_param_shapes=lambda model_config: nnx_model_param_shapes(jax, nnx, model_config),
            clear_caches=jax.clear_caches,
        )

    def _get_train_config(self) -> Any:
        cfg_mod = self._fork.config
        known = getattr(cfg_mod, "_CONFIGS_DICT", None)
        if known is not None and self.config_name not in known:
            raise ValueError(f"config {self.config_name!r} not found in this fork; known: {sorted(known)}")
        train_config = cfg_mod.get_config(self.config_name)
        model = train_config.model
        cls = getattr(self._fork.pi_behavior_config, "PiBehaviorConfig", None)
        if cls is not None and not isinstance(model, cls):
            raise ValueError(f"config {self.config_name!r} is not a PiBehaviorConfig model ({type(model).__name__})")
        da3 = getattr(model, "da3", None)
        if (da3 is not None and getattr(da3, "enabled", True)) or getattr(model, "use_spatial_action_cross_attention",
                                                                          False):
            raise ValueError(f"config {self.config_name!r} needs DA3 / spatial tokens; this backend sends RGB only")
        return train_config

    def _inspect_checkpoint(self, spec: CheckpointSpec) -> None:
        """Read the checkpoint's parameter shapes (orbax metadata): task-table rows and fork compatibility."""
        try:
            shapes = self._fork.checkpoint_param_shapes(pathlib.Path(spec.path) / "params")
        except Exception as exc:  # noqa: BLE001 - metadata layout differences must not block a loadable checkpoint
            logger.warning("checkpoint %r: could not read parameter metadata (%s); trusting the config", spec.name, exc)
            shapes = {}
        spec.shapes = shapes
        detected = detect_num_tasks(shapes)
        stage_rows = detect_stage_rows(shapes)
        expected_rows = sum(self.task_num_stages)
        if stage_rows is not None and stage_rows != expected_rows:
            other = {596: "the 2025 RLC fork (50 tasks)", 1120: "the 2026 JackLiu fork (100 tasks)"}.get(stage_rows,
                                                                                                         "another fork")
            raise ValueError(
                f"checkpoint {spec.name!r} has {stage_rows} stage-embedding rows but this fork's TASK_NUM_STAGES "
                f"sums to {expected_rows} ({self.fork_variant}); it belongs to {other}. Use the matching worker env.")
        if self.num_tasks is not None:
            if detected is not None and detected != self.num_tasks:
                raise ValueError(f"checkpoint {spec.name!r} has {detected} task rows but num_tasks={self.num_tasks}")
            spec.num_tasks = self.num_tasks
        else:
            spec.num_tasks = detected if detected is not None else len(self.task_num_stages)
        if spec.num_tasks > len(self.task_num_stages):
            raise ValueError(f"checkpoint {spec.name!r} has {spec.num_tasks} task rows but the fork's stage table only "
                             f"covers {len(self.task_num_stages)} tasks")
        logger.info("checkpoint %r: %s task rows%s", spec.name, spec.num_tasks,
                    "" if detected is not None else " (not read from metadata)")

    # ---- loading / LRU ------------------------------------------------------------------------------------------
    def _train_config_for(self, spec: CheckpointSpec) -> Any:
        train_config = self._base_train_config
        updates: dict[str, Any] = {}
        if spec.num_tasks is not None and spec.num_tasks != getattr(train_config.model, "num_tasks", None):
            updates["num_tasks"] = int(spec.num_tasks)
        if self.time_threshold_inpaint is not None:
            updates["time_threshold_inpaint"] = self.time_threshold_inpaint
        if updates:
            train_config = dataclasses.replace(train_config, model=dataclasses.replace(train_config.model, **updates))
        if self.asset_id is not None:
            data = train_config.data
            assets = dataclasses.replace(data.assets, asset_id=self.asset_id)
            train_config = dataclasses.replace(train_config, data=dataclasses.replace(data, assets=assets))
        return train_config

    def _norm_stats_source(self, train_config: Any, spec: CheckpointSpec) -> pathlib.Path:
        nsd = spec.norm_stats_dir or self.norm_stats_dir
        if nsd is not None:
            p = pathlib.Path(os.path.expanduser(nsd))
            return p if p.is_absolute() else pathlib.Path(spec.path) / p
        data = train_config.data
        asset_id = getattr(data.assets, "asset_id", None) or getattr(data, "repo_id", None)
        if not asset_id:
            raise ValueError("the config has no asset id; pass asset_id or norm_stats_dir")
        return pathlib.Path(spec.path) / "assets" / str(asset_id)

    def _load_norm_stats(self, train_config: Any, spec: CheckpointSpec) -> dict[str, Any]:
        source = self._norm_stats_source(train_config, spec)
        try:
            norm_stats = self._fork.normalize.load(source)
        except FileNotFoundError as exc:
            root = pathlib.Path(spec.path)
            found = sorted(str(p.relative_to(root)) for p in root.glob("assets/**/norm_stats.json"))
            raise FileNotFoundError(f"norm_stats.json not found under {source}; found in checkpoint: "
                                    f"{found or 'none'}. Set asset_id or norm_stats_dir.") from exc
        for key in ("state", "actions"):
            if key not in norm_stats:
                raise ValueError(f"norm stats at {source} lack {key!r} (keys: {sorted(norm_stats)})")
        state, actions = norm_stats["state"], norm_stats["actions"]
        if np.asarray(state.mean).shape[-1] < 23 or np.asarray(actions.mean).shape[-1] < C.ACTION_DIM:
            raise ValueError(f"norm stats at {source} have fewer than 23 state/action dims")
        model = train_config.model
        needs_correlation = bool(getattr(model, "use_correlated_noise", False))
        if needs_correlation and getattr(actions, "action_correlation_cholesky", None) is None:
            raise ValueError(f"norm stats at {source} have no action_correlation_cholesky (needed by "
                             "use_correlated_noise; RLC computes it with compute_norm_stats.py --correlation)")
        convention = base_qvel_stats_convention(state.std)
        self.base_qvel_convention[spec.name] = convention
        logger.info("checkpoint %r: norm stats %s (state std[0:3]=%s -> %s base_qvel convention)", spec.name, source,
                    np.round(np.asarray(state.std, dtype=np.float64).reshape(-1)[:3], 4).tolist(), convention)
        if convention == "2025":
            logger.warning("checkpoint %r normalizes base_qvel with 2025 stats (std ~0.01): unless it was trained on "
                           "2026 data with these stats, serve it with mask_base_qvel: true", spec.name)
        return norm_stats

    def _check_params(self, train_config: Any, spec: CheckpointSpec) -> None:
        if not self.strict_params:
            return
        got = spec.shapes or {}
        if not got:
            logger.warning("checkpoint %r: parameter check skipped (no metadata)", spec.name)
            return
        try:
            expected = self._fork.model_param_shapes(train_config.model)
        except Exception as exc:  # noqa: BLE001 - the fork's own load still validates
            logger.warning("checkpoint %r: could not build the abstract model for the parameter check (%s)",
                           spec.name, exc)
            return
        diff = compare_param_shapes(expected, got)
        if diff.common == 0:
            logger.warning("checkpoint %r: parameter paths do not match the model's naming; check skipped", spec.name)
            return
        problems = []
        if diff.mismatched:
            problems.append("shape mismatches: " + _fmt_paths([f"{'/'.join(p)} model {e} vs ckpt {g}"
                                                               for p, e, g in diff.mismatched]))
        if diff.missing:
            problems.append("missing in checkpoint: " + _fmt_paths(diff.missing))
        if diff.extra:
            spatial = any(h in "/".join(p).lower() for p in diff.extra for h in _SPATIAL_PARAM_HINTS)
            msg = ("unexpected checkpoint parameters (the fork would silently drop them"
                   + ("; looks like a DA3/spatial checkpoint, which needs inputs this backend never sends" if spatial
                      else "") + "): " + _fmt_paths(diff.extra))
            if self.allow_extra_params and not spatial:
                logger.warning("checkpoint %r: %s", spec.name, msg)
            else:
                problems.append(msg)
        if problems:
            raise ValueError(f"checkpoint {spec.name!r} does not match config {self.config_name!r} "
                             f"(num_tasks={spec.num_tasks}): " + "; ".join(problems))
        logger.info("checkpoint %r: parameter tree matches the model (%d arrays)", spec.name, diff.common)

    def _load_policy(self, spec: CheckpointSpec) -> Any:
        if self._fork is None:
            assert self._policy_factory is not None
            return self._policy_factory(spec)
        train_config = self._train_config_for(spec)
        self._check_params(train_config, spec)
        norm_stats = self._load_norm_stats(train_config, spec)
        policy = self._fork.policy_config.create_trained_policy(
            train_config, spec.path, sample_kwargs={"num_steps": self.num_steps}, norm_stats=norm_stats)
        try:
            params = inspect.signature(policy.infer).parameters
        except (TypeError, ValueError):
            params = {}
        if params and "initial_actions" not in params:
            raise RuntimeError(f"checkpoint {spec.name!r} loaded as {type(policy).__name__}, not PiBehaviorPolicy "
                               "(no initial_actions); is config_name a PiBehavior config?")
        return policy

    def _release(self, name: str, policy: Any) -> None:
        """Drop every reference to a policy's device arrays and free the memory (ZSB PolicyRouter / RLC switcher)."""
        for attr in ("_sample_actions", "_model", "_input_transform", "_output_transform", "_rng"):
            if hasattr(policy, attr):
                try:
                    delattr(policy, attr)
                except Exception:  # noqa: BLE001
                    pass
        del policy
        gc.collect()
        if self._fork is not None and self.clear_jax_caches:
            try:
                self._fork.clear_caches()
            except Exception as exc:  # noqa: BLE001
                logger.warning("jax.clear_caches failed: %s", exc)
            # Surviving resident models are simply re-traced on their next call (checked on CPU with both forks:
            # the inpainting correction matrix lives in the traced copy of the module, not in the policy's model).
            gc.collect()
        logger.info("unloaded checkpoint %r", name)

    def _policy_for(self, name: str) -> Any:
        if name in self._resident:
            self._resident.move_to_end(name)
            return self._resident[name]
        while len(self._resident) >= self.max_resident:
            old, pol = self._resident.popitem(last=False)
            self._release(old, pol)
        spec = self.specs[name]
        t0 = time.monotonic()
        logger.info("loading checkpoint %r from %s", name, spec.path)
        policy = self._load_policy(spec)
        self._resident[name] = policy
        self._load_count += 1
        logger.info("loaded checkpoint %r in %.1fs (resident: %s)", name, time.monotonic() - t0, list(self._resident))
        # Concurrent rollouts on tasks of different checkpoints make a small LRU reload (and recompile) on every plan.
        self._load_times.append(t0)
        recent = [t for t in self._load_times if t0 - t < THRASH_WINDOW_S]
        if len(recent) >= THRASH_LOADS and t0 - self._last_thrash_warning > THRASH_WINDOW_S:
            self._last_thrash_warning = t0
            logger.warning("checkpoint thrashing: %d loads in the last %.0f s with max_resident=%d; raise "
                           "max_resident (GPU memory permitting) or give these checkpoints separate workers",
                           len(recent), THRASH_WINDOW_S, self.max_resident)
        return policy

    @property
    def resident(self) -> list[str]:
        return list(self._resident)

    # ---- Backend API ------------------------------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        return {
            "flavor": self.flavor,
            "action_horizon": int(self.action_horizon),
            "image_size": IMAGE_SIZE,
            "num_stages": num_stages_table(self.task_num_stages, self.served_tasks),
            "supports_inpaint": True,
            "supports_stage": True,
            # Extras (not in the base contract): which task ids this worker can serve, and the fork.
            "supported_tasks": sorted(self.served_tasks),
            "fork": self.fork_variant,
        }

    def spec_for_task(self, task_id: int) -> str:
        t = int(task_id)
        name = self.task_to_spec.get(t)
        if name is None:
            if not 0 <= t < C.NUM_TASKS:
                raise ValueError(f"task_id {t} outside [0, {C.NUM_TASKS})")
            why = (f"this fork ({self.fork_variant}) has a {len(self.task_num_stages)}-task table"
                   if t >= len(self.task_num_stages) else "no checkpoint of this worker is mapped to it")
            raise ValueError(f"task {t} ({C.task(t).name}) is not served by this pibehavior worker: {why}. "
                             "Route it to another profile.")
        return name

    def _prepare(self, item: InferItem) -> tuple[str, dict[str, Any], np.ndarray | None]:
        name = self.spec_for_task(item.task_id)
        proprio, images = validate_item(item, IMAGE_SIZE)
        num_stages = int(self.task_num_stages[int(item.task_id)])
        stage = clamp_stage(item.stage, num_stages)
        if item.stage is not None and stage != int(item.stage) and (item.task_id, item.stage) not in self._warned:
            self._warned.add((item.task_id, item.stage))
            logger.warning("task %d: stage %s outside [0, %d); clamped to %d", item.task_id, item.stage, num_stages,
                           stage)
        initial = validate_initial_actions(item.initial_actions, self.action_horizon)
        if initial is not None and initial.shape[0] not in self._seen_inpaint_k:
            self._seen_inpaint_k.add(initial.shape[0])
            logger.info("inpainting with %d prefix steps (a new prefix length compiles the model once more)",
                        initial.shape[0])
        return name, build_example(item, stage, proprio, images, initial), initial

    def infer(self, items: list[InferItem]) -> list[ChunkOut]:
        if not items:
            return []
        prepared = [self._prepare(item) for item in items]  # validates every item before any model work
        groups: collections.OrderedDict[str, list[int]] = collections.OrderedDict()
        for i, (name, _, _) in enumerate(prepared):
            groups.setdefault(name, []).append(i)
        # Serve the checkpoints already in memory first, so a mixed batch swaps each checkpoint in at most once.
        order = sorted(groups, key=lambda n: (n not in self._resident, list(groups).index(n)))
        results: list[ChunkOut | None] = [None] * len(items)
        for name in order:
            policy = self._policy_for(name)
            for i in groups[name]:
                _, example, initial = prepared[i]
                if initial is None:
                    out = policy.infer(example)
                else:
                    out = policy.infer(example, initial_actions=initial)
                results[i] = ChunkOut(actions=postprocess_actions(out["actions"]),
                                      subtask_logits=sanitize_subtask_logits(out.get("subtask_logits")))
        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]

    def _dummy_item(self, task_id: int, initial_steps: int = 0) -> InferItem:
        blank = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        proprio = reset_pose_proprio()
        initial = None
        if initial_steps > 0:
            initial = np.tile(hold_action(proprio), (int(initial_steps), 1)).astype(np.float32)
        return InferItem(task_id=int(task_id), prompt="", proprio=proprio,
                         images={r: blank for r in CAMERA_KEYS.values()}, stage=0, initial_actions=initial)

    def warmup(self) -> float:
        """Load the warmup task's checkpoint and compile both sampling paths (plain and inpainting).

        With ``warmup_all`` every checkpoint is loaded and warmed once (the warmup task's checkpoint last, so it stays
        resident). Returns total milliseconds.
        """
        t0 = time.monotonic()
        default_name = self.spec_for_task(self.warmup_task_id)
        plan: list[tuple[str, int]] = []
        if self.warmup_all:
            first_task: dict[str, int] = {}
            for t, n in sorted(self.task_to_spec.items()):
                first_task.setdefault(n, t)
            plan = [(n, t) for n, t in first_task.items() if n != default_name]
        plan.append((default_name, self.warmup_task_id))
        k = min(self.warmup_inpaint_steps, self.action_horizon)
        for name, task in plan:
            for steps in ([0, k] if k > 0 else [0]):
                t1 = time.monotonic()
                out = self.infer([self._dummy_item(task, steps)])
                if len(out) != 1 or out[0].actions.ndim != 2 or out[0].actions.shape[1] != C.ACTION_DIM:
                    raise RuntimeError(f"warmup produced an invalid chunk for checkpoint {name!r}")
                logger.info("warmup %r task %d inpaint=%d: %.0f ms, chunk %s, logits %s", name, task, steps,
                            (time.monotonic() - t1) * 1e3, out[0].actions.shape,
                            None if out[0].subtask_logits is None else out[0].subtask_logits.shape)
        return (time.monotonic() - t0) * 1e3


__all__ = [
    "CAMERA_KEYS",
    "CheckpointSpec",
    "DEFAULT_CONFIG_NAME",
    "JACKLIU_COMMIT",
    "JACKLIU_REPO",
    "MASKED_LOGIT",
    "ParamDiff",
    "PiBehaviorBackend",
    "RLC_COMMIT",
    "RLC_OPENPI_COMMIT",
    "RLC_REPO",
    "base_qvel_stats_convention",
    "build_example",
    "check_rlc_state_extraction",
    "clamp_stage",
    "compare_param_shapes",
    "detect_num_tasks",
    "detect_stage_rows",
    "flatten_param_shapes",
    "num_stages_table",
    "parse_task_checkpoint_mapping",
    "sanitize_subtask_logits",
    "validate_initial_actions",
]
