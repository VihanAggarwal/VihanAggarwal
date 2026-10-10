"""Comet worker backend: ``sunshk/openpi_comet`` pi0.5 checkpoints (pt50, pt12) through ``mli0603/openpi-comet``.

Comet (2025 2nd place) pretrained pi0.5 on the 2025 50-task demos: ``pi05-b1kpt50-cs32`` (tasks 0-49) and
``pi05-b1kpt12-cs32`` (12 tasks), action chunk 32, served with the fork's ``pi05_b1k-base`` config
(``Pi0Config(pi05=True, action_horizon=32)``, ``LeRobotB1KDataConfig``: absolute actions, quantile norm stats
from ``<ckpt>/assets/behavior-1k/2025-challenge-demos``).

Per-item input dict, exactly what the fork's ``shared/eval_b1k_wrapper.py`` (and ZSB's 2026 ``serve_comet.py``)
pass to ``policy.infer``::

    {"observation/egocentric_camera": head (224,224,3) uint8,
     "observation/wrist_image_left": left wrist, "observation/wrist_image_right": right wrist,
     "observation/state": raw 61-D proprio (float32), "prompt": str}

The fork's ``B1kInputs`` extracts the 23-D state itself with ``PROPRIOCEPTION_INDICES`` imported from
``omnigibson.learning.utils.eval_utils``. That module does not exist in 2026 OmniGibson (it moved to
``omnigibson.eval``), so ``install_eval_utils_stub`` registers a stub with the 61-D indices from
``b1k26.constants`` before openpi is imported, as ZSB's ``serve_comet.py`` does. Comet's state order differs from
the action order: ``[base_qvel 3, trunk 4, left arm 7, right arm 7, left finger width, right finger width]``.

The 2025 demos have base velocity ~0 (world-frame joint velocity, mostly unused), while the 2026 evaluator reports
real robot-frame velocity. Profiles serving these checkpoints should set ``mask_base_qvel: true`` so the front
server zeroes ``proprio[0:3]``; this backend passes the proprio through unchanged.
"""

from __future__ import annotations

import collections
import logging
import sys
import types
from typing import Any

import numpy as np

from b1k26 import constants as C
from b1k26.backends.base import InferItem
from b1k26.backends.openpi_b1k import OpenPIBackendBase, resolve_prompt, validate_item

logger = logging.getLogger(__name__)

# Pinned upstream sources (scripts/envs/openpi_comet.sh installs exactly this commit).
OPENPI_COMET_REPO = "https://github.com/mli0603/openpi-comet"
OPENPI_COMET_COMMIT = "4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5"  # main head

EVAL_UTILS_MODULE = "omnigibson.learning.utils.eval_utils"
COMET_STATE_GRIPPER_COLUMNS = (21, 22)  # grippers come last in Comet's state

_P = C.PROPRIO_INDICES_2026


def comet_proprio_indices() -> dict[str, "collections.OrderedDict[str, slice]"]:
    """``PROPRIOCEPTION_INDICES`` in the shape ``b1k_policy`` expects: ``{"R1Pro": OrderedDict(name -> slice)}``."""
    return {"R1Pro": collections.OrderedDict(C.PROPRIO_INDICES_2026)}


def install_eval_utils_stub() -> types.ModuleType:
    """Register ``omnigibson.learning.utils.eval_utils`` with the 2026 61-D ``PROPRIOCEPTION_INDICES``.

    Must run before ``openpi.training.config`` / ``openpi.policies.b1k_policy`` are imported. Parent packages
    are only stubbed when absent from ``sys.modules``; the leaf module is always replaced so a stale 2025 layout
    (256-D) can never be picked up. Idempotent.
    """
    parts = EVAL_UTILS_MODULE.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = []  # behave like a package for submodule lookups
            sys.modules[name] = pkg
    mod = types.ModuleType(EVAL_UTILS_MODULE)
    mod.__doc__ = "b1k26 stub: the 61-D R1Pro proprio layout of omnigibson.eval.utils.eval_utils (2026)."
    mod.PROPRIOCEPTION_INDICES = comet_proprio_indices()
    mod.B1K26_STUB = True
    sys.modules[EVAL_UTILS_MODULE] = mod
    for i in range(1, len(parts)):
        parent = sys.modules[".".join(parts[:i])]
        try:
            setattr(parent, parts[i], sys.modules[".".join(parts[: i + 1])])
        except (AttributeError, TypeError):  # pragma: no cover - exotic module objects
            pass
    return mod


def comet_state_from_proprio(proprio: np.ndarray) -> np.ndarray:
    """Reference copy of Comet's ``b1k_policy.extract_state_from_proprio`` on the 61-D layout (23-D)."""
    p = np.asarray(proprio, dtype=np.float32)
    return np.concatenate(
        [
            p[..., _P["base_qvel"]],
            p[..., _P["trunk_qpos"]],
            p[..., _P["arm_left_qpos"]],
            p[..., _P["arm_right_qpos"]],
            p[..., _P["gripper_left_qpos"]].sum(axis=-1, keepdims=True),
            p[..., _P["gripper_right_qpos"]].sum(axis=-1, keepdims=True),
        ],
        axis=-1,
    ).astype(np.float32)


def check_comet_state_extraction(b1k_policy_module: Any) -> None:
    """Fail fast if the imported fork does not extract Comet's state from the 61-D layout."""
    indices = getattr(b1k_policy_module, "PROPRIOCEPTION_INDICES", None)
    if indices is None or dict(indices.get("R1Pro", {})) != dict(C.PROPRIO_INDICES_2026):
        raise RuntimeError(
            "openpi.policies.b1k_policy bound a PROPRIOCEPTION_INDICES that is not the 2026 61-D layout; "
            "install_eval_utils_stub() must run before openpi is imported"
        )
    probe = (np.arange(C.PROPRIO_DIM, dtype=np.float32) + 1.0) * 0.013
    got = np.asarray(b1k_policy_module.extract_state_from_proprio(probe), dtype=np.float32)
    want = comet_state_from_proprio(probe)
    if got.shape != want.shape or not np.allclose(got, want, atol=1e-6):
        raise RuntimeError(f"Comet state extraction mismatch:\n got  {got}\n want {want}")


class CometBackend(OpenPIBackendBase):
    """Comet pt50/pt12 (``pi05_b1k-base``) served through the openpi-comet fork.

    Constructor arguments are those of ``OpenPIBackendBase``; defaults: ``config_name="pi05_b1k-base"``,
    ``default_prompt_mode="comet2025"`` (the fork's ``scripts/task_mapping.json`` texts). The pt12 checkpoint uses
    the same serving config (its training config differs only in task list and LR).
    """

    flavor = "openpi_comet"
    default_config_name = "pi05_b1k-base"
    default_prompt_mode = "comet2025"
    recommended_mask_base_qvel = True  # 2025 demos: base_qvel ~0; mask the 2026 robot-frame velocity

    def _import_fork(self) -> types.SimpleNamespace:
        install_eval_utils_stub()
        import jax
        import jax.numpy as jnp

        # Module paths checked at mli0603/openpi-comet 4bb2aa7 (CPU integration run through the real fork).
        from openpi import transforms
        from openpi.models import model
        from openpi.policies import b1k_policy, policy, policy_config
        from openpi.shared import download, normalize
        from openpi.training import checkpoints
        from openpi.training import config

        check_comet_state_extraction(b1k_policy)
        return types.SimpleNamespace(
            jax=jax, jnp=jnp, transforms=transforms, model=model, b1k_policy=b1k_policy, policy=policy,
            policy_config=policy_config, download=download, normalize=normalize, checkpoints=checkpoints,
            config=config,
        )

    def _check_train_config(self, train_config: Any) -> None:
        cfg = self._fork.config
        rgbd_cls = getattr(cfg, "LeRobotB1KRGBDDataConfig", None)
        if rgbd_cls is not None and isinstance(train_config.data, rgbd_cls):
            raise ValueError(f"config {self.config_name!r} needs depth / point clouds; this backend sends RGB only")
        factory_cls = getattr(cfg, "LeRobotB1KDataConfig", None)
        if factory_cls is None or not isinstance(train_config.data, factory_cls):
            raise ValueError(f"config {self.config_name!r} does not use LeRobotB1KDataConfig")

    def _after_load(self) -> None:
        # Comet's state has raw finger widths (~[0, 0.1]) in columns 21-22; anything else means foreign stats.
        q99 = getattr(self.norm_stats["state"], "q99", None)
        if q99 is not None:
            q99 = np.asarray(q99)
            if q99.shape[-1] > max(COMET_STATE_GRIPPER_COLUMNS):
                highs = [float(q99[c]) for c in COMET_STATE_GRIPPER_COLUMNS]
                if any(h > 0.2 for h in highs):
                    logger.warning("state q99 at gripper columns 21-22 is %s; Comet expects widths <= ~0.1", highs)

    def build_example(self, item: InferItem) -> dict[str, Any]:
        proprio, images = validate_item(item)
        return {
            "observation/egocentric_camera": images["head"],
            "observation/wrist_image_left": images["left_wrist"],
            "observation/wrist_image_right": images["right_wrist"],
            "observation/state": proprio,
            "prompt": resolve_prompt(item.prompt, item.task_id, self.default_prompt_mode),
        }


__all__ = [
    "CometBackend",
    "OPENPI_COMET_COMMIT",
    "OPENPI_COMET_REPO",
    "check_comet_state_extraction",
    "comet_proprio_indices",
    "comet_state_from_proprio",
    "install_eval_utils_stub",
]
