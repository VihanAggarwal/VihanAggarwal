"""Backend interface: one model family's inference, hosted inside a worker process.

A backend receives already-prepared per-rollout inputs (images resized by the front server, raw 61-D proprio,
task id, prompt, optional stage and inpainting prefix) and returns absolute 23-D action chunks in the evaluator's
action space (see b1k26.constants). Everything else (queueing, compression, corrections, protocol) lives in the
front server so it is shared across families.
"""

from __future__ import annotations

import abc
import dataclasses
import importlib
from typing import Any

import numpy as np


@dataclasses.dataclass
class InferItem:
    task_id: int
    prompt: str
    proprio: np.ndarray  # (61,) float32; base_qvel already masked by the front server if the profile asks
    images: dict[str, np.ndarray]  # "head" / "left_wrist" / "right_wrist" -> (S, S, 3) uint8
    stage: int | None = None
    initial_actions: np.ndarray | None = None  # (k, 23) absolute actions for inpainting, if supported


@dataclasses.dataclass
class ChunkOut:
    actions: np.ndarray  # (T, 23) float32, absolute evaluator actions
    subtask_logits: np.ndarray | None = None  # (num_stages_max,) for stage-tracking models


class Backend(abc.ABC):
    """One loaded model (or a small LRU set of same-family checkpoints)."""

    flavor: str = "base"

    @abc.abstractmethod
    def info(self) -> dict[str, Any]:
        """Return {"flavor", "action_horizon", "image_size", "num_stages" (list[int] | None),
        "supports_inpaint", "supports_stage"}."""

    @abc.abstractmethod
    def warmup(self) -> float:
        """Run one dummy inference (triggers JIT compilation). Returns milliseconds."""

    @abc.abstractmethod
    def infer(self, items: list[InferItem]) -> list[ChunkOut]:
        """Run inference for a micro-batch. Must return one ChunkOut per item, in order."""


_REGISTRY = {
    "fake_hold": "b1k26.backends.fake:HoldBackend",
    "fake_sine": "b1k26.backends.fake:SineBackend",
    "fake_replay": "b1k26.backends.fake:ReplayBackend",
    "openpi_comet": "b1k26.backends.openpi_comet:CometBackend",
    "openpi_b1k": "b1k26.backends.openpi_b1k:OpenPIB1KBackend",
    "pibehavior": "b1k26.backends.pibehavior:PiBehaviorBackend",
    "gr00t": "b1k26.backends.gr00t:Gr00tBackend",
}


def create_backend(name: str, **kwargs: Any) -> Backend:
    """Instantiate a backend by registry name. Heavy dependencies are imported only here."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(_REGISTRY)}")
    module_name, cls_name = _REGISTRY[name].split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(**kwargs)
