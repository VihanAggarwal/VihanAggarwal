"""Shared helpers for the runtime tests: a front server (and in-process fake workers) on a background event loop,
config documents, and evaluator-shaped observations."""

from __future__ import annotations

import asyncio
import copy
import socket
import threading
import time
from typing import Any, Callable

import numpy as np

from b1k26 import constants as C
from b1k26.backends.base import Backend, create_backend
from b1k26.config import Config, parse_config
from b1k26.server import PolicyServer
from b1k26.worker import WorkerServer

_P = C.PROPRIO_INDICES_2026


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def base_doc(workers: dict[str, dict[str, Any]] | None = None, **profile_overrides: Any) -> dict[str, Any]:
    """A config document with one profile "p" on worker "w" (endpoint filled in by the harness)."""
    profile = {
        "worker": "w",
        "image_size": 32,
        "resize": "bilinear_pad",
        "prompt": "instruction",
        "execution": {"execute_steps": 20, "predicted_steps_to_use": 26, "keep_for_inpaint": 4},
        "corrections": True,
        "use_stage": False,
    }
    profile.update(profile_overrides)
    return {
        "server": {"host": "127.0.0.1", "ports": [0], "health_requires_warm": True},
        "engine": {"plan_timeout_s": 10, "max_batch": 8, "batch_wait_ms": 2},
        "workers": workers if workers is not None else {"w": {"endpoint": "ws://127.0.0.1:1"}},
        "profiles": {"p": profile},
        "routing": {"default": "p", "per_task": {}},
    }


class Harness:
    """Front server + in-process workers on one background event loop.

    ``workers`` maps worker name -> backend factory (or a registry name with kwargs as a tuple); each runs in a
    b1k26.worker.WorkerServer on an ephemeral port and the config's endpoint for that worker is filled in.
    """

    def __init__(self, doc: dict[str, Any], workers: dict[str, Callable[[], Backend] | tuple[str, dict]] | None = None,
                 n_ports: int = 1, start_workers: bool = True):
        self.doc = copy.deepcopy(doc)
        self.worker_factories = workers or {}
        self.n_ports = n_ports
        self.start_workers_now = start_workers
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="harness-loop", daemon=True)
        self.worker_servers: dict[str, WorkerServer] = {}
        self.worker_ports: dict[str, int] = {}
        self.server: PolicyServer | None = None
        self.config: Config | None = None

    # ---- loop helpers ----------------------------------------------------------------------------------------
    def call(self, coro: Any, timeout: float = 30.0) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def run(self, fn: Callable[[], Any], timeout: float = 10.0) -> Any:
        """Run a plain function on the loop thread (to touch loop-owned state safely)."""
        async def wrapper() -> Any:
            return fn()
        return self.call(wrapper(), timeout)

    # ---- lifecycle -----------------------------------------------------------------------------------------
    def _factory(self, spec: Callable[[], Backend] | tuple[str, dict]) -> Callable[[], Backend]:
        if isinstance(spec, tuple):
            name, kwargs = spec
            return lambda: create_backend(name, **kwargs)
        return spec

    def start(self, wait_warm: bool = True, timeout: float = 30.0) -> "Harness":
        self.thread.start()
        for name in self.worker_factories:
            self.worker_ports[name] = free_port()
            entry = dict(self.doc["workers"].get(name, {}))
            entry["endpoint"] = f"ws://127.0.0.1:{self.worker_ports[name]}"
            entry.setdefault("startup_timeout_s", 20)
            self.doc["workers"][name] = entry
        self.doc["server"]["ports"] = [0] * self.n_ports
        self.config = parse_config(self.doc)
        if self.start_workers_now:
            for name in self.worker_factories:
                self.start_worker(name)
        self.server = PolicyServer(self.config)
        self.call(self.server.start())
        if wait_warm:
            self.wait_warm(timeout)
        return self

    def start_worker(self, name: str) -> WorkerServer:
        ws = WorkerServer(self._factory(self.worker_factories[name]), host="127.0.0.1", port=self.worker_ports[name],
                          name=name)
        self.call(ws.start())
        self.call(ws.load())
        self.worker_servers[name] = ws
        return ws

    def wait_warm(self, timeout: float = 30.0) -> None:
        t0 = time.monotonic()
        while not self.run(lambda: self.server.engine.warm):
            if self.server.engine_failed.is_set():
                raise RuntimeError(f"engine failed: {self.server.engine.start_error}")
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("front server not warm")
            time.sleep(0.02)

    @property
    def ports(self) -> list[int]:
        assert self.server is not None
        return list(self.server.bound_ports)

    def backend(self, name: str = "w") -> Any:
        return self.worker_servers[name].backend

    def stop(self) -> None:
        try:
            if self.server is not None:
                self.call(self.server.close(), timeout=30)
            for ws in self.worker_servers.values():
                self.call(ws.stop(), timeout=10)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(10)

    def __enter__(self) -> "Harness":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


# ----------------------------------------------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------------------------------------------
def proprio_at(step: int, env: int = 0) -> np.ndarray:
    """A deterministic, plausible (61,) proprio that changes every step (independent of actions)."""
    p = np.zeros(C.PROPRIO_DIM, dtype=np.float32)
    p[_P["trunk_qpos"]] = [1.025, -1.45, -0.47, 0.0]
    phase = 0.01 * step + 0.3 * env
    p[_P["arm_left_qpos"]] = 0.1 * np.sin(phase + np.arange(7))
    p[_P["arm_right_qpos"]] = 0.1 * np.cos(phase + np.arange(7))
    p[_P["gripper_left_qpos"]] = 0.05
    p[_P["gripper_right_qpos"]] = 0.05
    p[_P["base_qvel"]] = [0.01 * np.sin(phase), 0.0, 0.0]
    return p


def env_obs_dict(step: int, task_id: int, env: int = 0, res: tuple[int, int] = (64, 64), depth: bool = True
                 ) -> dict[str, np.ndarray]:
    """One environment's observation, unbatched (as the evaluator holds it before batching)."""
    h, w = res
    out: dict[str, np.ndarray] = {}
    for i, role in enumerate(("head", "left_wrist", "right_wrist")):
        img = np.full((h, w, 4), 40 + 20 * i, dtype=np.uint8)
        img[..., 3] = 255
        img[(step * 3) % h, :, 0] = (step * 11 + env * 7) % 256
        out[C.rgb_key(role)] = img
        if depth:
            out[C.depth_key(role)] = np.full((h, w), 1.5, dtype=np.float32)
    out[C.PROPRIO_KEY] = proprio_at(step, env)
    out[C.CAM_REL_POSES_KEY] = np.zeros(21, dtype=np.float32)
    out[C.TASK_ID_KEY] = np.array([task_id], dtype=np.int64)
    return out


def batched_obs(step: int, task_ids: list[int], res: tuple[int, int] = (64, 64), depth: bool = True
                ) -> dict[str, np.ndarray]:
    """Batched (N, ...) observation like the single-port evaluator's _batch_obs()."""
    envs = [env_obs_dict(step, t, env=i, res=res, depth=depth) for i, t in enumerate(task_ids)]
    return {k: np.stack([e[k] for e in envs]) for k in envs[0]}


def to_torch(obs: dict[str, np.ndarray]) -> dict[str, Any]:
    import torch

    return {k: torch.from_numpy(np.array(v)) for k, v in obs.items()}
