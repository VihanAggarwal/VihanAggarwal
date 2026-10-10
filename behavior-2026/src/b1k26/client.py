"""Evaluator-faithful probe client: ``b1k26-probe``.

    b1k26-probe --host 127.0.0.1 --port 8000 --steps 200 --batch 1 --chunk 20 --res full --task-id 0

Mimics OmniGibson's ``WebsocketClientPolicy`` (eval/utils/network_utils.py, v3.9.3-post2 and 2026/eval):
polls ``/healthz``, connects with the same websocket options, receives the metadata frame, sends
``{"reset": True}`` (and checks that no reply comes back), then sends batched observations shaped like the
evaluator's (RGBA uint8 cameras at full 720/480 or 224 resolution, float32 ``depth_linear``, (N, 61) proprio,
(N, 21) cam_rel_poses, (N, 1) int64 task_id). With ``--chunk K > 1`` it adds ``__action_chunk_size__`` and
validates the reply exactly like ``WebsocketClientPolicy.act`` (``action_chunk`` shape ``(*action.shape[:-1], K,
23)`` and ``action_chunk[..., 0, :] == action`` bit for bit), then replays the chunk locally. Every action is also
checked for finiteness, limits and the "zero torso" mistake. A tiny kinematic model feeds the commanded joint
positions back into the next proprio so the server sees a moving robot.

Prints request latency percentiles and exits 0 when the server followed the protocol, 1 on any violation and 2 if
it could not connect.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from b1k26 import constants as C
from b1k26.engine import no_proxy_kwargs
from b1k26.protocol import Packer, unpackb

RESOLUTIONS = {"full": C.FULL_RES, "224": C.DEFAULT_WRAPPER_RES}
RESET_TRUNK = np.array([1.025, -1.45, -0.47, 0.0], dtype=np.float32)  # R1Pro reset pose (eval robot config)
_P = C.PROPRIO_INDICES_2026


class ProtocolViolation(RuntimeError):
    pass


@dataclass
class ProbeResult:
    steps: int = 0
    requests: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    metadata: Any = None
    health_wait_s: float = 0.0
    wall_s: float = 0.0
    actions: list[np.ndarray] = field(default_factory=list)  # executed (B, 23) actions, when recorded

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> dict[str, Any]:
        lat = np.asarray(self.latencies_ms, dtype=np.float64)

        def pct(q: float) -> float:
            return round(float(np.percentile(lat, q)), 2) if lat.size else 0.0

        return {
            "ok": self.ok,
            "steps": self.steps,
            "requests": self.requests,
            "latency_ms": {"mean": round(float(lat.mean()), 2) if lat.size else 0.0, "p50": pct(50), "p90": pct(90),
                           "p99": pct(99), "max": round(float(lat.max()), 2) if lat.size else 0.0},
            "steps_per_s": round(self.steps / self.wall_s, 2) if self.wall_s > 0 else 0.0,
            "health_wait_s": round(self.health_wait_s, 2),
            "metadata": self.metadata,
            "violations": self.violations[:20],
            "num_violations": len(self.violations),
        }


# ----------------------------------------------------------------------------------------------------------
# Synthetic robot / observations
# ----------------------------------------------------------------------------------------------------------
class FakeRobot:
    """Minimal kinematic stand-in for the R1Pro: joints follow the commanded positions with a first-order lag."""

    def __init__(self, n: int, seed: int = 0):
        self.n = n
        rng = np.random.default_rng(seed)
        self.proprio = np.zeros((n, C.PROPRIO_DIM), dtype=np.float32)
        self.proprio[:, _P["trunk_qpos"]] = RESET_TRUNK
        self.proprio[:, _P["arm_left_qpos"]] = rng.normal(0, 0.05, (n, 7))
        self.proprio[:, _P["arm_right_qpos"]] = rng.normal(0, 0.05, (n, 7))
        self.proprio[:, _P["gripper_left_qpos"]] = 0.05  # fully open: finger sum 0.1
        self.proprio[:, _P["gripper_right_qpos"]] = 0.05
        self.proprio[:, _P["eef_left_quat"]] = [0, 0, 0, 1]
        self.proprio[:, _P["eef_right_quat"]] = [0, 0, 0, 1]

    def apply(self, actions: np.ndarray, alpha: float = 0.5) -> None:
        a = np.asarray(actions, dtype=np.float32).reshape(self.n, C.ACTION_DIM)
        p = self.proprio
        for key, sl in (("trunk_qpos", C.ACTION_SLICES["torso"]), ("arm_left_qpos", C.ACTION_SLICES["left_arm"]),
                        ("arm_right_qpos", C.ACTION_SLICES["right_arm"])):
            q = p[:, _P[key]]
            p[:, _P[key]] = q + alpha * (a[:, sl] - q)
        p[:, _P["base_qvel"]] = a[:, C.ACTION_SLICES["base"]] * np.array([0.75, 0.75, 1.0], np.float32)
        for side, idx in (("left", C.LEFT_GRIPPER_ACTION_IDX), ("right", C.RIGHT_GRIPPER_ACTION_IDX)):
            width = (np.clip(a[:, idx], -1, 1) + 1.0) / 2.0 * C.GRIPPER_MAX_WIDTH
            p[:, _P[f"gripper_{side}_qpos"]] = (width / 2.0)[:, None]


class ObsFactory:
    """Evaluator-shaped observations. Images are fixed noise with a per-step stripe (cheap, but never repeat)."""

    def __init__(self, n: int, res: str = "full", depth: bool = True, seed: int = 0):
        if res not in RESOLUTIONS:
            raise ValueError(f"res must be one of {sorted(RESOLUTIONS)}")
        rng = np.random.default_rng(seed)
        self.n = n
        self.depth = depth
        self.base: dict[str, np.ndarray] = {}
        for role, (h, w) in RESOLUTIONS[res].items():
            img = rng.integers(0, 256, (n, h, w, 4), dtype=np.uint8)
            img[..., 3] = 255
            self.base[role] = img
        self.depth_base = {role: rng.uniform(0.2, 5.0, (n, h, w)).astype(np.float32)
                           for role, (h, w) in RESOLUTIONS[res].items()}

    def make(self, step: int, proprio: np.ndarray, task_ids: list[int], unbatched: bool = False) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        for role, base in self.base.items():
            img = base.copy()
            row = (step * 7) % img.shape[1]
            img[:, row, :, :3] = (step * 13) % 256
            obs[C.rgb_key(role)] = img
            if self.depth:
                obs[C.depth_key(role)] = self.depth_base[role]
        obs[C.PROPRIO_KEY] = np.ascontiguousarray(proprio, dtype=np.float32)
        obs[C.CAM_REL_POSES_KEY] = np.zeros((self.n, 21), dtype=np.float32)
        obs[C.TASK_ID_KEY] = np.asarray(task_ids, dtype=np.int64).reshape(self.n, 1)
        if unbatched:
            if self.n != 1:
                raise ValueError("unbatched observations need batch 1")
            obs = {k: v[0] for k, v in obs.items()}
        return obs


# ----------------------------------------------------------------------------------------------------------
# Protocol checks
# ----------------------------------------------------------------------------------------------------------
def check_reply(raw: Any, batch: int, chunk: int, unbatched: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate one reply like WebsocketClientPolicy.act; returns (action, chunk or None) as float32 arrays."""
    if isinstance(raw, str):
        raise ProtocolViolation(f"server sent a text frame: {raw[:200]!r}")
    reply = unpackb(raw)
    if not isinstance(reply, dict):
        raise ProtocolViolation(f"reply is not a map: {type(reply).__name__}")
    if "action" not in reply:
        raise ProtocolViolation(f"reply has no 'action' key (keys: {sorted(map(str, reply))})")
    action = np.array(reply["action"], dtype=np.float32, copy=True)
    allowed = {(batch, C.ACTION_DIM)} | ({(C.ACTION_DIM,)} if batch == 1 else set())
    if action.shape not in allowed:
        raise ProtocolViolation(f"action shape {action.shape}, expected {(batch, C.ACTION_DIM)}")
    act_chunk = None
    if chunk > 1:
        if "action_chunk" not in reply:
            raise ProtocolViolation("chunk requested but the reply has no 'action_chunk' (client would fall back)")
        act_chunk = np.array(reply["action_chunk"], dtype=np.float32, copy=True)
        expected = (*action.shape[:-1], chunk, action.shape[-1])
        if act_chunk.shape != expected:
            raise ProtocolViolation(f"action_chunk shape {act_chunk.shape}, expected {expected}")
        if not np.array_equal(act_chunk[..., 0, :], action):
            raise ProtocolViolation("action must exactly equal action_chunk[..., 0, :]")
    rows = (act_chunk if act_chunk is not None else action).reshape(-1, C.ACTION_DIM)
    if not np.all(np.isfinite(rows)):
        raise ProtocolViolation("non-finite action values")
    if np.any(np.abs(rows[:, C.ACTION_SLICES["base"]]) > C.BASE_ACTION_LIMIT + 1e-6):
        raise ProtocolViolation("base velocity outside [-1, 1]")
    grips = rows[:, [C.LEFT_GRIPPER_ACTION_IDX, C.RIGHT_GRIPPER_ACTION_IDX]]
    if np.any(np.abs(grips) > 1.0 + 1e-6):
        raise ProtocolViolation("gripper command outside [-1, 1]")
    if np.any(np.all(rows[:, C.ACTION_SLICES["torso"]] == 0.0, axis=1)):
        raise ProtocolViolation("all-zero torso command (a zero action stands the robot upright)")
    return action, act_chunk


def _keepalive_kwargs(connect_fn: Any) -> dict[str, Any]:
    """The 2026/eval client's keepalive (ping every 60 s, never time out on pongs) where the installed websockets
    sync client supports it (>= 15); older sync clients send no pings at all."""
    import inspect

    try:
        params = inspect.signature(connect_fn).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return {}
    return {"ping_interval": 60, "ping_timeout": None} if "ping_interval" in params else {}


def wait_healthy(host: str, port: int, timeout_s: float, poll_s: float = 1.0) -> float:
    """Poll http://host:port/healthz until 200; returns the seconds waited. Raises TimeoutError."""
    t0 = time.monotonic()
    url = f"http://{host}:{port}/healthz"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via an HTTP(S)_PROXY
    while True:
        try:
            with opener.open(url, timeout=2) as r:
                if r.status == 200:
                    return time.monotonic() - t0
        except (urllib.error.URLError, OSError, ValueError):
            pass
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(f"{url} not healthy after {timeout_s:.0f} s")
        time.sleep(poll_s)


def run_probe(
    host: str = "127.0.0.1",
    port: int = 8000,
    steps: int = 200,
    batch: int = 1,
    chunk: int = 0,
    res: str = "full",
    task_ids: list[int] | None = None,
    depth: bool = True,
    unbatched: bool = False,
    health_timeout_s: float = 600.0,
    reset_check_s: float = 0.3,
    recv_timeout_s: float = 600.0,
    record_actions: bool = False,
    seed: int = 0,
    log: Any = None,
    metadata_timeout_s: float = 10.0,
) -> ProbeResult:
    """Drive one rollout of ``steps`` evaluator steps. Protocol violations are collected in the result (the run
    stops at the first one); connection failures raise ConnectionError."""
    from websockets.sync.client import connect

    task_ids = list(task_ids if task_ids is not None else [0])
    if len(task_ids) == 1:
        task_ids = task_ids * batch
    if len(task_ids) != batch:
        raise ValueError(f"{len(task_ids)} task ids for batch {batch}")
    if unbatched and batch != 1:
        raise ValueError("--unbatched needs --batch 1")
    result = ProbeResult()
    try:
        result.health_wait_s = wait_healthy(host, port, health_timeout_s)
    except TimeoutError as e:
        raise ConnectionError(str(e)) from None
    packer = Packer()
    robot = FakeRobot(batch, seed=seed)
    factory = ObsFactory(batch, res=res, depth=depth, seed=seed)
    t_start = time.monotonic()
    with contextlib.ExitStack() as stack:
        try:
            ws = stack.enter_context(connect(f"ws://{host}:{port}", compression=None, max_size=None, open_timeout=10,
                                             **_keepalive_kwargs(connect), **no_proxy_kwargs(connect)))
        except Exception as e:
            raise ConnectionError(f"websocket connect failed: {e}") from None
        _drive(ws, result, packer, robot, factory, steps, batch, chunk, task_ids, unbatched, reset_check_s,
               recv_timeout_s, record_actions, log, metadata_timeout_s)
    result.wall_s = time.monotonic() - t_start
    return result


def _drive(ws: Any, result: ProbeResult, packer: Any, robot: FakeRobot, factory: ObsFactory, steps: int, batch: int,
           chunk: int, task_ids: list[int], unbatched: bool, reset_check_s: float, recv_timeout_s: float,
           record_actions: bool, log: Any, metadata_timeout_s: float) -> None:
    """The rollout loop of run_probe; records violations in ``result`` and returns at the first one."""
    try:
        try:
            meta_raw = ws.recv(timeout=metadata_timeout_s)
        except TimeoutError:
            result.violations.append(f"no metadata frame within {metadata_timeout_s:g} s of connecting "
                                     "(the 2026/eval client waits at most 10 s)")
            return
        if isinstance(meta_raw, str):
            result.violations.append("metadata frame is a text frame")
            return
        result.metadata = unpackb(meta_raw)
        if not isinstance(result.metadata, dict):
            result.violations.append(f"metadata is not a map: {result.metadata!r}")
            return

        ws.send(packer.pack({"reset": True}))
        if reset_check_s > 0:
            try:
                extra = ws.recv(timeout=reset_check_s)
                result.violations.append(f"server replied to reset ({len(extra)} bytes); replies would desync")
                return
            except TimeoutError:
                pass

        step = 0
        while step < steps:
            obs = factory.make(step, robot.proprio, task_ids, unbatched=unbatched)
            request = obs
            if chunk > 1:
                request = dict(obs)
                request[C.ACTION_CHUNK_REQUEST_KEY] = chunk
            data = packer.pack(request)
            t0 = time.monotonic()
            ws.send(data)
            try:
                raw = ws.recv(timeout=recv_timeout_s)
            except TimeoutError:
                result.violations.append(f"no reply within {recv_timeout_s:.0f} s at step {step}")
                return
            result.latencies_ms.append((time.monotonic() - t0) * 1e3)
            result.requests += 1
            try:
                action, act_chunk = check_reply(raw, batch, chunk, unbatched)
            except ProtocolViolation as e:
                result.violations.append(f"step {step}: {e}")
                return
            seq = [action] if act_chunk is None else [act_chunk[..., i, :] for i in range(chunk)]
            for a in seq:
                if step >= steps:
                    break
                robot.apply(a)
                if record_actions:
                    result.actions.append(np.array(a, copy=True).reshape(batch, C.ACTION_DIM))
                step += 1
            result.steps = step
            if log is not None and result.requests % 50 == 0:
                log(f"step {step}/{steps}: last request {result.latencies_ms[-1]:.1f} ms")
    except Exception as e:
        from websockets.exceptions import ConnectionClosed

        if isinstance(e, ConnectionClosed):
            result.violations.append(f"server closed the connection: {e}")
        else:
            raise


def _parse_task_ids(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
        else:
            out.append(C.task(part).task_id)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="b1k26-probe", description="Evaluator-faithful protocol probe for b1k26-serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--chunk", type=int, default=0, help="__action_chunk_size__ (<= 1: one request per step)")
    p.add_argument("--res", choices=sorted(RESOLUTIONS), default="full")
    p.add_argument("--task-id", default="0", help="task id or name; comma list for one per batch entry")
    p.add_argument("--no-depth", action="store_true", help="do not send depth_linear images")
    p.add_argument("--unbatched", action="store_true", help="v3.9.2-style unbatched observations (batch 1)")
    p.add_argument("--health-timeout", type=float, default=600.0)
    p.add_argument("--json", action="store_true", help="print the summary as JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        task_ids = _parse_task_ids(args.task_id)
        res = run_probe(args.host, args.port, steps=args.steps, batch=args.batch, chunk=args.chunk, res=args.res,
                        task_ids=task_ids, depth=not args.no_depth, unbatched=args.unbatched,
                        health_timeout_s=args.health_timeout,
                        log=lambda m: print(f"[probe] {m}", file=sys.stderr))
    except ConnectionError as e:
        print(f"b1k26-probe: {e}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as e:
        print(f"b1k26-probe: {e}", file=sys.stderr)
        return 2
    summary = res.summary()
    if args.json:
        print(json.dumps(summary, default=str))
    else:
        lat = summary["latency_ms"]
        print(f"steps={summary['steps']} requests={summary['requests']} steps/s={summary['steps_per_s']} "
              f"latency ms: mean={lat['mean']} p50={lat['p50']} p90={lat['p90']} p99={lat['p99']} max={lat['max']}")
        print(f"metadata={summary['metadata']}")
        for v in res.violations:
            print(f"VIOLATION: {v}")
        print("PASS" if res.ok else "FAIL")
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
