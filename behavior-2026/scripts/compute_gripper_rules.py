#!/usr/bin/env python
"""Derive per-task gripper-closure rules for b1k26.corrections from the BEHAVIOR-1K 2026 human demos.

Two phases, both resumable:

1. ``stats`` streams the LeRobot v3 data shards of one task at a time from the Hugging Face dataset
   ``behavior-1k/2026-challenge-demos`` (pinned revision) and reduces them to per-episode gripper statistics.
   Only the ``observation.state``, ``episode_index`` and ``frame_index`` column chunks are fetched, with HTTP
   range requests, into a sparse temporary file (parquet column projection over the network). The file is
   deleted right after it is read, so disk use stays at roughly one column chunk (< 0.6 GB). Per-task results
   are written atomically to ``<stats-dir>/task_TTT.json``; tasks that already have a result are skipped.

2. ``rules`` turns the cached statistics into ``src/b1k26/data/gripper_rules.json``. Tasks 0-49 take RLC's
   2025 hand tables verbatim (the 2026 demos for tasks 0-49 are the 2025 demos). Tasks 50-99 take rules derived
   from the statistics; a task without statistics gets no rule (absent == never corrected).

Definitions (per task, per side):
- A frame is *fully closed* when the two finger joint positions sum to less than ``--closed-sum-m`` (0.001 m),
  i.e. normalized gripper width ``2 * sum / 0.1 - 1 < -0.98`` (b1k26 ``closed_threshold``). Finger columns of
  observation.state: left 24:26, right 49:51.
- An episode *closes* when it has a fully-closed run of at least ``--min-run-frames`` consecutive frames
  (default 3, i.e. 0.1 s at 30 Hz; filters single-frame contact glitches). Its first-closure frame ``f`` is the
  start of the first such run.
- ``always_open`` when at most ``--always-open-max-frac`` of the episodes close (default 0.03 = 6 of 200
  episodes). The tolerance absorbs rare operator mistakes. It is calibrated on RLC's 2025 hand tables, which mark
  task 10 always-open on both sides although 4 and 6 of its 200 demos close fully (``--compare-rlc``).
  Exception, also from RLC's tables (tasks 41 and 46, right gripper): when those rare closures all happen late
  (first-closure progress >= ``--late-progress``, default 0.9), the side gets a ``min_progress`` gate from the
  earliest closure instead of ``always_open``.
- Otherwise ``min_progress`` = ``--progress-safety`` x the ``--percentile`` (default 1st) percentile of the
  per-episode first-closure progress, where progress is ``min(f / episode_length, f / human_mean_len)``. The
  runtime compares it with ``step / human_mean_len`` (see b1k26.corrections), so ``f / human_mean_len`` is the
  matching scale and the minimum with ``f / length`` keeps short demos from raising the threshold. The safety
  factor (default 0.6) covers a policy that reaches the first closure faster than the fastest demos, e.g. with
  26 -> 20 action compression (1.3x faster execution). A ``min_progress`` below ``--min-useful-progress``
  (default 0.05) is dropped (null), because it would almost never fire.
- Tasks 0-49 keep RLC's tables verbatim. Where RLC has a ``min_stage`` rule, the demo-derived ``min_progress``
  is added as well (unless ``--no-rlc-progress``); b1k26.corrections uses it only when no stage is known,
  so stage-tracking models behave exactly like RLC and stage-less models get a conservative fallback.

Usage:
    python scripts/compute_gripper_rules.py stats --tasks 50-99 --stats-dir /tmp/gripper_stats
    python scripts/compute_gripper_rules.py stats --tasks 0,1,30,38,46 --stats-dir /tmp/gripper_stats
    python scripts/compute_gripper_rules.py rules --stats-dir /tmp/gripper_stats --compare-rlc
    python scripts/compute_gripper_rules.py rules --stats-dir /tmp/gripper_stats --out src/b1k26/data/gripper_rules.json

Requirements: numpy, requests, pyarrow (pyarrow only for the ``stats`` phase).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as _dt
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ID = "behavior-1k/2026-challenge-demos"
# Commit of the published dataset (main == tag v3.0 on 2026-10-10). Pinned so the rules are reproducible.
REVISION = "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
COLUMNS = ("observation.state", "episode_index", "frame_index")
STATE_DIM = 61
FINGER_COLUMNS = {"left": (24, 26), "right": (49, 51)}  # PROPRIO_INDICES_2026 gripper_{left,right}_qpos
GRIPPER_MAX_WIDTH = 0.1
DEFAULT_CLOSED_SUM_M = 0.001  # == normalized width -0.98
MAX_RUNS_STORED = 64  # closed runs kept per episode and side in the stats cache
PIECE_BYTES = 8 << 20
STATS_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES_PATH = REPO_ROOT / "src" / "b1k26" / "data" / "gripper_rules.json"
TASKS_JSON = REPO_ROOT / "src" / "b1k26" / "data" / "tasks.json"

# --------------------------------------------------------------------------------------------------------------
# RLC 2025 tables, verbatim from shared/correction_rules.py of the 2025 1st-place solution (tasks 0-49).
# --------------------------------------------------------------------------------------------------------------
RLC_ALWAYS_OPEN_LEFT = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 47, 48}
RLC_ALWAYS_OPEN_RIGHT = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                         25, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 47, 48, 49}
RLC_MIN_STAGE_FOR_CLOSURE = {
    0: {"left": 2, "right": 2},
    30: {"right": 6},
    31: {"right": 8},
    32: {"right": 5},
    33: {"right": 11},
    40: {"right": 4},
    41: {"left": 14, "right": 14},
    45: {"right": 10},
    46: {"left": 8, "right": 8},
    49: {"left": 14},
}
RLC_RIGHT_GRIPPER_ALWAYS_ALLOWED = {38, 39}


def log(msg: str) -> None:
    print(f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_tasks(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [t for t in out if not 0 <= t < 100]
    if bad:
        raise SystemExit(f"task ids out of range: {bad}")
    return sorted(dict.fromkeys(out))


def load_task_table() -> list[dict[str, Any]]:
    return json.loads(TASKS_JSON.read_text())["tasks"]


# --------------------------------------------------------------------------------------------------------------
# Phase 1: streaming statistics
# --------------------------------------------------------------------------------------------------------------
class RemoteParquet:
    """Fetches selected column chunks of a remote parquet file into a sparse local file."""

    def __init__(self, session: Any, url: str, expected_size: int | None, workers: int, retries: int = 6):
        self.session = session
        self.workers = workers
        self.retries = retries
        resp = self._request("HEAD", url, allow_redirects=True)
        self.url = resp.url  # signed CDN URL after redirects; valid long enough for one file
        size = resp.headers.get("x-linked-size") or resp.headers.get("content-length")
        if size is None:
            raise RuntimeError(f"no size for {url}")
        self.size = int(size)
        if expected_size is not None and self.size != expected_size:
            raise RuntimeError(f"size mismatch for {url}: HEAD {self.size} vs tree {expected_size}")

    def _request(self, method: str, url: str, **kw: Any) -> Any:
        delay = 1.0
        for attempt in range(self.retries):
            try:
                resp = self.session.request(method, url, timeout=(30, 300), **kw)
                if resp.status_code in (200, 206):
                    return resp
                err: Exception = RuntimeError(f"HTTP {resp.status_code} for {method} {url[:100]}")
            except Exception as e:  # network errors: retry
                err = e
            if attempt + 1 < self.retries:
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise RuntimeError(f"{method} failed after {self.retries} attempts: {err}")

    def get_range(self, start: int, end_incl: int) -> bytes:
        for attempt in range(self.retries):
            resp = self._request("GET", self.url, headers={"Range": f"bytes={start}-{end_incl}"})
            data = resp.content
            if resp.status_code == 206 and len(data) == end_incl - start + 1:
                return data
            log(f"  short/whole read ({resp.status_code}, {len(data)} bytes), retry {attempt + 1}")
            time.sleep(1 + attempt)
        raise RuntimeError(f"range {start}-{end_incl} failed")

    def fetch_columns(self, path: Path, columns: tuple[str, ...]) -> int:
        """Materialize footer + the requested column chunks in a sparse file at ``path``. Returns bytes fetched."""
        import pyarrow.parquet as pq

        tail_len = min(self.size, 1 << 20)
        tail = self.get_range(self.size - tail_len, self.size - 1)
        if tail[-4:] != b"PAR1":
            raise RuntimeError("not a parquet file (bad magic)")
        footer_len = int.from_bytes(tail[-8:-4], "little")
        if footer_len + 8 > tail_len:
            tail_len = footer_len + 8
            tail = self.get_range(self.size - tail_len, self.size - 1)
        fetched = len(tail)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.ftruncate(fd, self.size)
            os.pwrite(fd, tail, self.size - tail_len)
            os.pwrite(fd, b"PAR1", 0)
            md = pq.ParquetFile(str(path)).metadata
            ranges: list[tuple[int, int]] = []
            for rg_i in range(md.num_row_groups):
                rg = md.row_group(rg_i)
                for c_i in range(rg.num_columns):
                    col = rg.column(c_i)
                    name = col.path_in_schema
                    if not any(name == c or name.startswith(c + ".") for c in columns):
                        continue
                    start = col.data_page_offset
                    if col.has_dictionary_page and col.dictionary_page_offset is not None:
                        start = min(start, col.dictionary_page_offset)
                    ranges.append((start, start + col.total_compressed_size))
            pieces = [(a, min(a + PIECE_BYTES, b)) for a, b in ranges for a in range(a, b, PIECE_BYTES)]

            def work(piece: tuple[int, int]) -> int:
                data = self.get_range(piece[0], piece[1] - 1)
                os.pwrite(fd, data, piece[0])
                return len(data)

            with cf.ThreadPoolExecutor(self.workers) as ex:
                fetched += sum(ex.map(work, pieces))
        finally:
            os.close(fd)
        return fetched


def list_task_files(session: Any, task_id: int, revision: str) -> list[dict[str, Any]]:
    url = f"{HF_ENDPOINT}/api/datasets/{REPO_ID}/tree/{revision}/data/chunk-{task_id:03d}"
    files: list[dict[str, Any]] = []
    while url:
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        files.extend(e for e in resp.json() if e.get("type") == "file" and e["path"].endswith(".parquet"))
        url = resp.links.get("next", {}).get("url")
    files.sort(key=lambda e: e["path"])
    if not files:
        raise RuntimeError(f"no parquet files under data/chunk-{task_id:03d}")
    return files


def finger_sums(table: Any) -> dict[str, np.ndarray]:
    """Reduce a pyarrow table to per-frame episode/frame indices and left/right finger-position sums."""
    import pyarrow as pa

    state = table.column("observation.state")
    if isinstance(state, pa.ChunkedArray):
        state = state.combine_chunks()
    offsets = state.offsets.to_numpy()
    lengths = np.diff(offsets)
    if lengths.size and not np.all(lengths == STATE_DIM):
        raise RuntimeError(f"observation.state rows are not {STATE_DIM}-D: {np.unique(lengths)[:5]}")
    flat = state.values.to_numpy(zero_copy_only=False)
    flat = flat[offsets[0]: offsets[-1]] if lengths.size else flat[:0]
    vals = flat.reshape(-1, STATE_DIM)
    out = {
        "episode": table.column("episode_index").to_numpy().astype(np.int64),
        "frame": table.column("frame_index").to_numpy().astype(np.int64),
    }
    for side, (a, b) in FINGER_COLUMNS.items():
        out[side] = vals[:, a:b].astype(np.float64).sum(axis=1)
    return out


def closed_runs(closed: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of every run of True in a 1-D bool array."""
    if closed.size == 0:
        return []
    d = np.diff(np.concatenate([[0], closed.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def episode_stats(per_frame: dict[str, np.ndarray], closed_sum_m: float) -> list[dict[str, Any]]:
    order = np.lexsort((per_frame["frame"], per_frame["episode"]))
    ep = per_frame["episode"][order]
    frame = per_frame["frame"][order]
    sums = {s: per_frame[s][order] for s in FINGER_COLUMNS}
    bounds = np.flatnonzero(np.diff(ep)) + 1
    out = []
    for lo, hi in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [ep.size]])):
        fr = frame[lo:hi]
        rec: dict[str, Any] = {
            "episode_index": int(ep[lo]),
            "length": int(hi - lo),
            "first_frame_index": int(fr[0]),
            "max_frame_index": int(fr[-1]),
            "contiguous": bool(np.all(np.diff(fr) == 1)),
        }
        for side in FINGER_COLUMNS:
            s = sums[side][lo:hi]
            closed = s < closed_sum_m
            # Runs are in row positions of the sorted episode; with contiguous frame indices these equal frames
            # relative to the first frame (0 for every published episode).
            runs = closed_runs(closed)
            rec[side] = {
                "min_sum": float(s.min()),
                "closed_frames": int(closed.sum()),
                "num_runs": len(runs),
                "runs": runs[:MAX_RUNS_STORED],
            }
        out.append(rec)
    return out


def compute_task_stats(session: Any, task_id: int, args: argparse.Namespace, tmp_dir: Path) -> dict[str, Any]:
    files = list_task_files(session, task_id, args.revision)
    parts: list[dict[str, np.ndarray]] = []
    file_log = []
    t0 = time.time()
    total_bytes = 0
    for entry in files:
        url = f"{HF_ENDPOINT}/datasets/{REPO_ID}/resolve/{args.revision}/{entry['path']}"
        rp = RemoteParquet(session, url, entry.get("size"), workers=args.workers)
        free = shutil.disk_usage(tmp_dir).free
        # Worst case the sparse file holds every byte of the shard.
        if free - rp.size < args.min_free_gb * 1e9:
            raise RuntimeError(
                f"refusing to fetch {entry['path']} ({rp.size / 1e9:.2f} GB): free space {free / 1e9:.1f} GB "
                f"would drop below --min-free-gb {args.min_free_gb}"
            )
        sparse = tmp_dir / f"task{task_id:03d}_{Path(entry['path']).name}"
        try:
            import pyarrow.parquet as pq

            nbytes = rp.fetch_columns(sparse, COLUMNS)
            table = pq.read_table(str(sparse), columns=list(COLUMNS), use_threads=True)
            parts.append(finger_sums(table))
            rows = table.num_rows
            del table
        finally:
            sparse.unlink(missing_ok=True)
        total_bytes += nbytes
        file_log.append({"path": entry["path"], "size": rp.size, "rows": rows, "fetched_bytes": nbytes})
        log(f"  task {task_id:3d} {entry['path']}: {rows} rows, {nbytes / 1e6:.0f} MB fetched")
    per_frame = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    episodes = episode_stats(per_frame, args.closed_sum_m)
    dt = time.time() - t0
    log(f"task {task_id:3d}: {len(episodes)} episodes, {per_frame['frame'].size} frames, "
        f"{total_bytes / 1e6:.0f} MB in {dt:.0f} s ({total_bytes / 1e6 / max(dt, 1e-6):.0f} MB/s)")
    return {
        "stats_version": STATS_VERSION,
        "task_id": task_id,
        "repo_id": REPO_ID,
        "revision": args.revision,
        "closed_sum_m": args.closed_sum_m,
        "finger_columns": {k: list(v) for k, v in FINGER_COLUMNS.items()},
        "n_frames": int(per_frame["frame"].size),
        "files": file_log,
        "episodes": episodes,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def stats_path(stats_dir: Path, task_id: int) -> Path:
    return stats_dir / f"task_{task_id:03d}.json"


def load_stats(stats_dir: Path, task_id: int, closed_sum_m: float | None = None) -> dict[str, Any] | None:
    p = stats_path(stats_dir, task_id)
    if not p.exists():
        return None
    try:
        st = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if st.get("stats_version") != STATS_VERSION:
        return None
    if closed_sum_m is not None and abs(st.get("closed_sum_m", -1) - closed_sum_m) > 1e-12:
        return None
    return st


def cmd_stats(args: argparse.Namespace) -> int:
    import requests

    stats_dir = Path(args.stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(args.tmp_dir) if args.tmp_dir else stats_dir
    tmp_root.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.max_minutes * 60 if args.max_minutes else None
    session = requests.Session()
    done, failed, skipped = [], [], []
    for task_id in parse_tasks(args.tasks):
        if load_stats(stats_dir, task_id, args.closed_sum_m) is not None:
            skipped.append(task_id)
            continue
        if deadline is not None and time.time() > deadline:
            log(f"time budget exhausted before task {task_id}; stopping (rerun to resume)")
            break
        with tempfile.TemporaryDirectory(dir=tmp_root, prefix="gripper_rules_") as td:
            try:
                st = compute_task_stats(session, task_id, args, Path(td))
            except Exception as e:  # keep going: a failed task simply gets no rule
                log(f"task {task_id}: FAILED: {e!r}")
                failed.append(task_id)
                if "refusing to fetch" in str(e):
                    break
                continue
        out = stats_path(stats_dir, task_id)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(st, separators=(",", ":")))
        os.replace(tmp, out)
        done.append(task_id)
    log(f"done: {done}; skipped (cached): {skipped}; failed: {failed}")
    return 1 if failed else 0


# --------------------------------------------------------------------------------------------------------------
# Phase 2: rules
# --------------------------------------------------------------------------------------------------------------
def side_summary(st: dict[str, Any], side: str, human_mean_len: float, min_run_frames: int) -> dict[str, Any]:
    """Closure statistics of one side over all episodes of a task."""
    n = len(st["episodes"])
    first_prog: list[float] = []
    any_closed = 0
    for ep in st["episodes"]:
        rec = ep[side]
        if rec["closed_frames"] > 0:
            any_closed += 1
        first = next((s for s, length in rec["runs"] if length >= min_run_frames), None)
        if first is None and rec["num_runs"] > len(rec["runs"]):
            # More runs than stored: fall back to the first stored run (conservative: earlier closure).
            first = rec["runs"][0][0] if rec["runs"] else None
        if first is not None:
            first_prog.append(min(first / max(ep["length"], 1), first / human_mean_len))
    out: dict[str, Any] = {
        "n_episodes": n,
        "n_closing": len(first_prog),
        "n_any_closed_frame": any_closed,
        "close_frac": round(len(first_prog) / n, 4) if n else 0.0,
    }
    if first_prog:
        arr = np.asarray(first_prog)
        for q in (0, 1, 5, 25, 50):
            out[f"first_close_progress_p{q:02d}"] = round(float(np.percentile(arr, q)), 4)
    return out


def derive_side_rule(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if summary["n_episodes"] == 0:
        return {"always_open": False, "min_stage": None, "min_progress": None}
    if summary["close_frac"] <= args.always_open_max_frac:
        earliest = summary.get("first_close_progress_p00")
        if earliest is not None and earliest >= args.late_progress:
            # Rare but only late closures: gate by the earliest one (RLC used late min_stage gates here).
            return {"always_open": False, "min_stage": None,
                    "min_progress": round(args.progress_safety * earliest, 4)}
        return {"always_open": True, "min_stage": None, "min_progress": None}
    arr_key = f"first_close_progress_p{args.percentile:02d}"
    if arr_key in summary:
        p = summary[arr_key]
    else:  # percentile not precomputed: recompute is not possible here, fall back to the minimum
        p = summary["first_close_progress_p00"]
    mp = round(args.progress_safety * p, 4)
    if mp < args.min_useful_progress:
        mp = None
    return {"always_open": False, "min_stage": None, "min_progress": mp}


def rlc_side_rule(task_id: int, side: str) -> dict[str, Any] | None:
    always = RLC_ALWAYS_OPEN_LEFT if side == "left" else RLC_ALWAYS_OPEN_RIGHT
    min_stage = RLC_MIN_STAGE_FOR_CLOSURE.get(task_id, {}).get(side)
    if task_id in always:
        return {"always_open": True, "min_stage": None, "min_progress": None}
    if min_stage is not None:
        return {"always_open": False, "min_stage": int(min_stage), "min_progress": None}
    return None


def build_rules(args: argparse.Namespace) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    stats_dir = Path(args.stats_dir) if args.stats_dir else None
    tasks = load_task_table()
    summaries: dict[int, dict[str, Any]] = {}
    if stats_dir is not None:
        for t in tasks:
            st = load_stats(stats_dir, t["task_id"])
            if st is None:
                continue
            summaries[t["task_id"]] = {
                side: side_summary(st, side, t["human_mean_len"], args.min_run_frames) for side in FINGER_COLUMNS
            }
    rules: dict[str, Any] = {}
    for t in tasks:
        tid = t["task_id"]
        entry: dict[str, Any] = {"name": t["name"]}
        if tid < 50:
            entry["source"] = "rlc2025"
            for side in ("left", "right"):
                r = rlc_side_rule(tid, side)
                if r is not None and r["min_stage"] is not None and tid in summaries and not args.no_rlc_progress:
                    # Stage-less fallback for RLC's stage rules (used by b1k26 only when no stage is known).
                    r["min_progress"] = derive_side_rule(summaries[tid][side], args)["min_progress"]
                if r is not None:
                    entry[side] = r
            entry["exempt_right"] = tid in RLC_RIGHT_GRIPPER_ALWAYS_ALLOWED
            if tid == 0:
                entry["rlc_task0_rule"] = True  # RLC task0_stage4_reset_to_stage2 (stage-tracking models only)
        elif tid in summaries:
            entry["source"] = "demos2026"
            for side in ("left", "right"):
                entry[side] = derive_side_rule(summaries[tid][side], args)
            entry["exempt_right"] = False
        else:
            continue  # no statistics: no rule (never corrected)
        if tid in summaries:
            entry["demo_stats"] = summaries[tid]
        rules[str(tid)] = entry
    doc = {
        "schema": "b1k26 gripper rules v1 (see b1k26/corrections.py and scripts/compute_gripper_rules.py)",
        "closed_threshold": round(2.0 * args.closed_sum_m / GRIPPER_MAX_WIDTH - 1.0, 6),
        "params": {
            "dataset": f"{REPO_ID}@{args.revision}",
            "closed_sum_m": args.closed_sum_m,
            "min_run_frames": args.min_run_frames,
            "always_open_max_frac": args.always_open_max_frac,
            "late_progress": args.late_progress,
            "percentile": args.percentile,
            "progress_safety": args.progress_safety,
            "min_useful_progress": args.min_useful_progress,
            "rlc_progress_fallback": not args.no_rlc_progress,
            "progress_definition": "min(first_close_frame / episode_length, first_close_frame / human_mean_len); "
                                   "runtime progress = step / human_mean_len",
        },
        "tasks": rules,
    }
    return doc, summaries


def compare_rlc(summaries: dict[int, dict[str, Any]], args: argparse.Namespace) -> None:
    """Print how the demo-derived rules compare with RLC's hand tables on tasks 0-49."""
    rows = []
    agree = total = 0
    for tid in sorted(t for t in summaries if t < 50):
        for side in ("left", "right"):
            s = summaries[tid][side]
            derived = derive_side_rule(s, args)
            rlc = rlc_side_rule(tid, side)
            exempt = side == "right" and tid in RLC_RIGHT_GRIPPER_ALWAYS_ALLOWED
            rlc_kind = "exempt" if exempt else ("always_open" if rlc and rlc["always_open"] else
                                                f"min_stage={rlc['min_stage']}" if rlc else "none")
            der_kind = "always_open" if derived["always_open"] else f"min_progress={derived['min_progress']}"
            ok = (rlc_kind == "always_open") == derived["always_open"]  # same kind: always-open vs gated/free
            agree += ok
            total += 1
            rows.append(f"task {tid:2d} {side:5s}: RLC {rlc_kind:14s} | demos {der_kind:22s} | "
                        f"close_frac {s['close_frac']:.3f} ({s['n_closing']}/{s['n_episodes']}, any-frame "
                        f"{s['n_any_closed_frame']}) p01 {s.get('first_close_progress_p01', '-')} "
                        f"p50 {s.get('first_close_progress_p50', '-')} {'OK' if ok else 'DIFF'}")
    print("\n".join(rows))
    if total:
        print(f"always_open agreement with RLC: {agree}/{total}")


def cmd_rules(args: argparse.Namespace) -> int:
    doc, summaries = build_rules(args)
    if args.compare_rlc:
        compare_rlc(summaries, args)
    missing = [t for t in range(50, 100) if str(t) not in doc["tasks"]]
    if missing:
        log(f"WARNING: no statistics (hence no rule) for tasks {missing}")
    if args.out:
        out = Path(args.out)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        os.replace(tmp, out)
        log(f"wrote {out} ({len(doc['tasks'])} tasks)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--stats-dir", default=None, help="per-task statistics cache (task_TTT.json)")
    common.add_argument("--closed-sum-m", type=float, default=DEFAULT_CLOSED_SUM_M)
    common.add_argument("--revision", default=REVISION)

    ps = sub.add_parser("stats", parents=[common], help="stream the demos and cache per-task statistics")
    ps.add_argument("--tasks", default="50-99", help="e.g. 50-99 or 0,1,30")
    ps.add_argument("--workers", type=int, default=16, help="parallel range requests per file")
    ps.add_argument("--tmp-dir", default=None, help="where sparse temporary files go (default: stats dir)")
    ps.add_argument("--min-free-gb", type=float, default=5.0)
    ps.add_argument("--max-minutes", type=float, default=0.0, help="stop starting new tasks after this (0 = off)")

    pr = sub.add_parser("rules", parents=[common], help="build gripper_rules.json from cached statistics")
    pr.add_argument("--out", default=None, help=f"output path (e.g. {DEFAULT_RULES_PATH})")
    pr.add_argument("--min-run-frames", type=int, default=3)
    pr.add_argument("--always-open-max-frac", type=float, default=0.03)
    pr.add_argument("--late-progress", type=float, default=0.9)
    pr.add_argument("--percentile", type=int, default=1, choices=(0, 1, 5, 25, 50))
    pr.add_argument("--progress-safety", type=float, default=0.6)
    pr.add_argument("--no-rlc-progress", action="store_true",
                    help="do not add demo-derived min_progress to RLC's min_stage rules (tasks 0-49)")
    pr.add_argument("--min-useful-progress", type=float, default=0.05)
    pr.add_argument("--compare-rlc", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "stats":
        if not args.stats_dir:
            ap.error("--stats-dir is required")
        return cmd_stats(args)
    return cmd_rules(args)


if __name__ == "__main__":
    sys.exit(main())
