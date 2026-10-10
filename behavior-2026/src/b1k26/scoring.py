"""Score rollout JSONs the way the organizers do, and validate a submission directory.

Two aggregate scores are reported because the organizers use two:

* ``leaderboard_q`` reproduces ``scripts/extract_self_reported_scores.py`` in the leaderboard Space: per task,
  the mean ``q_score.final`` over the JSONs that are present, summed over tasks and divided by 100. Missing tasks
  count as zero; missing instances inside a task are silently ignored.
* ``official_q`` applies the written rule ("missing rollout instances count as zero"): per task, the sum over the
  expected instance ids divided by the number of expected instances.

They agree when the submission is complete (100 tasks x 10 instances).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from b1k26 import constants

ROLLOUT_NAME_RE = re.compile(r"^(?P<task>[A-Za-z0-9_]+)_(?P<instance>\d+)_(?P<rollout>\d+)\.json$")


@dataclasses.dataclass(frozen=True)
class Rollout:
    task: str
    instance_id: int
    rollout_id: int
    success: bool
    q: float
    steps: int | None
    normalized_time: float | None
    agent_distance: dict
    normalized_agent_distance: dict
    failure_reason: str | None
    source: str

    @property
    def task_id(self) -> int | None:
        info = constants.task_by_name().get(self.task)
        return None if info is None else info.task_id


def _rollout_from_doc(doc: dict, source: str) -> Rollout | None:
    if not isinstance(doc, dict) or "q_score" not in doc or "success" not in doc or "task" not in doc:
        return None
    q = doc.get("q_score", {})
    q = q.get("final") if isinstance(q, dict) else q
    if q is None:
        return None
    name_match = ROLLOUT_NAME_RE.match(Path(source).name)
    instance_id = doc.get("instance_id", int(name_match["instance"]) if name_match else -1)
    rollout_id = doc.get("rollout_id", int(name_match["rollout"]) if name_match else 0)
    time_info = doc.get("time") or {}
    return Rollout(
        task=str(doc["task"]),
        instance_id=int(instance_id),
        rollout_id=int(rollout_id),
        success=bool(doc["success"]),
        q=float(q),
        steps=doc.get("steps"),
        normalized_time=time_info.get("normalized_time"),
        agent_distance=doc.get("agent_distance") or {},
        normalized_agent_distance=doc.get("normalized_agent_distance") or {},
        failure_reason=doc.get("failure_reason"),
        source=source,
    )


def iter_json_files(path: Path) -> Iterable[tuple[str, dict]]:
    """Yield (source, document) for every rollout JSON under a directory, inside a zip, or a single file."""
    path = Path(path)
    if path.is_dir():
        for p in sorted(path.rglob("*.json")):
            try:
                yield str(p), json.loads(p.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith(".json"):
                    try:
                        yield f"{path}!{name}", json.loads(zf.read(name))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
    elif path.suffix == ".json":
        yield str(path), json.loads(path.read_text())
    else:
        raise ValueError(f"not a directory, .zip or .json: {path}")


def load_rollouts(paths: Iterable[str | Path]) -> list[Rollout]:
    out = []
    for p in paths:
        for source, doc in iter_json_files(Path(p)):
            r = _rollout_from_doc(doc, source)
            if r is not None:
                out.append(r)
    return out


def by_task(rollouts: Iterable[Rollout]) -> dict[str, list[Rollout]]:
    groups: dict[str, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        groups[r.task].append(r)
    return dict(groups)


def leaderboard_q(rollouts: Iterable[Rollout], num_tasks: int = constants.NUM_TASKS) -> tuple[float, float]:
    """(Q, SR) exactly as the leaderboard Space's self-report extractor computes them."""
    groups = by_task(rollouts)
    q = sum(sum(r.q for r in eps) / len(eps) for eps in groups.values()) / num_tasks
    sr = sum(sum(r.success for r in eps) / len(eps) for eps in groups.values()) / num_tasks
    return q, sr


def official_q(
    rollouts: Iterable[Rollout],
    instance_ids: Iterable[int] = constants.REPORTED_INSTANCE_IDS,
    num_tasks: int = constants.NUM_TASKS,
) -> tuple[float, float]:
    """(Q, SR) with missing (task, instance) pairs counted as zero; only rollout_id 0 counts."""
    expected = sorted(set(instance_ids))
    best: dict[tuple[str, int], Rollout] = {}
    for r in rollouts:
        if r.rollout_id == 0 and r.instance_id in expected:
            best.setdefault((r.task, r.instance_id), r)
    q_total = sum(r.q for r in best.values())
    s_total = sum(r.success for r in best.values())
    denom = len(expected) * num_tasks
    return q_total / denom, s_total / denom


def per_task_table(rollouts: Iterable[Rollout]) -> list[dict]:
    groups = by_task(rollouts)
    rows = []
    for name, eps in sorted(groups.items(), key=lambda kv: (constants.task_by_name().get(kv[0]).task_id
                                                          if kv[0] in constants.task_by_name() else 10**6)):
        info = constants.task_by_name().get(name)
        rows.append({
            "task_id": info.task_id if info else None,
            "task": name,
            "n": len(eps),
            "q_mean": sum(r.q for r in eps) / len(eps),
            "success_rate": sum(r.success for r in eps) / len(eps),
            "failures": sum(r.failure_reason is not None for r in eps),
            "instances": sorted(r.instance_id for r in eps),
        })
    return rows


def efficiency(rollouts: Iterable[Rollout]) -> dict:
    """Tie-breaker metrics (score_utils): time_score = 3 - 2/normalized_time, plus mean normalized distances."""
    times, base, left, right = [], [], [], []
    for r in rollouts:
        nt = r.normalized_time
        if nt and nt > 0 and math.isfinite(nt):
            times.append(3 - 2 / nt)
        for key, acc in (("base", base), ("left", left), ("right", right)):
            v = r.normalized_agent_distance.get(key)
            if isinstance(v, (int, float)) and math.isfinite(v):
                acc.append(v)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    return {"time_score": mean(times), "base_distance": mean(base), "left_eef": mean(left), "right_eef": mean(right)}


REQUIRED_FIELDS = ("task", "instance_id", "rollout_id", "steps", "success", "q_score", "time", "agent_distance")


def validate_submission(
    metrics_dir: str | Path,
    videos_dir: str | Path | None = None,
    instance_ids: Iterable[int] = constants.REPORTED_INSTANCE_IDS,
    tasks: Iterable[str] | None = None,
) -> list[str]:
    """Return a list of problems (empty == valid) for a directory of rollout JSONs (and optional videos)."""
    problems: list[str] = []
    metrics_dir = Path(metrics_dir)
    expected_ids = set(instance_ids)
    task_names = set(tasks) if tasks is not None else {t.name for t in constants.tasks()}
    seen: dict[tuple[str, int], str] = {}
    files = sorted(metrics_dir.rglob("*.json"))
    if not files:
        return [f"no JSON files under {metrics_dir}"]
    for p in files:
        m = ROLLOUT_NAME_RE.match(p.name)
        if not m:
            problems.append(f"{p}: file name is not <task>_<instance_id>_<rollout_id>.json")
            continue
        try:
            doc = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"{p}: invalid JSON ({e})")
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in doc]
        if missing:
            problems.append(f"{p}: missing fields {missing}")
            continue
        task, inst, rid = m["task"], int(m["instance"]), int(m["rollout"])
        if doc["task"] != task or int(doc["instance_id"]) != inst or int(doc["rollout_id"]) != rid:
            problems.append(f"{p}: file name does not match task/instance_id/rollout_id inside the JSON")
        if task not in task_names:
            problems.append(f"{p}: unknown task {task!r}")
        if inst not in expected_ids:
            problems.append(f"{p}: instance_id {inst} is not in the reported set {sorted(expected_ids)}")
        if rid != 0:
            problems.append(f"{p}: rollout_id {rid} != 0 (only one rollout per instance may be submitted)")
        if (task, inst) in seen:
            problems.append(f"{p}: duplicate of {seen[(task, inst)]}")
        seen[(task, inst)] = str(p)
        q = doc["q_score"].get("final") if isinstance(doc["q_score"], dict) else None
        if not isinstance(q, (int, float)) or not 0.0 <= q <= 1.0:
            problems.append(f"{p}: q_score.final {q!r} not in [0, 1]")
    missing_pairs = [(t, i) for t in sorted(task_names) for i in sorted(expected_ids) if (t, i) not in seen]
    if missing_pairs:
        problems.append(f"{len(missing_pairs)} expected (task, instance) rollouts are missing, e.g. {missing_pairs[:5]}"
                        " (allowed for partial submissions, but they count as zero)")
    if videos_dir is not None:
        vids = {p.stem for p in Path(videos_dir).rglob("*.mp4")}
        jsons = {Path(s).stem for s in seen.values()}
        if jsons - vids:
            problems.append(f"{len(jsons - vids)} rollouts have no video, e.g. {sorted(jsons - vids)[:5]}")
        if vids - jsons:
            problems.append(f"{len(vids - jsons)} videos have no metrics JSON, e.g. {sorted(vids - jsons)[:5]}")
        empty = [p for p in Path(videos_dir).rglob("*.mp4") if p.stat().st_size == 0]
        if empty:
            problems.append(f"{len(empty)} videos are empty, e.g. {[str(p) for p in empty[:3]]}")
    return problems


def summarize(rollouts: list[Rollout], instance_ids: Iterable[int] = constants.REPORTED_INSTANCE_IDS) -> dict:
    lq, lsr = leaderboard_q(rollouts)
    oq, osr = official_q(rollouts, instance_ids)
    groups = by_task(rollouts)
    old = [r for r in rollouts if (r.task_id is not None and r.task_id < 50)]
    new = [r for r in rollouts if (r.task_id is not None and r.task_id >= 50)]
    return {
        "episodes": len(rollouts),
        "tasks": len(groups),
        "leaderboard_q": lq,
        "leaderboard_sr": lsr,
        "official_q": oq,
        "official_sr": osr,
        "q_tasks_0_49_per_covered_task": (leaderboard_q(old, num_tasks=len(by_task(old)))[0] if old else None),
        "q_tasks_50_99_per_covered_task": (leaderboard_q(new, num_tasks=len(by_task(new)))[0] if new else None),
        "failures": {k: v for k, v in _count(r.failure_reason for r in rollouts if r.failure_reason).items()},
        "efficiency": efficiency(rollouts),
    }


def _count(xs: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for x in xs:
        out[x] += 1
    return dict(out)


def parse_ids(spec: str) -> list[int]:
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            ids.extend(range(int(lo), int(hi) + 1))
        elif part:
            ids.append(int(part))
    return ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score BEHAVIOR-1K 2026 rollout JSONs and validate submissions.")
    ap.add_argument("paths", nargs="+", help="directories, zips or JSON files with rollout metrics")
    ap.add_argument("--instances", default="301-310", help="expected instance ids for official_q / validation")
    ap.add_argument("--per-task", action="store_true", help="print the per-task table")
    ap.add_argument("--validate", action="store_true", help="validate the first path as a submission metrics dir")
    ap.add_argument("--videos", default=None, help="videos dir for --validate")
    ap.add_argument("--json", default=None, help="write the summary (and per-task table) to this JSON file")
    ap.add_argument("--status-log", default=None,
                    help="b1k26-plan status.jsonl to report policy failures and over-budget rollouts from (with "
                         "--validate; default: status.jsonl next to the metrics dir, if any)")
    args = ap.parse_args(argv)

    ids = parse_ids(args.instances)
    rollouts = load_rollouts(args.paths)
    summary = summarize(rollouts, ids)
    print(json.dumps(summary, indent=2))
    table = per_task_table(rollouts)
    if args.per_task:
        for row in table:
            print(f"{row['task_id']!s:>3} {row['task']:<45} n={row['n']:<3} q={row['q_mean']:.3f} "
                  f"sr={row['success_rate']:.2f} fail={row['failures']}")
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "per_task": table}, indent=1))
    rc = 0
    if args.validate:
        problems = validate_submission(args.paths[0], args.videos, ids)
        for p in problems:
            print("PROBLEM:", p, file=sys.stderr)
        hard = [p for p in problems if "allowed for partial submissions" not in p]
        rc = 1 if hard else 0
        print(f"validation: {len(problems)} problem(s), {len(hard)} blocking", file=sys.stderr)
        status = Path(args.status_log) if args.status_log else Path(args.paths[0]).parent / "status.jsonl"
        if status.is_file():
            report_run_status(status)
    return rc


def report_run_status(status_log: Path) -> tuple[list[str], list[str]]:
    """Print (and return) the rollouts a status.jsonl marks as policy failures (not re-run, no metrics: zero) and as
    probably over the 2026 time budget (max_steps seconds after scene load)."""
    policy, over = [], []
    for line in status_log.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("policy_failure") and not rec.get("json_present"):
            policy.append(f"{rec.get('job')} ({rec['policy_failure']})")
        if rec.get("over_time_budget"):
            over.append(str(rec.get("job")))
    for job in policy:
        print(f"POLICY FAILURE (no metrics, counts as zero; not re-run): {job}", file=sys.stderr)
    for job in over:
        print(f"OVER TIME BUDGET (2026/eval would have cut it off): {job}", file=sys.stderr)
    print(f"status {status_log}: {len(policy)} policy failure(s), {len(over)} rollout(s) over the time budget",
          file=sys.stderr)
    return policy, over


if __name__ == "__main__":
    sys.exit(main())
