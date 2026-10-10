"""Plan, run and track evaluator jobs across GPU nodes.

A job is one (task, instance index) rollout, run as its own evaluator process with --num-envs 1. That is the
configuration with the fewest failure modes: one scene load per rollout, results recorded the moment the episode
ends, and a crash only loses that one rollout. Jobs are packed onto workers longest-first (LPT), using each task's
max_steps as the cost, which bounds the makespan by roughly 4/3 of optimal.

Commands:
  b1k26-plan plan    --instances 0-9 --workers 20 --out jobs/            (write worker job files)
  b1k26-plan run     --jobs jobs/worker_03.jsonl --output-dir runs/final  (run one worker's jobs; resumable)
  b1k26-plan status  runs/final [more dirs]                               (progress, failures, ETA)
  b1k26-plan collect runs/node*/ --into submission_run/                   (merge node outputs, refuse conflicts)

Self-evaluation is not cherry-picking-safe by construction unless you keep it that way: run each reported
(task, instance) exactly once. The runner re-runs a job only when the previous attempt produced no metrics JSON
(an infrastructure failure: simulator crash, OOM, node loss) and logs every attempt in status.jsonl.
"""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from b1k26 import constants
from b1k26.scoring import parse_ids

DEFAULT_EVAL_TEMPLATE = (
    "python -m omnigibson.eval.eval --task-name {task} --mode {mode} --instance-indices {index} "
    "--num-envs 1 --num-rollouts 1 --host {host} --port {port} --env-wrapper {wrapper} "
    "--output-dir {output_dir} {video_flag} {extra}"
)
OFFICIAL_WRAPPER = "omnigibson.eval.wrappers.RGBDFullResWrapper"


@dataclasses.dataclass(frozen=True)
class Job:
    task: str
    task_id: int
    index: int  # value passed to --instance-indices
    instance_id: int  # id in the output file name
    mode: str
    max_steps: int

    def json_name(self, rollout_id: int = 0) -> str:
        return f"{self.task}_{self.instance_id}_{rollout_id}.json"

    def key(self) -> str:
        return f"{self.task}:{self.mode}:{self.index}"


def instance_id_for(mode: str, index: int) -> int:
    if mode == "public_test":
        return constants.public_index_to_instance_id(index)
    if mode == "hidden_test":
        return 321 + index
    return index  # train mode takes raw instance ids


def make_jobs(tasks: Iterable[int | str], indices: Iterable[int], mode: str = "public_test") -> list[Job]:
    jobs = []
    for t in tasks:
        info = constants.task(t)
        for i in indices:
            jobs.append(Job(info.name, info.task_id, int(i), instance_id_for(mode, int(i)), mode, info.max_steps))
    return jobs


def lpt_assign(jobs: list[Job], workers: int, sec_per_step: float = 0.12, load_s: float = 240.0
               ) -> list[list[Job]]:
    """Longest-processing-time-first assignment of jobs to workers (min-heap on accumulated seconds)."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    heap = [(0.0, w) for w in range(workers)]
    buckets: list[list[Job]] = [[] for _ in range(workers)]
    for job in sorted(jobs, key=lambda j: (-j.max_steps, j.task, j.index)):
        load, w = heapq.heappop(heap)
        buckets[w].append(job)
        heapq.heappush(heap, (load + job.max_steps * sec_per_step + load_s, w))
    return buckets


def estimate_hours(bucket: list[Job], sec_per_step: float, load_s: float) -> float:
    return sum(j.max_steps * sec_per_step + load_s for j in bucket) / 3600.0


def cmd_plan(args: argparse.Namespace) -> int:
    tasks = parse_task_spec(args.tasks)
    jobs = make_jobs(tasks, parse_ids(args.instances), args.mode)
    buckets = lpt_assign(jobs, args.workers, args.sec_per_step, args.load_s)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for w, bucket in enumerate(buckets):
        with open(out / f"worker_{w:02d}.jsonl", "w") as f:
            for j in bucket:
                f.write(json.dumps(dataclasses.asdict(j)) + "\n")
    hours = [estimate_hours(b, args.sec_per_step, args.load_s) for b in buckets]
    total_steps = sum(j.max_steps for j in jobs)
    summary = {
        "jobs": len(jobs),
        "workers": args.workers,
        "worst_case_env_steps": total_steps,
        "worst_case_gpu_hours": round(sum(hours), 1),
        "makespan_hours_worst_case": round(max(hours), 2),
        "assumption": f"{args.sec_per_step} s/step (sim+render+policy) and {args.load_s} s scene load per job; "
                      "episodes that succeed early finish sooner (typically 5-10% fewer steps in total)",
    }
    (out / "plan_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


def parse_task_spec(spec: str) -> list[int]:
    if spec.startswith("@"):
        lines = Path(spec[1:]).read_text().splitlines()
        spec = ",".join(line.split("#", 1)[0].strip() for line in lines if line.split("#", 1)[0].strip())
    if spec in ("all", "*"):
        return [t.task_id for t in constants.tasks()]
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part in constants.task_by_name():
            out.append(constants.task(part).task_id)
        else:
            out.extend(parse_ids(part))
    return out


# --------------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------------
def load_jobs(path: str | Path) -> list[Job]:
    return [Job(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def _append_status(output_dir: Path, record: dict) -> None:
    with open(output_dir / "status.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def run_job(job: Job, args: argparse.Namespace, attempt: int, port: int) -> dict:
    output_dir = Path(args.output_dir)
    json_path = output_dir / "json" / job.json_name()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = args.eval_template.format(
        task=job.task, mode=job.mode, index=job.index, host=args.host, port=port, wrapper=args.wrapper,
        output_dir=shlex.quote(str(output_dir)), video_flag="--write-video" if args.write_video else "--no-write-video",
        extra=args.extra or "",
    )
    # Hard wall-clock cap per job: scene load allowance + max_steps at the rules' 1 s/step budget + slack.
    timeout = args.load_timeout_s + job.max_steps * args.max_sec_per_step + 600
    log_path = log_dir / f"{job.task}_{job.instance_id}_attempt{attempt}.log"
    start = time.time()
    rc: int | None = None
    timed_out = False
    with open(log_path, "w") as log:
        log.write(f"# {cmd}\n")
        log.flush()
        proc = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                rc = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                rc = proc.wait()
    wall = time.time() - start
    record = {
        "job": job.key(), "task": job.task, "instance_id": job.instance_id, "mode": job.mode, "attempt": attempt,
        "port": port, "start": start, "wall_s": round(wall, 1), "rc": rc, "timed_out": timed_out,
        "json_present": json_path.exists(), "log": str(log_path),
    }
    if json_path.exists():
        try:
            doc = json.loads(json_path.read_text())
            steps = int(doc.get("steps") or 0)
            record.update({"steps": steps, "q": doc.get("q_score", {}).get("final"), "success": doc.get("success"),
                           "failure_reason": doc.get("failure_reason"),
                           # Upper bound on wall seconds per step (includes scene load); must stay < 1.0.
                           "wall_s_per_step_upper": round(wall / max(steps, 1), 3)})
        except (json.JSONDecodeError, OSError):
            record["json_present"] = False
    return record


def cmd_run(args: argparse.Namespace) -> int:
    jobs = load_jobs(args.jobs)
    output_dir = Path(args.output_dir)
    (output_dir / "json").mkdir(parents=True, exist_ok=True)
    done = 0
    for job in jobs:
        if (output_dir / "json" / job.json_name()).exists():
            done += 1
            continue
        for attempt in range(1, args.max_attempts + 1):
            record = run_job(job, args, attempt, args.port)
            _append_status(output_dir, record)
            print(json.dumps(record), flush=True)
            if record["json_present"]:
                done += 1
                break
            # No metrics JSON == the rollout never produced a result (crash / OOM / timeout before recording).
            # Re-running is an infrastructure retry, not result selection; it is logged in status.jsonl.
            if args.health_cmd:
                subprocess.run(args.health_cmd, shell=True)
    print(f"{done}/{len(jobs)} jobs have metrics in {output_dir}")
    return 0 if done == len(jobs) else 1


# --------------------------------------------------------------------------------------------------
# Status / collect
# --------------------------------------------------------------------------------------------------
def cmd_status(args: argparse.Namespace) -> int:
    expected = None
    if args.jobs_dir:
        expected = [j for p in sorted(Path(args.jobs_dir).glob("worker_*.jsonl")) for j in load_jobs(p)]
    present: dict[str, Path] = {}
    attempts: list[dict] = []
    for d in args.dirs:
        d = Path(d)
        for p in d.glob("json/*.json"):
            present[p.name] = p
        status = d / "status.jsonl"
        if status.exists():
            attempts.extend(json.loads(line) for line in status.read_text().splitlines() if line.strip())
    failed_attempts = [a for a in attempts if not a.get("json_present")]
    slow = [a for a in attempts if (a.get("wall_s_per_step_upper") or 0) > 0.8]
    out = {"metrics_present": len(present), "attempts_logged": len(attempts),
           "attempts_without_metrics": len(failed_attempts),
           "rollouts_near_1fps_budget": [a["job"] for a in slow][:20]}
    if expected is not None:
        missing = [j for j in expected if j.json_name() not in present]
        out["expected"] = len(expected)
        out["missing"] = len(missing)
        out["missing_examples"] = [j.key() for j in missing[:10]]
        done_wall = [a["wall_s"] for a in attempts if a.get("json_present")]
        if done_wall and missing:
            per_step = sum(a["wall_s"] for a in attempts if a.get("json_present")) / max(
                1, sum(a.get("steps", 0) for a in attempts if a.get("json_present")))
            out["remaining_worst_case_gpu_hours"] = round(sum(j.max_steps for j in missing) * per_step / 3600, 1)
    print(json.dumps(out, indent=1))
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    into = Path(args.into)
    (into / "json").mkdir(parents=True, exist_ok=True)
    (into / "videos").mkdir(parents=True, exist_ok=True)
    conflicts = []
    copied = 0
    for d in args.dirs:
        d = Path(d)
        for p in sorted(d.glob("json/*.json")):
            dest = into / "json" / p.name
            if dest.exists():
                if dest.read_bytes() != p.read_bytes():
                    conflicts.append(f"{p} differs from already collected {dest}")
                continue
            shutil.copy2(p, dest)
            copied += 1
            video = d / "videos" / (p.stem + ".mp4")
            if video.exists():
                shutil.copy2(video, into / "videos" / video.name)
    for c in conflicts:
        print("CONFLICT:", c, file=sys.stderr)
    print(f"collected {copied} rollouts into {into}; {len(conflicts)} conflict(s)")
    # Two different results for the same (task, instance) mean two runs were mixed: that is not allowed in a
    # submission (one run of the final policy, one rollout per instance).
    return 1 if conflicts else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="write per-worker job files")
    p.add_argument("--tasks", default="all", help="'all', ids/ranges (0-49,72), task names, or @file")
    p.add_argument("--instances", default="0-9", help="public_test indices (0-9 reported, 10-19 held out)")
    p.add_argument("--mode", default="public_test", choices=["public_test", "train", "hidden_test"])
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--sec-per-step", type=float, default=0.12)
    p.add_argument("--load-s", type=float, default=240.0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="run one worker's jobs sequentially (resumable)")
    p.add_argument("--jobs", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--wrapper", default=OFFICIAL_WRAPPER)
    p.add_argument("--write-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--extra", default="", help="extra evaluator flags, e.g. '--replay-action-chunk-size 20'")
    p.add_argument("--eval-template", default=DEFAULT_EVAL_TEMPLATE)
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--max-sec-per-step", type=float, default=1.0, help="per-job timeout = load + max_steps * this")
    p.add_argument("--load-timeout-s", type=float, default=1200.0)
    p.add_argument("--health-cmd", default="", help="shell command run after a failed attempt (e.g. restart server)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="progress across output dirs")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--jobs-dir", default=None)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("collect", help="merge node outputs into one run directory")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--into", required=True)
    p.set_defaults(func=cmd_collect)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
