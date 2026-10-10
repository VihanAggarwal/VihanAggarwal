"""Plan, run and track evaluator jobs across GPU nodes.

A job is one (task, instance index) rollout, run as its own evaluator process with --num-envs 1. That is the
configuration with the fewest failure modes: one scene load per rollout, results recorded the moment the episode
ends, and a crash only loses that one rollout. Jobs are assigned to workers longest-first (LPT), using each task's
max_steps as the cost, which bounds the makespan by roughly 4/3 of optimal; each worker then runs its own jobs
shortest-first, so a run cut short by the deadline completes as many rollouts as possible.

Commands:
  b1k26-plan plan    --instances 10-11 --tasks @configs/probe_tasks.txt --workers 4 --out jobs/probe  (held out)
  b1k26-plan plan    --instances 0-9 --final --workers 20 --out jobs/final  (the reported run: needs --final)
  b1k26-plan run     --jobs jobs/final/worker_03.jsonl --output-dir runs/final_node03  (one worker; resumable)
  b1k26-plan status  runs/final_node* --jobs-dir jobs/final                (progress, failures, ETA)
  b1k26-plan collect runs/final_node* --into runs/final_all                (merge node outputs, refuse conflicts)

Self-evaluation is not cherry-picking-safe by construction unless you keep it that way: run each reported
(task, instance) exactly once. The runner re-runs a job only when the previous attempt produced no metrics JSON
because of an infrastructure failure (simulator crash, OOM, node loss). An attempt that failed because of the policy
(lost policy connection, bad reply, query timeout: the 2026 rules count these as failures) is never re-run unless
--retry-policy-failures is given (screening only). Every attempt is logged in status.jsonl with its command line.
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

# Evaluator log lines that mean the rollout failed because of the policy server (post1/post2 client:
# omnigibson/eval/utils/network_utils.py; 2026/eval: PolicyConnectionError / PolicyTimeoutError). The rules count such
# a rollout as a failure, so it must not be re-rolled. A client still waiting for /healthz never reached the policy.
POLICY_FAILURE_MARKERS = (
    "Websocket connection error",
    "Error in inference server",
    "Server response missing 'action' key",
    "Server returned action_chunk shape",
    "Server action must exactly equal",
    "PolicyConnectionError",
    "PolicyTimeoutError",
    "policy_connection_lost",
    "policy_query_timeout",
)

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
    """Longest-processing-time-first assignment of jobs to workers (min-heap on accumulated seconds). Each
    returned bucket is in execution order: shortest first (by max_steps, then task id and index). The order inside a
    bucket does not change its total time, but if the run must stop early, it leaves only the longest jobs (the
    fewest rollouts per lost hour) undone instead of every instance of the shortest tasks."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    heap = [(0.0, w) for w in range(workers)]
    buckets: list[list[Job]] = [[] for _ in range(workers)]
    for job in sorted(jobs, key=lambda j: (-j.max_steps, j.task, j.index)):
        load, w = heapq.heappop(heap)
        buckets[w].append(job)
        heapq.heappush(heap, (load + job.max_steps * sec_per_step + load_s, w))
    return [sorted(b, key=lambda j: (j.max_steps, j.task_id, j.index)) for b in buckets]


def reported_jobs(jobs: Iterable[Job]) -> list[Job]:
    """Jobs on the reported public instances 301-310 (what the submission's metrics are)."""
    return [j for j in jobs if j.mode == "public_test" and j.instance_id in constants.REPORTED_INSTANCE_IDS]


def _reported_banner(n: int, what: str) -> None:
    print(f"*** {what} {n} rollout(s) on the REPORTED instances 301-310: this is the final self-evaluation run. Run "
          "each one exactly once; never choose checkpoints or routes from these results. ***", file=sys.stderr)


def estimate_hours(bucket: list[Job], sec_per_step: float, load_s: float) -> float:
    return sum(j.max_steps * sec_per_step + load_s for j in bucket) / 3600.0


def cmd_plan(args: argparse.Namespace) -> int:
    tasks = parse_task_spec(args.tasks)
    jobs = make_jobs(tasks, parse_ids(args.instances), args.mode)
    reported = reported_jobs(jobs)
    if reported and not args.final:
        print(f"refusing: {len(reported)} job(s) target the reported instances 301-310 (public_test indices 0-9). "
              "Screening and confirmation use held-out indices (e.g. --instances 10-11); pass --final only for the "
              "one final run.", file=sys.stderr)
        return 2
    if reported:
        _reported_banner(len(reported), "planning")
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


def _stop_group(proc: subprocess.Popen, grace_s: float = 30.0) -> int | None:
    """SIGTERM the evaluator's process group, then SIGKILL it after ``grace_s``. Returns the exit status."""
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        try:
            return proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            continue
    return proc.poll()


def policy_failure_marker(log_path: Path) -> str | None:
    """The first POLICY_FAILURE_MARKERS entry found in an evaluator log, or None."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None
    for marker in POLICY_FAILURE_MARKERS:
        if marker in text:
            return marker
    return None


def run_job(job: Job, args: argparse.Namespace, attempt: int, port: int) -> dict:
    """Run one evaluator attempt and return its status record. If the runner is stopped meanwhile (signal), the
    evaluator is stopped too and an ``interrupted`` record is logged before the exception propagates."""
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
    if log_path.exists():  # never overwrite the log of an attempt that status.jsonl refers to
        log_path = log_dir / f"{job.task}_{job.instance_id}_attempt{attempt}_{int(time.time())}.log"
    start = time.time()
    rc: int | None = None
    timed_out = False
    interrupted = True
    with open(log_path, "w") as log:
        log.write(f"# {cmd}\n")
        log.flush()
        proc = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
            interrupted = False
        except subprocess.TimeoutExpired:
            timed_out = True
            rc = _stop_group(proc, 60.0)
            interrupted = False
        finally:
            # The runner is stopping (SIGTERM/SIGHUP/SIGINT, see cmd_run) or failed: never leave the evaluator
            # running in its own session. An orphan could later finish (or re-run) a reported rollout unseen.
            if proc.poll() is None:
                _stop_group(proc, 30.0)
            if interrupted:  # an operator stop, not a result: logged, and a resume runs the job again
                _append_status(output_dir, {
                    "job": job.key(), "task": job.task, "instance_id": job.instance_id, "mode": job.mode,
                    "attempt": attempt, "port": port, "start": start, "wall_s": round(time.time() - start, 1),
                    "rc": proc.poll(), "interrupted": True, "json_present": json_path.exists(), "log": str(log_path),
                    "wrapper": args.wrapper, "cmd": cmd})
    wall = time.time() - start
    record = {
        "job": job.key(), "task": job.task, "instance_id": job.instance_id, "mode": job.mode, "attempt": attempt,
        "port": port, "start": start, "wall_s": round(wall, 1), "rc": rc, "timed_out": timed_out,
        "json_present": json_path.exists(), "log": str(log_path), "wrapper": args.wrapper, "cmd": cmd,
    }
    if json_path.exists():
        try:
            doc = json.loads(json_path.read_text())
            steps = int(doc.get("steps") or 0)
            # 2026/eval gives each rollout max_steps seconds after scene load (the "1 step/s" rule). The load time is
            # not logged, so assume --load-allowance-s: over_time_budget flags rollouts that would likely have been
            # cut off there, so their q is not reproducible by the organizers.
            rollout_s = wall - args.load_allowance_s
            record.update({"steps": steps, "q": doc.get("q_score", {}).get("final"), "success": doc.get("success"),
                           "failure_reason": doc.get("failure_reason"), "max_steps": job.max_steps,
                           "rollout_s_estimate": round(rollout_s, 1),
                           "over_time_budget": bool(rollout_s > job.max_steps)})
        except (json.JSONDecodeError, OSError):
            record["json_present"] = False
    if not record["json_present"]:
        marker = policy_failure_marker(log_path)
        if marker is not None:
            record["policy_failure"] = marker
    return record


class _Stop(SystemExit):
    """Raised by the signal handlers of ``run`` so the running evaluator is stopped on the way out."""


def _install_stop_handlers() -> dict:
    def handler(signum, frame):  # noqa: ARG001
        raise _Stop(128 + signum)

    old = {}
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        try:
            old[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):  # not the main thread (tests calling main() from a thread)
            pass
    return old


def cmd_run(args: argparse.Namespace) -> int:
    old_handlers = _install_stop_handlers()
    try:
        return _run_jobs(args)
    finally:
        for sig, h in old_handlers.items():
            signal.signal(sig, h)


def _run_jobs(args: argparse.Namespace) -> int:
    jobs = load_jobs(args.jobs)
    output_dir = Path(args.output_dir)
    (output_dir / "json").mkdir(parents=True, exist_ok=True)
    reported = reported_jobs(jobs)
    if reported:
        _reported_banner(len(reported), "running")
    # Earlier invocations (resume): attempt counts, and jobs the policy already failed (never re-run).
    prior: dict[str, int] = {}
    policy_failed: set[str] = set()
    status_path = output_dir / "status.jsonl"
    if status_path.exists():
        for line in status_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "job" not in rec:
                continue
            prior[rec["job"]] = max(prior.get(rec["job"], 0), int(rec.get("attempt") or 0))
            if rec.get("policy_failure") and not rec.get("json_present"):
                policy_failed.add(rec["job"])
    done = failed = 0
    for job in jobs:
        if (output_dir / "json" / job.json_name()).exists():
            done += 1
            continue
        if job.key() in policy_failed and not args.retry_policy_failures:
            failed += 1
            continue
        for i in range(1, args.max_attempts + 1):
            record = run_job(job, args, prior.get(job.key(), 0) + i, args.port)
            _append_status(output_dir, record)
            print(json.dumps(record), flush=True)
            if record["json_present"]:
                done += 1
                break
            if record.get("policy_failure") and not args.retry_policy_failures:
                # The policy failed this rollout (the rules count it as a failure): re-running it would be a fresh
                # roll of a reported instance. It stays without metrics (= zero in official scoring).
                print(f"policy failure on {job.key()} ({record['policy_failure']}): not re-run", flush=True)
                failed += 1
                break
            # No metrics JSON and no sign of a policy failure: the rollout never produced a result (simulator crash,
            # OOM, timeout before recording). Re-running is an infrastructure retry, not result selection.
            if args.health_cmd:
                if not _health_check(args, output_dir):
                    print(f"{done}/{len(jobs)} jobs have metrics in {output_dir}; stopped: the policy server could not "
                          "be brought back (see status.jsonl)", flush=True)
                    return 1
    print(f"{done}/{len(jobs)} jobs have metrics in {output_dir}; {failed} failed because of the policy (not re-run)")
    return 0 if done + failed == len(jobs) else 1


def _health_check(args: argparse.Namespace, output_dir: Path) -> bool:
    """Run --health-cmd (restart the policy server if needed) with a timeout. False if it failed or hung."""
    try:
        rc = subprocess.run(args.health_cmd, shell=True, timeout=args.health_timeout_s).returncode
    except subprocess.TimeoutExpired:
        rc = None
    if rc == 0:
        return True
    _append_status(output_dir, {"event": "health_cmd_failed", "rc": rc, "time": time.time(),
                                "timeout_s": args.health_timeout_s})
    print(f"health command failed (rc={rc}); stopping instead of running jobs against a dead server", flush=True)
    return False


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
    attempts = [a for a in attempts if "job" in a]
    failed_attempts = [a for a in attempts if not a.get("json_present")]
    over = [a for a in attempts if a.get("over_time_budget")]
    policy = [a for a in failed_attempts if a.get("policy_failure")]
    out = {"metrics_present": len(present), "attempts_logged": len(attempts),
           "attempts_without_metrics": len(failed_attempts),
           # The policy failed these rollouts: not re-run, they count as zero (see the package README).
           "policy_failures": [f"{a['job']} ({a['policy_failure']})" for a in policy],
           # Probably slower than 2026/eval's max_steps seconds after scene load: the hidden run would cut them off.
           "rollouts_over_time_budget": [a["job"] for a in over],
           "wrappers": sorted({a["wrapper"] for a in attempts if a.get("wrapper")})}
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
    # One status log for the package: every attempt of every node (b1k26-package --status-log runs/<into>/status.jsonl).
    lines = []
    for d in args.dirs:
        status = Path(d) / "status.jsonl"
        if status.exists():
            lines += [line for line in status.read_text().splitlines() if line.strip()]
    (into / "status.jsonl").write_text("".join(line + "\n" for line in lines))
    for c in conflicts:
        print("CONFLICT:", c, file=sys.stderr)
    print(f"collected {copied} rollouts into {into} ({len(lines)} attempts in status.jsonl); "
          f"{len(conflicts)} conflict(s)")
    # Two different results for the same (task, instance) mean two runs were mixed: that is not allowed in a
    # submission (one run of the final policy, one rollout per instance).
    return 1 if conflicts else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="write per-worker job files")
    p.add_argument("--tasks", default="all", help="'all', ids/ranges (0-49,72), task names, or @file")
    p.add_argument("--instances", required=True,
                   help="public_test indices: 10-19 are held out (screening/confirmation); 0-9 are the reported "
                        "instances 301-310 and need --final")
    p.add_argument("--final", action="store_true",
                   help="allow jobs on the reported instances 301-310 (the one final self-evaluation run)")
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
    p.add_argument("--health-cmd", default="", help="shell command run after a failed attempt (e.g. restart server); "
                                                     "if it fails or times out, the run stops")
    p.add_argument("--health-timeout-s", type=float, default=3600.0, help="timeout of --health-cmd")
    p.add_argument("--retry-policy-failures", action="store_true",
                   help="also re-run attempts that failed because of the policy (screening only, never the final run)")
    p.add_argument("--load-allowance-s", type=float, default=300.0,
                   help="assumed scene-load time when estimating whether a rollout exceeded max_steps seconds")
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
