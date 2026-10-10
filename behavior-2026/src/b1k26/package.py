"""Build the submission package exactly as the 2026 rules ask for it.

Produces, under --out:
  metrics.zip        only the rollout JSONs (flat). The leaderboard extractor counts EVERY .json inside a zip that
                     has "success" and "q_score", so nothing else may go in this zip. Name contains "metric" so an
                     HF dataset upload is picked up by the extractor.
  package.zip        metrics JSONs + wrapper .py + robot config (or a statement that the default r1pro.yaml was
                     used) + README.md + MANIFEST.sha256 (the "final package" of submission.md)
  README.md          rendered evaluation README (full evaluator command, wrapper path, robot config, Docker
                     image, capacity, method, run provenance, score summary)
  videos_manifest.txt  every video file with size and sha256, for the separate video upload
  summary.json       scoring summary (official and leaderboard formulas) and validation problems
The JSON and video files are copied byte-for-byte; the rules forbid modifying them.

Refuses to build (exit 1, unless --force) when the package would misstate the run: placeholder Docker image / video
URL, no wrapper .py, a --replay-chunk-size that does not divide every routed profile's execute_steps (with
--config), or a status log whose attempts used another wrapper or chunk size than the README states.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import shlex
import shutil
import string
import sys
import zipfile
from pathlib import Path
from typing import Iterable

from b1k26 import constants
from b1k26.scoring import load_rollouts, summarize, validate_submission

OFFICIAL_WRAPPER = "omnigibson.eval.wrappers.RGBDFullResWrapper"
# Official wrapper class -> its source file inside the omnigibson package.
OFFICIAL_WRAPPER_FILES = {
    "omnigibson.eval.wrappers.RGBDFullResWrapper": "eval/wrappers/rgbd_full_res_wrapper.py",
    "omnigibson.eval.wrappers.DefaultWrapper": "eval/wrappers/default_wrapper.py",
}
JAX_BACKENDS = {"openpi_comet", "openpi_b1k", "pibehavior"}

README_TEMPLATE = string.Template(r"""# $team: BEHAVIOR-1K 2026 Challenge submission

Method: **$method**. Affiliation: $affiliation. Contact: $contact.

## Policy
$policy_description

Policy inputs: RGB (head + both wrists) and proprioception from `robot_r1::proprio`, plus the evaluator-provided
`task_id`. No depth, no privileged simulator state. All inference-time logic (action chunking, compression,
gripper corrections, routing) runs inside the policy server and only sees those observations.

## How to run (final evaluation)

**Policy server (Docker, recommended path).** Image: `$docker_image`
```
$docker_run
```
- Fits one 24 GB GPU (RTX 3090 / A5000 / TITAN RTX). Turing (sm_75) is supported: $turing_note
- Serves the BEHAVIOR websocket policy protocol on ports `$ports` (every port is an independent endpoint;
  `/healthz` returns 200 once every model is loaded and warmed up, usually within $warmup_note). Please start the
  evaluators after `/healthz` returns 200: under the 2026/eval client the wait otherwise counts against the first
  rollouts' time budget.
- Other ports, several copies on one host, or host networking / enroot (no network namespace): append
  `--ports <first>-<last>` after the image name (the entrypoint accepts it) or set `-e B1K26_PORTS=<first>-<last>`.
  The server's internal model-worker ports move to free loopback ports by themselves when taken.
- Capacity: $capacity_note

**Evaluator command (BEHAVIOR-1K $b1k_version, one process per task instance):**
```
$eval_command
```
- Wrapper: `$wrapper` ($wrapper_note)
- Robot config: $robot_config_note
- Chunk replay: $chunk_note
- Works with the v3.9.3-post1/post2 evaluator and with the 2026/eval `--policy-endpoints` multi-port evaluator.

## Self-evaluation run (what these metrics are)
$provenance

| metric | value |
|---|---|
| rollouts | $episodes |
| tasks covered | $tasks |
| Q (official: missing instances count as zero) | $official_q |
| Q (leaderboard extractor formula) | $leaderboard_q |
| full-task success rate | $official_sr |

$routing_section

## Files
- `metrics/`: one JSON per rollout, unmodified evaluator output, names `<task>_<instance_id>_0.json`.
- `$wrapper_file`: the evaluation wrapper used.
- `$robot_config_file`
- `MANIFEST.sha256`: checksums of every file in this package.
- Videos: uploaded separately ($video_url), one MP4 per rollout with the same base name as its JSON.

## Additional data
$extra_data
""")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_official_wrapper(wrapper: str) -> Path | None:
    """Source file of an official omnigibson wrapper class in the installed omnigibson (located, not imported)."""
    rel = OFFICIAL_WRAPPER_FILES.get(wrapper)
    if rel is None:
        return None
    try:
        spec = importlib.util.find_spec("omnigibson")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(list(spec.submodule_search_locations)[0]) / rel
    return path if path.is_file() else None


def routed_profiles(config_path: str) -> dict[str, tuple[int, str | None]]:
    """Routed profile name -> (execution.execute_steps, backend registry name of its launched worker)."""
    from b1k26.config import load_config

    cfg = load_config(config_path)
    out: dict[str, tuple[int, str | None]] = {}
    for name in dict.fromkeys([cfg.routing.default, *cfg.routing.per_task.values()]):
        prof = cfg.profiles[name]
        argv = cfg.workers[prof.worker].launch or []
        backend = argv[argv.index("--backend") + 1] if "--backend" in argv[:-1] else None
        out[name] = (prof.execution.execute_steps, backend)
    return out


def allowed_chunk_sizes(execute_steps: Iterable[int]) -> list[int]:
    """Replay chunk sizes K > 1 that keep every routed profile's plans intact (K divides each execute_steps)."""
    g = 0
    for n in execute_steps:
        g = math.gcd(g, int(n))
    return [k for k in range(2, g + 1) if g % k == 0]


def _last_flag(cmd: str, flag: str) -> str | None:
    """Value of the last ``flag value`` / ``flag=value`` in a command line (argparse keeps the last one)."""
    try:
        argv = shlex.split(cmd)
    except ValueError:
        argv = cmd.split()
    value = None
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            value = argv[i + 1]
        elif a.startswith(flag + "="):
            value = a.split("=", 1)[1]
    return value


def command_settings(cmd: str) -> tuple[str | None, int]:
    """(--env-wrapper, --replay-action-chunk-size or 0) of an evaluator command line."""
    raw = _last_flag(cmd, "--replay-action-chunk-size")
    try:
        k = int(raw) if raw is not None else 0
    except ValueError:
        k = -1  # unparseable: never equal to a valid setting
    if k in (0, 1):
        k = 0  # the evaluator treats 0 and 1 alike: one query per step
    return _last_flag(cmd, "--env-wrapper"), k


def run_settings(status_log: Path) -> dict:
    """Wrappers and chunk sizes used by the attempts in a status.jsonl, plus policy failures and rollouts that were
    probably over the 2026 time budget."""
    wrappers: set[str] = set()
    chunks: set[int] = set()
    policy, over, unknown = [], [], 0
    for line in status_log.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "job" not in rec:
            continue
        cmd = rec.get("cmd")
        if cmd is None and rec.get("log"):
            try:  # older records: the attempt log starts with "# <command>"
                first = Path(rec["log"]).read_text(errors="replace").splitlines()[0]
                cmd = first[2:] if first.startswith("# ") else None
            except (OSError, IndexError):
                cmd = None
        if cmd is None:
            unknown += 1
        else:
            w, k = command_settings(cmd)
            wrappers.add(rec.get("wrapper") or w or "?")
            chunks.add(k)
        if rec.get("policy_failure") and not rec.get("json_present"):
            policy.append(rec["job"])
        if rec.get("over_time_budget"):
            over.append(rec["job"])
    return {"wrappers": wrappers, "chunks": chunks, "policy_failures": policy, "over_time_budget": over,
            "attempts_without_command": unknown}


def turing_note_for(backends: set[str]) -> str:
    notes = []
    if backends & JAX_BACKENDS:
        notes.append("JAX models: XLA upcasts bf16 matmuls to fp32 on sm_75; no flash-attention is used.")
    if "gr00t" in backends:
        notes.append("GR00T (PyTorch): attention falls back to SDPA on sm_75 (no flash-attention).")
    return " ".join(notes) or "no Turing-specific kernels are used."


def chunk_note_for(k: int, allowed: list[int] | None) -> str:
    allowed_txt = f" (for this policy: {', '.join(map(str, allowed)) or 'none'}; or 0)" if allowed is not None else ""
    used = (f"these metrics were produced with `--replay-action-chunk-size {k}` (already in the command above); use "
            "exactly that value." if k > 1 else
            "these metrics were produced without `--replay-action-chunk-size` (one policy query per step); please run "
            "the same way.")
    return (used + " Other values change the executed plan: K must divide `execution.execute_steps` of every routed "
            f"profile{allowed_txt}, and K larger than execute_steps pads plans with hold actions.")


def build(args: argparse.Namespace) -> int:
    metrics_dir = Path(args.metrics)
    videos_dir = Path(args.videos) if args.videos else None
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        print(f"{out} is not empty (use --overwrite)", file=sys.stderr)
        return 2

    problems = validate_submission(metrics_dir, videos_dir)
    blocking = [p for p in problems if "allowed for partial submissions" not in p]
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)

    # ---- what the README will claim must match the run --------------------------------------------------------
    pkg_problems: list[str] = []
    k = int(args.replay_chunk_size or 0)
    k = k if k > 1 else 0
    eval_command = args.eval_command or (
        "python -m omnigibson.eval.eval --task-name <task> --mode public_test --instance-indices <i> --num-envs 1 "
        f"--num-rollouts 1 --host <host> --port <port> --env-wrapper {args.wrapper} --output-dir <out> --write-video")
    cmd_wrapper, cmd_k = command_settings(eval_command)
    if cmd_k != k and _last_flag(eval_command, "--replay-action-chunk-size") is None and k:
        eval_command += f" --replay-action-chunk-size {k}"
        cmd_k = k
    if cmd_k != k:
        pkg_problems.append(f"--eval-command uses --replay-action-chunk-size {cmd_k} but --replay-chunk-size is {k}")
    if cmd_wrapper is not None and cmd_wrapper != args.wrapper:
        pkg_problems.append(f"--eval-command uses --env-wrapper {cmd_wrapper} but --wrapper is {args.wrapper}")
    allowed: list[int] | None = None
    turing_note = args.turing_note
    if args.config:
        routed = routed_profiles(args.config)
        allowed = allowed_chunk_sizes(steps for steps, _ in routed.values())
        if k and k not in allowed:
            steps = {n: st for n, (st, _) in routed.items()}
            pkg_problems.append(f"--replay-chunk-size {k} does not divide execute_steps of every routed profile "
                                f"({steps}); allowed: {allowed or 'none'} or 0")
        if not turing_note:
            turing_note = turing_note_for({b for _, b in routed.values() if b})
    turing_note = turing_note or "see docs/BACKENDS.md (Turing settings of each model family)."
    chunk_note = args.chunk_note or chunk_note_for(k, allowed)

    if "<" in args.docker_image or "@sha256:" not in args.docker_image:
        pkg_problems.append(f"--docker-image {args.docker_image!r} is not a pushed image reference with a digest "
                            "(<registry>/<image>@sha256:<digest>)")
    if "<" in args.video_url or not args.video_url.strip():
        pkg_problems.append(f"--video-url {args.video_url!r} is a placeholder")

    wrapper_src = Path(args.wrapper_file) if args.wrapper_file else find_official_wrapper(args.wrapper)
    if wrapper_src is None or not wrapper_src.is_file():
        pkg_problems.append(f"no wrapper source to include for {args.wrapper} (pass --wrapper-file; the rules "
                            "require the .py wrapper used during evaluation)")

    status_src = Path(args.status_log) if args.status_log else None
    settings = None
    if status_src is not None and not status_src.is_file():
        pkg_problems.append(f"--status-log {status_src} does not exist")
        status_src = None
    if status_src is not None:
        settings = run_settings(status_src)
        if settings["wrappers"] and settings["wrappers"] != {args.wrapper}:
            pkg_problems.append(f"the run used wrapper(s) {sorted(settings['wrappers'])} but the README would "
                                f"state {args.wrapper} (--wrapper)")
        if settings["chunks"] and settings["chunks"] != {k}:
            pkg_problems.append(f"the run used --replay-action-chunk-size {sorted(settings['chunks'])} (0 = none) "
                                f"but the README would state {k} (--replay-chunk-size)")
        for job in settings["policy_failures"]:
            print(f"NOTE: policy failure without metrics (counts as zero): {job}", file=sys.stderr)
        for job in settings["over_time_budget"]:
            print(f"WARNING: rollout probably over the 2026 time budget (max_steps s): {job}", file=sys.stderr)
    else:
        print("WARNING: no --status-log: the package will not list the run's attempts", file=sys.stderr)

    for prob in pkg_problems:
        print("PROBLEM:", prob, file=sys.stderr)
    blocking += pkg_problems
    if blocking and not args.force:
        print(f"{len(blocking)} blocking problem(s); fix them or pass --force", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    json_files = sorted(metrics_dir.rglob("*.json"))
    rollouts = load_rollouts([metrics_dir])
    summary = summarize(rollouts)
    (out / "summary.json").write_text(json.dumps({"summary": summary, "problems": problems + pkg_problems},
                                                 indent=1))

    with zipfile.ZipFile(out / "metrics.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in json_files:
            zf.write(p, arcname=p.name)

    staging = out / "package"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "metrics").mkdir(parents=True)
    for p in json_files:
        shutil.copy2(p, staging / "metrics" / p.name)

    wrapper_file = f"(missing) wrapper source of {args.wrapper}"
    if wrapper_src is not None and wrapper_src.is_file():
        shutil.copy2(wrapper_src, staging / wrapper_src.name)
        wrapper_file = wrapper_src.name
    robot_config_file = "Robot config: the unchanged default `OmniGibson/omnigibson/eval/r1pro.yaml` (no --robot-config)."
    if args.robot_config:
        src = Path(args.robot_config)
        shutil.copy2(src, staging / src.name)
        robot_config_file = f"`{src.name}`: the exact robot config passed with --robot-config."

    routing_section = ""
    if args.routing_log and Path(args.routing_log).exists():
        shutil.copy2(args.routing_log, staging / Path(args.routing_log).name)
        routing_section = (f"## Checkpoint routing\nThe policy routes each task (by the evaluator's `task_id`) to one "
                           f"checkpoint. Routes were chosen on held-out instances only (311-320 / train-mode), never on "
                           f"the reported instances 301-310; see `{Path(args.routing_log).name}`.")

    provenance = args.provenance or (
        "Single evaluation run of the frozen final policy: every reported (task, instance) was rolled out once "
        "(rollout_id 0). Jobs that produced no metrics JSON because of an infrastructure failure (simulator "
        "crash, node loss) were re-run; rollouts that failed because of the policy were not re-run (they have no "
        "metrics and count as zero).")
    if status_src is not None and settings is not None:
        shutil.copy2(status_src, staging / "status.jsonl")
        provenance += " Every attempt (with its exact evaluator command) is listed in `status.jsonl`."
        if settings["policy_failures"]:
            provenance += (f"\n\nPolicy failures (not re-run, no metrics): "
                           + ", ".join(f"`{j}`" for j in settings["policy_failures"]) + ".")
        if settings["over_time_budget"]:
            provenance += (f"\n\nRollouts that probably exceeded max_steps seconds after scene load: "
                           + ", ".join(f"`{j}`" for j in settings["over_time_budget"]) + ".")

    # Bridge networking with published ports: several containers can run on one host without colliding.
    default_docker_run = f"docker run --gpus '\"device=0\"' -p {args.ports}:{args.ports} {args.docker_image}"
    fields = dict(
        team=args.team, method=args.method, affiliation=args.affiliation, contact=args.contact,
        policy_description=args.policy_description, docker_image=args.docker_image,
        docker_run=args.docker_run or default_docker_run,
        turing_note=turing_note, ports=args.ports, warmup_note=args.warmup_note, capacity_note=args.capacity_note,
        b1k_version=args.b1k_version, eval_command=eval_command, wrapper=args.wrapper,
        wrapper_note=args.wrapper_note, robot_config_note=robot_config_file, chunk_note=chunk_note,
        provenance=provenance, episodes=summary["episodes"], tasks=summary["tasks"],
        official_q=f"{summary['official_q']:.4f}", leaderboard_q=f"{summary['leaderboard_q']:.4f}",
        official_sr=f"{summary['official_sr']:.3f}", routing_section=routing_section,
        wrapper_file=wrapper_file, robot_config_file=robot_config_file, video_url=args.video_url,
        extra_data=args.extra_data,
    )
    readme = README_TEMPLATE.substitute(fields)
    (staging / "README.md").write_text(readme)
    (out / "README.md").write_text(readme)

    manifest_lines = [f"{sha256(p)}  {p.relative_to(staging)}" for p in sorted(staging.rglob("*")) if p.is_file()]
    (staging / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n")
    with zipfile.ZipFile(out / "package.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(staging.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(staging)))

    if videos_dir is not None:
        lines = [f"{sha256(p)}  {p.stat().st_size}  {p.name}" for p in sorted(videos_dir.rglob("*.mp4"))]
        (out / "videos_manifest.txt").write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=1))
    print(f"wrote {out/'metrics.zip'}, {out/'package.zip'}, {out/'README.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the BEHAVIOR-1K 2026 submission package.")
    ap.add_argument("--metrics", required=True, help="dir with the final run's rollout JSONs")
    ap.add_argument("--videos", default=None, help="dir with the final run's MP4s")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--force", action="store_true", help="package despite blocking validation problems")
    ap.add_argument("--team", required=True)
    ap.add_argument("--method", required=True, help="<= 25 characters on the portal")
    ap.add_argument("--affiliation", default="Independent")
    ap.add_argument("--contact", default="")
    ap.add_argument("--policy-description", default="(describe the model, checkpoints and training data)")
    ap.add_argument("--docker-image", default="<registry>/<image>@sha256:<digest>",
                    help="pushed image with its digest (required unless --force)")
    ap.add_argument("--docker-run", default="")
    ap.add_argument("--ports", default="8000-8049")
    ap.add_argument("--config", default=None,
                    help="the frozen serving config (configs/final.yaml): checks --replay-chunk-size against every "
                         "routed profile's execute_steps and derives the Turing note")
    ap.add_argument("--turing-note", default="", help="default: derived from --config's model families")
    ap.add_argument("--warmup-note", default="10 minutes")
    ap.add_argument("--capacity-note", default="one model copy shared by all ports; requests are micro-batched.")
    ap.add_argument("--b1k-version", default="v3.9.3-post2")
    ap.add_argument("--eval-command", default="",
                    help="default: the official eval command with --wrapper (and --replay-chunk-size if > 0)")
    ap.add_argument("--replay-chunk-size", type=int, default=0,
                    help="the --replay-action-chunk-size K used for these metrics (0 = none); stated in the README")
    ap.add_argument("--wrapper", default=OFFICIAL_WRAPPER, help="the --env-wrapper used for these metrics")
    ap.add_argument("--wrapper-file", default=None,
                    help="path of the wrapper .py to include (default: located in the installed omnigibson)")
    ap.add_argument("--wrapper-note", default="official wrapper, unmodified; the policy ignores depth")
    ap.add_argument("--robot-config", default=None)
    ap.add_argument("--chunk-note", default="", help="default: derived from --replay-chunk-size (and --config)")
    ap.add_argument("--provenance", default="")
    ap.add_argument("--status-log", default=None, help="status.jsonl from the run (all attempts); cross-checked")
    ap.add_argument("--routing-log", default=None, help="route_selection.md from b1k26-select")
    ap.add_argument("--video-url", default="<link>", help="link to the uploaded videos (required unless --force)")
    ap.add_argument("--extra-data", default="None. Only the official 2026 human demonstrations (via the public "
                                            "checkpoints we start from).")
    return build(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
