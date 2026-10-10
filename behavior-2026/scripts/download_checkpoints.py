#!/usr/bin/env python3
"""Download candidate checkpoints (params + assets only; optimizer state is skipped).

    python scripts/download_checkpoints.py --dest /workspace/ckpt comet_pt50 gr00t_multitask hoshipu_100t
    python scripts/download_checkpoints.py --dest /workspace/ckpt jackliu_meta100 --step 139999
    python scripts/download_checkpoints.py --dest /workspace/ckpt jackliu_sft --tasks 0-99
    python scripts/download_checkpoints.py --list

Gated repos (JackLiu0406/meta-SFT-checkpoints) need HF_TOKEN from an account whose access request was approved.
Sizes are approximate (params only).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

CANDIDATES = {
    # name: (repo_id, allow_patterns, approx size, note)
    "comet_pt50": ("sunshk/openpi_comet", ["pi05-b1kpt50-cs32/**"], "12 GB",
                   "Comet (2025 2nd place) pi0.5, 50 tasks; config pi05_b1k-base in openpi-comet"),
    "comet_pt12": ("sunshk/openpi_comet", ["pi05-b1kpt12-cs32/**"], "12 GB", "Comet pi0.5 trained on 12 tasks"),
    "gr00t_multitask": ("kmy17518/gr00t-n1.7-b1k-multitask", ["checkpoint-{step}/**", "README.md"], "7 GB",
                        "GR00T N1.7 on all 100 tasks; default --step 238000 (also try 200000)"),
    "hoshipu_100t": ("Hoshipu/pi05-b1k100t-2026-lr2.5e5", ["ckpt-{step}/**"], "12 GB",
                     "pi0.5 (wensi-ai openpi pi05_b1k) on 100 tasks; default --step 4000000"),
    "rlc_2025": ("IliaLarchenko/behavior_submission", ["checkpoint_*/params/**", "checkpoint_*/assets/**",
                                                       "README.md", "*.json"], "51 GB",
                 "RLC 2025 winner, 4 task-group checkpoints, tasks 0-49 only"),
    "rlc_50t": ("IliaLarchenko/behavior_50t_checkpoint", ["params/**", "assets/**", "README.md"], "13 GB",
                "RLC 2025 stage-1 50-task checkpoint"),
    "jackliu_meta100": ("JackLiu0406/meta-SFT-checkpoints",
                        ["meta100-1epoch/step{step}/params/**", "meta100-1epoch/step{step}/assets/**",
                         "meta100-1epoch/step{step}/_CHECKPOINT_METADATA", "norm-stats-fixed/**"], "13 GB",
                        "PiBehavior on all 100 tasks (1 epoch, 8xB300); default --step 139999. GATED"),
    "jackliu_sft": ("JackLiu0406/meta-SFT-checkpoints", None, "13 GB per task",
                    "per-task fine-tunes single-task-finetune/no-da3/<task>[-70k|-140k]. GATED"),
}
DEFAULT_STEPS = {"gr00t_multitask": "238000", "hoshipu_100t": "4000000", "jackliu_meta100": "139999"}


def parse_ids(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def jackliu_sft_folders(api, task_ids: list[int]) -> dict[int, str]:
    """Resolve task id -> folder under single-task-finetune/no-da3/ (plain id, else -140k, else -70k)."""
    files = api.list_repo_files("JackLiu0406/meta-SFT-checkpoints")
    names = {f.split("/")[2] for f in files if f.startswith("single-task-finetune/no-da3/") and f.count("/") >= 3}
    out = {}
    for t in task_ids:
        for cand in (f"{t}", f"{t}-140k", f"{t}-70k"):
            if cand in names:
                out[t] = cand
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--dest", default="/workspace/ckpt")
    ap.add_argument("--step", default=None, help="checkpoint step for repos with several snapshots")
    ap.add_argument("--tasks", default="0-99", help="task ids for jackliu_sft")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list or not args.names:
        for k, (repo, _, size, note) in CANDIDATES.items():
            print(f"{k:<18} {repo:<42} {size:>15}  {note}")
        return 0
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    os.makedirs(args.dest, exist_ok=True)
    for name in args.names:
        repo, patterns, _, note = CANDIDATES[name]
        dest = os.path.join(args.dest, name)
        if name == "jackliu_sft":
            folders = jackliu_sft_folders(api, parse_ids(args.tasks))
            missing = sorted(set(parse_ids(args.tasks)) - set(folders))
            if missing:
                print(f"no per-task fine-tune folder for tasks {missing}", file=sys.stderr)
            patterns = []
            for _, folder in sorted(folders.items()):
                base = f"single-task-finetune/no-da3/{folder}"
                patterns += [f"{base}/params/**", f"{base}/assets/**", f"{base}/_CHECKPOINT_METADATA", f"{base}/README.md"]
            with open(os.path.join(args.dest, "jackliu_sft_folders.txt"), "w") as f:
                for t, folder in sorted(folders.items()):
                    f.write(f"{t}\t{folder}\n")
        else:
            step = args.step or DEFAULT_STEPS.get(name, "")
            patterns = [p.format(step=step) for p in patterns]
        print(f"[download] {name}: {repo} {patterns} -> {dest}  ({note})")
        snapshot_download(repo_id=repo, allow_patterns=patterns, local_dir=dest, token=os.environ.get("HF_TOKEN"),
                          max_workers=8)
    return 0


if __name__ == "__main__":
    sys.exit(main())
