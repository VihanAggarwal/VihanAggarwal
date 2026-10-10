#!/usr/bin/env python3
"""Show the 2026 leaderboard's self-reported scores (the UI hides them since 10/09; the data file is public).

    python scripts/leaderboard.py            # latest entry per team, sorted by Q
    python scripts/leaderboard.py --all      # every submission
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

SPACE = "https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard"
SCORES = f"{SPACE}/resolve/main/data/self_reported_results.jsonl"
COMMITS = "https://huggingface.co/api/spaces/behavior-1k/2026-challenge-leaderboard/commits/main"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "b1k26-leaderboard/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def team_of(submission_id: str) -> str:
    # submission_id = <UTC ts>-<team slug>-<method slug>; the team slug is not delimited, so show the rest as-is.
    return submission_id.split("-", 1)[1] if "-" in submission_id else submission_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(line) for line in fetch(SCORES).decode().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("q_score") is not None]
    rows.sort(key=lambda r: r["submission_id"])
    if not args.all:
        # Keep the latest submission per (team+method slug prefix): the official ranking uses each team's latest.
        latest: dict[str, dict] = {}
        for r in rows:
            latest[team_of(r["submission_id"]).rsplit("-", 1)[0]] = r
        rows = list(latest.values())
    rows.sort(key=lambda r: -r["q_score"])
    print(f"{'submission':<60} {'Q':>7} {'SR':>6} {'tasks':>5} {'eps':>5}")
    for r in rows:
        print(f"{r['submission_id']:<60} {r['q_score']:7.4f} {r.get('success_rate', 0):6.3f} "
              f"{r.get('num_tasks', 0):5d} {r.get('num_episodes', 0):5d}")
    try:
        commits = json.loads(fetch(COMMITS))
        print(f"\nlast Space commit: {commits[0]['date']}  {commits[0]['title']}")
    except Exception as e:  # noqa: BLE001
        print(f"(could not read commits: {e})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
