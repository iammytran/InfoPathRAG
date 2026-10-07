"""Monitor a running InfoVQA ablation process and append progress samples."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def _process_snapshot(pid: int) -> dict:
    status = Path(f"/proc/{pid}/status")
    stat = Path(f"/proc/{pid}/stat")
    if not status.exists() or not stat.exists():
        return {"running": False}

    values = {}
    for line in status.read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"State", "VmRSS", "Threads"}:
            values[key] = value.strip()
    fields = stat.read_text().split()
    values.update({
        "running": True,
        "cpu_ticks": int(fields[13]) + int(fields[14]),
    })
    return values


def _output_snapshot(output_dir: Path) -> dict:
    files = {}
    for path in sorted(output_dir.glob("path_reranking*.jsonl")):
        files[path.name] = {
            "bytes": path.stat().st_size,
            "lines": sum(1 for _ in path.open()),
        }
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("algorithm_results/LILaC/InfoVQA/ablation"),
    )
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--log-file", type=Path, default=Path("debug/infovqa_progress.jsonl"))
    args = parser.parse_args()

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    while True:
        sample = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pid": args.pid,
            "process": _process_snapshot(args.pid),
            "outputs": _output_snapshot(args.output_dir),
        }
        with args.log_file.open("a") as handle:
            handle.write(json.dumps(sample) + "\n")
        print(json.dumps(sample), flush=True)
        if not sample["process"].get("running"):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
