"""Queue all ready public-data MLP tasks after the current MLP process exits.

Uses the same output and official resume checks to retain completed PaPaGei-S
jobs. This is a local process, not a scheduled task or a cloud allocation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_ppg_bench import process_alive
from scripts.ppg_bench_status import DATASET_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-pid", type=int, required=True, help="Existing MLP launcher to wait for; completed results are reused")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/ppg-bench-neural-mlp")
    args = parser.parse_args()
    if args.after_pid <= 0:
        parser.error("PID must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock
    with FileLock(args.output / ".continuation.lock", timeout=0):
        datasets = [name for name in DATASET_NAMES if name != "dreamt"]
        for name in datasets:
            for rate in (50, 64, 100, 125):
                metadata = ROOT / "datasets/ppg_bench_processed" / f"fs_{rate}" / DATASET_NAMES[name] / "preprocessing.json"
                if not metadata.is_file():
                    raise FileNotFoundError(f"The complete public-data MLP queue needs {metadata}")
        target = args.output / "continuation.json"
        state = {"datasets": datasets, "models": "all seven pinned benchmark objects", "methods": ["mlp"], "seeds": [42, 43, 44, 45, 46], "folds": [0, 1, 2, 3, 4], "published_search_trials": 48, "precision": "upstream", "wait_for_pid": args.after_pid, "full_reproduction_complete": False}

        def save(phase, **fields):
            state.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **fields)
            temporary = target.with_suffix(".tmp.json")
            temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temporary.replace(target)

        save("waiting_for_existing_mlp")
        while process_alive(args.after_pid):
            time.sleep(5)
        save("running_all_public_mlp")
        command = [sys.executable, "-u", str(ROOT / "scripts/run_ppg_bench.py"), "run", "--datasets", *datasets, "--methods", "mlp", "--output", str(args.output)]
        with (args.output / "continuation.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        save("complete" if result.returncode == 0 else "failed", exit_code=result.returncode)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
