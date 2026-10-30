"""Continue newly downloaded datasets without changing benchmark protocols.

This watcher prepares each public dataset when its download marker is present,
then runs all seven models with all five seeds/folds under linear probing.
Neural evaluation, cuML and DREAMT access are tracked separately, never treated
as completed by this frozen-probe stage.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    output = ROOT / ".local/ppg-bench-auto"
    output.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock
    with FileLock(output / ".watch.lock", timeout=0):
        state_path = output / "state.json"
        names = ["dalia", "bidmc", "butppg", "ecsmp", "wesad", "wildppg", "dreamt"]
        state = json.loads(state_path.read_text()) if state_path.exists() else {"datasets": {name: {"status": "waiting_for_download"} for name in names}, "scope": "linear only, all seven models, five seeds and five folds", "full_reproduction_complete": False}

        def save():
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            temp = state_path.with_suffix(".tmp.json")
            temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temp.replace(state_path)

        while True:
            for name in names:
                if state["datasets"][name]["status"] in ("complete", "failed"):
                    continue
                raw = ROOT / "datasets/ppg_bench_raw" / name
                # A single DREAMT CSV may be only a partial manual download.
                # The upstream downloader writes source.json after completion.
                ready = (raw / "source.json").is_file()
                if not ready:
                    continue
                state["datasets"][name] = {"status": "preprocessing"}
                save()
                target = output / name
                with (output / f"{name}.log").open("a", encoding="utf-8") as log:
                    command = [sys.executable, "-u", str(ROOT / "scripts/run_ppg_bench.py"), "preprocess", "--datasets", name, "--output", str(target)]
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode == 0:
                        state["datasets"][name] = {"status": "running_linear"}
                        save()
                        command = [sys.executable, "-u", str(ROOT / "scripts/run_ppg_bench.py"), "run", "--datasets", name, "--methods", "linear", "--output", str(target)]
                        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state["datasets"][name] = {"status": "complete" if result.returncode == 0 else "failed", "exit_code": result.returncode}
                save()
            save()
            if args.once or all(v["status"] in ("complete", "failed") for v in state["datasets"].values()):
                return
            time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    main()
