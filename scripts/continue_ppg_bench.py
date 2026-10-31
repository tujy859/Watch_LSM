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
NAMES = ["dalia", "bidmc", "butppg", "ecsmp", "wesad", "wildppg", "dreamt"]
MODELS = {"AnyPPG", "PaPaGei-S", "PaPaGei-P", "PulsePPG", "SigmaPPG", "PRVCtrl", "MOMENT-1-small"}


def complete_linear_scope(target, dataset):
    """A process exiting is insufficient evidence of a complete dataset scope."""
    try:
        state = json.loads((target / "run-status.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    scope = state.get("scope", {})
    expected = {f"{dataset}__{model}__linear" for model in MODELS}
    jobs = state.get("jobs", {})
    return (state.get("scope_complete") is True and scope.get("datasets") == [dataset]
            and set(scope.get("models", [])) == MODELS and scope.get("methods") == ["linear"]
            and set(jobs) == expected and all(job.get("status") == "complete" for job in jobs.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--parallel-datasets", type=int, default=1, help="Independent dataset queues; does not change benchmark splits or HPO")
    parser.add_argument("--adopt", action="append", default=[], help="Keep a verified existing dataset=PID launcher running when replacing this watcher")
    args = parser.parse_args()
    if args.parallel_datasets < 1:
        parser.error("Dataset concurrency must be positive")
    adopted = dict(item.split("=", 1) for item in args.adopt)
    if set(adopted) - set(NAMES) or len(adopted) > args.parallel_datasets:
        parser.error("Invalid adopted dataset scope")
    sys.path.insert(0, str(ROOT))
    from scripts.download_ppg_bench import process_alive
    output = ROOT / ".local/ppg-bench-auto"
    output.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock
    with FileLock(output / ".watch.lock", timeout=0):
        state_path = output / "state.json"
        names = NAMES
        state = json.loads(state_path.read_text()) if state_path.exists() else {"datasets": {name: {"status": "waiting_for_download"} for name in names}, "scope": "linear only, all seven models, five seeds and five folds", "full_reproduction_complete": False}
        state["parallel_datasets"] = args.parallel_datasets
        active = {}

        def save():
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            temp = state_path.with_suffix(".tmp.json")
            temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temp.replace(state_path)

        def start(name, stage):
            target = output / name
            log = (output / f"{name}.log").open("a", encoding="utf-8")
            command = [sys.executable, "-u", str(ROOT / "scripts/run_ppg_bench.py"), stage, "--datasets", name, "--output", str(target)]
            if stage == "run":
                command.extend(["--methods", "linear"])
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            active[name] = {"process": process, "pid": process.pid, "stage": stage, "log": log}
            state["datasets"][name] = {"status": "preprocessing" if stage == "preprocess" else "running_linear", "pid": process.pid}
            save()

        for name, pid in adopted.items():
            active[name] = {"process": None, "pid": int(pid), "stage": "run", "log": None}
            state["datasets"][name] = {"status": "running_linear", "adopted_pid": int(pid)}
        save()
        while True:
            for name, entry in list(active.items()):
                process = entry["process"]
                code = process.poll() if process else None if process_alive(entry["pid"]) else 0
                if code is None:
                    continue
                if entry["log"]:
                    entry["log"].close()
                del active[name]
                if entry["stage"] == "preprocess" and code == 0:
                    start(name, "run")
                else:
                    complete = code == 0 and complete_linear_scope(output / name, name)
                    state["datasets"][name] = {"status": "complete" if complete else "failed", "exit_code": code}
                    save()
            for name in names:
                if name in active:
                    continue
                if state["datasets"][name]["status"] in ("complete", "failed"):
                    continue
                raw = ROOT / "datasets/ppg_bench_raw" / name
                # A single DREAMT CSV may be only a partial manual download.
                # The upstream downloader writes source.json after completion.
                ready = (raw / "source.json").is_file()
                if not ready:
                    state["datasets"][name] = {"status": "waiting_for_download"}
                    continue
                state["datasets"][name] = {"status": "queued"}
                if len(active) < args.parallel_datasets:
                    start(name, "preprocess")
            save()
            if not active and (args.once or all(v["status"] in ("complete", "failed") for v in state["datasets"].values())):
                return
            time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    main()
