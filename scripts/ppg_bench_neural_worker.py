"""Run official neural HPO with explicit, recorded local execution limits.

Only candidate concurrency and evaluation batching change. Published training
batch candidates, trial counts, epochs, search spaces, splits and scores remain
unchanged. Serial TPE scheduling can choose different candidates than parallel
TPE; this execution difference is retained in both protocol and profile files.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-workers", type=int, default=1)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--precision", choices=["upstream", "float32"], default="upstream")
    parser.add_argument("--diagnostic-trials", type=int, help="Explicit diagnostic only; never report as the published search budget")
    limits, arguments = parser.parse_known_args()
    if min(limits.trial_workers, limits.evaluation_batch_size) < 1:
        parser.error("Execution limits must be positive")
    if "--output" not in arguments:
        parser.error("An explicit --output is required")
    output = Path(arguments[arguments.index("--output") + 1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if limits.diagnostic_trials is not None and limits.diagnostic_trials < 1:
        parser.error("Diagnostic trial count must be positive")
    profile = {"trial_workers_per_gpu": limits.trial_workers, "evaluation_batch_size": limits.evaluation_batch_size, "precision": limits.precision, "diagnostic_only": limits.diagnostic_trials is not None, "diagnostic_trials": limits.diagnostic_trials, "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "training_grids_and_search_budget_changed": limits.diagnostic_trials is not None, "note": "Serial and parallel TPE schedules may sample different candidate sequences. Float32 differs from the upstream AMP execution."}
    record = output / "execution-profile.json"
    if record.exists() and json.loads(record.read_text()) != profile:
        raise ValueError("Execution settings changed; use a new output directory")
    if not record.exists():
        record.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    runner = importlib.import_module("ppg_bench.run")
    original = runner.probe_config

    def local_config(method, seed):
        config = original(method, seed)
        if method in ("mlp", "finetune"):
            config["eval_batch_size"] = limits.evaluation_batch_size
            config["search"]["trial_workers_per_gpu"] = limits.trial_workers
            if limits.precision == "float32":
                config["amp"] = False
            if limits.diagnostic_trials is not None:
                config["search"]["n_trials"] = limits.diagnostic_trials
        return config

    runner.probe_config = local_config
    sys.argv = [sys.argv[0], "run", *arguments]
    from ppg_bench.__main__ import main as upstream_main
    return upstream_main()


if __name__ == "__main__":
    raise SystemExit(main())
