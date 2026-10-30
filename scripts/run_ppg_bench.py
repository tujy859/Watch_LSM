"""Run the pinned upstream benchmark, retaining its splits and search budgets.

Use the independent benchmark venv. Raw data, processed arrays, external model
code, weights and run outputs remain local; only this launcher and small config
files belong in Git. Missing tasks remain visible rather than being counted as
successful reproductions.
"""
import argparse
import atexit
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SHA = "2bca102b7be2c0a9816a3872314784d2ef92555b"
RAW_ROOTS = {
    "ppgbp": ROOT / "datasets/ppg_bp",
    "realworldppg": ROOT / "datasets/realworld_ppg",
    **{name: ROOT / "datasets/ppg_bench_raw" / name for name in
       ("dalia", "wildppg", "bidmc", "butppg", "wesad", "ecsmp", "dreamt")},
}


def write_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def recover_completed_jobs(destination):
    """Publish fully written jobs after a Windows rename/SQLite-handle failure.

    Call only after the upstream subprocess has exited. Its result and prediction
    files remain unchanged; rerunning the identical CLI updates summaries and
    skips these completed jobs using the upstream resume implementation.
    """
    manifest = destination / "run.json"
    if not manifest.is_file():
        return []
    allowed = set(json.loads(manifest.read_text())["jobs"])
    jobs_root = (destination / "jobs").resolve()
    recovered = []
    for working in jobs_root.glob("*.working"):
        job_id = working.name.removesuffix(".working")
        final = jobs_root / job_id
        record = working / "results.json"
        if job_id not in allowed or not record.is_file() or final.exists():
            continue
        result = json.loads(record.read_text())
        predictions = [working / item["predictions"] for item in result.get("results", [])]
        if not predictions or not all(path.is_file() for path in predictions):
            continue
        if not working.resolve().is_relative_to(jobs_root) or not final.resolve().is_relative_to(jobs_root):
            raise ValueError("Job finalization escaped its output directory")
        working.rename(final)
        recovered.append(job_id)
    return recovered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preprocess", "run", "status"])
    parser.add_argument("--datasets", nargs="+", choices=list(RAW_ROOTS), default=list(RAW_ROOTS))
    parser.add_argument("--models", nargs="+", help="Names from the model YAML; default all seven")
    parser.add_argument("--methods", nargs="+", choices=["linear", "rf", "mlp", "finetune"], default=["linear", "rf", "mlp", "finetune"])
    parser.add_argument("--config", type=Path, default=ROOT / "watch_lsm/configs/ppg_bench_models.yaml")
    parser.add_argument("--processed", type=Path, default=ROOT / "datasets/ppg_bench_processed")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/ppg-bench-reproduction")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32, help="Frozen embedding extraction only; published training grids are unchanged")
    parser.add_argument("--neural-trial-workers", type=int, help="Explicit local execution profile; preserves HPO budgets/grids")
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--precision", choices=["upstream", "float32"], default="upstream")
    args = parser.parse_args()
    if args.precision != "upstream" and args.neural_trial_workers is None:
        parser.error("A precision override requires an explicit --neural-trial-workers execution profile")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / f"{args.stage}-status.json"
    if args.stage == "status":
        for path in sorted(args.output.glob("*-status.json")):
            print(path.name)
            print(path.read_text(encoding="utf-8"))
        return 0
    from filelock import FileLock
    process_lock = FileLock(args.output / f".{args.stage}.lock")
    process_lock.acquire(timeout=0)
    atexit.register(process_lock.release)
    revision = subprocess.check_output(["git", "-C", str(ROOT / "repos/ppg-bench"), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_SHA:
        raise RuntimeError(f"Unexpected upstream revision: {revision}")
    import yaml
    from ppg_bench.datasets import DATASETS
    from ppg_bench.models import load_models
    from ppg_bench.preprocess import find_raw_root

    models = load_models(args.config)
    if args.models:
        unknown = set(args.models) - {m["name"] for m in models}
        if unknown:
            parser.error(f"Unknown model names: {sorted(unknown)}")
        models = [m for m in models if m["name"] in args.models]
    rates = sorted({m["fs"] for m in models})
    state = {"upstream_revision": revision, "started_at": datetime.now(timezone.utc).isoformat(), "scope": {"datasets": args.datasets, "models": [m["name"] for m in models], "methods": args.methods if args.stage == "run" else []}, "full_seeds": [42, 43, 44, 45, 46], "full_folds": [0, 1, 2, 3, 4], "jobs": {}}
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"})
    for dataset in args.datasets:
        if args.stage == "preprocess":
            job_id = dataset
            try:
                find_raw_root(dataset, RAW_ROOTS[dataset])
                command = [sys.executable, "-m", "ppg_bench", "preprocess", dataset, "--fs", *map(str, rates), "--raw-root", str(RAW_ROOTS[dataset]), "--output", str(args.processed)]
            except FileNotFoundError as error:
                state["jobs"][job_id] = {"status": "missing_raw_data", "error": str(error)}
                write_json(state_path, state)
                continue
            print(job_id, flush=True)
            state["jobs"][job_id] = {"status": "running"}
            write_json(state_path, state)
            with (args.output / f"preprocess-{job_id}.log").open("a", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            state["jobs"][job_id] = {"status": "complete" if result.returncode == 0 else "failed", "exit_code": result.returncode}
            write_json(state_path, state)
            continue
        for model in models:
            name = model["name"]
            config_path = args.output / f"model-{name}.yaml"
            # Runtime absolute paths are derived from the repository/config;
            # the tracked source config contains only relative paths.
            clean = {key: model[key] for key in ["name", "port", "fs", "embedding_dim", "init"]}
            config_path.write_text(yaml.safe_dump(clean), encoding="utf-8")
            prepared = args.processed / f"fs_{model['fs']}" / DATASETS[dataset]
            for method in args.methods:
                job_id = f"{dataset}__{name}__{method}"
                if not (prepared / "preprocessing.json").is_file():
                    state["jobs"][job_id] = {"status": "missing_processed_data"}
                elif method == "rf" and importlib.util.find_spec("cuml") is None:
                    state["jobs"][job_id] = {"status": "requires_cuml_environment"}
                else:
                    destination = args.output / dataset / name / method
                    head = [sys.executable, "-m", "ppg_bench", "run"]
                    if method in ("mlp", "finetune") and args.neural_trial_workers is not None:
                        head = [sys.executable, str(ROOT / "scripts/ppg_bench_neural_worker.py"), "--trial-workers", str(args.neural_trial_workers), "--evaluation-batch-size", str(args.evaluation_batch_size), "--precision", args.precision]
                    command = [*head, "--model", str(config_path), "--datasets", dataset, "--methods", method, "--processed", str(args.processed), "--output", str(destination), "--device", args.device, "--batch-size", str(args.batch_size)]
                    print(job_id, flush=True)
                    state["jobs"][job_id] = {"status": "running", "output": str(destination.relative_to(args.output))}
                    write_json(state_path, state)
                    recovered_jobs = []
                    # Each group has at most 26 tasks * 25 outer evaluations.
                    # Never repeat failed training unless a complete saved job
                    # was recovered, so genuine errors stop visibly.
                    for _ in range(651):
                        with (args.output / f"run-{job_id}.log").open("a", encoding="utf-8") as log:
                            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                        if result.returncode == 0:
                            break
                        recovered = recover_completed_jobs(destination)
                        if not recovered:
                            break
                        recovered_jobs.extend(recovered)
                    summary = destination / "status.json"
                    upstream = json.loads(summary.read_text()) if summary.exists() else {}
                    complete = result.returncode == 0 and upstream.get("complete") is True
                    state["jobs"][job_id] = {"status": "complete" if complete else "failed", "exit_code": result.returncode, "upstream": upstream, "recovered_completed_jobs": recovered_jobs}
                write_json(state_path, state)
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    state["scope_complete"] = bool(state["jobs"]) and all(v["status"] == "complete" for v in state["jobs"].values())
    state["full_reproduction_complete"] = (args.stage == "run" and state["scope_complete"] and set(args.datasets) == set(RAW_ROOTS) and len(models) == 7 and set(args.methods) == {"linear", "rf", "mlp", "finetune"} and args.precision == "upstream")
    write_json(state_path, state)
    return 0 if state["scope_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
