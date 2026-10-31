"""Report local PPG-Bench downloads and completed evaluations without loading data."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_NAMES = {"ppgbp": "PPG-BP", "realworldppg": "RealWorldPPG", "dalia": "Dalia", "bidmc": "BIDMC", "butppg": "BUT-PPG", "wesad": "WESAD", "ecsmp": "ECSMP", "wildppg": "WildPPG", "dreamt": "DREAMT"}


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def published_status(directory):
    """Count published jobs even when a process stopped before summary refresh.

    The upstream publishes a completed job by renaming its working directory;
    incomplete working folders and folders outside run.json never count here.
    Detailed prediction integrity is checked separately by the audit script.
    """
    manifest = read(directory / "run.json", {})
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        return None
    summary = read(directory / "status.json", {})
    completed = sum((directory / "jobs" / job / "results.json").is_file() for job in jobs)
    return {"completed": completed, "planned": len(jobs), "complete": completed == len(jobs), "upstream_reported_completed": summary.get("completed"), "summary_may_lag": summary.get("completed") != completed}


def main():
    local = ROOT / ".local"
    state = {"checked_at": datetime.now(timezone.utc).isoformat(), "full_paper_evaluations_planned": 14625, "nominal_report_cells": 15050, "prv_finetune_report_cells_pending": 425, "plan_note": "Exclude 17*25 parameter-free PRV finetuning cells from independent computations. Paper control reuse still needs protocol confirmation.", "full_reproduction_complete": False}
    state["downloads"] = read(local / "ppg-bench-repro/downloads-v2.json", {})
    for retry_path in (local / "ppg-bench-repro").glob("downloads-retry*.json"):
        retry = read(retry_path, {})
        for name, value in retry.get("datasets", {}).items():
            previous = state["downloads"].setdefault("datasets", {}).get(name, {})
            if value.get("updated_at", 0) >= previous.get("updated_at", 0):
                state["downloads"]["datasets"][name] = value
    for dataset, entry in state["downloads"].get("datasets", {}).items():
        raw = ROOT / "datasets/ppg_bench_raw" / dataset
        sizes = []
        if raw.exists():
            for path in raw.rglob("*"):
                try:
                    if path.is_file():
                        sizes.append(path.stat().st_size)
                except FileNotFoundError:
                    pass  # Atomic publication can rename an in-flight download.
        entry["local_files"] = len(sizes)
        entry["local_bytes_including_archives_and_partials"] = sum(sizes)
    state["model_validation"] = read(local / "ppg-bench-repro/model-validation.json", {})
    state["gradient_validation"] = read(local / "ppg-bench-repro/training-validation.json", {})
    state["preprocessing"] = {}
    for name, display in DATASET_NAMES.items():
        rates = {}
        for rate in (50, 64, 100, 125):
            prepared = ROOT / "datasets/ppg_bench_processed" / f"fs_{rate}" / display
            metadata = read(prepared / "preprocessing.json")
            rates[str(rate)] = {"ready": metadata is not None, "groups": len(list(prepared.glob("*/ppg.npy"))) if metadata else 0}
        state["preprocessing"][name] = rates
    state["evaluations"] = []
    roots = [local / "ppg-bench-reproduction-py312", local / "ppg-bench-reproduction-extra", local / "ppg-bench-auto", local / "ppg-bench-neural-mlp"]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("run.json")):
            value = published_status(path.parent)
            if value:
                state["evaluations"].append({"path": path.parent.relative_to(ROOT).as_posix(), **value})
    state["completed_evaluations"] = sum(item["completed"] for item in state["evaluations"])
    state["queued_evaluations_with_existing_manifests"] = sum(item["planned"] for item in state["evaluations"])
    state["automatic_continuation"] = read(local / "ppg-bench-auto/state.json", {})
    state["mlp_continuation"] = read(local / "ppg-bench-neural-mlp/continuation.json", {})
    state["mlp_capacity_run"] = read(local / "ppg-bench-mlp-capacity/status.json", {"complete": False})
    state["amp_kernel_diagnostic"] = read(local / "ppg-bench-amp-kernel-diagnostic/report.json", {})
    state["adapted_backend_pilot"] = {"status": published_status(local / "ppg-bench-amp-native-pilot"), "backend": read(local / "ppg-bench-amp-native-pilot/backend-profile.json", {}), "finalization": read(local / "ppg-bench-repro/amp-native-pilot-finalization.json", {})}
    state["adapted_precision_runs"] = []
    for path in (local / "ppg-bench-neural-fp32").rglob("run.json"):
        value = published_status(path.parent)
        if value:
            state["adapted_precision_runs"].append({"path": path.parent.relative_to(ROOT).as_posix(), "precision": "float32", **value})
    state["remaining_external_requirements"] = ["DREAMT 2.1.0: user must log in and sign the dataset agreement", "Official RF: a compatible Linux/cuML environment", "Original AMP finetuning produced NaNs on this GPU; Float32 runs are tracked separately, not counted as original-precision reproduction"]
    target = local / "ppg-bench-status.json"
    temporary = target.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({"completed_evaluations": state["completed_evaluations"], "full_paper_evaluations_planned": state["full_paper_evaluations_planned"], "prv_report_cells_pending": 425, "all_models_passed": all(v.get("status") == "passed" for v in state["model_validation"].get("models", {}).values()) and len(state["model_validation"].get("models", {})) == 7, "downloads": {k: v["status"] for k, v in state["downloads"].get("datasets", {}).items()}, "full_reproduction_complete": False, "report": str(target)}, indent=2))


if __name__ == "__main__":
    main()
