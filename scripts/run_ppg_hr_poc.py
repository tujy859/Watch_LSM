"""Fixed subject-separated HR, label-budget and source-only transfer evaluation.

This is a small research runner. Frozen encoder features are produced separately;
no foundation model is trained here. All selections use source validation people.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from ppg_hr_baselines import BASELINE_PROTOCOL, predict_hr_windows
from ppg_hr_poc_data import ROOT, sha256, write_json

MODELS = ("AnyPPG", "PaPaGei-S", "PulsePPG", "MOMENT-1-small")
SEEDS = (42, 43, 44)
ALPHAS = (0.01, 0.1, 1., 10., 100., 1000.)
DATASETS = ("Dalia", "WildPPG")


def make_splits(subjects_by_dataset):
    """Nested label budgets include validation; train/val subsets also nest."""
    jobs = []
    for dataset, subjects in subjects_by_dataset.items():
        subjects = np.asarray(sorted(subjects))
        for fold, (outer, test) in enumerate(KFold(5, shuffle=True, random_state=20261104).split(subjects)):
            for seed in SEEDS:
                ordered = np.random.default_rng(seed + 100 * fold).permutation(subjects[outer]).tolist()
                for budget, n, val_positions in (("3", 3, (0,)), ("6", 6, (0, 3)), ("full", len(outer), (0, 3, 6))):
                    val = [ordered[i] for i in val_positions]
                    train = [s for s in ordered[:n] if s not in val]
                    jobs.append({"id": f"{dataset}-fold{fold}-seed{seed}-labels{budget}", "scenario": "within", "source": dataset, "target": dataset, "fold": fold, "seed": seed, "budget": budget, "train": train, "validation": val, "test": subjects[test].tolist()})
    for seed in SEEDS:
        ordered = np.random.default_rng(seed).permutation(sorted(subjects_by_dataset["Dalia"])).tolist()
        jobs.append({"id": f"Dalia-to-WildPPG-seed{seed}", "scenario": "transfer", "source": "Dalia", "target": "WildPPG", "fold": -1, "seed": seed, "budget": "full", "train": ordered[3:], "validation": ordered[:3], "test": sorted(subjects_by_dataset["WildPPG"])})
    for job in jobs:
        validate_split(job)
    return jobs


def validate_split(job):
    train, val, test = (set(job[k]) for k in ("train", "validation", "test"))
    if not train or not val or not test or train & val:
        raise ValueError("Missing subjects or training/validation leakage")
    if job["source"] == job["target"] and (train | val) & test:
        raise ValueError("Held-out subject leakage")
    if any(len(job[k]) != len(set(job[k])) for k in ("train", "validation", "test")):
        raise ValueError("Duplicate subject identity")


def weights(subjects):
    _, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    return len(subjects) / (len(counts) * counts[inverse])


def macro_mae(y, pred, subjects):
    return float(np.mean([np.mean(np.abs(y[subjects == s] - pred[subjects == s])) for s in np.unique(subjects)]))


def fit_ridge(x, y, subjects, val_x, val_y, val_subjects, alphas=ALPHAS):
    """Subject-weighted standardization + ridge; one covariance for all alphas."""
    if set(subjects) & set(val_subjects):
        raise ValueError("Training/validation subjects overlap")
    x, val_x, y = np.asarray(x, dtype=np.float64), np.asarray(val_x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if not all(np.isfinite(a).all() for a in (x, val_x, y, val_y)):
        raise ValueError("Nonfinite ridge data")
    w = weights(subjects)
    mean = np.average(x, axis=0, weights=w)
    z = x - mean
    scale = np.sqrt(np.average(z * z, axis=0, weights=w))
    scale[scale < 1e-12] = 1.
    z /= scale
    center = float(np.average(y, weights=w))
    gram = z.T @ (z * w[:, None])
    rhs = z.T @ (w * (y - center))
    eigval, eigvec = np.linalg.eigh(gram)
    coefficients = eigvec @ ((eigvec.T @ rhs)[:, None] / (np.maximum(eigval, 0)[:, None] + np.asarray(alphas)[None, :]))
    predictions = ((val_x - mean) / scale) @ coefficients + center
    scores = [macro_mae(val_y, predictions[:, i], val_subjects) for i in range(len(alphas))]
    chosen = int(np.argmin(scores))
    coef = coefficients[:, chosen] / scale
    intercept = center - mean @ coef
    return coef, float(intercept), {"alpha": float(alphas[chosen]), "validation_subject_mae_bpm": scores[chosen], "alpha_scores": dict(zip(map(str, alphas), scores)), "train_subject_mean_hr": center}


def protocol(root):
    manifests = {d: json.loads((root / "data" / d / "manifest.json").read_text()) for d in DATASETS}
    if "reference_correction" not in manifests["WildPPG"]:
        raise ValueError("Recompute/apply the official WildPPG ECG reference before evaluation")
    scripts = [Path(__file__), *[ROOT / "scripts" / name for name in ("ppg_hr_baselines.py", "ppg_hr_cnn.py", "ppg_hr_poc_data.py", "ppg_hr_wild_reference.py", "ppg_hr_poc_wild_data.py")]]
    return {
        "version": "wrist-hr-poc-v1", "fixed_before_new_model_results": True,
        "models": list(MODELS), "seeds": list(SEEDS), "outer_folds": 5, "outer_split_seed": 20261104,
        "label_budgets": {"3": "2 training + 1 validation people", "6": "4 training + 2 validation people", "full": "all outer training people, including 3 validation people"},
        "transfer": "Dalia 12 training / 3 validation people -> all 16 WildPPG people; no target labels or target-trained parameters used for selection",
        "primary_metric": "mean participant MAE in bpm, first average repeated seed MAEs within participant",
        "primary_comparisons": "each PPG foundation model versus small CNN and the two fixed traditional methods; MOMENT is a generic representation control",
        "practical_margin_bpm": 1.0,
        "inference": "paired participant bootstrap 10000 samples, exploratory 95% intervals; not multiplicity-adjusted or a population-wide superiority claim",
        "ridge": {"alphas": list(ALPHAS), "fit": "subject-balanced standardization and squared error; select validation subject MAE; no train+validation refit"},
        "cnn": {"parameters": 27065, "epochs": 20, "patience": 5, "batch": 1024, "lr": 0.001, "fit": "subject-balanced L1, best validation subject MAE, FP32, no refit"},
        "traditional": BASELINE_PROTOCOL,
        "input": "PPG-only, aligned 8 second windows, native model resampling/preprocessing; ECG reference and ACC descriptive strata only",
        "coverage": "retain every finite PPG window with positive finite HR reference; report failed traditional outputs separately and paired comparisons on common finite windows",
        "pulseppg_note": "upstream whole-subject unlabeled mean/std; offline record normalization, not online evaluation",
        "data_manifest_sha256": {d: sha256(root / "data" / d / "manifest.json") for d in DATASETS},
        "data_rows_sha256": {d: manifests[d]["hashes"]["rows.npz"] for d in DATASETS},
        "code_sha256": {p.name: sha256(p) for p in scripts},
        "jobs": make_splits({d: manifests[d]["subjects"] for d in DATASETS}),
    }


def ensure_protocol(root):
    path = root / "protocol.json"
    current = protocol(root)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != current:
            raise ValueError("Protocol or code changed after registration; use a fresh run root")
    else:
        write_json(path, current)
    return current


def load_rows(root, dataset):
    with np.load(root / "data" / dataset / "rows.npz") as data:
        return {k: data[k] for k in data.files}


def result_identity(root, job, method):
    identity = {"protocol_sha256": sha256(root / "protocol.json"), "job": job, "method": method}
    if method in MODELS:
        identity["feature_metadata_sha256"] = {d: sha256(root / "features" / d / f"{method}.json") for d in set((job["source"], job["target"]))}
    return identity


def run_baselines(root, rows):
    for dataset in DATASETS:
        destination = root / "traditional" / dataset
        destination.mkdir(parents=True, exist_ok=True)
        identity = {"data_manifest_sha256": sha256(root / "data" / dataset / "manifest.json"), "algorithm_sha256": sha256(ROOT / "scripts/ppg_hr_baselines.py")}
        if (destination / "complete.json").exists():
            if json.loads((destination / "complete.json").read_text())["identity"] != identity:
                raise ValueError("Traditional cache identity changed")
            continue
        start = time.monotonic()
        predictions = predict_hr_windows(np.load(root / "data" / dataset / "x64.npy", mmap_mode="r"), 64)
        np.savez_compressed(destination / "predictions.npz", **predictions)
        write_json(destination / "complete.json", {"identity": identity, "elapsed_seconds": time.monotonic() - start, "coverage": {k: float(np.isfinite(v).mean()) for k, v in predictions.items()}, "sha256": sha256(destination / "predictions.npz")})
        print(f"Traditional {dataset} complete", flush=True)


def run_job(root, job, method, rows, device):
    destination = root / "runs" / job["id"] / method
    identity = result_identity(root, job, method)
    complete = destination / "complete.json"
    if complete.exists():
        previous = json.loads(complete.read_text())
        if previous["identity"] != identity or sha256(destination / "predictions.npz") != previous["predictions_sha256"]:
            raise ValueError("Completed result identity/content changed")
        return
    destination.mkdir(parents=True, exist_ok=True)
    source, target = rows[job["source"]], rows[job["target"]]
    train = np.flatnonzero(np.isin(source["subject"], job["train"]))
    val = np.flatnonzero(np.isin(source["subject"], job["validation"]))
    test = np.flatnonzero(np.isin(target["subject"], job["test"]))
    start = time.monotonic()
    if method == "TrainMean":
        mean = float(np.average(source["hr"][train], weights=weights(source["subject"][train])))
        pred = np.full(len(test), mean)
        metadata = {"train_subject_mean_hr": mean}
    elif method in MODELS:
        x = np.load(root / "features" / job["source"] / f"{method}.npy", mmap_mode="r")
        tx = x if job["source"] == job["target"] else np.load(root / "features" / job["target"] / f"{method}.npy", mmap_mode="r")
        coef, intercept, metadata = fit_ridge(x[train], source["hr"][train], source["subject"][train], x[val], source["hr"][val], source["subject"][val])
        pred = np.asarray(tx[test], dtype=np.float64) @ coef + intercept
        np.savez(destination / "ridge.npz", coef=coef, intercept=intercept)
    elif method == "SmallCNN":
        import torch
        from ppg_hr_cnn import fit_cnn, predict_cnn, cpu_state_dict
        torch.set_num_threads(4)
        x = np.load(root / "data" / job["source"] / "x64.npy", mmap_mode="r")
        tx = x if job["source"] == job["target"] else np.load(root / "data" / job["target"] / "x64.npy", mmap_mode="r")
        model, metadata = fit_cnn(x[train], source["hr"][train], source["subject"][train], val_x=x[val], val_y=source["hr"][val], val_subjects=source["subject"][val], seed=job["seed"], device=device)
        pred = predict_cnn(model, tx[test])
        torch.save(cpu_state_dict(model), destination / "weights.pt")
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        raise ValueError(method)
    if pred.shape != (len(test),) or not np.isfinite(pred).all():
        raise ValueError("Invalid learned model output")
    np.savez_compressed(destination / "predictions.npz", row=test, subject=target["subject"][test], start_s=target["start_s"][test], reference=target["hr"][test], prediction=pred)
    write_json(complete, {"identity": identity, "train_windows": len(train), "validation_windows": len(val), "test_windows": len(test), "subject_mae_bpm": macro_mae(target["hr"][test], pred, target["subject"][test]), "fit": metadata, "elapsed_seconds": time.monotonic() - start, "predictions_sha256": sha256(destination / "predictions.npz")})
    print(f"{job['id']}/{method}: {time.monotonic() - start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / ".local/ppg-hr-poc")
    parser.add_argument("--stage", choices=("register", "traditional", "learned"), default="register")
    parser.add_argument("--methods", nargs="+", default=["TrainMean", *MODELS, "SmallCNN"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--job", help="Run only a specific registered job")
    args = parser.parse_args()
    plan = ensure_protocol(args.root)
    if args.stage == "register":
        print(f"Registered {len(plan['jobs'])} subject splits before new results")
        return
    rows = {d: load_rows(args.root, d) for d in DATASETS}
    with threadpool_limits(limits=4):
        if args.stage == "traditional":
            run_baselines(args.root, rows)
        else:
            for job in plan["jobs"]:
                if args.job and args.job != job["id"]:
                    continue
                for method in args.methods:
                    run_job(args.root, job, method, rows, args.device)


if __name__ == "__main__":
    main()
