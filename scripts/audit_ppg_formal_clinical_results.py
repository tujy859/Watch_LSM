"""Reload every formal clinical prediction and summarize paired patient errors."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import resample_poly
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ppg_formal_clinical import (
    DEFAULT, SHARED, METHODS, MODELS, RANDOM, HEADS, SEEDS, ALPHAS, register,
    load_features, indices, observation_views, cnn_predict, read_json, project,
)
from scripts.ppg_hr_poc_features import sha256, write_json
from scripts.ppg_sqi_poc import weights


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def paired_interval(values):
    x = np.asarray(values, float)
    rng = np.random.default_rng(20261105)
    sample = x[rng.integers(0, len(x), size=(10000, len(x)))].mean(1)
    return np.quantile(sample, [.025, .975]).tolist()


def signflip_p(values):
    x = np.asarray(values, float)
    rng = np.random.default_rng(20261105)
    hits, trials = 0, 100000
    for _ in range(trials // 1000):
        signs = rng.integers(0, 2, size=(1000, len(x))) * 2 - 1
        hits += int((np.abs(signs @ x) >= abs(x.sum()) - 1e-12).sum())
    return (hits + 1) / (trials + 1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_dataset(folder):
    import torch
    from scripts.ppg_hr_cnn import SmallHRCNN, normalize_windows
    torch.set_num_threads(2)
    data, plan = register(folder); subjects = data["subject"]
    targets = {"resp_manual": data["y"]} if folder.name == "BIDMC" else {"sbp": data["y"][:, 0], "dbp": data["y"][:, 1]}
    feature_cache, rows, maxima, refits = {}, [], defaultdict(float), []
    views = observation_views(resample_poly(data["x"], 64, 125, axis=-1).astype(np.float32), 64)
    tensor = normalize_windows(torch.tensor(views).reshape(-1, 1, 512)).reshape(len(views), views.shape[1], 1, 512)
    checks = 0
    for job in plan["jobs"]:
        train, val, test = indices(subjects, job)
        for target, y in targets.items():
            pairs = [(method, head) for method in (*METHODS, "HandFeatures") for head in HEADS]
            pairs += [("TrainMean", ""), ("SmallCNN", "")]
            if folder.name == "BIDMC":
                pairs += [(m, "") for m in ("Baseline_modulation", "Amplitude_modulation", "Interval_modulation", "Median_fusion")]
            for method, head in pairs:
                dest = folder / "runs" / target / job["id"] / method
                if head:
                    dest /= head
                meta = read_json(dest / "complete.json")
                require(meta["job"] == job and meta["protocol_sha256"] == sha256(folder / "protocol.json"), "Output protocol mismatch")
                require(sha256(dest / "predictions.npz") == meta["prediction_sha256"], "Prediction hash mismatch")
                with np.load(dest / "predictions.npz", allow_pickle=False) as saved:
                    require(np.array_equal(saved["row"], test), "Test row identity mismatch")
                    require(np.array_equal(saved["subject"], subjects[test]), "Test patient identity mismatch")
                    require(np.array_equal(saved["reference"], y[test]), "Test label mismatch")
                    pred = saved["prediction"].copy()
                require(np.isfinite(pred).all(), "Nonfinite predictions")
                if head:
                    key = (method, job["seed"] if method == RANDOM else 42)
                    if key not in feature_cache:
                        # Match the production sufficient-statistics precision.
                        # sklearn otherwise keeps float32 embeddings in a poorly
                        # conditioned dual/primal solve for the smallest alphas.
                        feature_cache[key] = np.asarray(load_features(folder, method, job["seed"], data)[0], dtype=np.float64)
                    x = feature_cache[key]
                    with np.load(dest / "projection.npz") as projection:
                        transform = {key: projection[key] for key in projection.files}
                    z = x if head == "Native-Ridge" else project(x, transform)
                    if head == "PCA16-MLP":
                        model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
                        model.load_state_dict(torch.load(dest / "model.pt", map_location="cpu", weights_only=True)); model.eval()
                        with torch.inference_mode():
                            recalculated = model(torch.tensor(z[test], dtype=torch.float32)).squeeze(-1).numpy() * meta["fit"]["target_scale"] + meta["fit"]["target_center"]
                    else:
                        with np.load(dest / "model.npz") as model:
                            recalculated = z[test] @ model["coef"] + model["intercept"]
                    # A real-data independent direct sklearn check on three
                    # encoders and both linear heads per target, first job only.
                    if job["id"] == "fold0-seed42" and method in ("AnyPPG", "PaPaGei-P", RANDOM) and head != "PCA16-MLP":
                        if head == "PCA16-Ridge":
                            first = StandardScaler().fit(x[train])
                            pc = PCA(16, svd_solver="full").fit(first.transform(x[train]))
                            last = StandardScaler().fit(pc.transform(first.transform(x[train])))
                            direct = last.transform(pc.transform(first.transform(x)))
                            signs = np.sign((direct[train] * z[train]).sum(0))
                            error = float(np.max(np.abs(direct[train] * signs - z[train])))
                            require(error < 2e-4, "PCA coordinates differ from independent fit")
                        else:
                            direct = x
                        w = weights(subjects[train])
                        scaler = StandardScaler().fit(direct[train], sample_weight=w)
                        choices = []
                        for alpha in ALPHAS:
                            regression = Ridge(alpha=alpha).fit(scaler.transform(direct[train]), y[train], sample_weight=w)
                            validation = regression.predict(scaler.transform(direct[val]))
                            score = np.mean([np.mean(np.abs(y[val][subjects[val] == s] - validation[subjects[val] == s])) for s in np.unique(subjects[val])])
                            choices.append((score, alpha, regression))
                        _, alpha, regression = min(choices, key=lambda item: item[0])
                        prediction = regression.predict(scaler.transform(direct[test]))
                        difference = float(np.max(np.abs(prediction - pred)))
                        require(alpha == meta["fit"]["alpha"] and difference < 2e-4, "Independent Ridge mismatch")
                        refits.append({"target": target, "model": method, "head": head, "alpha": alpha,
                                       "refit_precision": "float64", "max_prediction_difference": difference})
                elif method == "SmallCNN":
                    model = SmallHRCNN()
                    model.load_state_dict(torch.load(dest / "model.pt", map_location="cpu", weights_only=True)); model.eval()
                    recalculated = cnn_predict(model, tensor[test], meta["fit"]["target_center"], meta["fit"]["target_scale"])
                elif method == "TrainMean":
                    value = float(np.mean([y[train][subjects[train] == s].mean() for s in np.unique(subjects[train])]))
                    recalculated = np.full(len(test), value)
                else:
                    col = ("Baseline_modulation", "Amplitude_modulation", "Interval_modulation", "Median_fusion").index(method)
                    recalculated = data["traditional"][test, col]
                difference = float(np.max(np.abs(recalculated - pred)))
                maxima[method] = max(maxima[method], difference)
                require(difference < (1e-3 if method == "SmallCNN" else 1e-6), "Saved model does not reproduce predictions")
                losses = []
                for patient in np.unique(subjects[test]):
                    mask = subjects[test] == patient
                    error = pred[mask] - y[test][mask]
                    mae = float(np.mean(np.abs(error))); losses.append(mae)
                    rows.append({"dataset": folder.name, "target": target, "method": method, "head": head,
                                 "seed": job["seed"], "fold": job["fold"], "subject": patient,
                                 "mae": mae, "bias": float(error.mean()), "error_sd": float(error.std(ddof=1)) if len(error) > 1 else 0.,
                                 "windows": int(mask.sum()), "coverage": 1.})
                require(abs(float(np.mean(losses)) - meta["test_patient_mae"]) < 1e-10, "Published patient metric mismatch")
                checks += 1
    return rows, {"dataset": folder.name, "completed_outputs": checks, "maximum_model_reload_errors": dict(maxima),
                  "independent_real_data_ridge_refits": refits, "input_patients": len(set(subjects)),
                  "data_sha256": sha256(folder / "data.npz"), "protocol_sha256": sha256(folder / "protocol.json")}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("dataset", "target", "method", "head", "subject")); groups[key].append(row)
    participants, measures = [], defaultdict(dict)
    for key, entries in groups.items():
        require(len(entries) == 3 and {r["seed"] for r in entries} == set(SEEDS), "Incomplete seed replicates")
        entry = dict(zip(("dataset", "target", "method", "head", "subject"), key))
        entry.update(mae=float(np.mean([r["mae"] for r in entries])), bias=float(np.mean([r["bias"] for r in entries])),
                     windows=entries[0]["windows"], coverage=1.)
        participants.append(entry); measures[key[:-1]][key[-1]] = entry["mae"]
    summary = []
    for key, people in measures.items():
        low, high = paired_interval(list(people.values()))
        row = dict(zip(("dataset", "target", "method", "head"), key))
        row.update(mae=float(np.mean(list(people.values()))), ci_low=low, ci_high=high, patients=len(people), coverage=1.,
                   unit="breaths/min" if key[1] == "resp_manual" else "mmHg")
        summary.append(row)
    pairs = []
    for dataset, target in sorted({key[:2] for key in measures}):
        family = []
        for method, control, control_head in [*((m, "SmallCNN", "") for m in MODELS), ("AnyPPG", RANDOM, "PCA16-MLP")]:
            a = measures[(dataset, target, method, "PCA16-MLP")]; b = measures[(dataset, target, control, control_head)]
            require(a.keys() == b.keys(), "Paired patient mismatch")
            delta = [a[p] - b[p] for p in sorted(a)]; low, high = paired_interval(delta)
            family.append({"dataset": dataset, "target": target, "method": method, "head": "PCA16-MLP", "control": control,
                           "control_head": control_head, "delta_mae": float(np.mean(delta)), "ci_low": low, "ci_high": high,
                           "patients": len(delta), "p_signflip_montecarlo": signflip_p(delta)})
        order = np.argsort([r["p_signflip_montecarlo"] for r in family]); running = 0.
        for rank, index in enumerate(order):
            running = max(running, min(1., (len(family) - rank) * family[index]["p_signflip_montecarlo"]))
            family[index]["p_holm_within_target"] = running
        pairs.extend(family)
    save_csv(SHARED / "per_patient.csv", participants); save_csv(SHARED / "summary.csv", summary)
    save_csv(SHARED / "paired_primary.csv", pairs)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--datasets", nargs="+", choices=["BIDMC", "PulseDB"], default=["BIDMC", "PulseDB"])
    args = parser.parse_args()
    rows, checks, features, artifacts = [], [], [], []
    with threadpool_limits(limits=2):
        for dataset in args.datasets:
            r, c = audit_dataset(args.root / dataset); rows.extend(r); checks.append(c)
            print(json.dumps(c), flush=True)
            folder = args.root / dataset
            for meta_path in sorted((folder / "features").glob("*.json")):
                features.append({"dataset": dataset, "name": meta_path.stem, **read_json(meta_path)})
            for complete in sorted((folder / "runs").rglob("complete.json")):
                artifacts.append({"path": complete.relative_to(ROOT).as_posix(), "sha256": sha256(complete),
                                  "predictions_sha256": sha256(complete.parent / "predictions.npz")})
        summarize(rows)
    write_json(SHARED / "feature_provenance.json", features)
    write_json(SHARED / "prediction_manifest.json", artifacts)
    write_json(SHARED / "results_audit.json", {"status": "passed", "checks": checks,
               "code_sha256": sha256(Path(__file__)), "inference": {
                   "unit": "patient, three seed-specific errors averaged before comparison",
                   "intervals": "10000 paired patient bootstrap draws, unadjusted 95% percentile intervals",
                   "p_values": "100000 two-sided patient sign flips; Holm within target across seven predeclared pairs",
                   "limits": "Fixed cohort/folds, possible pretraining exposure, finger PPG; no clinical device claim"}})


if __name__ == "__main__":
    main()
