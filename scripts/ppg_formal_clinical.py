"""Frozen six-model clinical regression with identical observations and heads.

Run only after prepare_ppg_formal_clinical.py and independent reference audit.
There is deliberately no SpO2 task. Predictions concern manual respiratory
rate and official clinical finger-PPG blood-pressure labels only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import resample_poly
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ppg_formal_hr import sufficient_statistics, projection_from_stats, project, ridge_from_stats, ALPHAS
from scripts.ppg_model_expansion_pilot import MODELS, extract_features, fit_small_head, macro_mae
from scripts.ppg_hr_poc_features import sha256, write_json
from scripts.ppg_sqi_poc import handcrafted, weights

DEFAULT = ROOT / ".local/ppg-formal-expansion/clinical"
SHARED = ROOT / "docs/results/ppg_formal_expansion/clinical"
SEEDS = (42, 43, 44)
RANDOM = "AnyPPG-random-frozen"
METHODS = (*MODELS, RANDOM)
HEADS = ("Native-Ridge", "PCA16-Ridge", "PCA16-MLP")
PROTOCOL = {
    "version": "formal-clinical-v1", "models": list(METHODS), "heads": list(HEADS), "seeds": list(SEEDS),
    "BIDMC": "Fixed five patient folds; next fold validation; 32 s includes 4 nonoverlapping 8 s views, equal mean embeddings",
    "PulseDB": "Fixed official Train/CalFree cohort; 10 s includes 8 s views starting at 0 and 2 s, equal mean embeddings",
    "representation": "Native preprocessing at each model sampling rate; all models observe identical view times",
    "projection": "Training rows only StandardScaler/PCA16/StandardScaler; no label, validation or test fit",
    "ridge": {"alphas": list(ALPHAS), "person_balanced": True, "selection": "validation patient MAE, no train+validation refit"},
    "mlp": {"width": 16, "parameters": 289, "epochs": 40, "learning_rate": .01, "weight_decay": 1e-4,
            "loss": "patient-balanced scaled squared error", "selection": "validation patient MAE"},
    "cnn": {"architecture": "SmallHRCNN", "parameters": 27065, "epochs_max": 20, "patience": 5,
            "learning_rate": .001, "batch_size": 128, "loss": "patient-balanced scaled absolute error",
            "observation": "same 8 s views, per-view z score, mean predictions before loss"},
    "primary": "Six pretrained PCA16-MLP versus SmallCNN and AnyPPG versus same-architecture random PCA16-MLP; Holm per target across seven pairs",
    "traditional_RR": "Four fixed PPG modulation estimates all reported; predeclared Median_fusion is primary conventional comparator, no test-based selection",
    "statistics": "Average absolute error per person and seed, then paired person bootstrap; no windows as independent observations",
    "encoder_training": False, "SpO2_task": False,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def feature_key(method, seed):
    return f"{method}-seed{seed}" if method == RANDOM else method


def observation_views(x, fs=125):
    x = np.asarray(x)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Finite waveform matrix required")
    seconds = x.shape[1] / fs
    if seconds == 32:
        starts = (0, 8, 16, 24)
    elif seconds == 10:
        starts = (0, 2)
    else:
        raise ValueError("Formal clinical inputs must span 10 or 32 seconds")
    return np.stack([x[:, int(start * fs):int((start + 8) * fs)] for start in starts], axis=1)


def load_data(folder):
    report = read_json(folder / "report.json")
    if sha256(folder / "data.npz") != report["data_sha256"] or sha256(folder / "rows.json") != report["rows_sha256"]:
        raise ValueError("Prepared clinical input changed")
    with np.load(folder / "data.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    rows = read_json(folder / "rows.json")
    if len(rows) != len(data["x"]):
        raise ValueError("Input and row counts differ")
    for i, row in enumerate(rows):
        subject = row["subject"] if "subject" in row else row["patient_id"]
        if subject != data["subject"][i] or hashlib.sha256(data["x"][i].tobytes()).hexdigest() != row["ppg_sha256"]:
            raise ValueError("Clinical patient/waveform identity changed")
    return data, report


def register(folder):
    data, report = load_data(folder)
    audit = read_json(SHARED / "clinical_reference_audit.json")
    match = next(c for c in audit["checks"] if c["dataset"] == folder.name)
    if audit["status"] != "passed" or match["data_sha256"] != report["data_sha256"]:
        raise ValueError("Independent reference audit is missing or stale")
    if folder.name == "BIDMC":
        splits = read_json(folder / "splits.json")
    else:
        splits = [{"fold": 0, **{role: sorted(set(data["subject"][data["role"] == role]))
                                  for role in ("train", "validation", "test")}}]
    jobs = [{**split, "seed": seed, "id": f"fold{split['fold']}-seed{seed}"} for split in splits for seed in SEEDS]
    plan = {**PROTOCOL, "dataset": folder.name, "jobs": jobs, "data_sha256": report["data_sha256"],
            "rows_sha256": report["rows_sha256"], "source_preparation_sha256": sha256(folder / "report.json")}
    target = folder / "protocol.json"
    if target.exists() and read_json(target) != plan:
        raise ValueError("Registered clinical protocol changed")
    if not target.exists():
        write_json(target, plan)
    return data, plan


def make_features(folder, methods, device):
    data, plan = register(folder)
    views = observation_views(data["x"])
    stats = np.repeat(data["recording_stats"], views.shape[1], axis=0)
    for method in methods:
        for seed in (SEEDS if method == RANDOM else (42,)):
            name = feature_key(method, seed); dest = folder / "features"; dest.mkdir(exist_ok=True)
            meta, path = dest / f"{name}.json", dest / f"{name}.npy"
            identity = {"protocol_sha256": sha256(folder / "protocol.json"), "data_sha256": plan["data_sha256"],
                        "method": method, "seed": seed if method == RANDOM else None,
                        "extractor_sha256": sha256(ROOT / "scripts/ppg_model_expansion_pilot.py")}
            if meta.exists():
                previous = read_json(meta)
                if previous["identity"] != identity or sha256(path) != previous["features_sha256"]:
                    raise ValueError("Feature cache changed")
                continue
            z, evidence = extract_features("AnyPPG" if method == RANDOM else method,
                                           views.reshape(-1, 1000), 125, stats=stats,
                                           random_init=method == RANDOM, seed=seed, device=device, batch_size=32)
            z = z.reshape(len(views), views.shape[1], -1).mean(axis=1)
            if not np.isfinite(z).all():
                raise ValueError("Nonfinite averaged representation")
            np.save(path, z)
            write_json(meta, {"identity": identity, "shape": list(z.shape), "features_sha256": sha256(path),
                              "encoder": evidence, "views_per_observation": views.shape[1]})
            print(json.dumps({"dataset": folder.name, "features": name, "shape": list(z.shape)}), flush=True)


def load_features(folder, method, seed, data):
    if method == "HandFeatures":
        if "hand_features" in data:
            return data["hand_features"], {"source": "PPG-only pulse/spectral morphology, no ABP or ECG"}
        x = np.column_stack([handcrafted(data["x"], 125), data["traditional"]])
        if not np.isfinite(x).all():
            raise ValueError("Handcrafted representation has failed output; do not silently fill")
        return x, {"source": "PPG-only quality/shape features plus all four respiratory modulation estimates"}
    name = feature_key(method, seed); dest = folder / "features"
    meta = read_json(dest / f"{name}.json")
    expected = {"method": method, "seed": seed if method == RANDOM else None,
                "protocol_sha256": sha256(folder / "protocol.json")}
    if any(meta["identity"].get(key) != value for key, value in expected.items()):
        raise ValueError("Feature protocol/model/initialization identity differs")
    encoder = meta["encoder"]
    if (encoder["encoder_trainable_parameters"] != 0
            or encoder["state_before_sha256"] != encoder["state_after_sha256"]
            or encoder["pretrained_weights_loaded"] != (method != RANDOM)):
        raise ValueError("Frozen/pretrained/random encoder state contract failed")
    if sha256(dest / f"{name}.npy") != meta["features_sha256"]:
        raise ValueError("Feature file changed")
    return np.load(dest / f"{name}.npy"), meta


def indices(subjects, job):
    result = tuple(np.flatnonzero(np.isin(subjects, job[role])) for role in ("train", "validation", "test"))
    sets = [set(subjects[ix]) for ix in result]
    if not all(len(ix) for ix in result) or any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Patient split leakage or empty fold")
    return result


def cnn_predict(model, x, center, scale, batch_size=128):
    import torch
    output = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            part = x[start:start + batch_size]
            output.append(model(part.reshape(-1, 1, 512)).reshape(len(part), part.shape[1]).mean(1).cpu().numpy())
    return np.concatenate(output) * scale + center


def fit_cnn_control(x64, y, subjects, train, val, seed, device):
    import torch
    from scripts.ppg_hr_cnn import SmallHRCNN, normalize_windows
    torch.set_num_threads(2); torch.manual_seed(seed)
    views = observation_views(x64, 64)
    tensor = normalize_windows(torch.tensor(views, dtype=torch.float32).reshape(-1, 1, 512)).reshape(len(views), views.shape[1], 1, 512).to(device)
    center, scale = float(y[train].mean()), max(float(y[train].std()), 1.)
    targets = torch.tensor((y - center) / scale, dtype=torch.float32, device=device)
    weight = torch.tensor(weights(subjects[train]), dtype=torch.float32, device=device)
    model = SmallHRCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rng = np.random.default_rng(seed); best = (float("inf"), None, 0); history = []
    for epoch in range(1, 21):
        model.train(); order = rng.permutation(len(train))
        for start in range(0, len(train), 128):
            local = order[start:start + 128]; selected = train[local]; part = tensor[selected]
            optimizer.zero_grad(set_to_none=True)
            pred = model(part.reshape(-1, 1, 512)).reshape(len(part), part.shape[1]).mean(1)
            loss = ((pred - targets[selected]).abs() * weight[local]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite CNN loss")
            loss.backward(); optimizer.step()
        model.eval()
        score = macro_mae(y[val], cnn_predict(model, tensor[val], center, scale), subjects[val])
        history.append({"epoch": epoch, "validation_mae": score})
        if score < best[0]:
            best = (score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch)
        if epoch - best[2] >= 5:
            break
    model.load_state_dict(best[1]); model.eval()
    predictions = cnn_predict(model, tensor, center, scale)
    details = {"selected_epoch": best[2], "target_center": center, "target_scale": scale,
               "history": history, "parameters": sum(p.numel() for p in model.parameters()), "seed": seed}
    return predictions, details, best[1]


def publish(folder, dest, job, method, head, target, data, y, test, predictions, fit):
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "predictions.npz"
    if not np.isfinite(predictions).all() or len(predictions) != len(test):
        raise ValueError("Invalid clinical predictions")
    np.savez_compressed(path, row=test, subject=data["subject"][test], reference=y[test], prediction=predictions)
    report = {"dataset": folder.name, "target": target, "job": job, "method": method, "head": head, "fit": fit,
              "test_patient_mae": macro_mae(y[test], predictions, data["subject"][test]),
              "protocol_sha256": sha256(folder / "protocol.json"), "prediction_sha256": sha256(path),
              "runner_sha256": sha256(Path(__file__))}
    write_json(dest / "complete.json", report)


def completed(dest, folder, job):
    path = dest / "complete.json"
    if not path.exists():
        return False
    meta = read_json(path)
    if meta["protocol_sha256"] != sha256(folder / "protocol.json") or meta["job"] != job or sha256(dest / "predictions.npz") != meta["prediction_sha256"]:
        raise ValueError("Completed output identity changed")
    return True


def fit_heads(folder, methods, controls, cnn, device):
    import torch
    torch.set_num_threads(2)
    data, plan = register(folder)
    subjects = data["subject"]
    targets = {"resp_manual": data["y"]} if folder.name == "BIDMC" else {"sbp": data["y"][:, 0], "dbp": data["y"][:, 1]}
    cache = {}
    x64 = resample_poly(data["x"], 64, 125, axis=-1).astype(np.float32) if cnn else None
    for job in plan["jobs"]:
        train, val, test = indices(subjects, job)
        for target, y in targets.items():
            for method in (*methods, *(('HandFeatures',) if controls else ())):
                key = (method, job["seed"] if method == RANDOM else 42)
                if key not in cache:
                    cache[key] = load_features(folder, method, job["seed"], data)
                x, evidence = cache[key]
                stats = sufficient_statistics(x, y, subjects)
                projection = projection_from_stats(stats, job["train"], 16)
                z = project(x, projection)
                for head in HEADS:
                    dest = folder / "runs" / target / job["id"] / method / head
                    if completed(dest, folder, job):
                        continue
                    dest.mkdir(parents=True, exist_ok=True)
                    np.savez(dest / "projection.npz", **projection)
                    if head == "PCA16-MLP":
                        model, details = fit_small_head(z[train], y[train], subjects[train], z[val], y[val], subjects[val], seed=job["seed"], epochs=40)
                        with torch.inference_mode():
                            pred = model(torch.tensor(z[test], dtype=torch.float32)).squeeze(-1).numpy() * details["target_scale"] + details["target_center"]
                        torch.save(model.state_dict(), dest / "model.pt")
                    else:
                        features = x if head == "Native-Ridge" else z
                        new_stats = stats if head == "Native-Ridge" else sufficient_statistics(z, y, subjects)
                        coef, intercept, details = ridge_from_stats(new_stats, job["train"], features[val], y[val], subjects[val])
                        pred = features[test] @ coef + intercept
                        np.savez(dest / "model.npz", coef=coef, intercept=intercept)
                    details["feature_evidence"] = evidence
                    publish(folder, dest, job, method, head, target, data, y, test, pred, details)
            if controls:
                dest = folder / "runs" / target / job["id"] / "TrainMean"
                if not completed(dest, folder, job):
                    mean = float(np.average(y[train], weights=weights(subjects[train])))
                    publish(folder, dest, job, "TrainMean", "", target, data, y, test, np.full(len(test), mean), {"training_patient_mean": mean})
                if folder.name == "BIDMC":
                    for col, method in enumerate(("Baseline_modulation", "Amplitude_modulation", "Interval_modulation", "Median_fusion")):
                        dest = folder / "runs" / target / job["id"] / method
                        if not completed(dest, folder, job):
                            publish(folder, dest, job, method, "", target, data, y, test, data["traditional"][test, col], {"fixed_algorithm": True})
            if cnn:
                dest = folder / "runs" / target / job["id"] / "SmallCNN"
                if not completed(dest, folder, job):
                    prediction, details, state = fit_cnn_control(x64, y, subjects, train, val, job["seed"], device)
                    dest.mkdir(parents=True, exist_ok=True); torch.save(state, dest / "model.pt")
                    publish(folder, dest, job, "SmallCNN", "", target, data, y, test, prediction[test], details)
            print(json.dumps({"dataset": folder.name, "target": target, "job": job["id"], "status": "complete_selected_methods"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--datasets", nargs="+", choices=["BIDMC", "PulseDB"], default=["BIDMC", "PulseDB"])
    parser.add_argument("--stage", choices=["register", "features", "heads"], default="register")
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=list(METHODS))
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--cnn", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        for dataset in args.datasets:
            folder = args.root / dataset
            if args.stage == "register":
                _, protocol = register(folder)
                print(json.dumps({"dataset": dataset, "jobs": len(protocol["jobs"])}))
            elif args.stage == "features":
                make_features(folder, args.methods, args.device)
            else:
                fit_heads(folder, args.methods, args.controls, args.cnn, args.device)


if __name__ == "__main__":
    main()
