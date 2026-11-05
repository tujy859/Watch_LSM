"""Small, explicitly nonconfirmatory model/head integration check.

Run with the existing .local/ppg-bench-venv Python. Encoders are frozen.
The CLI deliberately caps data and epochs: it is not a full benchmark runner.
Five PPG checkpoint variants, MOMENT, and a fresh random AnyPPG architecture
share the same subject/window identities. Standardization and PCA see training
features only. The two heads consume the same 16 coordinates.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.signal import resample_poly
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from scripts.ppg_hr_poc_features import UPSTREAM_COMMIT, sha256, write_json

MODELS = ("AnyPPG", "PaPaGei-S", "PaPaGei-P", "PulsePPG", "SigmaPPG", "MOMENT-1-small")
METHODS = (*MODELS, "AnyPPG-random-frozen")
PCA_DIM = 16


def macro_mae(y, prediction, subjects):
    """Keep CPU-only head checks independent of the optional data checkout."""
    return float(np.mean([np.mean(np.abs(y[subjects == s] - prediction[subjects == s]))
                          for s in np.unique(subjects)]))


def _upstream():
    upstream = ROOT / "repos/ppg-bench"
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError("Pinned PPG-Bench checkout changed")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from ppg_bench.models import load_models
    configs = {c["name"]: c for c in load_models(ROOT / "watch_lsm/configs/ppg_bench_models.yaml")}
    return configs, revision


def _state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _weights(config):
    checkpoint = Path(config["init"]["ckpt_path"])
    files = [checkpoint] if checkpoint.is_file() else sorted(p for p in checkpoint.rglob("*") if p.is_file())
    actual = {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
    expected = {}
    for manifest_name in ("model_assets.json", "ppg_bench_assets.json"):
        manifest = json.loads((ROOT / "data_manifests" / manifest_name).read_text(encoding="utf-8"))
        def walk(value):
            if isinstance(value, dict):
                if "path" in value and "sha256" in value:
                    expected[value["path"]] = value["sha256"]
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(manifest)
    for path, digest in actual.items():
        if path in expected and expected[path] != digest:
            raise ValueError(f"Checkpoint hash differs from registered asset: {path}")
    if not actual or not any(path in expected for path in actual):
        raise ValueError("No checkpoint file is covered by the registered asset manifests")
    return actual


def extract_features(model_name, windows, fs, stats=None, random_init=False,
                     device="cuda", batch_size=16, seed=42):
    """Return frozen native features and auditable metadata for small arrays.

    ``windows`` are raw single-channel PPG [N,T] at integer ``fs`` Hz. Native
    model filtering/normalization is preserved. PulsePPG requires explicitly
    supplied [N,2] unlabeled recording mean/std; callers document its scope.
    MOMENT accepts exactly 8 seconds. SigmaPPG's native trailing partial-patch
    removal is reported, so callers cannot assume that no samples were lost.
    Random initialization is supported only for a fresh official AnyPPG Net1D.
    """
    if model_name not in MODELS or (random_init and model_name != "AnyPPG"):
        raise ValueError("Unknown model, or unsupported random architecture")
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or int(fs) != fs or fs <= 0:
        raise ValueError("Require finite nonempty [N,T] windows and positive integer fs")
    if batch_size < 1 or batch_size > 32:
        raise ValueError("Pilot batch_size must be between 1 and 32")
    configs, revision = _upstream()
    config = configs[model_name]
    target_fs = config["fs"]
    if int(fs) != target_fs:
        x = resample_poly(x, target_fs, int(fs), axis=-1).astype(np.float32)
    if model_name == "MOMENT-1-small" and x.shape[1] != 512:
        raise ValueError("MOMENT pilot requires an 8-second input, without implicit padding")
    if model_name == "PulsePPG" and stats is None:
        raise ValueError("PulsePPG requires explicit unlabeled recording mean/std")
    if stats is None:
        stats = np.column_stack([x.mean(axis=1), x.std(axis=1)])
    stats = np.asarray(stats, dtype=np.float32)
    if stats.shape != (len(x), 2) or not np.isfinite(stats).all() or (stats[:, 1] < 0).any():
        raise ValueError("Invalid recording normalization statistics")
    import torch
    from ppg_bench.models import external_module, get_port
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    torch.set_num_threads(4)
    cuda_devices = [torch.cuda.current_device()] if str(device).startswith("cuda") else []
    started = time.monotonic()
    port = get_port(config)
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if random_init:
            from ppg_bench.encoders.anyppg import ANYPPG_CFG
            source = external_module(config["init"]["source_root"], "load_anyppg.resnet1d")
            encoder = source.Net1D(**ANYPPG_CFG)
            weights = {}
        else:
            weights = _weights(config)
            encoder = port.load_encoder()
    encoder.eval().requires_grad_(False).to(device)
    before = _state_hash(encoder)
    native_parameters = sum(p.numel() for p in encoder.parameters())
    output = []
    for start in range(0, len(x), batch_size):
        stop = start + batch_size
        batch = {"x": torch.tensor(x[start:stop]),
                 "ppg_mean": torch.tensor(stats[start:stop, 0]),
                 "ppg_std": torch.tensor(stats[start:stop, 1])}
        with torch.inference_mode():
            features = encoder(port.prepare_batch(batch, config, device))
        if not isinstance(features, torch.Tensor):
            raise ValueError("Encoder did not return a tensor")
        output.append(features.detach().float().cpu().numpy())
    output = np.concatenate(output)
    if output.shape != (len(x), config["embedding_dim"]) or not np.isfinite(output).all():
        raise ValueError("Unexpected encoder feature shape/nonfinite output")
    after = _state_hash(encoder)
    if before != after or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Frozen encoder state changed during inference")
    metadata = {"model": model_name, "random_init": bool(random_init), "seed": seed,
                "pretrained_weights_loaded": not random_init,
                "random_initialization": "Fresh official Net1D constructor, including fresh BatchNorm buffers; no checkpoint read" if random_init else None,
                "shape": list(output.shape), "native_fs": target_fs, "native_input_samples": x.shape[1],
                "native_used_samples": x.shape[1] // 100 * 100 if model_name == "SigmaPPG" else x.shape[1],
                "encoder_parameters": native_parameters, "encoder_trainable_parameters": 0,
                "state_before_sha256": before, "state_after_sha256": after,
                "checkpoint_sha256": weights, "upstream_commit": revision,
                "native_port": config["port"], "device": str(device), "precision": "float32; no autocast",
                "native_input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                "record_statistics_sha256": hashlib.sha256(stats.tobytes()).hexdigest(),
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
                "elapsed_seconds": time.monotonic() - started}
    del encoder, port
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return output, metadata


def select_pilot_rows(rows, windows_per_subject=24):
    """Choose identities only: first six sorted people, evenly spaced windows.

    No labels or predictions participate in sample selection. This intentionally
    fixed small sample is a plumbing check, not a population sample estimate.
    """
    if not 1 <= windows_per_subject <= 48:
        raise ValueError("Pilot permits 1-48 windows per person only")
    subjects = np.asarray(rows["subject"]).astype(str)
    starts = np.asarray(rows["start_s"])
    keys = list(zip(subjects.tolist(), starts.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate subject/start identity")
    people = sorted(set(subjects))[:6]
    if len(people) != 6:
        raise ValueError("Pilot needs six distinct subjects")
    split = {"train": people[:3], "validation": people[3:4], "test": people[4:6]}
    selected = []
    for subject in people:
        indices = np.flatnonzero(subjects == subject)
        indices = indices[np.argsort(starts[indices], kind="stable")]
        take = np.linspace(0, len(indices) - 1, min(windows_per_subject, len(indices)), dtype=int)
        selected.extend(indices[take].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    roles = np.asarray([next(role for role, ids in split.items() if subject in ids)
                        for subject in subjects[selected]])
    validate_roles(subjects[selected], roles)
    return selected, roles, split


def validate_roles(subjects, roles):
    subjects, roles = np.asarray(subjects), np.asarray(roles)
    if subjects.ndim != 1 or subjects.shape != roles.shape:
        raise ValueError("Subject and split arrays must be equal vectors")
    if set(roles.tolist()) != {"train", "validation", "test"}:
        raise ValueError("All three train/validation/test splits are required")
    sets = {role: set(subjects[roles == role].tolist()) for role in set(roles.tolist())}
    if any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Subject leakage across train/validation/test")


def fit_projection(train_x, n_components=PCA_DIM):
    """Fit all feature transforms on training people only; no validation input."""
    train_x = np.asarray(train_x, dtype=np.float64)
    if train_x.ndim != 2 or not np.isfinite(train_x).all() or min(train_x.shape) <= n_components:
        raise ValueError("Need finite training features with sufficient rows/dimensions for PCA")
    scale = StandardScaler().fit(train_x)
    pca = PCA(n_components=n_components, svd_solver="full", whiten=False).fit(scale.transform(train_x))
    pca_scale = StandardScaler().fit(pca.transform(scale.transform(train_x)))
    return scale, pca, pca_scale


def project(x, transforms):
    scale, pca, pca_scale = transforms
    return pca_scale.transform(pca.transform(scale.transform(np.asarray(x, dtype=np.float64))))


def fit_small_head(train_x, train_y, train_subjects, val_x, val_y, val_subjects,
                   seed=42, epochs=40):
    """CPU 16->16->1 MLP, 289 parameters; select epoch by validation MAE."""
    import torch
    if not 1 <= epochs <= 40:
        raise ValueError("Pilot MLP permits at most 40 epochs")
    if set(train_subjects) & set(val_subjects):
        raise ValueError("Training and validation subject overlap")
    arrays = [np.asarray(v) for v in (train_x, train_y, val_x, val_y)]
    if not all(np.isfinite(v).all() for v in arrays) or train_x.shape[1] != PCA_DIM or val_x.shape[1] != PCA_DIM:
        raise ValueError("Require finite 16-dimensional feature matrices and targets")
    center, target_scale = float(np.mean(train_y)), max(float(np.std(train_y)), 1.)
    tx, ty = torch.tensor(train_x, dtype=torch.float32), torch.tensor((train_y-center)/target_scale, dtype=torch.float32)
    vx = torch.tensor(val_x, dtype=torch.float32)
    _, inverse, counts = np.unique(train_subjects, return_inverse=True, return_counts=True)
    w = torch.tensor(len(train_subjects) / (len(counts) * counts[inverse]), dtype=torch.float32)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(PCA_DIM, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        initial = _state_hash(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        best_score, best_state, best_epoch = float("inf"), None, None
        history = []
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(tx).squeeze(-1)-ty).square()*w).mean()
            loss.backward()
            optimizer.step()
            with torch.inference_mode():
                prediction = model(vx).squeeze(-1).numpy()*target_scale+center
            score = macro_mae(val_y, prediction, np.asarray(val_subjects))
            if not np.isfinite(score):
                raise ValueError("Nonfinite validation score")
            history.append({"epoch": epoch, "train_scaled_mse": float(loss.item()), "validation_mae_bpm": score})
            if score < best_score:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
    model.eval()
    return model, {"parameters": sum(p.numel() for p in model.parameters()), "epochs_executed": epochs,
                   "selected_epoch": best_epoch, "validation_mae_bpm": best_score, "seed": seed,
                   "target_center": center, "target_scale": target_scale,
                   "initial_state_sha256": initial, "selected_state_sha256": _state_hash(model), "history": history}


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    from scripts.run_ppg_hr_poc import fit_ridge
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/ppg-expansion-pilot/models")
    parser.add_argument("--shared", type=Path, default=ROOT / "docs/results/ppg_expansion_pilot")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--windows-per-person", type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.shared.mkdir(parents=True, exist_ok=True)
    configs, revision = _upstream()
    plan = {"purpose": "small-sample integration only; not an efficacy benchmark", "seed": 42,
            "models": list(METHODS), "subjects_per_dataset": {"training": 3, "validation": 1, "testing": 2},
            "windows_per_person": args.windows_per_person, "window_seconds": 8,
            "heads": {"PCA16-Ridge": "17 coefficients; validation chooses alpha from 0.1/1/10/100",
                      "PCA16-MLP": "16->16->1, 289 trainable parameters, Adam 0.01, at most 40 epochs; validation selects epoch"},
            "projection": "input standardization, PCA16 and output standardization fit only on training features; shared between heads",
            "full_evaluation_started": False, "upstream_commit": revision,
            "selection": "first six sorted IDs; evenly spaced windows using subject/start identity only",
            "pulse_normalization": "existing whole-subject unlabeled recording statistics, including target record; offline evaluation",
            "data": {}}
    datasets = {}
    for dataset in ("Dalia", "WildPPG"):
        source = ROOT / ".local/ppg-hr-poc/data" / dataset
        with np.load(source / "rows.npz", allow_pickle=False) as f:
            rows = {k: f[k] for k in f.files}
        selected, roles, split = select_pilot_rows(rows, args.windows_per_person)
        datasets[dataset] = (source, rows, selected, roles)
        plan["data"][dataset] = {"subjects": split, "selected_rows": selected.tolist(),
                                 "rows_sha256": sha256(source / "rows.npz"),
                                 "preparation_manifest_sha256": sha256(source / "manifest.json")}
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Pilot plan changed; choose a new output directory")
    write_json(plan_path, plan)
    import torch
    from scripts.ppg_hr_baselines import predict_hr_windows
    from scripts.ppg_hr_cnn import fit_cnn, predict_cnn
    torch.set_num_threads(4)
    scores, evidence = [], []
    started = time.monotonic()
    with threadpool_limits(limits=4):
        for dataset, (source, rows, selected, roles) in datasets.items():
            destination = args.output / dataset
            destination.mkdir(exist_ok=True)
            y, subjects = rows["hr"][selected].astype(float), rows["subject"][selected]
            train, val, test = (roles == role for role in ("train", "validation", "test"))
            np.savez_compressed(destination / "rows.npz", row=selected, subject=subjects,
                                start_s=rows["start_s"][selected], role=roles, reference=y)
            for name in METHODS:
                random_init = name.endswith("-random-frozen")
                model_name = "AnyPPG" if random_init else name
                fs = configs[model_name]["fs"]
                wave = np.load(source / f"x{fs}.npy", mmap_mode="r")[selected]
                stats = np.load(source / f"stats{fs}.npy", mmap_mode="r")[selected]
                features, info = extract_features(model_name, wave, fs, stats, random_init, args.device)
                np.save(destination / f"{name}-features.npy", features)
                info["dataset"] = dataset
                info["features_sha256"] = sha256(destination / f"{name}-features.npy")
                evidence.append(info)
                transforms = fit_projection(features[train])
                z = project(features, transforms)
                scale, pca, output_scale = transforms
                np.savez(destination / f"{name}-projection.npz", mean=scale.mean_, scale=scale.scale_,
                         pca_mean=pca.mean_, components=pca.components_, output_mean=output_scale.mean_, output_scale=output_scale.scale_)
                coef, intercept, ridge_info = fit_ridge(z[train], y[train], subjects[train], z[val], y[val], subjects[val], alphas=(0.1, 1., 10., 100.))
                ridge_prediction = z[test] @ coef + intercept
                np.savez(destination / f"{name}-ridge.npz", coef=coef, intercept=intercept)
                mlp, mlp_info = fit_small_head(z[train], y[train], subjects[train], z[val], y[val], subjects[val])
                with torch.inference_mode():
                    mlp_prediction = mlp(torch.tensor(z[test], dtype=torch.float32)).squeeze(-1).numpy()*mlp_info["target_scale"]+mlp_info["target_center"]
                torch.save(mlp.state_dict(), destination / f"{name}-mlp.pt")
                for head, prediction, metadata, count in (("PCA16-Ridge", ridge_prediction, ridge_info, 17), ("PCA16-MLP", mlp_prediction, mlp_info, 289)):
                    path = destination / f"{name}-{head}-prediction.npz"
                    if not np.isfinite(prediction).all():
                        raise ValueError("Nonfinite prediction")
                    np.savez_compressed(path, row=selected[test], subject=subjects[test], reference=y[test], prediction=prediction)
                    write_json(destination / f"{name}-{head}.json", metadata)
                    scores.append({"dataset": dataset, "method": name, "head": head, "head_parameters": count,
                                   "train_people": 3, "validation_people": 1, "test_people": 2,
                                   "train_windows": int(train.sum()), "validation_windows": int(val.sum()), "test_windows": int(test.sum()),
                                   "test_mae_bpm": macro_mae(y[test], prediction, subjects[test]), "coverage": 1.,
                                   "prediction_sha256": sha256(path), "status": "pilot_only_not_for_model_ranking"})
                print(json.dumps({"dataset": dataset, "model": name, "status": "two_heads_passed", "elapsed_seconds": round(time.monotonic()-started, 1)}), flush=True)
            x64 = np.load(source / "x64.npy", mmap_mode="r")[selected]
            cnn, cnn_info = fit_cnn(x64[train], y[train], subjects[train], val_x=x64[val], val_y=y[val],
                                    val_subjects=subjects[val], seed=42, device="cpu", max_epochs=5, patience=5, batch_size=24)
            cnn_prediction = predict_cnn(cnn, x64[test])
            torch.save(cnn.state_dict(), destination / "SmallCNN.pt")
            write_json(destination / "SmallCNN.json", cnn_info)
            controls = {"SmallCNN": cnn_prediction, **predict_hr_windows(x64[test], 64),
                        "TrainMean": np.full(test.sum(), np.mean(y[train]))}
            for name, prediction in controls.items():
                valid = np.isfinite(prediction)
                path = destination / f"{name}-prediction.npz"
                np.savez_compressed(path, row=selected[test], subject=subjects[test], reference=y[test], prediction=prediction)
                scores.append({"dataset": dataset, "method": name, "head": "task_control", "head_parameters": 27065 if name == "SmallCNN" else 0,
                               "train_people": 3 if name in ("SmallCNN", "TrainMean") else 0, "validation_people": 1 if name == "SmallCNN" else 0,
                               "test_people": 2, "train_windows": int(train.sum()) if name in ("SmallCNN", "TrainMean") else 0,
                               "validation_windows": int(val.sum()) if name == "SmallCNN" else 0, "test_windows": int(test.sum()),
                               "test_mae_bpm": macro_mae(y[test][valid], prediction[valid], subjects[test][valid]), "coverage": float(valid.mean()),
                               "prediction_sha256": sha256(path), "status": "pilot_only_not_for_model_ranking"})
    _write_csv(args.shared / "models_pilot_scores.csv", scores)
    report = {"status": "passed", "purpose": plan["purpose"], "full_evaluation_started": False,
              "plan": plan, "feature_runs": evidence, "prediction_runs": len(scores), "all_frozen_states_unchanged": True,
              "all_subject_splits_disjoint": True, "new_downloads": 0, "elapsed_seconds": time.monotonic()-started,
              "script_sha256": sha256(Path(__file__)),
              "limitations": ["Two test people and one seed per dataset; scores are workflow checks, not model ranking.",
                              "SmallCNN trained for only five epochs; no efficacy comparison is justified.",
                              "Random frozen AnyPPG is an architecture/feature control, not a trained supervised baseline.",
                              "PCA fixes head input dimension but can discard useful information; native embedding widths and preprocessing differ.",
                              "Existing complete unlabeled subject statistics are retained for PulsePPG; this is offline."]}
    write_json(args.shared / "models_pilot.json", report)
    print(json.dumps({"status": "passed", "prediction_runs": len(scores), "elapsed_seconds": round(time.monotonic()-started, 1)}), flush=True)


if __name__ == "__main__":
    main()
