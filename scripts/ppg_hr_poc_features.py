"""Frozen native PPG-Bench features for the wrist HR PoC, with resumable files.

Run with .local/ppg-bench-venv/Scripts/python.exe. No model is fitted here.
Published .npy files are complete; partial outputs have a separate suffix.
"""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("AnyPPG", "PaPaGei-S", "PulsePPG", "MOMENT-1-small")
UPSTREAM_COMMIT = "2bca102b7be2c0a9816a3872314784d2ef92555b"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest_files(paths, base):
    return {str(p.relative_to(base)).replace("\\", "/"): sha256(p) for p in sorted(paths)}


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def verify_identity(existing, identity):
    require(existing["identity_sha256"] == identity_digest(identity) and existing["identity"] == identity,
            "Cache identity differs from current data, weights, source, or execution settings; use a new output directory")


def row_mapping(old_rows, subjects, starts):
    keys = [(str(s), float(t)) for s, t in zip(subjects, starts)]
    require(len(keys) == len(set(keys)), "Duplicate PoC subject/start keys")
    lookup = {key: i for i, key in enumerate(keys)}
    old_keys = [(r["subject"], float(r["start_sec"])) for r in old_rows]
    require(len(old_keys) == len(set(old_keys)), "Duplicate legacy subject/start keys")
    require(all(key in lookup for key in old_keys), "Legacy row absent from PoC data")
    return np.asarray([lookup[key] for key in old_keys], dtype=np.int64)


def validate_data(directory, fs):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    names = ["rows.npz", f"x{fs}.npy", f"stats{fs}.npy"]
    hashes = {name: sha256(directory / name) for name in names}
    for name, digest in hashes.items():
        if name in manifest["hashes"]:
            require(digest == manifest["hashes"][name], f"Data differs from preparation manifest: {directory / name}")
    with np.load(directory / "rows.npz", allow_pickle=False) as record:
        rows = {key: record[key] for key in record.files}
    x = np.load(directory / f"x{fs}.npy", mmap_mode="r", allow_pickle=False)
    stats = np.load(directory / f"stats{fs}.npy", mmap_mode="r", allow_pickle=False)
    require(x.shape == (manifest["windows"], 8 * fs), "Input dimensions disagree with eight-second window protocol")
    require(stats.shape == (len(x), 2) and len(rows["subject"]) == len(x), "Input row/statistics dimensions disagree")
    require(np.isfinite(x).all() and np.isfinite(stats).all() and (stats[:, 1] >= 0).all(), "Non-finite inputs or invalid normalization statistics")
    return x, stats, rows, {"manifest_sha256": sha256(manifest_path), "files_sha256": hashes,
                             "preparation_sha256": manifest["preparation_sha256"], "raw_sources": manifest["sources"]}


def forward_rows(torch, port, encoder, config, x, stats, indices, device):
    batch = {"x": torch.from_numpy(np.asarray(x[indices], dtype=np.float32).copy()),
             "ppg_mean": torch.from_numpy(np.asarray(stats[indices, 0], dtype=np.float32).copy()),
             "ppg_std": torch.from_numpy(np.asarray(stats[indices, 1], dtype=np.float32).copy())}
    with torch.inference_mode():
        prepared = port.prepare_batch(batch, config, device)
        result = encoder(prepared)
    require(isinstance(result, torch.Tensor) and result.shape == (len(indices), config["embedding_dim"]), "Native encoder output has unexpected dimensions")
    result = result.detach().float().cpu().numpy()
    require(np.isfinite(result).all(), "Native encoder produced non-finite features")
    return result


def legacy_features(torch, port, encoder, config, x, stats, rows, device, audit_dir):
    if config["name"] not in ("AnyPPG", "PaPaGei-S"):
        return None, None, {"used": False, "reason": "No audited legacy cache for this model"}
    report = json.loads((audit_dir / "audit.json").read_text())
    info = next(v for v in report["reusable_features"] if v["dataset"] == "dalia" and v["model"] == config["name"])
    features_path, rows_path = Path(info["features"]), Path(info["rows"])
    require(sha256(features_path) == info["features_sha256"] and sha256(rows_path) == info["rows_sha256"], "Audited legacy export changed")
    with rows_path.open(newline="", encoding="utf-8") as stream:
        old_rows = list(csv.DictReader(stream))
    mapping = row_mapping(old_rows, rows["subject"], rows["start_s"])
    require(np.all(rows["official_include"][mapping]), "Legacy cache mapped to a nonlegacy window")
    require(np.allclose(rows["hr"][mapping], [float(r["hr"]) for r in old_rows], atol=1e-5, rtol=1e-6), "Legacy labels differ")
    # Check all reused input waveforms, not just the rows selected for GPU validation.
    for subject in np.unique(rows["subject"][mapping]):
        positions = np.flatnonzero(rows["subject"][mapping] == subject)
        original = np.load(ROOT / "datasets/ppg_bench_processed/fs_125/Dalia" / subject / "ppg.npy", mmap_mode="r")
        original_indices = [int(old_rows[i]["processed_row"]) for i in positions]
        require(np.array_equal(x[mapping[positions]], original[original_indices]), f"Legacy waveform differs for {subject}")
    values = np.load(features_path, mmap_mode="r", allow_pickle=False)
    require(values.shape == (len(mapping), config["embedding_dim"]) and np.isfinite(values).all(), "Legacy feature shape or values invalid")
    sample = []
    for subject in np.unique(rows["subject"][mapping]):
        positions = np.flatnonzero(rows["subject"][mapping] == subject)
        sample.extend(positions[np.linspace(0, len(positions) - 1, 3, dtype=int)])
    sample = np.asarray(sample)
    observed = forward_rows(torch, port, encoder, config, x, stats, mapping[sample], device)
    expected = values[sample]
    rtol, atol = 1e-4, 1e-5
    passed = bool(np.allclose(observed, expected, rtol=rtol, atol=atol))
    evidence = {"used": passed, "source_cache_sha256": info["source_cache_sha256"],
                "export_features_sha256": info["features_sha256"], "export_rows_sha256": info["rows_sha256"],
                "all_reused_waveforms_exactly_equal": True, "validation_rows": mapping[sample].tolist(),
                "validation_subjects": sorted(set(rows["subject"][mapping[sample]].tolist())),
                "validation_windows_per_subject": 3, "rtol": rtol, "atol": atol,
                "maximum_absolute_difference": float(np.max(np.abs(observed - expected))),
                "maximum_tolerance_ratio": float(np.max(np.abs(observed - expected) / (atol + rtol * np.abs(expected)))),
                "validation_allclose": passed, "reused_windows": len(mapping) if passed else 0}
    return (mapping, values, evidence) if passed else (None, None, evidence)


def extract_one(torch, config, dataset, args, source_info):
    from ppg_bench.models import get_port

    started = time.monotonic()
    destination = args.output / dataset
    destination.mkdir(parents=True, exist_ok=True)
    name = config["name"]
    final, metadata = destination / f"{name}.npy", destination / f"{name}.json"
    partial, done_path = destination / f"{name}.partial.npy", destination / f"{name}.done.npy"
    progress_path = destination / f"{name}.progress.json"
    x, stats, rows, data_identity = validate_data(args.data / dataset, config["fs"])
    checkpoint = Path(config["init"]["ckpt_path"])
    checkpoint_files = [checkpoint] if checkpoint.is_file() else list(checkpoint.rglob("*"))
    checkpoint_files = [p for p in checkpoint_files if p.is_file()]
    weights = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in checkpoint_files}
    identity = {"dataset": dataset, "model": config, "data": data_identity, "weight_files_sha256": weights,
                "source": source_info, "device": "cuda", "precision": "float32", "autocast": False,
                "batch_size_requested": args.batch_size, "cudnn_deterministic": True, "cudnn_benchmark": False,
                "torch": torch.__version__, "numpy": np.__version__, "scipy": version("scipy"),
                "transformers": version("transformers"), "momentfm": version("momentfm")}
    if metadata.is_file():
        existing = json.loads(metadata.read_text())
        verify_identity(existing, identity)
        require(existing["status"] == "complete" and final.is_file() and sha256(final) == existing["features_sha256"], "Published feature cache is incomplete or modified")
        print(json.dumps({"dataset": dataset, "model": name, "status": "verified_existing"}), flush=True)
        return existing
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        verify_identity(progress, identity)
        require(done_path.is_file() and (partial.is_file() or final.is_file()), "Incomplete resume files")
        features = np.load(partial if partial.is_file() else final, mmap_mode="r+", allow_pickle=False)
        done = np.load(done_path, mmap_mode="r+", allow_pickle=False)
        require(features.shape == (len(x), config["embedding_dim"]) and done.shape == (len(x),), "Resume dimensions differ")
        require(np.isfinite(features[done]).all(), "Previously completed cache rows are non-finite")
    else:
        require(not partial.exists() and not final.exists() and not done_path.exists(), "Unidentified feature files exist; use a fresh output location")
        features = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(x), config["embedding_dim"]))
        features[:] = np.nan
        features.flush()
        done = np.lib.format.open_memmap(done_path, mode="w+", dtype=bool, shape=(len(x),))
        done[:] = False
        done.flush()
        progress = {"status": "running", "identity": identity, "identity_sha256": identity_digest(identity),
                    "started_at": datetime.now(timezone.utc).isoformat(), "completed_windows": 0,
                    "batch_size": args.batch_size, "elapsed_previous_seconds": 0.0}
        write_json(progress_path, progress)
    port = get_port(config)
    encoder = port.load_encoder().float().to(args.device).eval()
    encoder.requires_grad_(False)
    if dataset == "Dalia" and "legacy_reuse" not in progress and not np.any(done):
        mapping, legacy, info = legacy_features(torch, port, encoder, config, x, stats, rows, args.device, args.audit)
        if mapping is not None:
            features[mapping] = legacy
            features.flush()
            done[mapping] = True
            done.flush()
        progress["legacy_reuse"] = info
        progress["completed_windows"] = int(done.sum())
        write_json(progress_path, progress)
        print(json.dumps({"dataset": dataset, "model": name, "legacy_reuse": info}), flush=True)
    remaining = np.flatnonzero(~done)
    offset, batch_size = 0, int(progress["batch_size"])
    last_report = 0.0
    while offset < len(remaining):
        indices = remaining[offset:offset + batch_size]
        try:
            result = forward_rows(torch, port, encoder, config, x, stats, indices, args.device)
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 32:
                raise
            batch_size = 32
            torch.cuda.empty_cache()
            progress["batch_size"] = batch_size
            write_json(progress_path, progress)
            continue
        features[indices] = result
        features.flush()
        done[indices] = True
        done.flush()
        offset += len(indices)
        now = time.monotonic()
        if now - last_report >= 30 or offset == len(remaining):
            progress["completed_windows"] = int(done.sum())
            progress["elapsed_seconds_this_process"] = now - started
            progress["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(progress_path, progress)
            print(json.dumps({"dataset": dataset, "model": name, "completed": int(done.sum()), "total": len(done),
                              "elapsed_seconds": round(now - started, 1), "batch_size": batch_size}), flush=True)
            last_report = now
    require(bool(np.all(done)) and bool(np.isfinite(features).all()), "Features are not complete and finite")
    shape = list(features.shape)
    del features, done, encoder, port
    gc.collect()
    torch.cuda.empty_cache()
    if partial.is_file():
        partial.replace(final)
    result = {**progress, "status": "complete", "shape": shape, "dtype": "float32", "features_sha256": sha256(final),
              "completed_windows": len(x), "finished_at": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds_this_process": time.monotonic() - started,
              "freeze_verified": "eval mode; requires_grad false; torch.inference_mode; no optimizer",
              "normalization": "Native port unchanged. PulsePPG uses complete unlabeled subject recording statistics; this is offline inference."}
    write_json(metadata, result)
    write_json(progress_path, result)
    print(json.dumps({"dataset": dataset, "model": name, "status": "complete", "elapsed_seconds": round(result["elapsed_seconds_this_process"], 1)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("Dalia", "WildPPG"), default=["Dalia", "WildPPG"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--data", type=Path, default=ROOT / ".local/ppg-hr-poc/data")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/ppg-hr-poc/features")
    parser.add_argument("--audit", type=Path, default=ROOT / ".local/ppg-hr-poc-audit")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    require(args.batch_size in (32, 64), "Use batch size 64 or 32 for the frozen FP32 protocol")
    upstream = ROOT / "repos/ppg-bench"
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    require(commit == UPSTREAM_COMMIT, "Upstream checkout changed")
    sys.path.insert(0, str(upstream))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from ppg_bench.models import load_models

    require(torch.cuda.is_available(), "CUDA is required for this explicitly selected execution profile")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    config_path = ROOT / "watch_lsm/configs/ppg_bench_models.yaml"
    configs = {c["name"]: c for c in load_models(config_path)}
    source_info = {"upstream_commit": commit, "upstream_python_files_sha256": digest_files((upstream / "ppg_bench").rglob("*.py"), upstream),
                   "config_sha256": sha256(config_path), "extraction_script_sha256": sha256(Path(__file__)),
                   "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    for dataset in args.datasets:
        for model in args.models:
            extract_one(torch, configs[model], dataset, args, source_info)


if __name__ == "__main__":
    main()
