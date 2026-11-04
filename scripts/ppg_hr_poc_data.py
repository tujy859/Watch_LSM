"""Prepare identity-preserving wrist HR PoC arrays without changing benchmark data."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repos/ppg-bench"))
from ppg_bench.preprocess import _load_pickled_subject, _resample_signal

RATES = (50, 64, 125)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def windows(trace, fs, stride):
    return np.lib.stride_tricks.sliding_window_view(np.asarray(trace).squeeze(), 8 * fs)[::stride * fs]


def motion_windows(acc, fs, starts):
    """RMS of demeaned acceleration, used only for descriptive stratification."""
    acc = np.asarray(acc, dtype=np.float64)
    sums = np.vstack([np.zeros((1, acc.shape[1])), np.cumsum(acc, axis=0)])
    squares = np.vstack([np.zeros((1, acc.shape[1])), np.cumsum(acc * acc, axis=0)])
    lo = np.rint(starts * fs).astype(int)
    hi = lo + int(round(8 * fs))
    if hi.max() > len(acc):
        raise ValueError("Acceleration and PPG time coverage differ")
    means = (sums[hi] - sums[lo]) / (hi - lo)[:, None]
    variance = (squares[hi] - squares[lo]) / (hi - lo)[:, None] - means ** 2
    return np.sqrt(np.maximum(variance, 0).sum(axis=1)).astype(np.float32)


def prepare_dataset(name, destination):
    destination = Path(destination) / name
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["preparation_sha256"] != sha256(__file__):
            raise ValueError("Preparation changed; use a fresh PoC directory")
        return manifest
    destination.mkdir(parents=True, exist_ok=True)
    prepared = ROOT / "datasets/ppg_bench_processed"
    subjects = sorted(p.name for p in (prepared / "fs_64" / name).iterdir() if (p / "ppg.npy").is_file())
    expected = 15 if name == "Dalia" else 16
    if len(subjects) != expected:
        raise ValueError(f"Expected {expected} subjects for {name}, found {len(subjects)}")
    all_arrays = {rate: [] for rate in RATES}
    row_parts, stat_parts = [], {rate: [] for rate in RATES}
    coverage = []
    sources = []
    for subject in subjects:
        base = prepared / "fs_64" / name / subject
        with (base / "metadata.csv").open() as stream:
            old_rows = list(csv.DictReader(stream))
        old_y = np.load(base / "hr.npy").astype(np.float32)
        old_starts = np.array([float(r.get("start_sec", r.get("start_s"))) for r in old_rows])
        if len(old_rows) != len(old_y):
            raise ValueError("Metadata/target length mismatch")
        if name == "Dalia":
            raw_root = ROOT / "datasets/ppg_bench_raw/dalia/PPG_FieldStudy"
            raw_file = raw_root / subject / f"{subject}.pkl"
            source = _load_pickled_subject(subject, raw_root)
            trace = np.asarray(source["signal"]["wrist"]["BVP"]).squeeze()
            y = np.asarray(source["label"], dtype=np.float32).squeeze()
            starts = 2.0 * np.arange(len(y))
            activities = windows(source["activity"], 4, 2)
            if len(activities) != len(y) or len(windows(trace, 64, 2)) != len(y):
                raise ValueError("DaLiA original label alignment failed")
            activity = np.where(np.ptp(activities, axis=1) == 0, activities[:, 0], -2).astype(int)
            motion = motion_windows(source["signal"]["wrist"]["ACC"], 32, starts)
            official = np.isin(starts, old_starts)
            old_index = (old_starts / 2).astype(int)
            if not np.array_equal(y[old_index], old_y):
                raise ValueError("DaLiA original/processed HR mismatch")
            native_fs, stride = 64, 2
            rate_arrays = {rate: windows(_resample_signal(trace, 64, rate), rate, stride)[:len(y)].astype(np.float32) for rate in RATES}
        else:
            raw_file = ROOT / "datasets/ppg_bench_raw/wildppg/WildPPG/data" / f"WildPPG_Part_{subject}.mat"
            source = loadmat(raw_file, struct_as_record=False, squeeze_me=True)
            wrist = source["wrist"]
            native_fs = int(wrist.ppg_g.fs)
            trace = np.asarray(wrist.ppg_g.v, dtype=np.float32).squeeze()
            starts, y = old_starts, old_y
            activity = np.full(len(y), -1)
            acc = np.column_stack([getattr(wrist, axis).v for axis in ("acc_x", "acc_y", "acc_z")])
            motion = motion_windows(acc, float(wrist.acc_x.fs), starts)
            official = np.ones(len(y), dtype=bool)
            old_index = np.arange(len(y))
            rate_arrays = {rate: np.load(prepared / f"fs_{rate}" / name / subject / "ppg.npy").astype(np.float32) for rate in RATES}
            # Raw-channel/time alignment sample: beginning, middle, end, without modifying labels.
            reference = _resample_signal(trace, native_fs, 64)
            for i in (0, len(y) // 2, len(y) - 1):
                lo = round(starts[i] * 64)
                if not np.array_equal(reference[lo:lo + 512], rate_arrays[64][i]):
                    raise ValueError("WildPPG raw wrist green PPG/time mismatch")
        valid = np.isfinite(y) & (y > 0)
        for rate, x in rate_arrays.items():
            if len(x) != len(y):
                raise ValueError("Native-rate row identities differ")
            valid &= np.isfinite(x).all(axis=1)
            legacy = np.load(prepared / f"fs_{rate}" / name / subject / "ppg.npy")
            if not np.array_equal(x[old_index], legacy, equal_nan=True):
                raise ValueError(f"{name}/{subject}/{rate} differs from existing processed waveforms")
            stats = json.loads((prepared / f"fs_{rate}" / name / subject / "stats.json").read_text())
            stat_parts[rate].append(np.tile([stats["ppg_mean"], stats["ppg_std"]], (int(valid.sum()), 1)))
        # Rebuild lengths after all sampling-rate validity checks.
        for rate, x in rate_arrays.items():
            stats = json.loads((prepared / f"fs_{rate}" / name / subject / "stats.json").read_text())
            stat_parts[rate][-1] = np.tile([stats["ppg_mean"], stats["ppg_std"]], (int(valid.sum()), 1))
            all_arrays[rate].append(x[valid])
        row_parts.extend({"subject": subject, "start_s": float(starts[i]), "hr": float(y[i]), "activity": int(activity[i]), "official_include": bool(official[i]), "motion": float(motion[i])} for i in np.flatnonzero(valid))
        coverage.append({"subject": subject, "candidate_windows": len(y), "included": int(valid.sum()), "invalid_reference_or_signal": int((~valid).sum()), "legacy_included": int((valid & official).sum()), "additional_windows": int((valid & ~official).sum())})
        sources.append({"path": raw_file.relative_to(ROOT).as_posix(), "bytes": raw_file.stat().st_size, "sha256": sha256(raw_file)})
        print(f"prepare {name}/{subject}: {int(valid.sum())} windows", flush=True)
    with (destination / "rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row_parts[0]))
        writer.writeheader(); writer.writerows(row_parts)
    arrays = {key: np.asarray([r[key] for r in row_parts]) for key in row_parts[0]}
    finite_motion = arrays["motion"][np.isfinite(arrays["motion"])]
    cutpoints = np.quantile(finite_motion, [1/3, 2/3])
    arrays["motion_bin"] = np.digitize(arrays["motion"], cutpoints)
    np.savez(destination / "rows.npz", **arrays)
    hashes = {}
    for rate in RATES:
        np.save(destination / f"x{rate}.npy", np.concatenate(all_arrays[rate]))
        np.save(destination / f"stats{rate}.npy", np.concatenate(stat_parts[rate]).astype(np.float32))
        hashes[f"x{rate}.npy"] = sha256(destination / f"x{rate}.npy")
    hashes["rows.npz"] = sha256(destination / "rows.npz")
    manifest = {"dataset": name, "subjects": subjects, "windows": len(row_parts), "coverage": coverage, "sources": sources, "hashes": hashes, "preparation_sha256": sha256(__file__), "motion_terciles": cutpoints.tolist(), "motion_note": "Within-dataset unlabeled acceleration terciles for descriptive reporting only; ACC is never model input", "pulseppg_normalization": "Retain upstream whole-subject unlabeled mean/std; offline record normalization, not online inference"}
    write_json(manifest_path, manifest)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/ppg-hr-poc/data")
    parser.add_argument("--datasets", nargs="+", default=["Dalia", "WildPPG"])
    args = parser.parse_args()
    for dataset in args.datasets:
        prepare_dataset(dataset, args.output)
