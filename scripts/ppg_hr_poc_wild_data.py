"""Apply independently recomputed official WildPPG ECG references to the PoC.

The previous PoC view is retained alongside the corrected view. Neither raw
data nor the original PPG-Bench preprocessing outputs are modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ppg_hr_poc_data import ROOT, RATES, motion_windows, sha256, write_json


def apply_references(root):
    root = Path(root).resolve()
    destination = root / "data/WildPPG"
    reference_root = root / "wild-reference"
    reference_summary = reference_root / "summary.json"
    if not reference_summary.is_file():
        raise ValueError("Complete the independent official ECG reference audit first")
    identity = {"reference_summary_sha256": sha256(reference_summary), "correction_script_sha256": sha256(__file__)}
    previous = json.loads((destination / "manifest.json").read_text())
    if previous.get("reference_correction") == identity:
        return previous
    if any((root / "runs").glob("*/*/complete.json")) or any((root / "features/WildPPG").glob("*.json")):
        raise ValueError("Do not change reference rows after evaluation has started")
    archive = root / "data/WildPPG-before-reference-audit"
    if archive.exists():
        raise ValueError("Previous reference archive already exists; inspect state before proceeding")
    for path in (destination, archive):
        if not path.resolve().is_relative_to(root):
            raise ValueError("Archive path outside the PoC workspace")
    destination.rename(archive)
    destination.mkdir()
    parts = {rate: [] for rate in RATES}
    stats_parts = {rate: [] for rate in RATES}
    rows, coverage = [], []
    for subject in previous["subjects"]:
        reference_path = reference_root / f"{subject}.npz"
        with np.load(reference_path) as ref:
            starts = ref["start_s"]
            y = ref["reference_official"].astype(np.float32)
            old = ref["reference_old"]
            valid = ref["valid_reference"].copy()
        processed = ROOT / "datasets/ppg_bench_processed"
        with (processed / "fs_64/WildPPG" / subject / "metadata.csv").open() as stream:
            legacy_starts = np.array([float(r["start_s"]) for r in csv.DictReader(stream)])
        if not np.array_equal(starts, legacy_starts):
            raise ValueError("Corrected reference and original PPG times differ")
        signals = {rate: np.load(processed / f"fs_{rate}/WildPPG" / subject / "ppg.npy", mmap_mode="r") for rate in RATES}
        for x in signals.values():
            if len(x) != len(y):
                raise ValueError("PPG/reference row count mismatch")
            valid &= np.isfinite(x).all(axis=1)
        raw = ROOT / "datasets/ppg_bench_raw/wildppg/WildPPG/data" / f"WildPPG_Part_{subject}.mat"
        wrist = loadmat(raw, struct_as_record=False, squeeze_me=True)["wrist"]
        motion = motion_windows(np.column_stack([getattr(wrist, k).v for k in ("acc_x", "acc_y", "acc_z")]), float(wrist.acc_x.fs), starts)
        for rate, x in signals.items():
            stats = json.loads((processed / f"fs_{rate}/WildPPG" / subject / "stats.json").read_text())
            parts[rate].append(x[valid])
            stats_parts[rate].append(np.tile([stats["ppg_mean"], stats["ppg_std"]], (int(valid.sum()), 1)))
        rows.extend({"subject": subject, "start_s": float(starts[i]), "hr": float(y[i]), "activity": -1, "official_include": True, "motion": float(motion[i])} for i in np.flatnonzero(valid))
        old_valid = np.isfinite(old) & (old > 0)
        coverage.append({"subject": subject, "candidate_windows": len(y), "included": int(valid.sum()), "invalid_reference_or_signal": int((~valid).sum()), "legacy_included": int((valid & old_valid).sum()), "additional_windows": int((valid & ~old_valid).sum()), "old_valid_now_excluded": int((old_valid & ~valid).sum())})
    with (destination / "rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    cutpoints = np.quantile(arrays["motion"][np.isfinite(arrays["motion"])], [1/3, 2/3])
    arrays["motion_bin"] = np.digitize(arrays["motion"], cutpoints)
    np.savez(destination / "rows.npz", **arrays)
    hashes = {"rows.npz": sha256(destination / "rows.npz")}
    for rate in RATES:
        np.save(destination / f"x{rate}.npy", np.concatenate(parts[rate]))
        np.save(destination / f"stats{rate}.npy", np.concatenate(stats_parts[rate]).astype(np.float32))
        for name in (f"x{rate}.npy", f"stats{rate}.npy"):
            hashes[name] = sha256(destination / name)
    manifest = {**previous, "windows": len(rows), "coverage": coverage, "hashes": hashes, "motion_terciles": cutpoints.tolist(), "reference_correction": identity, "reference_note": "Automatic sternum ECG RR references from unmodified official WildPPG functions; not manually adjudicated ECG truth", "prior_manifest_sha256": sha256(archive / "manifest.json")}
    write_json(destination / "manifest.json", manifest)
    print(f"Corrected WildPPG: {len(rows)} windows; previous view retained at {archive}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / ".local/ppg-hr-poc")
    apply_references(parser.parse_args().root)
