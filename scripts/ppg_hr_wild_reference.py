"""Recompute WildPPG ECG HR with the unmodified released reference functions.

The released file also contains a top-level data-conversion loop. Only its five
reference function definitions are loaded, so that loop is never executed. This
script writes separate PoC references and does not change PPG-Bench artifacts.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy.io
import scipy.ndimage
import scipy.signal

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SOURCE = ROOT / "datasets/ppg_bench_raw/wildppg/WildPPG/2024-wildppg-neurips-code-master/src/supervised_baselines/data_preprocessing.py"
FUNCTIONS = (
    "panPeakDetect", "pan_tompkins_detector", "quotient_filter",
    "butter_bandpass", "butter_bandpass_filter",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_official_reference(path=OFFICIAL_SOURCE):
    """Load exact source definitions, excluding imports and all top-level work."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    if sorted(node.name for node in definitions) != sorted(FUNCTIONS):
        raise ValueError("Released WildPPG reference function set changed")
    namespace = {"np": np, "scipy": __import__("scipy")}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def derive_reference(ecg, ecg_fs, starts, functions):
    """Keep raw ECG precision and the released detector/quotient-filter logic."""
    peaks = functions["pan_tompkins_detector"](ecg, ecg_fs)
    good_peaks, rr, rr_midpoints = functions["quotient_filter"](
        peaks, outlier_over=5, sampling_rate=ecg_fs, tol=0.75
    )
    reference = np.full(len(starts), np.nan, dtype=np.float32)
    n_rr = np.zeros(len(starts), dtype=np.int64)
    for i, start_s in enumerate(starts):
        # Open interval boundaries and 60 / mean(RR) follow released code.
        in_window = (rr_midpoints > start_s * ecg_fs) & (rr_midpoints < (start_s + 8) * ecg_fs)
        window_rr = rr[in_window]
        n_rr[i] = len(window_rr)
        if len(window_rr) > 1:
            reference[i] = 60 * len(window_rr) / (np.sum(window_rr) / ecg_fs)
    return {
        "reference_official": reference,
        "valid_reference": np.isfinite(reference) & (reference > 0),
        "n_rr_intervals": n_rr,
        "r_peak_samples": np.asarray(peaks, dtype=np.int64),
        "r_peak_times_s": np.asarray(peaks, dtype=np.float64) / ecg_fs,
        "good_peak_samples": np.asarray(good_peaks, dtype=np.int64),
        "good_peak_times_s": np.asarray(good_peaks, dtype=np.float64) / ecg_fs,
        "rr_samples": np.asarray(rr, dtype=np.int64),
        "rr_midpoints_samples": np.asarray(rr_midpoints, dtype=np.float64),
    }


def prepare_subject(subject, output, functions, official_hash, preparation_hash):
    raw_file = ROOT / "datasets/ppg_bench_raw/wildppg/WildPPG/data" / f"WildPPG_Part_{subject}.mat"
    legacy = ROOT / "datasets/ppg_bench_processed/fs_64/WildPPG" / subject
    identity = {
        "subject": subject,
        "official_source_sha256": official_hash,
        "preparation_sha256": preparation_hash,
        "raw_path": raw_file.relative_to(ROOT).as_posix(),
        "raw_bytes": raw_file.stat().st_size,
        "raw_mtime_ns": raw_file.stat().st_mtime_ns,
        "old_hr_sha256": sha256(legacy / "hr.npy"),
        "old_metadata_sha256": sha256(legacy / "metadata.csv"),
    }
    manifest_path = output / f"{subject}.json"
    arrays_path = output / f"{subject}.npz"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["identity"] != identity or sha256(arrays_path) != existing["arrays_sha256"]:
            raise ValueError(f"Reference cache identity/content changed for {subject}")
        return existing
    started = time.monotonic()
    raw = scipy.io.loadmat(raw_file, variable_names=["sternum", "wrist"], struct_as_record=False, squeeze_me=True)
    ecg = np.asarray(raw["sternum"].ecg.v).squeeze()
    ppg = np.asarray(raw["wrist"].ppg_g.v).squeeze()
    ecg_fs, ppg_fs = (float(raw["sternum"].ecg.fs), float(raw["wrist"].ppg_g.fs))
    if not ecg_fs.is_integer() or not ppg_fs.is_integer():
        raise ValueError("Released reference expects integer sampling rates")
    ecg_fs, ppg_fs = int(ecg_fs), int(ppg_fs)
    if not np.isfinite(ecg).all():
        raise ValueError(f"Nonfinite raw ECG in {subject}; do not silently alter detector input")
    n_candidates = len(ppg) // (8 * ppg_fs)
    starts = 8.0 * np.arange(n_candidates)
    finite_ppg = np.isfinite(ppg[:n_candidates * 8 * ppg_fs].reshape(n_candidates, 8 * ppg_fs)).all(axis=1)
    old_y = np.load(legacy / "hr.npy")
    old_rows = list(csv.DictReader((legacy / "metadata.csv").open()))
    if len(old_rows) != len(old_y):
        raise ValueError("Legacy HR and row identities disagree")
    old_starts = np.asarray([float(row["start_s"]) for row in old_rows])
    old_index = (old_starts / 8).astype(np.int64)
    if not np.array_equal(starts[old_index], old_starts) or len(np.unique(old_index)) != len(old_index):
        raise ValueError("Legacy window identities are not a unique 8-second grid")
    old_reference = np.full(n_candidates, np.nan, dtype=np.float32)
    old_reference[old_index] = old_y
    old_present = np.zeros(n_candidates, dtype=bool)
    old_present[old_index] = True
    result = derive_reference(ecg, ecg_fs, starts, functions)
    result.update({
        "candidate_index": np.arange(n_candidates, dtype=np.int64),
        "start_s": starts, "end_s": starts + 8,
        "reference_old": old_reference,
        "old_valid_reference": np.isfinite(old_reference) & (old_reference > 0),
        "old_window_present": old_present,
        "finite_native_ppg": finite_ppg,
        "ecg_fs": np.asarray(ecg_fs), "ppg_fs": np.asarray(ppg_fs),
    })
    old_valid, new_valid = result["old_valid_reference"], result["valid_reference"]
    common = old_valid & new_valid
    absolute_difference = np.abs(result["reference_official"][common] - old_reference[common])
    np.savez_compressed(arrays_path, **result)
    summary = {
        "identity": identity,
        "candidate_windows": n_candidates,
        "native_ppg_nonfinite_windows": int((~finite_ppg).sum()),
        "legacy_missing_windows": int((~old_present).sum()),
        "old_valid_reference": int(old_valid.sum()),
        "official_valid_reference": int(new_valid.sum()),
        "gained_valid_reference": int((~old_valid & new_valid).sum()),
        "lost_valid_reference": int((old_valid & ~new_valid).sum()),
        "common_valid_windows": int(common.sum()),
        "changed_common_labels": int(np.count_nonzero(absolute_difference)),
        "mean_abs_hr_difference_common_bpm": float(absolute_difference.mean()),
        "max_abs_hr_difference_common_bpm": float(absolute_difference.max()),
        "raw_ecg_dtype": str(ecg.dtype), "native_ecg_fs": ecg_fs, "native_ppg_fs": ppg_fs,
        "arrays_sha256": sha256(arrays_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(manifest_path, summary)
    print(f"{subject}: official valid {new_valid.sum()}/{n_candidates}; old {old_valid.sum()}; changed common {summary['changed_common_labels']}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/ppg-hr-poc/wild-reference")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    functions = load_official_reference()
    official_hash, preparation_hash = sha256(OFFICIAL_SOURCE), sha256(__file__)
    subjects = sorted(path.name for path in (ROOT / "datasets/ppg_bench_processed/fs_64/WildPPG").iterdir() if (path / "hr.npy").is_file())
    if len(subjects) != 16:
        raise ValueError(f"Expected 16 WildPPG people, found {len(subjects)}")
    summaries = [prepare_subject(subject, args.output, functions, official_hash, preparation_hash) for subject in subjects]
    totals = {key: sum(item[key] for item in summaries) for key in (
        "candidate_windows", "native_ppg_nonfinite_windows", "legacy_missing_windows",
        "old_valid_reference", "official_valid_reference", "gained_valid_reference", "lost_valid_reference",
        "common_valid_windows", "changed_common_labels",
    )}
    summary = {
        "complete": True, "subjects": subjects, "official_source": OFFICIAL_SOURCE.relative_to(ROOT).as_posix(),
        "official_source_sha256": official_hash, "preparation_sha256": preparation_hash,
        "loaded_function_names": list(FUNCTIONS), "function_modifications": "none; exact AST function definitions",
        "totals": totals, "by_subject": summaries,
        "reference_protocol": "Raw native-precision sternum ECG; released Pan-Tompkins functions; quotient filter 5 peaks / tolerance 0.75; 8-second non-overlapping PPG windows; strict RR-midpoint boundaries; HR=60/mean(RR), requiring at least two accepted RR intervals",
        "invalid_representation": "Released conversion uses 0; this artifact uses NaN plus an explicit validity mask. No artificial HR clipping or PPG quality filtering.",
        "tail_note": "All complete wrist PPG windows are retained as candidates; windows beyond accepted ECG beats have invalid references, instead of truncating the candidate grid at the last accepted beat.",
        "known_upstream_port_differences": [
            "PPG-Bench port omits released threshold weight update from 0.125 to 0.0125 after more than 100 signal peaks.",
            "PPG-Bench port computes missed-RR threshold as int(1.66*mean(RR)); released code uses int(1.66*int(mean(RR))).",
            "PPG-Bench port casts ECG to float32 and clamps peak-refinement boundary slices; this recomputation keeps the released raw input precision and exact slices.",
        ],
        "limitations": [
            "This is an automated ECG-derived reference, not manually adjudicated beat annotations.",
            "Reference availability is selected by the released ECG quality filter, so accuracy applies to windows with valid reference and does not measure all-day coverage.",
            "Raw file identity here uses size and mtime; full raw SHA-256 values are recorded by the separate PoC data preparation manifest.",
        ],
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
