"""Read Biagetti wrist raw PPG and the EMBC21 expert onset annotations.

CSV is the runtime annotation source. Raw ADC intensity is retained: maxima
correspond to diastolic pulse onset; negation gives conventional blood-volume
polarity. The expert's artifact ranges are excluded, including both endpoints.
No intervals or beats are joined across an artifact gap or recording boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

FS = 400.0
ANNOTATION_URL = "https://www.eti.uni-siegen.de/ubicomp/home/datasets/embc21/ppgopt_annot.zip"
RAW_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6971339/supplementaryFiles"
METHODS_URL = "https://www.eti.uni-siegen.de/ubicomp/papers/ubi_embc2021.pdf"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_integer_csv(path: Path, columns: int) -> np.ndarray:
    if not path.stat().st_size:
        return np.empty((0, columns), dtype=np.int64)
    with path.open(newline="", encoding="utf-8") as f:
        rows = [[int(value) for value in row] for row in csv.reader(f, delimiter=";")]
    if any(len(row) != columns for row in rows):
        raise ValueError(f"Unexpected annotation shape: {path}")
    return np.asarray(rows, dtype=np.int64).reshape(-1, columns)


def valid_mask_from_artifacts(length: int, artifacts: np.ndarray) -> np.ndarray:
    """Conservatively exclude the official inclusive sample-index ranges."""
    valid = np.ones(length, dtype=bool)
    previous_end = -1
    for start, end in np.asarray(artifacts, dtype=np.int64).reshape(-1, 2):
        if not 0 <= start <= end < length or start <= previous_end:
            raise ValueError("Artifact intervals are unsorted, overlapping, or out of bounds")
        valid[start : end + 1] = False
        previous_end = int(end)
    return valid


def mask_intervals_s(valid: np.ndarray, fs: float = FS) -> np.ndarray:
    """Return half-open [start, end) seconds for contiguous valid sample runs."""
    transitions = np.diff(np.r_[False, np.asarray(valid, dtype=bool), False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return np.column_stack((starts, ends)).astype(np.float64) / fs


def load_records(data_root: str | Path) -> list[dict[str, Any]]:
    """Return all 105 records, including the fully invalid S5/squat3.

    `onset_times_s` contains only labels within the valid mask; the original
    21,806 labels remain accessible as `all_onset_times_s` and `onset_samples`.
    `annotated_intervals_s` is a conservative manual-coverage mask, independent
    of any model or signal-quality test. All times are relative to record start.
    """
    root = Path(data_root)
    records = []
    paths = sorted((root / "PPG_ACC_dataset").glob("S*/*_ppg.mat"))
    if len(paths) != 105:
        raise ValueError(f"Expected 105 raw PPG records, found {len(paths)}")
    for path in paths:
        subject = path.parent.name
        session = path.name.removesuffix("_ppg.mat")
        activity = session.rstrip("0123456789")
        record_id = f"{subject}_{session}"
        annotation_stem = f"{subject.lower()}_{session}"
        annotation_dir = root / "ppgopt_annot" / "csv" / subject
        onset_path = annotation_dir / f"{annotation_stem}_pks.csv"
        artifact_path = annotation_dir / f"{annotation_stem}_atf.csv"
        data = np.asarray(loadmat(path)["PPG"], dtype=np.float64)
        if data.ndim != 2 or data.shape[1] not in (2, 3) or not np.isfinite(data[:, 0]).all():
            raise ValueError(f"Invalid native waveform: {path}")
        if data.shape[1] == 3 and not (data[:, 2] == 0).all():
            raise ValueError(f"Unexpected nonzero extra column: {path}")
        source_time_s, raw_ppg = data[:, 0], data[:, 1]
        time_s = source_time_s - source_time_s[0]
        if not np.allclose(time_s, np.arange(len(data)) / FS, rtol=0, atol=1e-9):
            raise ValueError(f"Native timestamp grid is not 400 Hz from zero: {path}")
        onsets = read_integer_csv(onset_path, 1).ravel()
        artifacts = read_integer_csv(artifact_path, 2)
        if np.any(np.diff(onsets) <= 0) or np.any(onsets < 0) or np.any(onsets >= len(data)):
            raise ValueError(f"Invalid or duplicate onset indices: {onset_path}")
        valid = valid_mask_from_artifacts(len(data), artifacts)
        # A no-label recording cannot supply negatives even if a future copy
        # accidentally omits its explicit all-record artifact annotation.
        if not len(onsets):
            valid[:] = False
        manual_valid = valid.copy()
        finite = np.isfinite(raw_ppg)
        if not finite.any() or np.any(np.diff(np.flatnonzero(~finite)) == 1):
            raise ValueError(f"Unexpected absent or consecutive missing samples: {path}")
        # S2/squat3 contains 39 isolated missing samples at integer seconds.
        # Preserve these in raw_ppg; only the explicitly named input copy is
        # interpolated. No training/evaluation window may cross a missing sample.
        input_ppg = raw_ppg.copy()
        if not finite.all():
            input_ppg[~finite] = np.interp(time_s[~finite], time_s[finite], raw_ppg[finite])
        valid &= finite
        records.append({
            "subject": subject,
            "record_id": record_id,
            "activity": activity,
            "fs": FS,
            "time_s": time_s,
            "source_time_origin_s": float(source_time_s[0]),
            "raw_ppg": raw_ppg,
            "input_ppg": input_ppg,
            "source_finite_mask": finite,
            "source_mat_columns": data.shape[1],
            "onset_samples": onsets,
            "all_onset_times_s": time_s[onsets],
            "onset_times_s": time_s[onsets[valid[onsets]]],
            "artifact_intervals_samples_inclusive": artifacts,
            "annotated_intervals_s": mask_intervals_s(valid),
            "manual_annotated_intervals_s": mask_intervals_s(manual_valid),
            "manual_valid_mask": manual_valid,
            "valid_mask": valid,
            "duration_s": len(data) / FS,
            "raw_polarity": "sensor light intensity; diastolic onset is a raw maximum",
            "source_paths": [path, onset_path, artifact_path],
        })
    return records


def eligible_windows(record: dict[str, Any], duration_s: float, stride_s: float) -> np.ndarray:
    """Recording-zero grid of windows fully covered by one manual interval."""
    fs = record["fs"]
    width = int(round(duration_s * fs))
    stride = int(round(stride_s * fs))
    if width <= 0 or stride <= 0 or abs(width / fs - duration_s) > 1e-10 or abs(stride / fs - stride_s) > 1e-10:
        raise ValueError("Window duration and stride must be positive native-sample multiples")
    valid = record["valid_mask"]
    starts = np.arange(0, max(0, len(valid) - width + 1), stride, dtype=np.int64)
    invalid_cumsum = np.r_[0, np.cumsum(~valid)]
    starts = starts[invalid_cumsum[starts + width] == invalid_cumsum[starts]]
    return np.column_stack((starts, starts + width)).astype(np.float64) / fs


class _BuiltinsOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError("Global objects are forbidden in annotation audit")

    def persistent_load(self, pid: Any) -> Any:
        raise pickle.UnpicklingError("Persistent objects are forbidden in annotation audit")


def audit(data_root: Path, output_dir: Path) -> dict[str, Any]:
    records = load_records(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    windows = []
    for record in records:
        stem = record["record_id"].lower()
        pkl_path = data_root / "ppgopt_annot" / "pkl" / record["subject"] / f"{stem}.p"
        # No global construction or persistent object loading is permitted.
        obj = _BuiltinsOnlyUnpickler(io.BytesIO(pkl_path.read_bytes())).load()
        if set(obj) != {"peaks", "artifacts"}:
            raise ValueError(f"Unexpected pickle keys: {pkl_path}")
        pickle_peaks = [] if obj["peaks"] is None else obj["peaks"]
        if not np.array_equal(np.asarray(pickle_peaks), record["onset_samples"]):
            raise ValueError(f"CSV/pickle onset mismatch: {pkl_path}")
        pickle_artifacts = [] if obj["artifacts"] is None else obj["artifacts"]
        if not np.array_equal(np.asarray(pickle_artifacts).reshape(-1, 2), record["artifact_intervals_samples_inclusive"]):
            raise ValueError(f"CSV/pickle artifact mismatch: {pkl_path}")
        row = {key: record[key] for key in ("subject", "record_id", "activity", "duration_s")}
        row.update({
            "native_samples": len(record["raw_ppg"]),
            "raw_annotation_count": len(record["all_onset_times_s"]),
            "valid_annotation_count": len(record["onset_times_s"]),
            "manual_valid_seconds": float(record["valid_mask"].sum() / FS),
            "source_missing_samples": int((~record["source_finite_mask"]).sum()),
            "source_mat_columns": record["source_mat_columns"],
            "source_time_origin_s": record["source_time_origin_s"],
            "artifact_intervals": len(record["artifact_intervals_samples_inclusive"]),
            "manual_intervals": len(record["annotated_intervals_s"]),
        })
        for duration, stride in [(8, 6), (60, 60), (300, 300)]:
            eligible = eligible_windows(record, duration, stride)
            row[f"windows_{duration}s_stride{stride}s"] = len(eligible)
            for start, end in eligible:
                windows.append({"subject": record["subject"], "record_id": record["record_id"], "activity": record["activity"], "duration_s": duration, "stride_s": stride, "start_s": start, "end_s": end})
        rows.append(row)
    for filename, table in [("record_audit.csv", rows), ("eligible_windows.csv", windows)]:
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    paths = sorted(p for p in data_root.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "dataset": "Biagetti raw wrist PPG + EMBC21 expert diastolic onset annotations",
        "version": "embc21-native-v1",
        "created_date": "2026-11-04",
        "sources": {"annotations": ANNOTATION_URL, "waveforms_supplement": RAW_URL, "original_article": "https://doi.org/10.1016/j.dib.2019.105044", "annotation_methods": METHODS_URL},
        "license": {"annotations": "scientific non-commercial research with attribution to both source papers", "original_article_and_supplement": "CC BY 4.0 (original article)"},
        "subject_count": 7,
        "record_count": len(records),
        "records_with_valid_labels": sum(bool(len(r["onset_times_s"])) for r in records),
        "fully_annotated_records": sum(bool(r["manual_valid_mask"].all()) for r in records),
        "fully_usable_records": sum(bool(r["valid_mask"].all()) for r in records),
        "native_fs": FS,
        "raw_annotation_count": sum(row["raw_annotation_count"] for row in rows),
        "valid_annotation_count": sum(row["valid_annotation_count"] for row in rows),
        "duration_s": sum(row["duration_s"] for row in rows),
        "manual_valid_seconds": sum(row["manual_valid_seconds"] for row in rows),
        "source_missing_samples": sum(row["source_missing_samples"] for row in rows),
        "windows": {f"{duration}s_stride{stride}s": sum(row[f"windows_{duration}s_stride{stride}s"] for row in rows) for duration, stride in [(8, 6), (60, 60), (300, 300)]},
        "semantics": {
            "onsets": "0-based native 400 Hz sample indices; maximum in raw inverse-blood-volume sensor intensity",
            "index_origin_evidence": "S5/squat3 official artifacts=[0,24384] spans exactly all 24385 native samples; CSV identical to Python list annotations",
            "artifact_boundary_policy": "conservatively exclude both official endpoints; valid intervals are half-open sample-time spans",
            "raw_labels_in_artifacts": "Preserved in source/all_onset_times_s, excluded from onset_times_s and scoring; never treat unannotated segments as negatives",
            "complete_window_policy": "zero-aligned recording grid, no stitching across artifact gaps or recordings",
            "missing_sample_policy": "raw_ppg preserves NaN; input_ppg linearly fills only isolated missing samples for finite whole-record inference; all missing samples excluded from valid_mask/annotated_intervals_s, so no complete training/evaluation window crosses them. S2/squat3 has 39 such samples (integer seconds 1 to 39).",
            "extra_column_policy": "S4 all 15 MAT records have a third column containing only zero, verified and ignored; column 0 is time and column 1 is raw PPG throughout",
            "time_origin_policy": "Labels are zero-based array indices. Source time origin is subtracted for all record-relative times; S5/squat2 starts at 0.0225 s and S5/step1 starts at 0.045 s in MAT. Original origins preserved as source_time_origin_s.",
            "duration_note": "The paper's 7.73 min excluded amount omits the entirely rejected 60.96 s S5/squat3; complete CSV artifact masks cover about 8.745 min including this record",
            "label_limitations": "Single expert, no inter-rater agreement and no ECG; noisy/ambiguous intervals excluded; permits native PPG onset/PRV agreement, not ECG HRV validation",
            "population": "7 healthy adults; 3 male and 4 female, ages 20-52; firmly secured MAXREFDES100 wrist sensor; rest/squat/step",
        },
        "checks": {"finite_input_ppg_and_timestamps": True, "native_missing_samples_preserved_and_excluded": True, "native_timestamp_grid_400Hz": True, "label_bounds_order_duplicates": True, "artifact_bounds_order": True, "csv_pickle_exact_match_records": 105},
        "files": [{"path": p.relative_to(data_root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths],
        "adapter_sha256": sha256(Path(__file__)),
        "audit_files": {p.name: sha256(p) for p in [output_dir / "record_audit.csv", output_dir / "eligible_windows.csv"]},
    }
    (data_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "audit.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/embc21"))
    parser.add_argument("--output-dir", type=Path, default=Path(".local/ppg-phase2/embc21"))
    args = parser.parse_args()
    result = audit(args.data_root, args.output_dir)
    print(json.dumps({k: result[k] for k in ("record_count", "records_with_valid_labels", "raw_annotation_count", "valid_annotation_count", "duration_s", "manual_valid_seconds", "windows")}, indent=2))
