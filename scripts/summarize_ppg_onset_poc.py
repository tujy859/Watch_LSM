"""Independent manual PPG-onset/PP-variability scoring for EMBC21.

This evaluator never aligns detections to test labels, crosses annotation gaps,
joins acquisition records, rejects erroneous predicted beats, or trains models.
The reference is expert PPG onset timing, so interval statistics are PP/PRV,
not ECG RR/NN or clinical HRV. Sixty-second PRV is algorithm agreement only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ppg_beat_metrics import match_events, matched_rr_errors, rr_features
from ppg_embc21_data import load_records

METHODS = {"AnyPPG": "AnyPPG", "SmallCNN": "SmallCNN", "MSPTDfast": "MSPTD"}
GROUPS = ("all", "rest", "squat", "step")
TOLERANCES = (.025, .05, .1)
PP_FEATURES = {"mean_pp_ms": "mean_rr_ms", "sdpp_ms": "sdnn_rr_ms", "rmssd_ms": "rmssd_rr_ms"}
MIN_INTERVALS = 20


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, records):
    if not records:
        raise ValueError("Refuse to publish empty table")
    keys = list(dict.fromkeys(k for row in records for k in row if not k.startswith("_")))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in keys} for row in records)


def finite_mean(values):
    x = np.asarray(values, dtype=float)
    return float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else float("nan")


def bootstrap(values, samples=10000):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    # A single supported participant cannot estimate between-person uncertainty.
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(20261104)
    values = x[rng.integers(0, len(x), (samples, len(x)))].mean(axis=1)
    return tuple(float(q) for q in np.quantile(values, [.025, .975]))


def validate_events(events, duration, label):
    x = np.asarray(events, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or (np.diff(x) <= 0).any() or (x < 0).any() or (x >= duration).any():
        raise ValueError(f"Invalid {label} event times")
    return x


def validate_intervals(intervals, duration):
    values = np.asarray(intervals, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Annotation intervals must have shape (N,2)")
    if not np.isfinite(values).all() or (values[:, 0] < 0).any() or (values[:, 1] > duration + 1e-9).any() or (values[:, 1] <= values[:, 0]).any():
        raise ValueError("Invalid annotated intervals")
    if len(values) > 1 and (values[1:, 0] < values[:-1, 1]).any():
        raise ValueError("Overlapping or reversed annotated intervals")
    return values


def between(events, start, end):
    a, b = np.searchsorted(events, [start, end], side="left")
    return events[a:b]


def score_record(record, method, prediction, inference_start, inference_end):
    """Score intervals and complete blocks, retaining every unsupported block."""
    duration = float(record["duration_s"])
    if not np.isfinite([inference_start, inference_end]).all() or not 0 <= inference_start < inference_end <= duration + 1e-8:
        raise ValueError("Invalid inference boundaries")
    reference = validate_events(record["onset_times_s"], duration, "reference")
    prediction = validate_events(prediction, duration, "prediction")
    if (prediction < inference_start).any() or (prediction >= inference_end).any():
        raise ValueError("Prediction outside declared inference bounds")
    annotated = validate_intervals(record["annotated_intervals_s"], duration)
    base = {"dataset": "EMBC21", "subject": str(record["subject"]), "activity": record["activity"],
            "record_id": record["record_id"], "method": method}
    intervals, windows = [], []
    annotated_seconds = float(np.sum(annotated[:, 1] - annotated[:, 0]))
    evaluated_seconds = 0.
    for index, (lower, upper) in enumerate(annotated):
        start, end = max(float(lower), inference_start), min(float(upper), inference_end)
        if end <= start:
            continue
        evaluated_seconds += end - start
        ref, pred = between(reference, start, end), between(prediction, start, end)
        row = {**base, "interval_index": index, "start_s": start, "end_s": end,
               "n_reference_events": len(ref), "n_prediction_events": len(pred)}
        for tolerance in TOLERANCES:
            matched = match_events(ref, pred, tolerance)
            suffix = f"{int(round(tolerance * 1000))}ms"
            for key in ("tp", "fp", "fn", "f1", "precision", "recall", "timing_mae_ms"):
                row[f"{key}_{suffix}"] = matched[key]
            if tolerance == .05:
                pp = matched_rr_errors(ref, pred, matched)
                row.update(matched_pp_count=pp["n_rr_errors"], reference_pp_count=max(0, len(ref) - 1),
                           matched_pp_mae_ms=pp["rr_mae_ms"], matched_pp_coverage=pp["reference_pair_coverage"],
                           _reference_pair_index=pp["reference_pair_start_index"], _pp_abs_errors=np.abs(pp["rr_error_ms"]))
        intervals.append(row)
        for block in (60, 300):
            count = int(np.floor((end - start + 1e-9) / block))
            for number in range(count):
                lo = start + number * block
                hi = lo + block
                ref_block, pred_block = between(ref, lo, hi), between(pred, lo, hi)
                ref_features, pred_features = rr_features(ref_block), rr_features(pred_block)
                valid_ref = ref_features["n_intervals"] >= MIN_INTERVALS
                valid_pred = pred_features["n_intervals"] >= MIN_INTERVALS
                item = {**base, "interval_index": index, "start_s": lo, "duration_s": block,
                        "reference_intervals": ref_features["n_intervals"], "prediction_intervals": pred_features["n_intervals"],
                        "reference_eligible": valid_ref, "prediction_eligible": valid_pred}
                for name, key in PP_FEATURES.items():
                    item["reference_" + name] = ref_features[key]
                    item["prediction_" + name] = pred_features[key]
                    item[name + "_absolute_error"] = abs(pred_features[key] - ref_features[key]) if valid_ref and valid_pred else float("nan")
                windows.append(item)
    coverage = {**base, "record_seconds": duration, "annotated_seconds": annotated_seconds,
                "inference_seconds": inference_end - inference_start, "evaluated_seconds": evaluated_seconds,
                "annotated_intervals": len(annotated), "evaluated_intervals": len(intervals),
                "annotated_events": sum(len(between(reference, a, b)) for a, b in annotated),
                "evaluated_reference_events": sum(r["n_reference_events"] for r in intervals),
                "annotated_fraction_of_record": annotated_seconds / duration,
                "evaluated_fraction_of_annotated": evaluated_seconds / annotated_seconds if annotated_seconds else 0.}
    return intervals, windows, coverage


def participant_metrics(intervals, windows, subject, method, group):
    def belongs(row):
        return row["subject"] == subject and row["method"] == method and (group == "all" or row["activity"] == group)
    events = [r for r in intervals if belongs(r)]
    blocks = [r for r in windows if belongs(r)]
    common = {"dataset": "EMBC21", "group": group, "subject": subject, "method": method}
    result = []
    for tolerance in (25, 50, 100):
        tp, fp, fn = [sum(r[f"{key}_{tolerance}ms"] for r in events) for key in ("tp", "fp", "fn")]
        values = {"f1": 2 * tp / (2*tp + fp + fn) if 2*tp + fp + fn else float("nan"),
                  "precision": tp / (tp + fp) if tp + fp else float("nan"),
                  "recall": tp / (tp + fn) if tp + fn else float("nan")}
        for metric, value in values.items():
            result.append({**common, "metric": f"{metric}_{tolerance}ms", "duration_s": 0, "value": value,
                           "support": "all annotated interval intersections; empty predictions retained", "n_supported": tp,
                           "n_eligible": tp + fn, "coverage": tp / (tp + fn) if tp + fn else 0.})
    count = sum(r["matched_pp_count"] for r in events)
    total = sum(r["reference_pp_count"] for r in events)
    error = sum(r["matched_pp_mae_ms"] * r["matched_pp_count"] for r in events if r["matched_pp_count"]) / count if count else float("nan")
    for metric, value in (("matched_rr_mae_ms", error), ("matched_rr_coverage", count / total if total else float("nan"))):
        result.append({**common, "metric": metric, "duration_s": 0, "value": value,
                       "support": "PP intervals consecutive in both matched sequences at 50ms; legacy rr key means PP here",
                       "n_supported": count, "n_eligible": total, "coverage": count / total if total else 0.})
    for duration in (60, 300):
        eligible = [r for r in blocks if r["duration_s"] == duration and r["reference_eligible"]]
        for name in PP_FEATURES:
            errors = [r[name + "_absolute_error"] for r in eligible]
            n = int(np.isfinite(errors).sum())
            coverage = n / len(eligible) if eligible else float("nan")
            for suffix, value in (("mae", finite_mean(errors)), ("coverage", coverage)):
                result.append({**common, "metric": name + "_" + suffix, "duration_s": duration, "value": value,
                               "support": "continuous expert PPG PP windows; both sequences >=20 intervals; conditional error with coverage",
                               "n_supported": n, "n_eligible": len(eligible), "coverage": coverage})
    return result


def paired_participant(intervals, windows, subject, comparator, group, metric, duration):
    def belongs(row, method):
        return row["subject"] == subject and row["method"] == method and (group == "all" or row["activity"] == group)
    common = {"dataset": "EMBC21", "group": group, "subject": subject, "comparator": comparator,
              "metric": metric, "duration_s": duration}
    if metric.endswith("_mae") and duration:
        name = metric[:-4]
        mapped = []
        for method in ("AnyPPG", comparator):
            mapped.append({(r["record_id"], r["interval_index"], r["start_s"]): r for r in windows
                           if belongs(r, method) and r["duration_s"] == duration and r["reference_eligible"]})
        if set(mapped[0]) != set(mapped[1]):
            raise ValueError("PRV comparator reference windows differ")
        difference = []
        for key in sorted(mapped[0]):
            a, b = (m[key][name + "_absolute_error"] for m in mapped)
            if np.isfinite(a) and np.isfinite(b):
                difference.append(a - b)
        supported, eligible = len(difference), len(mapped[0])
        value = finite_mean(difference)
        support = "common finite continuous PRV windows"
    elif metric == "matched_rr_mae_ms":
        mapped = [{(r["record_id"], r["interval_index"]): r for r in intervals if belongs(r, method)} for method in ("AnyPPG", comparator)]
        if set(mapped[0]) != set(mapped[1]):
            raise ValueError("PP comparator annotated intervals differ")
        difference, eligible = [], 0
        for key in sorted(mapped[0]):
            a, b = mapped[0][key], mapped[1][key]
            if a["reference_pp_count"] != b["reference_pp_count"] or a["start_s"] != b["start_s"] or a["end_s"] != b["end_s"]:
                raise ValueError("PP comparator support differs")
            eligible += a["reference_pp_count"]
            _, ia, ib = np.intersect1d(a["_reference_pair_index"], b["_reference_pair_index"], assume_unique=True, return_indices=True)
            difference.extend((a["_pp_abs_errors"][ia] - b["_pp_abs_errors"][ib]).tolist())
        supported, value = len(difference), finite_mean(difference)
        support = "common consecutive expert-reference PP intervals at 50ms"
    else:
        pair = []
        for method in ("AnyPPG", comparator):
            pair.append(next(r for r in participant_metrics(intervals, windows, subject, method, group)
                             if r["metric"] == metric and r["duration_s"] == duration))
        value = pair[0]["value"] - pair[1]["value"]
        supported = eligible = pair[0]["n_eligible"]
        support = "common complete annotated support; failure/coverage differences retained"
    return {**common, "difference": value, "n_common_supported": supported, "n_eligible": eligible,
            "common_support_fraction": supported / eligible if eligible else 0., "support": support}


def aggregate_people(people, paired=False):
    groups = defaultdict(list)
    identity = ("dataset", "group", "comparator" if paired else "method", "metric", "duration_s")
    for row in people:
        groups[tuple(row[k] for k in identity)].append(row)
    output = []
    value_key = "difference" if paired else "value"
    for key, rows in groups.items():
        values = [r[value_key] for r in rows]
        lo, hi = bootstrap(values)
        output.append({**dict(zip(identity, key)), "n_subjects": len(rows),
                       "n_finite_subjects": int(np.isfinite(values).sum()),
                       "difference" if paired else "mean": finite_mean(values),
                       "ci95_low": lo, "ci95_high": hi, "support": rows[0]["support"],
                       "n_supported": sum(r["n_common_supported" if paired else "n_supported"] for r in rows),
                       "n_eligible": sum(r["n_eligible"] for r in rows),
                       "participant_mean_coverage": finite_mean([r["common_support_fraction" if paired else "coverage"] for r in rows])})
    return output


def write_results(output, intervals, windows, coverage, subjects):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    people, pairs = [], []
    event_index, window_index = defaultdict(list), defaultdict(list)
    for row in intervals:
        event_index[row["subject"]].append(row)
    for row in windows:
        window_index[row["subject"]].append(row)
    for subject in subjects:
        events, blocks = event_index[subject], window_index[subject]
        for group in GROUPS:
            own = []
            for method in METHODS.values():
                own.extend(participant_metrics(events, blocks, subject, method, group))
            people.extend(own)
            for comparator in ("SmallCNN", "MSPTD"):
                for row in own:
                    if row["method"] != "AnyPPG":
                        continue
                    pairs.append(paired_participant(events, blocks, subject, comparator, group, row["metric"], row["duration_s"]))
    tables = {"intervals.csv": intervals, "prv_windows.csv": windows, "coverage.csv": coverage,
              "per_subject.csv": people, "paired_per_subject.csv": pairs,
              "summary.csv": aggregate_people(people), "paired.csv": aggregate_people(pairs, paired=True)}
    for filename, records in tables.items():
        write_csv(output / filename, records)
    return tables


def independent_record_identity(record):
    """Recompute the runner's declared byte identity without importing it."""
    arrays = {}
    for key in ("raw_ppg", "input_ppg", "onset_times_s", "annotated_intervals_s"):
        value = np.ascontiguousarray(record[key])
        arrays[key] = {"dtype": str(value.dtype), "shape": list(value.shape),
                       "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    identity = {"record_id": str(record["record_id"]), "subject": str(record["subject"]),
                "activity": str(record["activity"]), "fs": float(record["fs"]),
                "duration_s": float(record["duration_s"]), "arrays": arrays}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {**identity, "record_sha256": digest}


def verify_protocol(protocol, records):
    subjects = sorted({str(r["subject"]) for r in records})
    identities = {str(r["record_id"]): independent_record_identity(r) for r in records}
    if len(identities) != len(records) or len(subjects) != 7 or len(records) != 105:
        raise ValueError("Expected 105 distinct acquisition records from seven people")
    if protocol["version"] != "manual-wrist-onset-poc-v1" or protocol["models"] != list(METHODS):
        raise ValueError("Unrecognized manual onset protocol or model set")
    if protocol["records"] != identities:
        raise ValueError("Protocol record bytes or identities changed")
    if any(r["activity"] not in GROUPS[1:] for r in records):
        raise ValueError("Unrecognized acquisition activity")
    jobs = protocol["jobs"]
    if len(jobs) != 7 or len({j["id"] for j in jobs}) != 7:
        raise ValueError("Expected seven distinct LOSO jobs")
    for fold, subject in enumerate(subjects):
        remaining = [s for s in subjects if s != subject]
        validation = str(np.random.default_rng(42 + fold).choice(remaining))
        expected = {"id": f"EMBC21-fold{fold}-seed42", "fold": fold, "seed": 42,
                    "scenario": "within", "source": "EMBC21", "target": "EMBC21", "test": [subject],
                    "validation": [validation], "train": [s for s in remaining if s != validation]}
        if jobs[fold] != expected:
            raise ValueError("LOSO split or validation participant differs from the fixed protocol")
    return subjects


def verify_data_and_code(protocol, data_root):
    manifest_path = data_root / "manifest.json"
    if sha256(manifest_path) != protocol["data_manifest_sha256"]:
        raise ValueError("Data manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["adapter_sha256"] != sha256(ROOT / "scripts/ppg_embc21_data.py"):
        raise ValueError("Data adapter changed after the native-data audit")
    checked = []
    for item in manifest["files"]:
        path = (data_root / item["path"]).resolve()
        if not path.is_relative_to(data_root.resolve()) or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Source file identity changed: {item['path']}")
        checked.append(item["path"])
    for name, digest in protocol["code_sha256"].items():
        if sha256(ROOT / "scripts" / name) != digest:
            raise ValueError(f"Registered experimental code changed: {name}")
    for path, key in (("models/weights/anyppg/anyppg_ckpt.pth", "weight_sha256"),
                      ("repos/anyppg/load_anyppg/resnet1d.py", "encoder_source_sha256")):
        if sha256(ROOT / path) != protocol[key]:
            raise ValueError(f"Frozen encoder asset changed: {path}")
    return checked


def verify_frame_cache(root, protocol_hash, records):
    directory = root / "frames"
    complete = json.loads((directory / "complete.json").read_text())
    if complete["protocol_sha256"] != protocol_hash:
        raise ValueError("Frame cache protocol changed")
    for filename, digest in complete["sha256"].items():
        if sha256(directory / filename) != digest:
            raise ValueError(f"Frame cache bytes changed: {filename}")
    with np.load(directory / "rows.npz", allow_pickle=False) as z:
        rows = {k: z[k] for k in ("subject", "record_id", "start_s")}
    expected = []
    for record in sorted(records, key=lambda r: r["record_id"]):
        n = max(0, int(np.floor((len(record["raw_ppg"]) / record["fs"] - 8) / 6)) + 1)
        for start in 6. * np.arange(n):
            if any(start >= left and start + 8 <= right for left, right in record["annotated_intervals_s"]):
                expected.append((str(record["subject"]), str(record["record_id"]), start))
    actual = list(zip(rows["subject"].tolist(), rows["record_id"].tolist(), rows["start_s"].tolist()))
    if actual != expected or complete["frames"] != len(expected):
        raise ValueError("Frame identities cross annotation gaps or differ from native sampling")
    return rows, complete["sha256"]


def selected_indices(rows, subjects, limit):
    result = []
    for subject in subjects:
        candidates = np.flatnonzero(rows["subject"] == subject)
        if not len(candidates):
            raise ValueError("Training/validation participant has no annotated context")
        ranks = np.unique(np.linspace(0, len(candidates) - 1, min(limit, len(candidates))).astype(int))
        result.extend(candidates[ranks].tolist())
    return result


def verify_fit(directory, fit, job, method, protocol_hash, rows, protocol):
    if fit["identity"] != {"protocol_sha256": protocol_hash, "job": job, "method": method}:
        raise ValueError("Fit identity or participant split changed")
    if method == "MSPTDfast":
        if fit["traditional"] != protocol["traditional"]:
            raise ValueError("Traditional configuration changed")
        return
    if sha256(directory / "head.pt") != fit["head_sha256"]:
        raise ValueError("Fitted task head changed")
    for label, subjects, limit in (("train", job["train"], 256), ("validation", job["validation"], 128)):
        indices = selected_indices(rows, subjects, limit)
        if fit[label + "_frame_indices"] != indices or fit[label + "_frames"] != len(indices):
            raise ValueError("Fitting frame selection or participant budget changed")
    expected_counts = {s: int(np.sum(rows["subject"][fit["train_frame_indices"]] == s)) for s in job["train"]}
    if fit["train_frames_by_person"] != expected_counts:
        raise ValueError("Training participant weights/counts changed")
    history, thresholds = fit["history"], protocol["fit"]["thresholds"]
    if not history or len(history) > 20:
        raise ValueError("Invalid fitting history")
    best_index = int(np.argmax([r["validation_f1_50ms"] for r in history]))
    for epoch, row in enumerate(history, 1):
        scores = [row["threshold_scores"][str(t)] for t in thresholds]
        winner = int(np.argmax(scores))
        if row["epoch"] != epoch or not np.isfinite(scores).all() or row["threshold"] != thresholds[winner] or row["validation_f1_50ms"] != scores[winner]:
            raise ValueError("Validation threshold selection does not use the declared first maximum")
    selected = history[best_index]
    if fit["best_epoch"] != selected["epoch"] or fit["threshold"] != selected["threshold"] or fit["validation_f1_50ms"] != selected["validation_f1_50ms"]:
        raise ValueError("Task head epoch/threshold was not selected by declared validation rule")


def verify_prediction(path, metadata, record, fit_hash, method, fit):
    identity = {"fit_sha256": fit_hash, "subject": str(record["subject"]), "record_id": str(record["record_id"]),
                "target": "EMBC21", "record_sha256": independent_record_identity(record)["record_sha256"]}
    if metadata["identity"] != identity or sha256(path) != metadata["sha256"]:
        raise ValueError("Prediction identity or bytes changed")
    with np.load(path, allow_pickle=False) as z:
        events = np.asarray(z["event_times_s"], dtype=float)
        start, end = float(z["start_s"]), float(z["end_s"])
        n = max(0, int(np.floor((len(record["raw_ppg"]) / record["fs"] - 8) / 6)) + 1)
        if start != 1. or end != 1. + 6 * n or metadata["events"] != len(events):
            raise ValueError("Unequal inference boundaries or invalid event count")
        if method != "MSPTDfast":
            from scipy.signal import find_peaks
            probability = z["probability"]
            if float(z["probability_fs"]) != 64 or probability.shape != (n * 384,) or not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
                raise ValueError("Invalid dense onset probabilities")
            expected = start + find_peaks(probability, height=fit["threshold"], distance=18)[0] / 64.
            if not np.array_equal(events, expected):
                raise ValueError("Event times differ from fixed continuous probability peak extraction")
    return events, start, end


def summarize(root, data_root, output):
    root, data_root, output = Path(root), Path(data_root), Path(output)
    protocol_path = root / "protocol.json"
    protocol, protocol_hash = json.loads(protocol_path.read_text(encoding="utf-8")), sha256(protocol_path)
    source_files = verify_data_and_code(protocol, data_root)
    records = load_records(data_root)
    subjects = verify_protocol(protocol, records)
    rows, cache_hashes = verify_frame_cache(root, protocol_hash, records)
    intervals, windows, coverage, artifacts = [], [], [], {}
    scored = set()
    for job in protocol["jobs"]:
        test_records = [r for r in records if str(r["subject"]) in job["test"]]
        for method, display in METHODS.items():
            directory = root / "runs" / job["id"] / method
            fit_path = directory / "fit.json"
            fit, fit_hash = json.loads(fit_path.read_text()), sha256(fit_path)
            verify_fit(directory, fit, job, method, protocol_hash, rows, protocol)
            artifacts[fit_path.relative_to(root).as_posix()] = fit_hash
            if method != "MSPTDfast":
                artifacts[(directory / "head.pt").relative_to(root).as_posix()] = fit["head_sha256"]
            expected_names = {r["record_id"] + ".npz" for r in test_records}
            if {p.name for p in directory.glob("*.npz")} != expected_names:
                raise ValueError("Unexpected/missing test acquisition prediction files")
            for record in test_records:
                key = (record["record_id"], method)
                if key in scored:
                    raise ValueError("Acquisition evaluated more than once")
                scored.add(key)
                path = directory / (record["record_id"] + ".npz")
                meta_path = path.with_suffix(".json")
                meta = json.loads(meta_path.read_text())
                prediction, start, end = verify_prediction(path, meta, record, fit_hash, method, fit)
                events, blocks, support = score_record(record, display, prediction, start, end)
                intervals.extend(events)
                windows.extend(blocks)
                coverage.append(support)
                artifacts[path.relative_to(root).as_posix()] = meta["sha256"]
                artifacts[meta_path.relative_to(root).as_posix()] = sha256(meta_path)
    if len(scored) != 315:
        raise ValueError("Not all 105 acquisitions have all three methods")
    tables = write_results(output, intervals, windows, coverage, subjects)
    supports = []
    for subject in subjects:
        for group in GROUPS:
            for duration in (60, 300):
                own = [w for w in windows if w["subject"] == subject and w["method"] == "AnyPPG" and w["duration_s"] == duration and (group == "all" or w["activity"] == group)]
                supports.append({"subject": subject, "group": group, "duration_s": duration,
                                 "complete_windows": len(own), "reference_eligible_windows": sum(w["reference_eligible"] for w in own)})
    write_csv(output / "reference_support.csv", supports)
    audit = {"status": "passed", "dataset": "EMBC21", "participants": subjects,
             "test_each_participant_exactly_once": True, "record_method_predictions": len(scored),
             "protocol_sha256": protocol_hash, "source_files_verified": len(source_files),
             "frame_cache_sha256": cache_hashes, "artifacts_sha256": artifacts,
             "scoring_sha256": sha256(Path(__file__)), "metric_code_sha256": sha256(ROOT / "scripts/ppg_beat_metrics.py"),
             "tables": {k: {"rows": len(v), "sha256": sha256(output / k)} for k, v in tables.items()},
             "support_table_sha256": sha256(output / "reference_support.csv"),
             "reference": "Single expert PPG diastolic onset; valid annotated regions only; fixed original clock, no test alignment",
             "prv": "60s continuous PP agreement is exploratory, not clinical ECG HRV; no deletion of predicted beats; >=20 intervals in each sequence",
             "window_grid": "Each annotated/inference intersection anchors its own complete nonoverlapping 60/300s blocks; never joins records or gaps",
             "coverage": "F1 includes all empty predictions; PP MAE is conditional and paired on common reference pairs; PRV paired only on common finite windows",
             "uncertainty": "10000 participant bootstrap draws, equal participant weight, seed20261104; finite-participant conditional support explicitly reported; CI unavailable with fewer than two finite participants; exploratory, no multiplicity adjustment",
             "bootstrap_300s_warning": "Too few people with continuous 300s support for a seven-person standard 5min comparison; inspect reference_support.csv and n_finite_subjects"}
    write_json(output / "audit.json", audit)
    print(json.dumps({"status": "passed", "participants": len(subjects), "record_method_predictions": len(scored),
                      "output": str(output)}, indent=2))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / ".local/ppg-phase2/onsets")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/embc21")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/ppg-phase2/onsets/summary")
    args = parser.parse_args()
    summarize(args.root, args.data, args.output)


if __name__ == "__main__":
    main()
