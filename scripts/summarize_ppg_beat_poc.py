"""Independent event/RR-variability scoring with explicit reference coverage."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ppg_beat_metrics import evaluate_event_frame
from ppg_hr_wild_reference import sha256, write_json

METHODS = ("AnyPPG", "SmallCNN", "Elgendi")
SCOPES = ("reference_valid", "reference_strict99", "all_reference")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot publish an empty result table")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_true(value):
    return str(value).lower() in ("true", "1")


def finite_mean(values):
    values = np.asarray(values, float)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float("nan")


def bootstrap(values, samples=10000):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(20261104)
    return np.quantile(values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1), [.025, .975]).tolist()


def reference_eligible(row, scope):
    if scope not in SCOPES:
        raise ValueError("Unknown reference scope")
    if scope == "all_reference":
        return True
    if not row["reference_valid"]:
        return False
    if scope == "reference_strict99" and row["dataset"] == "WildPPG":
        return float(row.get("xqrs_agreement_f1", float("nan"))) >= .99
    return True


def participant_result(rows, dataset, scenario, subject, method, duration, scope):
    selected = [r for r in rows if r["dataset"] == dataset and r["scenario"] == scenario
                and r["subject"] == subject and r["method"] == method and r["duration_s"] == duration]
    eligible = [r for r in selected if r["inside_inference"] and reference_eligible(r, scope)]
    result = {"dataset": dataset, "scenario": scenario, "subject": subject, "method": method,
              "duration_s": duration, "scope": scope, "candidate_windows": len(selected),
              "reference_valid_windows": sum(r["reference_valid"] for r in selected),
              "scope_reference_valid_windows": sum(reference_eligible(r, scope) for r in selected),
              "inside_inference_windows": sum(r["inside_inference"] for r in selected),
              "evaluated_windows": len(eligible)}
    result["reference_valid_fraction"] = result["reference_valid_windows"] / len(selected) if selected else 0.
    result["scope_reference_valid_fraction"] = result["scope_reference_valid_windows"] / len(selected) if selected else 0.
    result["inference_window_fraction"] = result["inside_inference_windows"] / len(selected) if selected else 0.
    result["evaluated_window_fraction"] = len(eligible) / len(selected) if selected else 0.
    for tolerance in (50, 100, 150):
        suffix = f"{tolerance}ms"
        tp, fp, fn = [sum(r.get(f"{k}_{suffix}", 0) for r in eligible) for k in ("tp", "fp", "fn")]
        result.update({f"tp_{suffix}": tp, f"fp_{suffix}": fp, f"fn_{suffix}": fn,
                       f"precision_{suffix}": tp / (tp + fp) if tp + fp else float("nan"),
                       f"recall_{suffix}": tp / (tp + fn) if tp + fn else float("nan"),
                       f"f1_{suffix}": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")})
    rr_count = sum(r.get("n_rr_errors", 0) for r in eligible)
    ref_pairs = sum(max(0, r.get("n_reference_events", 0) - 1) for r in eligible)
    result["matched_rr_count"] = rr_count
    result["matched_rr_mae_ms"] = sum(r["rr_mae_ms"] * r["n_rr_errors"] for r in eligible if r.get("n_rr_errors", 0)) / rr_count if rr_count else float("nan")
    result["matched_rr_coverage"] = rr_count / ref_pairs if ref_pairs else float("nan")
    for name in ("mean_rr_ms", "sdnn_rr_ms", "rmssd_rr_ms"):
        key = name + "_absolute_error"
        values = [r.get(key, float("nan")) for r in eligible]
        result[name + "_mae"] = finite_mean(values)
        result[name + "_coverage"] = float(np.isfinite(values).mean()) if values else 0.
        result[name + "_bias"] = finite_mean([r.get("prediction_" + name, float("nan")) - r.get("reference_" + name, float("nan")) for r in eligible])
    return result


def paired_participant(rows, dataset, scenario, subject, comparator, duration, scope, metric):
    """Compare identical windows, or identical reference pairs for matched RR.

Every unsupported window/interval stays in the reported coverage denominator.
PRV errors use only common finite predictions, while F1 retains empty outputs as
failures. This is conditional accuracy plus coverage, not an imputed error for
missing estimates. No reference-valid window is selected using PPG performance.
"""
    selected = [r for r in rows if (r["dataset"], r["scenario"], r["subject"], r["duration_s"]) ==
                (dataset, scenario, subject, duration) and r["method"] in ("AnyPPG", comparator)]
    by_method = {}
    for method in ("AnyPPG", comparator):
        windows = {r["start_s"]: r for r in selected if r["method"] == method}
        if len(windows) != sum(r["method"] == method for r in selected):
            raise ValueError("Duplicate subject/method/window")
        by_method[method] = windows
    left, right = by_method["AnyPPG"], by_method[comparator]
    if set(left) != set(right):
        raise ValueError("Comparator candidate windows differ")
    eligible = []
    for start in sorted(left):
        a, b = left[start], right[start]
        if a["reference_valid"] != b["reference_valid"] or reference_eligible(a, scope) != reference_eligible(b, scope):
            raise ValueError("Comparator reference eligibility differs")
        if a["inside_inference"] and b["inside_inference"] and reference_eligible(a, scope):
            eligible.append((a, b))
    result = {"dataset": dataset, "scenario": scenario, "subject": subject, "comparator": comparator,
              "duration_s": duration, "scope": scope, "metric": metric,
              "candidate_windows": len(left), "eligible_common_windows": len(eligible),
              "common_finite_windows": 0, "common_reference_pairs": 0,
              "pair_denominator": len(eligible), "support": "common_reference_valid_windows"}
    difference = float("nan")
    if metric == "matched_rr_mae_ms":
        errors = []
        denominator = 0
        supported_windows = 0
        for a, b in eligible:
            if a["n_reference_events"] != b["n_reference_events"]:
                raise ValueError("Comparator reference event counts differ")
            denominator += max(0, a["n_reference_events"] - 1)
            ai, bi = a["_rr_reference_pair_index"], b["_rr_reference_pair_index"]
            common, ia, ib = np.intersect1d(ai, bi, assume_unique=True, return_indices=True)
            if len(common):
                supported_windows += 1
                errors.extend((a["_rr_abs_error_ms"][ia] - b["_rr_abs_error_ms"][ib]).tolist())
        difference = float(np.mean(errors)) if errors else float("nan")
        result.update({"common_finite_windows": supported_windows, "common_reference_pairs": len(errors),
                       "pair_denominator": denominator, "support": "common_consecutive_reference_pairs"})
    elif metric in ("mean_rr_ms_mae", "sdnn_rr_ms_mae", "rmssd_rr_ms_mae"):
        key = metric[:-4] + "_absolute_error"
        errors = [a[key] - b[key] for a, b in eligible if np.isfinite(a[key]) and np.isfinite(b[key])]
        difference = float(np.mean(errors)) if errors else float("nan")
        result.update({"common_finite_windows": len(errors), "support": "common_finite_PRV_windows"})
    else:
        # F1/coverage must include all common eligible windows, even when a
        # method produces no event or no finite variability estimate.
        common_rows = [r for pair in eligible for r in pair]
        a = participant_result(common_rows, dataset, scenario, subject, "AnyPPG", duration, scope)
        b = participant_result(common_rows, dataset, scenario, subject, comparator, duration, scope)
        difference = a[metric] - b[metric]
        result["common_finite_windows"] = len(eligible)
    support_count = result["common_reference_pairs"] if metric == "matched_rr_mae_ms" else result["common_finite_windows"]
    result["common_support_fraction"] = support_count / result["pair_denominator"] if result["pair_denominator"] else 0.
    result["difference"] = difference
    return result


def paired_summary(records, samples=10000):
    values = np.asarray([r["difference"] for r in records])
    ci = bootstrap(values, samples)
    return {"n_subjects": len(values), "n_finite_subjects": int(np.isfinite(values).sum()),
            "difference": finite_mean(values), "ci95_low": ci[0], "ci95_high": ci[1],
            "common_finite_windows": sum(r["common_finite_windows"] for r in records),
            "eligible_common_windows": sum(r["eligible_common_windows"] for r in records),
            "common_reference_pairs": sum(r["common_reference_pairs"] for r in records),
            "participant_mean_common_support": finite_mean([r["common_support_fraction"] for r in records]),
            "support": records[0]["support"] if records else "none"}


def summarize(root, reference, output):
    root, reference, output = Path(root), Path(reference), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((root / "protocol.json").read_text())
    phash = sha256(root / "protocol.json")
    verified = {"protocol.json": phash}
    expected_people = {dataset: {p.parent.name for p in reference.glob(f"{dataset}/*/manifest.json")}
                       for dataset in ("Dalia", "WildPPG")}
    observed_people = {dataset: [] for dataset in expected_people}
    for job in protocol["jobs"]:
        if job["source"] != "Dalia" or (job["scenario"], job["target"]) not in (("within", "Dalia"), ("transfer", "WildPPG")):
            raise ValueError("Unexpected phase-two scenario")
        observed_people[job["target"]].extend(job["test"])
    if any(len(observed_people[d]) != len(set(observed_people[d])) or set(observed_people[d]) != expected_people[d] for d in expected_people):
        raise ValueError("Test participant coverage duplicated or incomplete")
    for relative, digest in protocol["reference_manifests"].items():
        path = reference / relative
        if sha256(path) != digest:
            raise ValueError("Reference manifest changed")
        verified["reference/" + relative] = digest
        manifest = json.loads(path.read_text())
        for name, expected in manifest["outputs_sha256"].items():
            if sha256(path.parent / name) != expected:
                raise ValueError("Reference arrays changed")
            verified["reference/" + (path.parent / name).relative_to(reference).as_posix()] = expected
    rows, completed = [], 0
    for job in protocol["jobs"]:
        train, validation, test = (set(job[k]) for k in ("train", "validation", "test"))
        if train & validation or (job["source"] == job["target"] and (train | validation) & test):
            raise ValueError("Subject leakage")
        for method in METHODS:
            directory = root / "runs" / job["id"] / method
            fit = json.loads((directory / "fit.json").read_text())
            if fit["identity"] != {"protocol_sha256": phash, "job": job, "method": method}:
                raise ValueError("Wrong fitted method/job identity")
            verified[(directory / "fit.json").relative_to(root).as_posix()] = sha256(directory / "fit.json")
            if method == "Elgendi" and set(fit["pat"]["subjects"]) != set(job["train"]):
                raise ValueError("PAT used wrong people")
            if method != "Elgendi" and sha256(directory / "head.pt") != fit["head_sha256"]:
                raise ValueError("Head changed")
            if method != "Elgendi":
                verified[(directory / "head.pt").relative_to(root).as_posix()] = fit["head_sha256"]
            for subject in job["test"]:
                path = directory / f"{subject}.npz"
                meta = json.loads((directory / f"{subject}.json").read_text())
                if meta["identity"] != {"fit_sha256": sha256(directory / "fit.json"), "subject": subject, "target": job["target"]} or sha256(path) != meta["sha256"]:
                    raise ValueError("Prediction identity/hash changed")
                verified[path.relative_to(root).as_posix()] = meta["sha256"]
                verified[(directory / f"{subject}.json").relative_to(root).as_posix()] = sha256(directory / f"{subject}.json")
                with np.load(path) as data:
                    predicted = data["event_times_s"]
                    lo, hi = float(data["start_s"]), float(data["end_s"])
                if predicted.ndim != 1 or not np.isfinite([lo, hi]).all() or lo < 0 or hi <= lo or not np.isfinite(predicted).all() or (np.diff(predicted) <= 0).any() or (predicted < lo).any() or (predicted >= hi).any():
                    raise ValueError("Invalid predicted event sequence")
                ref_dir = reference / job["target"] / subject
                with np.load(ref_dir / "beats.npz") as data:
                    ref = data["r_peak_times_s"]
                    if "duration_s" in data and hi > float(data["duration_s"]) + 1e-8:
                        raise ValueError("Inference extends beyond the reference recording")
                if ref.ndim != 1 or not np.isfinite(ref).all() or (np.diff(ref) <= 0).any():
                    raise ValueError("Invalid reference event sequence")
                for window in read_csv(ref_dir / "windows.csv"):
                    start, duration = float(window["start_s"]), int(float(window["duration_s"]))
                    inside = start >= lo and start + duration <= hi
                    result = {"dataset": job["target"], "scenario": job["scenario"], "subject": subject,
                              "method": method, "job": job["id"], "start_s": start, "duration_s": duration,
                              "reference_valid": is_true(window["reference_valid"]),
                              "reference_quality": window["reference_quality"],
                              "xqrs_agreement_f1": float(window.get("xqrs_agreement_f1", "") or "nan"),
                              "reference_exclusion_reason": window["exclusion_reason"], "inside_inference": inside}
                    if inside:
                        ra, rb = np.searchsorted(ref, [start, start + duration], side="left")
                        pa, pb = np.searchsorted(predicted, [start, start + duration], side="left")
                        result.update(evaluate_event_frame(ref[ra:rb], predicted[pa:pb], start, start + duration, return_rr_detail=True))
                        # Independently computed interval metrics must agree with the reference manifest.
                        for output_key, reference_key in (("reference_mean_rr_ms", "mean_rr_ms"),
                                                          ("reference_sdnn_rr_ms", "sdnn_ms"),
                                                          ("reference_rmssd_rr_ms", "rmssd_ms")):
                            expected = float(window[reference_key] or "nan")
                            if not np.isclose(result[output_key], expected, rtol=1e-9, atol=1e-7, equal_nan=True):
                                raise ValueError("Independent RR reference recalculation disagrees")
                    rows.append(result)
                completed += 1
                print(f"Scored {completed}/93: {job['target']} {subject} {method}", flush=True)
    if completed != 93:
        raise ValueError(f"Expected 93 subject-method predictions, got {completed}")
    keys = list(dict.fromkeys(k for row in rows for k in row if not k.startswith("_")))
    write_csv(output / "windows.csv", [{k: r.get(k, "") for k in keys} for r in rows])
    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["scenario"], row["subject"]), []).append(row)
    people = []
    for dataset, scenario in (("Dalia", "within"), ("WildPPG", "transfer")):
        subjects = sorted({r["subject"] for r in rows if r["dataset"] == dataset})
        for subject in subjects:
            for method in METHODS:
                for duration in (60, 300):
                    for scope in SCOPES:
                        people.append(participant_result(grouped[(dataset, scenario, subject)], dataset, scenario, subject, method, duration, scope))
    write_csv(output / "per_subject.csv", people)
    metrics = ("f1_50ms", "f1_100ms", "f1_150ms", "precision_100ms", "recall_100ms", "matched_rr_mae_ms", "matched_rr_coverage",
               "mean_rr_ms_mae", "sdnn_rr_ms_mae", "rmssd_rr_ms_mae", "mean_rr_ms_coverage", "sdnn_rr_ms_coverage", "rmssd_rr_ms_coverage")
    coverage_metrics = ("reference_valid_fraction", "scope_reference_valid_fraction", "inference_window_fraction", "evaluated_window_fraction")
    summaries, paired, paired_people = [], [], []
    for dataset, scenario in (("Dalia", "within"), ("WildPPG", "transfer")):
        for duration in (60, 300):
            for scope in SCOPES:
                subset = [r for r in people if (r["dataset"], r["duration_s"], r["scope"]) == (dataset, duration, scope)]
                for method in METHODS:
                    selected = [r for r in subset if r["method"] == method]
                    for metric in (*metrics, *coverage_metrics):
                        values = np.asarray([r[metric] for r in selected])
                        ci = bootstrap(values)
                        summaries.append({"dataset": dataset, "scenario": scenario, "duration_s": duration,
                                          "scope": scope, "method": method, "metric": metric,
                                          "n_subjects": len(values), "n_finite_subjects": int(np.isfinite(values).sum()),
                                          "mean": finite_mean(values), "ci95_low": ci[0], "ci95_high": ci[1]})
                for comparator in ("SmallCNN", "Elgendi"):
                    for metric in metrics:
                        subjects = sorted({r["subject"] for r in subset})
                        comparison = [paired_participant(grouped[(dataset, scenario, s)], dataset, scenario, s, comparator, duration, scope, metric) for s in subjects]
                        paired_people.extend(comparison)
                        paired.append({"dataset": dataset, "scenario": scenario, "duration_s": duration, "scope": scope,
                                       "comparator": comparator, "metric": metric, **paired_summary(comparison)})
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired.csv", paired)
    write_csv(output / "paired_per_subject.csv", paired_people)
    audit = {"status": "complete", "subject_method_predictions": completed,
             "reference_manifests": len(protocol["reference_manifests"]), "verified_predictions": verified,
             "scoring_code_sha256": {p.name: sha256(p) for p in (Path(__file__), ROOT / "scripts/ppg_beat_metrics.py")},
             "table_sha256": {p.name: sha256(p) for p in output.glob("*.csv")},
             "metric_support": {"single_method_PRV_MAE": "Own finite windows, conditional on reported prediction coverage",
                                "paired_PRV_MAE": "Common finite windows first, then participant mean difference",
                                "paired_matched_RR_MAE": "Common consecutive reference intervals first, then participant mean difference",
                                "F1": "All common eligible windows, including empty predictions as failures",
                                "scope": "reference_valid is primary (Wild ECG detector F1>=.95); reference_strict99 additionally requires Wild F1>=.99 while retaining DaLiA manual policy; all_reference sensitivity includes unreliable ECG labels"},
             "limitation": "Participant bootstrap conditional on fixed single seed/folds; no multiplicity correction. Matched RR error is conditional and requires its coverage. Raw RR/PP statistics are not NN or equivalent physiological HRV."}
    write_json(output / "audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / ".local/ppg-phase2/beats")
    parser.add_argument("--reference", type=Path, default=ROOT / ".local/ppg-phase2/reference-v3")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/ppg-phase2/beats/summary")
    args = parser.parse_args()
    summarize(args.root, args.reference, args.output)


if __name__ == "__main__":
    main()
