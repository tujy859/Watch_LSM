"""Describe the fixed DaLiA -> WildPPG HR transfer failure without refitting.

All diagnosis is post hoc. Target labels are used to describe errors only, never
to select models. Transfer and within-Wild predictions are compared on identical
target rows. This separates target population composition from the observed
transfer penalty, but does not identify device, cohort or acquisition causality.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("AnyPPG", "SmallCNN", "MOMENT-1-small", "TrainMean")
SEEDS = (42, 43, 44)
HR_EDGES = np.array([60., 80., 100., 120.])
HR_NAMES = ("lt60", "60to80", "80to100", "100to120", "ge120")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as values:
        return {k: values[k] for k in values.files}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, records):
    keys = list(dict.fromkeys(k for row in records for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def subject_weights(subjects):
    """Weights sum to one, with equal total mass per participant."""
    _, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    return 1. / (len(counts) * counts[inverse])


def weighted_quantile(values, weights, quantiles):
    order = np.argsort(values, kind="stable")
    x, w = np.asarray(values)[order], np.asarray(weights)[order]
    if not np.isfinite(x).all() or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid weighted distribution")
    cumulative = np.cumsum(w) / w.sum()
    return x[np.searchsorted(cumulative, quantiles, side="left").clip(0, len(x) - 1)]


def describe_distribution(values, subjects):
    w = subject_weights(subjects)
    mean = float(w @ values)
    q = weighted_quantile(values, w, [.01, .05, .25, .5, .75, .95, .99])
    return {"subjects": int(len(np.unique(subjects))), "windows": len(values),
            "mean": mean, "sd": float(np.sqrt(w @ ((values - mean) ** 2))),
            "min": float(np.min(values)), "max": float(np.max(values)),
            **{f"p{int(p)}": float(v) for p, v in zip((1, 5, 25, 50, 75, 95, 99), q)}}


def paired_bootstrap(values, n=10000):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite participant differences")
    samples = np.random.default_rng(20261104).choice(x, size=(n, len(x)), replace=True).mean(1)
    lo, hi = np.quantile(samples, [.025, .975])
    return float(x.mean()), float(lo), float(hi)


def calibration(y, predictions, subjects):
    """Descriptive regression only; never apply this regression to predictions."""
    w = subject_weights(subjects)
    ym, pm = float(w @ y), float(w @ predictions)
    vy, vp = float(w @ ((y - ym) ** 2)), float(w @ ((predictions - pm) ** 2))
    cov = float(w @ ((y - ym) * (predictions - pm)))
    return {"reference_mean_bpm": ym, "prediction_mean_bpm": pm,
            "reference_sd_bpm": np.sqrt(vy), "prediction_sd_bpm": np.sqrt(vp),
            "bias_bpm": pm - ym,
            "prediction_on_reference_slope": cov / vy if vy else None,
            "correlation": cov / np.sqrt(vy * vp) if vy and vp else None,
            "mae_bpm": float(w @ np.abs(y - predictions))}


def stratified_pair(y, within, transfer, subjects, stratum, levels):
    """Exact additive contribution to full macro MAE gap, plus conditional MAE.

Conditional MAEs average only participants who have windows in that stratum.
Contribution gives each participant the same full-window denominator; summing
over exhaustive strata equals full participant-macro transfer-minus-within MAE.
"""
    output = []
    people = np.unique(subjects)
    for name, value in levels:
        contributions, conditional, frequencies = [], [], []
        nwindows = 0
        for person in people:
            own = subjects == person
            chosen = own & (stratum == value)
            n = int(chosen.sum())
            nwindows += n
            frequencies.append(n / int(own.sum()))
            difference = np.abs(transfer[chosen] - y[chosen]) - np.abs(within[chosen] - y[chosen])
            contributions.append(float(difference.sum() / own.sum()))
            if n:
                conditional.append([np.mean(np.abs(within[chosen] - y[chosen])),
                                    np.mean(np.abs(transfer[chosen] - y[chosen])),
                                    np.mean(within[chosen] - y[chosen]),
                                    np.mean(transfer[chosen] - y[chosen])])
        averages = np.asarray(conditional).mean(0) if conditional else [None] * 4
        output.append({"stratum": name, "n_subjects": len(conditional), "windows": nwindows,
                       "participant_mean_fraction": float(np.mean(frequencies)),
                       "within_mae_bpm": averages[0], "transfer_mae_bpm": averages[1],
                       "within_bias_bpm": averages[2], "transfer_bias_bpm": averages[3],
                       "gap_contribution_bpm": float(np.mean(contributions))})
    return output


def aggregate_seed_rows(records, keys):
    groups = {}
    for row in records:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for key, rows in groups.items():
        item = dict(zip(keys, key))
        item["n_seeds"] = len(rows)
        for field in rows[0]:
            if field in keys or field == "seed":
                continue
            values = [r[field] for r in rows if r[field] is not None]
            item[field] = float(np.mean(values)) if values else None
        result.append(item)
    return result


class Inputs:
    def __init__(self, root):
        self.root = Path(root)
        self.hashes = {}
        self.plan = self.json(self.root / "protocol.json")
        self.rows = {}
        for dataset in ("Dalia", "WildPPG"):
            path = self.root / "data" / dataset / "rows.npz"
            self.verify(path, self.plan["data_rows_sha256"][dataset])
            self.rows[dataset] = load_npz(path)
        self.predictions = {}

    def digest(self, path):
        path = Path(path)
        key = path.relative_to(self.root).as_posix()
        if key not in self.hashes:
            self.hashes[key] = sha256(path)
        return self.hashes[key]

    def json(self, path):
        self.digest(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def verify(self, path, expected):
        if self.digest(path) != expected:
            raise ValueError(f"Input hash mismatch: {path}")

    def prediction(self, scenario, dataset, method, seed):
        key = scenario, dataset, method, seed
        if key in self.predictions:
            return self.predictions[key]
        rows = self.rows[dataset]
        result = np.full(len(rows["hr"]), np.nan)
        count = np.zeros(len(result), int)
        jobs = [j for j in self.plan["jobs"] if j["scenario"] == scenario and j["target"] == dataset
                and j["budget"] == "full" and j["seed"] == seed]
        if len(jobs) != (5 if scenario == "within" else 1):
            raise ValueError("Incomplete protocol jobs")
        for job in jobs:
            folder = self.root / "runs" / job["id"] / method
            metadata = self.json(folder / "complete.json")
            identity = metadata["identity"]
            if identity["job"] != job or identity["method"] != method or identity["protocol_sha256"] != self.digest(self.root / "protocol.json"):
                raise ValueError("Prediction identity mismatch")
            path = folder / "predictions.npz"
            self.verify(path, metadata["predictions_sha256"])
            data = load_npz(path)
            index = data["row"]
            expected = np.flatnonzero(np.isin(rows["subject"], job["test"]))
            if not np.array_equal(index, expected):
                raise ValueError("Prediction row ordering mismatch")
            for field, other in (("subject", "subject"), ("start_s", "start_s"), ("reference", "hr")):
                if not np.array_equal(data[field], rows[other][index]):
                    raise ValueError("Prediction reference mismatch")
            result[index] = data["prediction"]
            count[index] += 1
        if not np.isfinite(result).all() or not (count == 1).all():
            raise ValueError("Predictions incomplete or repeated")
        self.predictions[key] = result
        return result


def feature_distance(features, mean, scale, chunk=4096):
    distance = np.empty(len(features))
    for start in range(0, len(features), chunk):
        z = (np.asarray(features[start:start + chunk], dtype=float) - mean) / scale
        distance[start:start + chunk] = np.sqrt(np.mean(z * z, axis=1))
    return distance


def feature_diagnosis(inputs):
    """Source-train only diagonal distance, calibrated on source validation."""
    features = {}
    for dataset in ("Dalia", "WildPPG"):
        directory = inputs.root / "features" / dataset
        metadata = inputs.json(directory / "AnyPPG.json")
        inputs.verify(directory / "AnyPPG.npy", metadata["features_sha256"])
        if metadata["identity"]["data"]["files_sha256"]["rows.npz"] != inputs.plan["data_rows_sha256"][dataset]:
            raise ValueError("Feature row identity mismatch")
        features[dataset] = np.load(directory / "AnyPPG.npy", mmap_mode="r")
    records, summaries = [], []
    for job in [j for j in inputs.plan["jobs"] if j["scenario"] == "transfer"]:
        source, target = inputs.rows["Dalia"], inputs.rows["WildPPG"]
        train = np.isin(source["subject"], job["train"])
        val = np.isin(source["subject"], job["validation"])
        x = np.asarray(features["Dalia"][train], dtype=float)
        w = subject_weights(source["subject"][train])
        mean = np.average(x, weights=w, axis=0)
        scale = np.sqrt(np.average((x - mean) ** 2, weights=w, axis=0))
        near_constant = int((scale < 1e-8).sum())
        scale = np.maximum(scale, 1e-8)
        del x
        val_distance = feature_distance(features["Dalia"][val], mean, scale)
        target_distance = feature_distance(features["WildPPG"], mean, scale)
        cutoff = float(weighted_quantile(val_distance, subject_weights(source["subject"][val]), [.95])[0])
        for dataset, selected_rows, distance in (("Dalia_validation", {k: v[val] for k, v in source.items()}, val_distance),
                                                  ("WildPPG_test", target, target_distance)):
            for person in np.unique(selected_rows["subject"]):
                own = selected_rows["subject"] == person
                records.append({"seed": job["seed"], "dataset": dataset, "subject": str(person),
                                "windows": int(own.sum()), "mean_standardized_rms": float(distance[own].mean()),
                                "fraction_above_source_validation_p95": float((distance[own] > cutoff).mean()),
                                "source_validation_p95": cutoff})
        summaries.append({"seed": job["seed"], "near_constant_dimensions": near_constant,
                          "source_validation_p95": cutoff,
                          "source_train_mean_standardized_rms": "not used as evaluation",
                          "target_above_cutoff_fraction": float(np.mean([np.mean(target_distance[target["subject"] == s] > cutoff) for s in np.unique(target["subject"])])),
                          "target_mean_standardized_rms": float(subject_weights(target["subject"]) @ target_distance),
                          "validation_mean_standardized_rms": float(subject_weights(source["subject"][val]) @ val_distance)})
    return records, summaries


def analyze(root, output):
    inputs = Inputs(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"version": "hr-transfer-diagnosis-v1", "analysis_status": "post_hoc_descriptive",
                "no_refitting_or_target_label_selection": True, "models": list(METHODS), "seeds": list(SEEDS),
                "hr_bins_bpm": ["<60", "[60,80)", "[80,100)", "[100,120)", ">=120"],
                "motion": "original within-dataset ACC RMS terciles; raw units not compared across devices",
                "primary_pair": "transfer versus within-Wild on identical valid-reference windows and participants",
                "feature_distance": "AnyPPG diagonal standardized RMS; source-training subject-balanced mean/std, validation participant-weighted 95th percentile",
                "inference": "exploratory paired participant bootstrap after averaging seed errors; not causal, not multiplicity-adjusted",
                "limitations": ["Reference availability selection and automatic Wild ECG errors may affect estimates.",
                                "Within-Wild uses target training labels; comparison measures transfer penalty, not matched training population or cost.",
                                "HR/ACC strata and embedding distance are correlated descriptions, not identified causal mechanisms.",
                                "Embedding RMS is a simple distance proxy, not a calibrated out-of-distribution detector."]}
    write_json(output / "protocol.json", protocol)
    distributions, distribution_bins, calibration_rows, subject_rows, strata = [], [], [], [], []
    for dataset, rows in inputs.rows.items():
        for person in ("all", *np.unique(rows["subject"]).tolist()):
            selected = np.ones(len(rows["hr"]), bool) if person == "all" else rows["subject"] == person
            desc = describe_distribution(rows["hr"][selected], rows["subject"][selected])
            distributions.append({"dataset": dataset, "subject": person, **desc})
        for number, name in enumerate(HR_NAMES):
            membership = np.digitize(rows["hr"], HR_EDGES) == number
            distribution_bins.append({"dataset": dataset, "hr_bin": name,
                                      "windows": int(membership.sum()),
                                      "participant_mean_fraction": float(subject_weights(rows["subject"]) @ membership)})
    target = inputs.rows["WildPPG"]
    hr_bins = np.digitize(target["hr"], HR_EDGES)
    for method in METHODS:
        for seed in SEEDS:
            for dataset, scenario in (("Dalia", "within"), ("WildPPG", "within"), ("WildPPG", "transfer")):
                rows = inputs.rows[dataset]
                prediction = inputs.prediction(scenario, dataset, method, seed)
                calibration_rows.append({"method": method, "seed": seed, "dataset": dataset, "scenario": scenario,
                                         **calibration(rows["hr"], prediction, rows["subject"])})
            within = inputs.prediction("within", "WildPPG", method, seed)
            transfer = inputs.prediction("transfer", "WildPPG", method, seed)
            for person in np.unique(target["subject"]):
                own = target["subject"] == person
                y = target["hr"][own]
                wm, tm = np.mean(np.abs(within[own] - y)), np.mean(np.abs(transfer[own] - y))
                subject_rows.append({"method": method, "seed": seed, "subject": str(person), "windows": int(own.sum()),
                                     "reference_mean_bpm": float(y.mean()), "within_mae_bpm": float(wm),
                                     "transfer_mae_bpm": float(tm), "gap_bpm": float(tm - wm),
                                     "within_bias_bpm": float(np.mean(within[own] - y)),
                                     "transfer_bias_bpm": float(np.mean(transfer[own] - y))})
            for group, bins, names in (("hr", hr_bins, HR_NAMES), ("motion", target["motion_bin"], ("low", "middle", "high")),
                                      ("hr_by_motion", hr_bins * 3 + target["motion_bin"],
                                       tuple(f"{h}:{m}" for h in HR_NAMES for m in ("low", "middle", "high")))):
                records = stratified_pair(target["hr"], within, transfer, target["subject"], bins, list(zip(names, range(len(names)))))
                for record in records:
                    strata.append({"method": method, "seed": seed, "group": group, **record})
    subjects = aggregate_seed_rows(subject_rows, ("method", "subject"))
    strata_mean = aggregate_seed_rows(strata, ("method", "group", "stratum"))
    calibration_mean = aggregate_seed_rows(calibration_rows, ("method", "dataset", "scenario"))
    summary = []
    for method in METHODS:
        selected = [r for r in subjects if r["method"] == method]
        mean, lo, hi = paired_bootstrap([r["gap_bpm"] for r in selected])
        summary.append({"method": method, "n_subjects": len(selected), "within_wild_mae_bpm": float(np.mean([r["within_mae_bpm"] for r in selected])),
                        "transfer_mae_bpm": float(np.mean([r["transfer_mae_bpm"] for r in selected])),
                        "paired_gap_bpm": mean, "ci95_low_bpm": lo, "ci95_high_bpm": hi,
                        "participants_worse_transfer": int(sum(r["gap_bpm"] > 0 for r in selected))})
    ranges = []
    source = inputs.rows["Dalia"]
    for job in [j for j in inputs.plan["jobs"] if j["scenario"] == "transfer"]:
        selected = np.isin(source["subject"], job["train"])
        stats = describe_distribution(source["hr"][selected], source["subject"][selected])
        outside = (target["hr"] < stats["min"]) | (target["hr"] > stats["max"])
        central = (target["hr"] < stats["p1"]) | (target["hr"] > stats["p99"])
        ranges.append({"seed": job["seed"], **stats,
                       "target_fraction_outside_train_min_max": float(subject_weights(target["subject"]) @ outside),
                       "target_fraction_outside_train_p1_p99": float(subject_weights(target["subject"]) @ central)})
    distance_records, distance_summary = feature_diagnosis(inputs)
    distance_people = aggregate_seed_rows([r for r in distance_records if r["dataset"] == "WildPPG_test"], ("dataset", "subject"))
    any_people = {r["subject"]: r for r in subjects if r["method"] == "AnyPPG"}
    rho = float(spearmanr([r["mean_standardized_rms"] for r in distance_people], [any_people[r["subject"]]["gap_bpm"] for r in distance_people]).statistic)
    traditional = []
    for dataset, rows in inputs.rows.items():
        directory = inputs.root / "traditional" / dataset
        metadata = inputs.json(directory / "complete.json")
        inputs.verify(directory / "predictions.npz", metadata["sha256"])
        arrays = load_npz(directory / "predictions.npz")
        for method in ("elgendi", "welch"):
            prediction = arrays[method]
            for group, bins, names in (("all", np.zeros(len(prediction), int), ("all",)),
                                      ("hr", np.digitize(rows["hr"], HR_EDGES), HR_NAMES),
                                      ("motion", rows["motion_bin"], ("low", "middle", "high"))):
                for number, name in enumerate(names):
                    conditional, coverage, nwindows = [], [], 0
                    for person in np.unique(rows["subject"]):
                        selection = (rows["subject"] == person) & (bins == number)
                        good = selection & np.isfinite(prediction)
                        nwindows += int(selection.sum())
                        if selection.any():
                            coverage.append(float(good.sum() / selection.sum()))
                        if good.any():
                            error = prediction[good] - rows["hr"][good]
                            conditional.append([np.mean(np.abs(error)), np.mean(error)])
                    values = np.mean(conditional, axis=0) if conditional else [None, None]
                    traditional.append({"dataset": dataset, "method": method, "group": group,
                                        "stratum": name, "windows": nwindows,
                                        "subjects_with_estimate": len(conditional),
                                        "participant_mean_coverage": float(np.mean(coverage)) if coverage else None,
                                        "conditional_mae_bpm": values[0], "conditional_bias_bpm": values[1]})
    tables = {"hr_distributions.csv": distributions, "hr_distribution_bins.csv": distribution_bins,
              "traditional_strata.csv": traditional, "calibration.csv": calibration_mean,
              "calibration_by_seed.csv": calibration_rows, "subject_penalties.csv": subjects,
              "subject_penalties_by_seed.csv": subject_rows, "strata.csv": strata_mean,
              "strata_by_seed.csv": strata, "transfer_summary.csv": summary,
              "source_training_ranges.csv": ranges, "feature_distance_by_subject_seed.csv": distance_records,
              "feature_distance_by_subject.csv": distance_people}
    for filename, records in tables.items():
        write_csv(output / filename, records)
    report = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": sha256(Path(__file__)), "source_protocol_sha256": inputs.digest(inputs.root / "protocol.json"),
              "transfer_summary": summary, "source_training_ranges": ranges, "feature_distance": distance_summary,
              "feature_distance_vs_transfer_gap_subject_spearman": rho,
              "input_sha256": inputs.hashes, "table_sha256": {f: sha256(output / f) for f in tables},
              "protocol_sha256": sha256(output / "protocol.json"), "limitations": protocol["limitations"]}
    write_json(output / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / ".local/ppg-hr-poc")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/ppg-phase2/transfer")
    args = parser.parse_args()
    report = analyze(args.root, args.output)
    print(json.dumps({k: report[k] for k in ("status", "transfer_summary", "feature_distance", "feature_distance_vs_transfer_gap_subject_spearman")}, indent=2))


if __name__ == "__main__":
    main()
