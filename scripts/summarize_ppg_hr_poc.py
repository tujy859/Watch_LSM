"""Audit registered wrist-HR PoC predictions and report participant-level results.

No models are fitted here. Formal summaries require all 558 learned results and
both traditional caches. --allow-partial tolerates missing completed artifacts,
never corrupt ones; every output then records whether the complete study exists.
Bootstrap units are participants after averaging seed-specific paired errors.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Dalia", "WildPPG")
MODELS = ("AnyPPG", "PaPaGei-S", "PulsePPG", "MOMENT-1-small")
METHODS = ("TrainMean", *MODELS, "SmallCNN")
FOUNDATIONS = MODELS[:3]
CONTROLS = ("SmallCNN", "welch", "elgendi", "MOMENT-1-small")
SEEDS = (42, 43, 44)
GROUP = ("scenario", "source", "target", "budget")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def validate_plan(plan, subjects):
    """Check complete split coverage and nesting without trusting runner checks."""
    require(tuple(plan["models"]) == MODELS and tuple(plan["seeds"]) == SEEDS, "Unexpected registered models/seeds")
    require(plan["outer_folds"] == 5, "Expected five outer folds")
    jobs = plan["jobs"]
    require(len(jobs) == 93 and len({j["id"] for j in jobs}) == 93, "Expected 93 unique registered splits")
    within, transfers = {}, {}
    for job in jobs:
        source, target = job["source"], job["target"]
        require(source in DATASETS and target in DATASETS, "Unexpected split dataset")
        train, val, test = (set(job[key]) for key in ("train", "validation", "test"))
        require(all((train, val, test)) and not train & val, "Training/validation leakage or missing subjects")
        require(all(len(job[k]) == len(set(job[k])) for k in ("train", "validation", "test")), "Duplicate split subjects")
        require(train | val <= subjects[source] and test <= subjects[target], "Unknown split subject")
        require(job["seed"] in SEEDS, "Unexpected split seed")
        if job["scenario"] == "within":
            require(source == target and not (train | val) & test, "Held-out subject leakage")
            require(job["budget"] in ("3", "6", "full") and job["fold"] in range(5), "Unexpected budget/fold")
            expected_n = {"3": (2, 1), "6": (4, 2), "full": (len(subjects[source] - test) - 3, 3)}[job["budget"]]
            require((len(train), len(val)) == expected_n, "Label budget does not include registered validation people")
            if job["budget"] == "full":
                require(train | val | test == subjects[source], "Full split omits subjects")
            key = (source, job["seed"], job["budget"], job["fold"])
            require(key not in within, "Duplicate within configuration")
            within[key] = job
        else:
            require(job["scenario"] == "transfer" and source == "Dalia" and target == "WildPPG", "Unexpected transfer direction")
            require(job["budget"] == "full" and job["fold"] == -1, "Unexpected transfer budget/fold")
            require(len(val) == 3 and train | val == subjects[source] and test == subjects[target], "Incomplete source-only transfer split")
            require(job["seed"] not in transfers, "Duplicate transfer seed")
            transfers[job["seed"]] = job
    require(len(within) == 90 and set(transfers) == set(SEEDS), "Incomplete scenario coverage")
    for dataset in DATASETS:
        canonical = {fold: set(within[(dataset, 42, "full", fold)]["test"]) for fold in range(5)}
        flattened = [s for fold in canonical.values() for s in fold]
        require(len(flattened) == len(set(flattened)) and set(flattened) == subjects[dataset], "Outer test folds overlap or omit participants")
        for seed in SEEDS:
            for fold in range(5):
                previous_train, previous_val = set(), set()
                for budget in ("3", "6", "full"):
                    job = within[(dataset, seed, budget, fold)]
                    require(set(job["test"]) == canonical[fold], "Test people change with seed or label budget")
                    require(previous_train <= set(job["train"]) and previous_val <= set(job["validation"]), "Label-budget subsets are not nested")
                    previous_train, previous_val = set(job["train"]), set(job["validation"])


class Auditor:
    def __init__(self, root, repository=ROOT):
        self.root, self.repository = Path(root), Path(repository)
        self.hashes = {}
        self.plan = read_json(self.root / "protocol.json")
        self.protocol_hash = self.digest(self.root / "protocol.json")
        self.rows, self.manifests = {}, {}
        self.feature_metadata = {}
        self.missing = []

    def digest(self, path):
        path = Path(path)
        if path not in self.hashes:
            self.hashes[path] = sha256(path)
        return self.hashes[path]

    def check_hash(self, path, expected):
        require(self.digest(path) == expected, f"Hash mismatch: {path}")

    def data(self):
        for filename, expected in self.plan["code_sha256"].items():
            self.check_hash(self.repository / "scripts" / filename, expected)
        for dataset in DATASETS:
            directory = self.root / "data" / dataset
            self.check_hash(directory / "manifest.json", self.plan["data_manifest_sha256"][dataset])
            manifest = read_json(directory / "manifest.json")
            require(manifest["dataset"] == dataset and len(manifest["subjects"]) == {"Dalia": 15, "WildPPG": 16}[dataset], "Unexpected dataset/participant coverage")
            for filename, expected in manifest["hashes"].items():
                self.check_hash(directory / filename, expected)
            self.check_hash(directory / "rows.npz", self.plan["data_rows_sha256"][dataset])
            rows = load_npz(directory / "rows.npz")
            n = manifest["windows"]
            for key in ("subject", "start_s", "hr", "activity", "official_include", "motion_bin"):
                require(key in rows and rows[key].shape == (n,), f"Invalid rows field {dataset}/{key}")
            require(set(rows["subject"].tolist()) == set(manifest["subjects"]), "Manifest subject list disagrees with rows")
            require(np.isfinite(rows["hr"]).all() and (rows["hr"] > 0).all(), "Invalid HR references")
            require(np.isfinite(rows["start_s"]).all() and (rows["start_s"] >= 0).all(), "Invalid window starts")
            require(len(set(zip(rows["subject"].tolist(), rows["start_s"].tolist()))) == n, "Duplicate subject/time window")
            require(np.isin(rows["motion_bin"], (0, 1, 2)).all(), "Unexpected motion strata")
            require(np.isin(rows["official_include"], (0, 1)).all(), "Invalid official_include flags")
            self.rows[dataset], self.manifests[dataset] = rows, manifest
        validate_plan(self.plan, {d: set(m["subjects"]) for d, m in self.manifests.items()})

    def feature(self, dataset, method):
        key = (dataset, method)
        if key in self.feature_metadata:
            return self.feature_metadata[key]
        directory = self.root / "features" / dataset
        metadata_path = directory / f"{method}.json"
        if not metadata_path.is_file():
            self.missing.append(f"features/{dataset}/{method}")
            return None
        metadata = read_json(metadata_path)
        identity = metadata["identity"]
        require(metadata["status"] == "complete", f"Unfinished published feature cache: {key}")
        require(identity["dataset"] == dataset and identity["model"]["name"] == method, "Feature identity mismatch")
        require(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == metadata["identity_sha256"], "Feature identity digest mismatch")
        n = len(self.rows[dataset]["hr"])
        require(metadata["completed_windows"] == n and metadata["shape"][0] == n, "Feature cache row count mismatch")
        require(metadata["shape"] == [n, identity["model"]["embedding_dim"]], "Feature dimension mismatch")
        self.check_hash(directory / f"{method}.npy", metadata["features_sha256"])
        values = np.load(directory / f"{method}.npy", mmap_mode="r", allow_pickle=False)
        require(list(values.shape) == metadata["shape"] and str(values.dtype) == metadata["dtype"], "Feature array metadata mismatch")
        for start in range(0, n, 4096):
            require(np.isfinite(values[start:start + 4096]).all(), "Nonfinite completed features")
        data_identity = identity["data"]
        self.check_hash(self.root / "data" / dataset / "manifest.json", data_identity["manifest_sha256"])
        require("rows.npz" in data_identity["files_sha256"], "Feature identity omits row identity")
        for filename, expected in data_identity["files_sha256"].items():
            self.check_hash(self.root / "data" / dataset / filename, expected)
        require(data_identity["raw_sources"] == self.manifests[dataset]["sources"], "Feature raw provenance differs")
        # The cache publication hash and input hashes establish the evaluated
        # artifact. Rehash weights too, without needing to load any model/GPU.
        for filename, expected in identity["weight_files_sha256"].items():
            self.check_hash(self.repository / filename, expected)
        require(bool(identity["weight_files_sha256"]), "Feature identity has no weight hashes")
        source = identity["source"]
        self.check_hash(self.repository / "scripts/ppg_hr_poc_features.py", source["extraction_script_sha256"])
        self.check_hash(self.repository / "watch_lsm/configs/ppg_bench_models.yaml", source["config_sha256"])
        require(bool(source["upstream_python_files_sha256"]), "Feature identity omits upstream source hashes")
        for filename, expected in source["upstream_python_files_sha256"].items():
            self.check_hash(self.repository / "repos/ppg-bench" / filename, expected)
        self.feature_metadata[key] = self.digest(metadata_path)
        return self.feature_metadata[key]

    def traditional(self, dataset):
        directory = self.root / "traditional" / dataset
        if not (directory / "complete.json").is_file():
            self.missing.append(f"traditional/{dataset}")
            return {}
        metadata = read_json(directory / "complete.json")
        expected = {"data_manifest_sha256": self.plan["data_manifest_sha256"][dataset], "algorithm_sha256": self.plan["code_sha256"]["ppg_hr_baselines.py"]}
        require(metadata["identity"] == expected, "Traditional completion identity mismatch")
        self.check_hash(directory / "predictions.npz", metadata["sha256"])
        predictions = load_npz(directory / "predictions.npz")
        require(set(predictions) == {"welch", "elgendi"}, "Traditional methods missing or unexpected")
        n = len(self.rows[dataset]["hr"])
        for method, pred in predictions.items():
            require(pred.shape == (n,) and not np.isinf(pred).any(), "Invalid traditional output")
            require(np.isclose(np.isfinite(pred).mean(), metadata["coverage"][method], atol=1e-12, rtol=0), "Traditional coverage mismatch")
        return predictions

    def learned(self, job, method):
        directory = self.root / "runs" / job["id"] / method
        if not (directory / "complete.json").is_file():
            self.missing.append(f"runs/{job['id']}/{method}")
            return None
        metadata = read_json(directory / "complete.json")
        expected = {"protocol_sha256": self.protocol_hash, "job": job, "method": method}
        if method in MODELS:
            feature_hashes = {d: self.feature(d, method) for d in set((job["source"], job["target"]))}
            require(all(feature_hashes.values()), "A completed learned result has no completed feature cache")
            expected["feature_metadata_sha256"] = feature_hashes
        require(metadata["identity"] == expected, "Learned completion identity mismatch")
        self.check_hash(directory / "predictions.npz", metadata["predictions_sha256"])
        predictions = load_npz(directory / "predictions.npz")
        source, target = self.rows[job["source"]], self.rows[job["target"]]
        test = np.flatnonzero(np.isin(target["subject"], job["test"]))
        for key, reference in (("row", test), ("subject", target["subject"][test]), ("start_s", target["start_s"][test]), ("reference", target["hr"][test])):
            require(key in predictions and np.array_equal(predictions[key], reference), f"Prediction row identity mismatch: {job['id']}/{method}/{key}")
        pred = predictions["prediction"]
        require(pred.shape == (len(test),) and np.isfinite(pred).all(), "Invalid learned predictions")
        for split, count_key in (("train", "train_windows"), ("validation", "validation_windows"), ("test", "test_windows")):
            rows = target if split == "test" else source
            require(metadata[count_key] == int(np.isin(rows["subject"], job[split]).sum()), "Completed split window count mismatch")
        macro = np.mean([np.abs(pred[target["subject"][test] == s] - target["hr"][test][target["subject"][test] == s]).mean() for s in job["test"]])
        require(np.isclose(macro, metadata["subject_mae_bpm"], atol=1e-9, rtol=1e-10), "Stored MAE cannot be reproduced")
        return pred


def strata_masks(rows, dataset):
    masks = [("all", "all", np.ones(len(rows["hr"]), dtype=bool))]
    masks.extend(("motion_bin", str(value), rows["motion_bin"] == value) for value in (0, 1, 2))
    masks.extend(("official_include", str(value), rows["official_include"] == bool(value)) for value in (0, 1))
    if dataset == "Dalia":
        masks.extend(("activity", str(value), rows["activity"] == value) for value in np.unique(rows["activity"]))
    return masks


def subject_records(job, rows, predictions):
    """One row per person/seed/method/stratum; failed estimates stay in counts."""
    output = []
    for stratum, level, stratum_mask in strata_masks(rows, job["target"]):
        for subject in np.unique(rows["subject"][stratum_mask]):
            selected = stratum_mask & (rows["subject"] == subject)
            n = int(selected.sum())
            for method, prediction in predictions.items():
                finite = selected & np.isfinite(prediction)
                valid = int(finite.sum())
                output.append({**{key: job[key] for key in GROUP}, "method": method, "stratum": stratum, "level": level,
                               "subject": str(subject), "seed": job["seed"], "windows": n, "valid_windows": valid,
                               "mae_bpm": float(np.abs(prediction[finite] - rows["hr"][finite]).mean()) if valid else None})
    return output


def paired_records(job, rows, predictions):
    """Paired estimands use exactly the same finite windows for both methods."""
    output = []
    for foundation in FOUNDATIONS:
        if foundation not in predictions:
            continue
        for control in CONTROLS:
            if control not in predictions:
                continue
            for subject in np.unique(rows["subject"]):
                selected = rows["subject"] == subject
                common = selected & np.isfinite(predictions[foundation]) & np.isfinite(predictions[control])
                fm_mae = float(np.abs(predictions[foundation][common] - rows["hr"][common]).mean()) if common.any() else None
                control_mae = float(np.abs(predictions[control][common] - rows["hr"][common]).mean()) if common.any() else None
                output.append({**{key: job[key] for key in GROUP}, "foundation": foundation, "control": control,
                               "subject": str(subject), "seed": job["seed"], "windows": int(selected.sum()),
                               "common_windows": int(common.sum()), "foundation_mae_bpm": fm_mae,
                               "control_mae_bpm": control_mae, "difference_bpm": fm_mae - control_mae if fm_mae is not None else None})
    return output


def grouped(records, keys):
    result = defaultdict(list)
    for record in records:
        result[tuple(record[key] for key in keys)].append(record)
    return result


def participant_aggregate(records):
    keys = (*GROUP, "method", "stratum", "level", "subject")
    output = []
    for identity, rows in sorted(grouped(records, keys).items()):
        require(len({r["seed"] for r in rows}) == len(rows), "Participant has duplicate held-out predictions within a seed")
        require(len({r["windows"] for r in rows}) == 1, "Participant window population changes across seeds")
        errors = [r["mae_bpm"] for r in rows if r["mae_bpm"] is not None]
        output.append({**dict(zip(keys, identity)), "n_seeds": len(rows), "n_seeds_with_estimate": len(errors),
                       "windows": rows[0]["windows"], "mean_valid_windows": float(np.mean([r["valid_windows"] for r in rows])),
                       "coverage": float(np.mean([r["valid_windows"] / r["windows"] for r in rows])),
                       "conditional_mae_bpm": float(np.mean(errors)) if errors else None,
                       "seed_mae_sd_bpm": float(np.std(errors, ddof=1)) if len(errors) > 1 else None})
    return output


def summary_aggregate(participants, seed_records):
    keys = (*GROUP, "method", "stratum", "level")
    per_seed_groups = grouped(seed_records, (*keys, "seed"))
    output = []
    for identity, rows in sorted(grouped(participants, keys).items()):
        errors = [r["conditional_mae_bpm"] for r in rows if r["conditional_mae_bpm"] is not None]
        total = sum(r["windows"] for r in rows)
        valid = sum(r["mean_valid_windows"] for r in rows)
        seed_errors = []
        for seed in SEEDS:
            values = [r["mae_bpm"] for r in per_seed_groups.get((*identity, seed), []) if r["mae_bpm"] is not None]
            if values:
                seed_errors.append(float(np.mean(values)))
        output.append({**dict(zip(keys, identity)), "n_subjects": len(rows), "n_subjects_with_estimate": len(errors),
                       "unique_windows": total, "mean_valid_windows": valid, "window_coverage": valid / total,
                       "participant_coverage": float(np.mean([r["coverage"] for r in rows])),
                       "conditional_participant_mae_bpm": float(np.mean(errors)) if errors else None,
                       "full_coverage_participant_mae_bpm": float(np.mean(errors)) if valid == total and len(errors) == len(rows) else None,
                       "min_seeds_per_subject": min(r["n_seeds"] for r in rows),
                       "seed_macro_mae_sd_bpm": float(np.std(seed_errors, ddof=1)) if len(seed_errors) > 1 else None,
                       "seed_macro_mae_min_bpm": min(seed_errors) if seed_errors else None,
                       "seed_macro_mae_max_bpm": max(seed_errors) if seed_errors else None,
                       "traditional_reused_without_refit": identity[4] in ("welch", "elgendi")})
    return output


def bootstrap_mean(values, samples=10000, seed=20261104):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "Invalid bootstrap participant values")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        estimates[start:start + count] = values[rng.integers(0, len(values), (count, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def paired_aggregate(records, samples=10000):
    keys = (*GROUP, "foundation", "control")
    participants = []
    for identity, rows in sorted(grouped(records, (*keys, "subject")).items()):
        require(len({r["seed"] for r in rows}) == len(rows), "Duplicate participant seed in paired inference")
        require(len({r["windows"] for r in rows}) == 1, "Paired population changes across seeds")
        paired_rows = [r for r in rows if r["difference_bpm"] is not None]
        participants.append({**dict(zip((*keys, "subject"), identity)), "n_seeds": len(rows), "n_seeds_with_pair": len(paired_rows), "windows": rows[0]["windows"],
                             "common_windows": float(np.mean([r["common_windows"] for r in rows])),
                             **{key: float(np.mean([r[key] for r in paired_rows])) if paired_rows else None for key in ("foundation_mae_bpm", "control_mae_bpm", "difference_bpm")}})
    result = []
    for identity, rows in sorted(grouped(participants, keys).items()):
        paired_rows = [r for r in rows if r["difference_bpm"] is not None]
        deltas = np.array([r["difference_bpm"] for r in paired_rows])
        low, high = bootstrap_mean(deltas, samples) if len(deltas) else (None, None)
        result.append({**dict(zip(keys, identity)), "n_subjects": len(paired_rows), "n_subjects_total": len(rows),
                       "min_seeds_per_subject": min(r["n_seeds"] for r in rows), "min_seeds_with_pair": min(r["n_seeds_with_pair"] for r in rows),
                       "unique_windows": sum(r["windows"] for r in rows), "mean_common_windows": sum(r["common_windows"] for r in rows),
                       "paired_window_coverage": sum(r["common_windows"] for r in rows) / sum(r["windows"] for r in rows),
                       "foundation_common_mae_bpm": float(np.mean([r["foundation_mae_bpm"] for r in paired_rows])) if paired_rows else None,
                       "control_common_mae_bpm": float(np.mean([r["control_mae_bpm"] for r in paired_rows])) if paired_rows else None,
                       "difference_bpm": float(deltas.mean()) if len(deltas) else None, "ci95_low_bpm": low, "ci95_high_bpm": high,
                       "participant_bootstrap_samples": samples, "direction": "negative favours foundation",
                       "exploratory_not_multiplicity_adjusted": True})
    return result, participants


def write_csv(path, records):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        if records:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def summarize(root, output=None, allow_partial=False, repository=ROOT, bootstrap_samples=10000):
    auditor = Auditor(root, repository)
    auditor.data()
    traditional = {dataset: auditor.traditional(dataset) for dataset in DATASETS}
    # Check publication of all eight caches even if learned jobs are not ready.
    for dataset in DATASETS:
        for method in MODELS:
            auditor.feature(dataset, method)
    seed_records, pair_records, completed = [], [], 0
    for job in auditor.plan["jobs"]:
        rows = auditor.rows[job["target"]]
        selected = np.flatnonzero(np.isin(rows["subject"], job["test"]))
        test_rows = {key: value[selected] for key, value in rows.items()}
        predictions = {method: value[selected] for method, value in traditional[job["target"]].items()}
        for method in METHODS:
            prediction = auditor.learned(job, method)
            if prediction is not None:
                predictions[method] = prediction
                completed += 1
        seed_records.extend(subject_records(job, test_rows, predictions))
        pair_records.extend(paired_records(job, test_rows, predictions))
    missing = sorted(set(auditor.missing))
    complete = not missing and completed == 558
    require(allow_partial or complete, f"Formal summary requires all results: {completed}/558 learned, {len(missing)} missing artifacts; use --allow-partial only for monitoring")
    participants = participant_aggregate(seed_records)
    aggregated = summary_aggregate(participants, seed_records)
    pairs, paired_subjects = paired_aggregate(pair_records, bootstrap_samples)
    tables = {"summary.csv": [r for r in aggregated if r["stratum"] == "all"],
              "per_subject.csv": [r for r in participants if r["stratum"] == "all"],
              "strata.csv": [r for r in aggregated if r["stratum"] != "all"],
              "paired.csv": pairs, "paired_per_subject.csv": paired_subjects,
              "per_subject_strata.csv": [r for r in participants if r["stratum"] != "all"]}
    for records in tables.values():
        for record in records:
            record["study_complete"] = complete
    output = Path(output) if output else Path(root) / "summary"
    output.mkdir(parents=True, exist_ok=True)
    for name, records in tables.items():
        write_csv(output / name, records)
    audit = {"status": "complete" if complete else "partial", "audited_at": datetime.now(timezone.utc).isoformat(),
             "protocol_sha256": auditor.protocol_hash, "registered_splits": len(auditor.plan["jobs"]),
             "expected_learned_results": 558, "completed_learned_results": completed, "missing_artifacts": missing,
             "verified_feature_caches": len(auditor.feature_metadata), "verified_file_count": len(auditor.hashes),
             "feature_metadata_sha256": {f"{d}/{m}": digest for (d, m), digest in sorted(auditor.feature_metadata.items())},
             "verified_artifact_set_sha256": hashlib.sha256(json.dumps(sorted((str(path.relative_to(auditor.root)) if path.is_relative_to(auditor.root) else str(path), digest) for path, digest in auditor.hashes.items())).encode()).hexdigest(),
             "checks": ["registered code/data hashes", "subject split isolation, outer coverage and budget nesting", "complete feature hashes/inputs/weights and finite values", "completion identity and prediction hashes", "exact prediction row/subject/start/reference alignment", "finite learned predictions and stored MAE recomputation"],
             "estimand": "Within each participant compute window MAE per seed, average seeds, then average participants equally. Traditional missing outputs remain in coverage; conditional MAE is labelled explicitly.",
             "paired_estimand": "For each participant/seed use common finite windows, subtract control MAE from foundation MAE, average seeds, bootstrap participants only.",
             "bootstrap_samples": bootstrap_samples, "bootstrap_seed": 20261104,
             "inference_note": "Exploratory percentile intervals, no multiplicity adjustment. Partial results may have unequal seed coverage; do not treat as final evidence.",
             "strata_note": "Motion terciles are within-dataset descriptive ACC strata; activity only in Dalia. official_include=1 is legacy benchmark-window sensitivity; zero is added-window stratum.",
             "traditional_note": "Traditional predictions reused across seeds/budgets and transfer; n_subjects and unique_windows never count this reuse as new observations.",
             "dataset_counts": {d: {"subjects": len(auditor.manifests[d]["subjects"]), "windows": len(auditor.rows[d]["hr"])} for d in DATASETS},
             "table_sha256": {name: sha256(output / name) for name in tables}}
    (output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / ".local/ppg-hr-poc")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    audit = summarize(args.root, args.output, args.allow_partial)
    print(json.dumps({"status": audit["status"], "completed_learned_results": audit["completed_learned_results"],
                      "expected_learned_results": audit["expected_learned_results"], "missing_artifacts": len(audit["missing_artifacts"]),
                      "verified_feature_caches": audit["verified_feature_caches"], "output": str(args.output or args.root / "summary")}, indent=2))


if __name__ == "__main__":
    main()
