"""Audit saved wrist-HR results without retraining or changing upstream outputs.

Partial scope is reported explicitly: absent runs are not failed experiments.
Feature exports preserve the upstream loader's ordering, and provide source row
references for reuse in a separately defined PoC. No waveform identity is inferred
from an HR value alone.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"dalia": "Dalia", "wildppg": "WildPPG"}
EXPECTED_MODELS = ("AnyPPG", "PaPaGei-S", "PaPaGei-P", "PulsePPG", "SigmaPPG", "PRVCtrl", "MOMENT-1-small")
SPLITS = {(seed, fold) for seed in range(42, 47) for fold in range(5)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cohort(directory):
    """Match BenchmarkDataset's float32 conversion and finite-label selection."""
    cohort = {}
    for subject in sorted(p for p in directory.iterdir() if (p / "ppg.npy").is_file()):
        labels = np.load(subject / "hr.npy", allow_pickle=False).astype(np.float32).reshape(-1)
        signals = np.load(subject / "ppg.npy", mmap_mode="r", allow_pickle=False)
        require(signals.ndim == 2 and len(labels) == len(signals), f"Invalid shapes: {subject}")
        valid = np.flatnonzero(np.isfinite(labels))
        require(len(valid) > 0, f"No valid HR: {subject}")
        cohort[subject.name] = {"hr": labels[valid], "source_rows": valid, "raw_windows": len(labels), "signal_shape": list(signals.shape)}
    require(bool(cohort), f"Empty cohort: {directory}")
    return cohort


def expected_split(subjects, dataset, seed, fold):
    """Independent reproduction of the fixed upstream subject split algorithm."""
    shuffled = sorted(subjects)
    digest = hashlib.sha256(f"{seed}:{dataset}".encode()).hexdigest()
    random.Random(int(digest[:16], 16)).shuffle(shuffled)
    offset = sum(len(shuffled) // 5 + (i < len(shuffled) % 5) for i in range(fold))
    size = len(shuffled) // 5 + (fold < len(shuffled) % 5)
    test = shuffled[offset:offset + size]
    return [s for s in shuffled if s not in set(test)], test


def ordered_targets(cohort, subjects):
    return (np.concatenate([cohort[s]["hr"] for s in subjects]),
            np.concatenate([np.repeat(s, len(cohort[s]["hr"])) for s in subjects]))


def audit_result(directory, model, dataset, cohort, enforce_official_split=True):
    record_path = directory / "results.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    require(record["model"] == model and record["method"] == "linear" and record["task"] == f"{dataset}-hr", f"Wrong identity: {directory}")
    split = (record["seed"], record["fold"])
    require(split in SPLITS, f"Unexpected seed/fold: {directory}")
    train, test = record["train_groups"], record["test_groups"]
    require(len(train) == len(set(train)) and len(test) == len(set(test)), "Repeated subject in split")
    require(not set(train) & set(test), "Train/test subject overlap")
    require(set(train) | set(test) == set(cohort), "Incomplete subject coverage")
    if enforce_official_split:
        require((train, test) == expected_split(cohort, dataset, *split), "Split differs from fixed upstream protocol")
    require(len(record["results"]) == 1 and record["results"][0]["repeat"] == 0, "Unexpected linear repeat count")
    item = record["results"][0]
    prediction = (directory / item["predictions"]).resolve()
    require(prediction.is_relative_to(directory.resolve()), "Prediction escaped result directory")
    true, expected_subjects = ordered_targets(cohort, test)
    with np.load(prediction, allow_pickle=False) as arrays:
        require(set(arrays.files) == {"subjects", "y_true", "y_pred"}, "Unexpected prediction fields")
        subjects = arrays["subjects"].astype(str)
        pred, saved_true = arrays["y_pred"], arrays["y_true"]
        require(pred.shape == saved_true.shape == subjects.shape == true.shape, "Prediction shape mismatch")
        require(np.isfinite(pred).all() and np.isfinite(saved_true).all(), "Non-finite predictions or labels")
        require(np.array_equal(subjects, expected_subjects), "Prediction subject order differs from source")
        require(np.array_equal(saved_true, true), "Prediction labels differ from ordered source labels")
        error = np.abs(pred.astype(np.float64) - true.astype(np.float64))
    mae = float(error.mean())
    require(np.isclose(mae, item["test"]["mae"], rtol=1e-6, atol=1e-5), "Saved MAE disagrees with predictions")
    training_hr, _ = ordered_targets(cohort, train)
    baseline = float(np.abs(true.astype(np.float64) - float(training_hr.mean())).mean())
    require(np.isclose(baseline, record["baseline"]["mae"], rtol=1e-6, atol=1e-5), "Saved training-mean baseline disagrees")
    rows = [{"dataset": dataset, "model": model, "seed": split[0], "fold": split[1], "subject": s, "windows": int(np.sum(subjects == s)), "mae_bpm": float(error[subjects == s].mean())} for s in test]
    return {"split": split, "record": record, "mae": mae, "baseline": baseline,
            "subject_rows": rows, "result_sha256": sha256(record_path), "prediction_sha256": sha256(prediction)}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_features(manifest, run, model, dataset, cohort, output, source_root):
    """Export one complete cached outer partition; the encoder was frozen."""
    job = next((j for j in run["jobs"] if f"__{dataset}-hr__linear__seed42__fold0" in j), None)
    if job is None or not (manifest.parent / "jobs" / job / "results.json").is_file():
        return None
    record = json.loads((manifest.parent / "jobs" / job / "results.json").read_text())
    model_id = job.split("__")[0]
    cache = manifest.parent / "features" / f"{model_id}__{dataset}-hr__seed42__fold0.npz"
    if not cache.is_file():
        return {"available": False, "expected_cache": str(cache)}
    by_subject = {}
    expected_dim = next(m["embedding_dim"] for m in run["identity"]["models"] if m["name"] == model)
    with np.load(cache, allow_pickle=False) as arrays:
        require(set(arrays.files) == {"train", "test"}, f"Unexpected cache keys: {cache}")
        for part in ("train", "test"):
            values = arrays[part]
            groups = record[f"{part}_groups"]
            require(values.ndim == 2 and values.shape[1] == expected_dim and np.isfinite(values).all(), f"Invalid cached features: {cache}")
            require(len(values) == sum(len(cohort[s]["hr"]) for s in groups), f"Cache/source length mismatch: {cache}")
            offset = 0
            for subject in groups:
                count = len(cohort[subject]["hr"])
                by_subject[subject] = values[offset:offset + count].copy()
                offset += count
    destination = output / "reusable_features" / dataset
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in sorted(cohort):
        with (source_root / subject / "metadata.csv").open(newline="", encoding="utf-8") as stream:
            metadata = list(csv.DictReader(stream))
        require(len(metadata) == cohort[subject]["raw_windows"], f"Metadata/source length mismatch: {subject}")
        for hr, source_row in zip(cohort[subject]["hr"], cohort[subject]["source_rows"]):
            row = metadata[int(source_row)]
            require(np.isclose(float(row["hr"]), float(hr), atol=1e-5, rtol=1e-6), f"Metadata HR differs: {subject}/{source_row}")
            rows.append({"feature_row": len(rows), "subject": subject, "processed_row": int(source_row),
                         "window_idx": int(row["window_idx"]), "start_sec": float(row.get("start_sec", row.get("start_s"))),
                         "end_sec": float(row.get("end_sec", row.get("end_s"))), "hr": float(hr)})
    rows_path = destination / "rows.csv"
    write_csv(rows_path, rows)
    feature_dir = destination / model
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_dir / "features.npy"
    features = np.concatenate([by_subject[s] for s in sorted(cohort)])
    np.save(feature_path, features)
    return {"available": True, "source_cache": str(cache), "source_cache_sha256": sha256(cache),
            "features": str(feature_path), "features_sha256": sha256(feature_path), "shape": list(features.shape),
            "rows": str(rows_path), "rows_sha256": sha256(rows_path), "train_groups": record["train_groups"],
            "test_groups": record["test_groups"], "ordering": "lexicographic subject, then original processed row; finite float32 HR only",
            "identity_limit": "Cache has no row IDs. Association follows the audited source loader and matching source fingerprints/split sizes; no new encoder forward pass was run."}


def audit(roots, processed, output):
    output.mkdir(parents=True, exist_ok=True)
    cohorts = {name: load_cohort(processed / folder) for name, folder in DATASETS.items()}
    summaries, per_subject, evidence, exports, pending = [], [], [], [], []
    found = set()
    artifact_digest = hashlib.sha256()
    for root in roots:
        for manifest in sorted(root.glob("**/linear/run.json")):
            run = json.loads(manifest.read_text(encoding="utf-8"))
            model = manifest.parent.parent.name
            relevant = [j for j in run["jobs"] if any(f"__{d}-hr__" in j for d in DATASETS)]
            if not relevant:
                continue
            require(model in EXPECTED_MODELS, f"Unexpected model: {model}")
            require(set(run["seeds"]) == set(range(42, 47)) and set(run["folds"]) == set(range(5)), "Changed run scope")
            require(run["protocol"]["linear"]["inner_folds"] == 4, "Changed inner CV protocol")
            for dataset in DATASETS:
                jobs = [j for j in relevant if f"__{dataset}-hr__" in j]
                if not jobs:
                    continue
                require((dataset, model) not in found, f"Duplicate run for {dataset}/{model}")
                found.add((dataset, model))
                source = processed / DATASETS[dataset]
                fingerprints = run["identity"]["datasets"].get(str(source.resolve()))
                require(fingerprints is not None, f"Manifest source differs from selected data: {manifest}")
                for relative, size, mtime_ns in fingerprints:
                    stat = (source / relative).stat()
                    require(stat.st_size == size and stat.st_mtime_ns == mtime_ns, f"Source changed since original run: {relative}")
                splits = {}
                group_rows = []
                for job in jobs:
                    directory = manifest.parent / "jobs" / job
                    require(directory.resolve().is_relative_to((manifest.parent / "jobs").resolve()), "Job path escaped manifest")
                    if not (directory / "results.json").is_file():
                        pending.append({"dataset": dataset, "model": model, "job": job})
                        continue
                    checked = audit_result(directory, model, dataset, cohorts[dataset])
                    split = checked["split"]
                    require(job.endswith(f"__{dataset}-hr__linear__seed{split[0]}__fold{split[1]}"), "Result split differs from declared job identity")
                    require(split not in splits, f"Duplicate split: {dataset}/{model}/{split}")
                    splits[split] = checked
                    group_rows.extend(checked["subject_rows"])
                    for key in ("result_sha256", "prediction_sha256"):
                        artifact_digest.update(f"{dataset}/{model}/{job}/{key}:{checked[key]}\n".encode())
                complete = set(splits) == SPLITS
                for seed in range(42, 47):
                    if all((seed, fold) in splits for fold in range(5)):
                        test_sets = [set(splits[seed, fold]["record"]["test_groups"]) for fold in range(5)]
                        require(sum(map(len, test_sets)) == len(set.union(*test_sets)) == len(cohorts[dataset]), "Test folds overlap or omit subjects")
                if splits:
                    summaries.append({"dataset": dataset, "model": model, "method": "linear", "outer_evaluations": len(splits), "complete_5_seeds_5_folds": complete,
                                      "window_mae_mean_over_folds_seeds_bpm": float(np.mean([v["mae"] for v in splits.values()])),
                                      "subject_macro_mae_bpm": float(np.mean([r["mae_bpm"] for r in group_rows])) if complete else None,
                                      "training_mean_baseline_window_mae_bpm": float(np.mean([v["baseline"] for v in splits.values()]))})
                per_subject.extend(group_rows)
                evidence.append({"dataset": dataset, "model": model, "manifest": str(manifest), "manifest_sha256": sha256(manifest),
                                 "upstream_code_sha256": run["identity"]["code_sha256"], "versions": run["identity"]["versions"],
                                 "source_size_mtime_fingerprints_matched": len(fingerprints)})
                exported = export_features(manifest, run, model, dataset, cohorts[dataset], output, source)
                if exported is not None:
                    exports.append({"dataset": dataset, "model": model, **exported})
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "per_subject.csv", per_subject)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "passed_existing_artifact_checks": True,
              "scope": "Existing published DaLiA/WildPPG HR linear outputs only; missing runs remain missing",
              "completed_evaluations": sum(r["outer_evaluations"] for r in summaries), "full_scope_complete": not pending and len(summaries) == len(EXPECTED_MODELS) * len(DATASETS) and all(r["complete_5_seeds_5_folds"] for r in summaries),
              "missing_runs": [{"dataset": d, "model": m} for d in DATASETS for m in EXPECTED_MODELS if (d, m) not in found],
              "pending_jobs": pending, "summary": summaries, "artifact_set_sha256": artifact_digest.hexdigest(),
              "source_cohorts": {name: {"subjects": len(cohort), "raw_windows": sum(c["raw_windows"] for c in cohort.values()),
                                         "finite_hr_windows": sum(len(c["hr"]) for c in cohort.values()),
                                         "by_subject": {s: {"raw_windows": c["raw_windows"], "finite_hr_windows": len(c["hr"]), "signal_shape": c["signal_shape"]} for s, c in cohort.items()}} for name, cohort in cohorts.items()},
              "checks": ["manifest source file sizes and modification times unchanged", "fixed upstream subject splits independently reproduced", "no train/test overlap; complete folds cover each subject once per seed", "ordered prediction subjects and labels match finite-label source rows", "finite predictions and recomputed window HR MAE", "constant training-mean baseline recomputed", "canonical cached features have finite values and expected dimensions"],
              "limitations": ["Only published linear results were audited; absence is not a measured failure.", "Processed DaLiA excludes windows according to upstream preprocessing; this is not all raw motion data.", "Repeated overlapping windows and seeds are not independent subjects; no population CI is claimed here.", "The original inner CV implementation was not rerun; outer splits and saved artifacts were checked.", "Original run dataset fingerprints use size and mtime, not waveform content hashes. Canonical feature export hashes are included."],
              "environment_evidence": evidence, "reusable_features": exports}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_roots = [p for p in (ROOT / ".local").glob("ppg-bench-*") if p.is_dir() and not any(x in p.name for x in ("venv", "py311", "interrupted"))]
    parser.add_argument("--roots", type=Path, nargs="+", default=default_roots)
    parser.add_argument("--processed", type=Path, default=ROOT / "datasets/ppg_bench_processed/fs_125")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/ppg-hr-poc-audit")
    args = parser.parse_args()
    report = audit(args.roots, args.processed, args.output)
    print(json.dumps({"passed": True, "completed_evaluations": report["completed_evaluations"], "missing_runs": len(report["missing_runs"]), "report": str(args.output / "audit.json")}))


if __name__ == "__main__":
    main()
