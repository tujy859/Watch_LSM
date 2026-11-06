"""Read saved SDS artifacts; emit aggregate diagnostics without fitting models.

Run from any working directory with the existing ppg-bench Python environment.
Prints JSON to stdout by default. --output may create a new file under .local;
existing files are never overwritten. No participant identifiers or individual
participant rows are included in the output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".local/ppg-formal-expansion/health/ECSMP"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def set_digest(entries):
    """Hash sorted path/SHA pairs without publishing participant path entries."""
    encoded = json.dumps(sorted(entries), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def average(values):
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Create a new JSON file under .local; default: stdout")
    args = parser.parse_args()
    output = None
    if args.output is not None:
        output = (ROOT / args.output).resolve()
        if not output.is_relative_to((ROOT / ".local").resolve()):
            parser.error("File output must stay under the project .local directory")
        if output.exists():
            parser.error("Refusing to overwrite an existing output file")
    with np.load(SOURCE / "data.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((SOURCE / "results.json").read_text(encoding="utf-8"))
    assert sha256(SOURCE / "data.npz") == manifest["data_npz_sha256"]
    subjects = data["subjects"]
    y = data["targets"].ravel()
    source_files = [SOURCE / name for name in ("data.npz", "manifest.json", "results.json")]
    source_files += [
        ROOT / "scripts/ppg_formal_health.py",
        ROOT / "scripts/ppg_health_increment_pilot.py",
        ROOT / "docs/results/ppg_formal_expansion/health/ECSMP_summary.csv",
        ROOT / "docs/results/ppg_formal_expansion/health/ECSMP_comparisons.json",
    ]
    folds = []
    mean_errors, median_errors = [], []
    for split in manifest["split"]:
        train, validation, test = [
            np.flatnonzero(np.isin(subjects, split[role]))
            for role in ("train", "validation", "test")
        ]
        assert not (set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test))
        mean_error = np.abs(y[test] - y[train].mean())
        median_error = np.abs(y[test] - np.median(y[train]))
        mean_errors.extend(mean_error.tolist())
        median_errors.extend(median_error.tolist())
        folds.append({
            "fold": split["fold"],
            "train_people": len(train), "validation_people": len(validation), "test_people": len(test),
            "train_mean": float(y[train].mean()), "train_median": float(np.median(y[train])),
            "validation_mean": float(y[validation].mean()), "test_mean": float(y[test].mean()),
            "test_population_std": float(y[test].std()),
            "train_mean_test_mae": average(mean_error), "train_median_test_mae": average(median_error),
        })

    pca_rows, native_inputs = [], []
    for feature_path in sorted((SOURCE / "features").glob("*.npz")):
        model, seed_tag = feature_path.stem.split("__")
        seed = int(seed_tag[4:])
        with np.load(feature_path, allow_pickle=False) as blob:
            assert np.array_equal(blob["subjects"], subjects)
            raw = blob["person_features"].astype(float)
        evidence_path = feature_path.with_suffix(".json")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        assert sha256(feature_path) == evidence["output_sha256"]
        source_files.extend([feature_path, evidence_path])
        ratios, first_pc_ratios = [], []
        for split in manifest["split"]:
            train = np.flatnonzero(np.isin(subjects, split["train"]))
            file = SOURCE / "predictions" / f"{model}__embedding__sds_self_report_score__linear__s{seed}__f{split['fold']}.npz"
            with np.load(file, allow_pickle=False) as saved:
                z = (raw[train] - saved["scale_mean"]) / saved["scale_scale"] - saved["pca_mean"]
                projected = z @ saved["pca_components"].T
            total_ss = np.square(z).sum()
            ratios.append(float(np.square(projected).sum() / total_ss))
            first_pc_ratios.append(float(np.square(projected[:, 0]).sum() / total_ss))
        pca_rows.append({
            "model": model, "encoder_seed": seed, "raw_dimensions": raw.shape[1], "components": 8,
            "training_variance_retained_by_fold": ratios, "training_variance_retained_mean": average(ratios),
            "first_component_variance_retained_mean": average(first_pc_ratios),
        })
        native_inputs.append({key: evidence[key] for key in (
            "model", "random_init", "seed", "native_fs", "native_input_samples",
            "feature_shape", "padding_samples", "cropped_samples", "trainable_encoder_parameters",
        )})

    grouped = defaultdict(list)
    prediction_sources = []
    for row in rows:
        file = SOURCE.parent / row["prediction_file"]
        digest = sha256(file)
        assert digest == row["prediction_sha256"]
        prediction_sources.append((relative(file), digest))
        if row["head"] != "constant":
            grouped[(row["model"], row["inputs"], row["head"])].append(row)

    saved_head_diagnostics, validation_curves = [], []
    max_test_reconstruction_difference = 0.0
    max_mae_reconstruction_difference = 0.0
    for (model, inputs, head), group in sorted(grouped.items()):
        role_errors = defaultdict(list)
        prediction_by_seed, truth_by_seed = defaultdict(list), defaultdict(list)
        active_columns = set()
        for row in group:
            with np.load(SOURCE.parent / row["prediction_file"], allow_pickle=False) as saved:
                z = saved["prepared_features"]
                active_columns.add(int(np.sum(np.std(z, axis=0) > 1e-10)))
                if head == "linear":
                    predicted = z @ saved["coef"] + saved["intercept"]
                else:
                    hidden = np.maximum(z.astype(np.float32) @ saved["w1"].T + saved["b1"], 0)
                    predicted = (hidden @ saved["w2"].T + saved["b2"]).ravel().astype(float)
                    predicted = predicted * saved["target_scale"] + saved["target_center"]
                for role, key in (("train", "train_subjects"), ("validation", "validation_subjects"), ("test", "subjects")):
                    indices = np.flatnonzero(np.isin(subjects, saved[key]))
                    value = float(np.mean(np.abs(predicted[indices] - y[indices])))
                    role_errors[role].append(value)
                    if role == "test":
                        assert np.array_equal(subjects[indices], saved["subjects"])
                        difference = float(np.max(np.abs(predicted[indices] - saved["y_pred"])))
                        max_test_reconstruction_difference = max(max_test_reconstruction_difference, difference)
                        max_mae_reconstruction_difference = max(max_mae_reconstruction_difference, abs(value - row["mae"]))
                        prediction_by_seed[row["seed"]].extend(predicted[indices].tolist())
                        truth_by_seed[row["seed"]].extend(y[indices].tolist())
        r2s, prediction_stds = [], []
        for seed in sorted(prediction_by_seed):
            predicted = np.array(prediction_by_seed[seed])
            truth = np.array(truth_by_seed[seed])
            r2s.append(float(1 - np.square(predicted - truth).sum() / np.square(truth - truth.mean()).sum()))
            prediction_stds.append(float(predicted.std()))
        item = {
            "model": model, "inputs": inputs, "head": head, "fits": len(group),
            "active_nonconstant_input_columns": sorted(active_columns),
            "allocated_head_parameters": group[0]["trainable_head_parameters"],
            "mae_mean_over_folds_and_seeds": {role: average(values) for role, values in role_errors.items()},
            "oof_prediction_population_std_mean_over_seeds": average(prediction_stds),
            "oof_r2_mean_over_seeds": average(r2s),
        }
        if head == "linear":
            item["selected_alpha_counts_seed42"] = dict(sorted(Counter(str(r["alpha"]) for r in group if r["seed"] == 42).items()))
        else:
            epochs = np.array([r["epochs"] for r in group])
            curves = np.array([r["validation_mae_by_epoch"] for r in group], dtype=float)
            assert curves.shape == (15, 100)
            last_minus_best = curves[:, -1] - curves.min(axis=1)
            item.update({
                "best_epoch_median": float(np.median(epochs)),
                "best_epoch_counts": dict(sorted(Counter(map(str, epochs)).items(), key=lambda x: int(x[0]))),
                "last_minus_best_validation_mae_mean": average(last_minus_best),
                "last_minus_best_validation_mae_median": float(np.median(last_minus_best)),
            })
            validation_curves.append({
                "model": model, "inputs": inputs, "fit_count": len(group),
                "epoch_indexing": "1 through 100",
                "mean_validation_mae_by_epoch": curves.mean(axis=0).tolist(),
            })
        saved_head_diagnostics.append(item)
    assert max_test_reconstruction_difference < 1e-4

    metadata_sources = []
    available_seconds, video_seconds = [], []
    for file in sorted((ROOT / "datasets/ppg_bench_processed/fs_125/ECSMP").glob("*/metadata.csv")):
        digest = sha256(file)
        assert digest == manifest["source_hashes"][relative(file)]
        metadata_sources.append((relative(file), digest))
        with file.open(newline="") as stream:
            records = list(csv.DictReader(stream))
        neutral = [r for r in records if r["stimulus"] == "0" and r["has_video_label"] == "1"]
        spans = sorted(set((float(r["video_start_sec"]), float(r["video_end_sec"])) for r in neutral))
        start = min(float(r["start_sec"]) for r in neutral)
        available_seconds.append(min(end for begin, end in spans) - start)
        video_seconds.append(sum(end - begin for begin, end in spans))
    assert len(available_seconds) == len(subjects) == 60
    assert min(available_seconds) >= 232

    mlp_rows = [r for r in rows if r["head"] == "mlp"]
    epochs = np.array([r["epochs"] for r in mlp_rows])
    embedding_linear = [r for r in rows if r["head"] == "linear" and r["seed"] == 42 and r["inputs"] == "embedding" and r["model"] not in ("shared_control", "AnyPPG-random-frozen")]
    summary_path = ROOT / "docs/results/ppg_formal_expansion/health/ECSMP_summary.csv"
    with summary_path.open(newline="") as stream:
        formal_summary = list(csv.DictReader(stream))
    report = {
        "schema_version": "sds-readonly-diagnostic-v1",
        "analysis_date": "2026-11-06",
        "scope": "Post hoc descriptive diagnostics of existing formal SDS artifacts; no new model fitting, encoder extraction, preprocessing fit, hypothesis selection, or replacement of formal outputs.",
        "privacy": "No participant identifiers, per-person predictions, or per-person label rows are emitted.",
        "execution": {"script": relative(Path(__file__)), "script_sha256": sha256(Path(__file__)), "python_environment": ".local/ppg-bench-venv", "command": ".local/ppg-bench-venv/Scripts/python.exe -B scripts/diagnose_ppg_sds_readonly.py" + (" --output " + relative(output) if output is not None else "")},
        "source_files": [{"path": relative(path), "sha256": sha256(path)} for path in source_files],
        "source_sets": {
            "prediction_npz": {"root": relative(SOURCE / "predictions"), "count": len(prediction_sources), "sorted_path_sha256_pairs_sha256": set_digest(prediction_sources), "all_match_results_manifest": True},
            "neutral_metadata": {"root": "datasets/ppg_bench_processed/fs_125/ECSMP", "pattern": "*/metadata.csv", "count": len(metadata_sources), "sorted_path_sha256_pairs_sha256": set_digest(metadata_sources), "all_match_formal_manifest": True},
            "digest_encoding": "SHA256 of UTF-8 JSON sorted list of [repository-relative POSIX path, SHA256] pairs, ensure_ascii=False and separators=(',', ':')",
        },
        "protocol_facts": {
            "people": len(subjects), "waveform_clips": len(data["windows"]), "seconds_per_person": 120,
            "total_observed_hours": len(subjects) * 120 / 3600, "clips_per_person": 15, "seconds_per_clip": 8,
            "pooling": "Arithmetic mean of fifteen separately encoded eight-second vectors; no between-clip order supplied to the prediction head.",
            "projection": "Saved training-person standardization followed by saved PCA8; no PCA fit performed in this diagnostic.",
            "traditional_features": "Ten averaged eight-second shape/spectral descriptors plus five descriptors computed from continuous 120-second PPG: HR mean/SD, interval SD, RMSSD and pulse rate; no ECG validation.",
            "folds": folds, "initialization_seeds": [42, 43, 44], "seeds_change_person_folds": False,
            "native_inputs": native_inputs,
        },
        "label_distribution": {
            "mean": float(y.mean()), "population_std": float(y.std()), "sample_std": float(y.std(ddof=1)),
            "quantile_percentiles": [0, 10, 25, 50, 75, 90, 100],
            "quantile_values": np.percentile(y, [0, 10, 25, 50, 75, 90, 100]).tolist(),
            "distinct_score_values": len(np.unique(y)),
            "aggregate_score_counts": {str(value): int(count) for value, count in sorted(Counter(y.tolist()).items())},
            "age_counts": {str(value): int(count) for value, count in sorted(Counter(data["covariates"][:, 0].tolist()).items())},
            "male_count": int(data["covariates"][:, 1].sum()), "female_count": int(len(subjects) - data["covariates"][:, 1].sum()),
        },
        "constant_diagnostics": {
            "training_mean_oof_mae": average(mean_errors), "training_median_oof_mae": average(median_errors),
            "training_median_status": "New descriptive calculation using existing training roles only; not a new formal model result or winner-selection criterion.",
        },
        "pca8_training_variance_retained": pca_rows,
        "saved_head_diagnostics": saved_head_diagnostics,
        "mlp_validation_curves": validation_curves,
        "fit_selection_aggregate": {
            "mlp_fit_count": len(epochs), "mlp_best_epoch_median": float(np.median(epochs)),
            "mlp_best_epoch_eq1": int((epochs == 1).sum()), "mlp_best_epoch_le5": int((epochs <= 5).sum()), "mlp_best_epoch_eq100": int((epochs == 100).sum()),
            "pretrained_embedding_linear_independent_model_folds": len(embedding_linear),
            "pretrained_embedding_linear_alpha_eq1000": sum(r["alpha"] == 1000 for r in embedding_linear),
            "pretrained_embedding_linear_alpha_ge100": sum(r["alpha"] >= 100 for r in embedding_linear),
        },
        "available_neutral_duration": {
            "method": "Read fixed stimulus metadata; from earliest fully labelled neutral-window start to earliest neutral video end; do not read model scores to select time.",
            "people": len(available_seconds), "usable_seconds_min_median_max": np.percentile(available_seconds, [0, 50, 100]).tolist(),
            "neutral_video_seconds_min_median_max": np.percentile(video_seconds, [0, 50, 100]).tolist(),
            "potential_uniform_seconds": 232, "potential_uniform_8s_clips": 29,
            "all_people_have_metadata_coverage_for_232_seconds": True,
            "status": "Feasibility from timestamps only; 232-second waveform extraction and evaluation were not performed.",
        },
        "checks": {"data_hash_matches_formal_manifest": True, "prediction_hashes_match_formal_results": True, "metadata_hashes_match_formal_manifest": True, "max_saved_prediction_reconstruction_absolute_difference": max_test_reconstruction_difference, "max_recorded_mae_reconstruction_absolute_difference": max_mae_reconstruction_difference},
        "formal_summary_reference": [{"model": row["model"], "inputs": row["inputs"], "head": row["head"], "mae": float(row["mae"]), "rmse": float(row["rmse"])} for row in formal_summary],
        "interpretation_limits": [
            "Retained feature variance is not retained SDS-predictive information; no predictive utility test of discarded PCA directions was performed.",
            "Fixed folds contain only 36 training, 12 validation and 12 test people; initialization seeds do not create new independent participants or cohort splits.",
            "Early selected MLP epochs and worsening later validation curves are consistent with overfitting/unstable small-sample selection, not proof of a unique causal failure mechanism.",
            "OOF R2 uses the global held-out cohort mean only as a descriptive scoring denominator, not as an available deployment predictor.",
            "Saved best heads were evaluated without fitting; training trajectories were not recorded and cannot be reconstructed from best weights.",
            "Aggregate diagnosis and proposed next controls are exploratory after inspecting existing test results; independent cohorts are required for confirmatory health-increment claims.",
            "The 232-second option is a modest extension of the same experimental neutral period, not repeated multiday health observation.",
        ],
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output is None:
        print(encoded, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
        print(json.dumps({"output": relative(output), "bytes": output.stat().st_size,
                          "sha256": sha256(output), "checks": report["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
