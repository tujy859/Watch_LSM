"""Small subject-separated health-information increment checks, not benchmarks.

ECSMP: predict the self-reported SDS symptom score from the first 120 seconds
of the neutral stimulus (30 people: 18 train, 6 validation, 6 test).
PPG-BP: predict cuff SBP/DBP from three independent 2.1-second fingertip segments
(60 people: 40 train, 10 validation, 10 test). No PRV is computed from these
short, noncontiguous clips. All model encoders are frozen; only Ridge heads fit.

The CLI intentionally has no full-dataset switch. Full evaluation needs the
user's separate confirmation. Inputs/outputs with person identities stay local.
Run using the existing .local/ppg-bench-venv environment; no new dependencies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import warnings
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.signal import resample_poly, welch
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.ppg_beat_metrics import elgendi_peaks
from scripts.ppg_hr_poc_features import sha256

SEED = 20261105
MODELS = ("AnyPPG", "PaPaGei-P")
ALPHAS = (.1, 1., 10., 100., 1000.)
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def workbook_rows(path, sheet_name=None):
    """Read simple XLSX cells without introducing a spreadsheet dependency."""
    with ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            for item in ET.fromstring(z.read("xl/sharedStrings.xml")):
                strings.append("".join(t.text or "" for t in item.iter("{" + NS["s"] + "}t")))
        target = "xl/worksheets/sheet1.xml"
        if sheet_name:
            relationships = {x.attrib["Id"]: x.attrib["Target"] for x in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
            sheets = ET.fromstring(z.read("xl/workbook.xml")).findall("s:sheets/s:sheet", NS)
            matches = [s for s in sheets if s.attrib["name"] == sheet_name]
            if len(matches) != 1:
                raise ValueError("Workbook sheet missing or ambiguous")
            key = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            rel = relationships[matches[0].attrib[key]]
            target = rel.lstrip("/") if rel.startswith("/") else "xl/" + rel
        rows = []
        for row in ET.fromstring(z.read(target)).findall("s:sheetData/s:row", NS):
            cells = {}
            for cell in row:
                letters = "".join(c for c in cell.attrib["r"] if c.isalpha())
                column = 0
                for letter in letters:
                    column = 26 * column + ord(letter) - ord("A") + 1
                value = cell.find("s:v", NS)
                text = value.text if value is not None else "".join(t.text or "" for t in cell.findall("s:is/s:t", NS))
                if cell.attrib.get("t") == "s":
                    text = strings[int(text)]
                cells[column - 1] = text
            rows.append(cells)
    return rows


def deterministic_split(subjects, sizes, seed=SEED):
    """Choose identities without examining outcomes; return disjoint roles."""
    people = np.asarray(sorted(set(map(str, subjects))))
    if len(people) != len(subjects) or sum(sizes) > len(people) or min(sizes) < 1:
        raise ValueError("Need distinct people and feasible positive split sizes")
    chosen = np.random.default_rng(seed).permutation(people)[:sum(sizes)]
    a, b = sizes[0], sizes[0] + sizes[1]
    return {"train": chosen[:a].tolist(), "validation": chosen[a:b].tolist(), "test": chosen[b:].tolist()}


def check_roles(subjects, roles):
    if len(subjects) != len(roles) or len(subjects) != len(set(subjects)):
        raise ValueError("One row per distinct subject is required")
    if set(roles) != {"train", "validation", "test"}:
        raise ValueError("Need train, validation, and test roles")


def continuous_core(signal, fs, start_s, duration_s=120., pad_s=2.):
    """Slice one raw continuous interval; never concatenate selected windows."""
    x = np.asarray(signal)
    start, stop = int(round(start_s * fs)), int(round((start_s + duration_s) * fs))
    pad = int(round(pad_s * fs))
    if x.ndim != 1 or start < pad or stop + pad > len(x):
        raise ValueError("Continuous interval or its fixed padding unavailable")
    extended = np.asarray(x[start - pad:stop + pad], dtype=np.float64)
    if not np.isfinite(extended).all():
        raise ValueError("Nonfinite samples in raw continuous interval")
    return extended[pad:-pad].copy(), extended, {"start_sample": start, "stop_sample": stop, "sampling_hz": fs, "padding_samples": pad}


def signal_features(signal, fs):
    """Fixed PPG shape/spectral descriptors; not a complete morphology package."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or len(x) < fs * 2 or not np.isfinite(x).all():
        raise ValueError("Require a finite PPG segment at least two seconds long")
    deviation = x.std()
    if deviation <= 0:
        return np.full(10, np.nan)
    z = (x - x.mean()) / deviation
    f, p = welch(z, fs=fs, nperseg=min(len(z), int(4 * fs)))
    cardiac = (f >= .5) & (f <= 4)
    total = p.sum()
    dominant = f[np.flatnonzero(cardiac)[np.argmax(p[cardiac])]] * 60
    derivative = np.diff(z) * fs
    second = np.diff(z, n=2) * fs * fs
    return np.asarray([dominant, skew(z, bias=False), kurtosis(z, bias=False), np.ptp(z),
                       np.mean(np.abs(derivative)), np.sqrt(np.mean(derivative ** 2)),
                       np.sqrt(np.mean(second ** 2)), np.mean(z > 0),
                       p[(f >= .5) & (f <= 2)].sum() / total,
                       p[(f > 2) & (f <= 8)].sum() / total], dtype=float)


def pulse_features(extended, fs, duration_s=120., pad_s=2.):
    """PRV descriptors from raw continuous PPG, without ECG or clinical claims."""
    times = elgendi_peaks(extended, fs)
    times = times[(times >= pad_s) & (times < pad_s + duration_s)]
    rr = np.diff(times) * 1000
    if len(rr) < 2:
        return np.asarray([np.nan, np.nan, np.nan, np.nan, len(times) / duration_s])
    # No label-selected correction, ectopic filtering, or interpolation.
    return np.asarray([60000 / rr.mean(), np.std(60000 / rr, ddof=1),
                       np.std(rr, ddof=1), np.sqrt(np.mean(np.diff(rr) ** 2)), len(times) / duration_s])


def fit_preprocessor(training):
    x = np.asarray(training, dtype=float)
    if x.ndim != 2 or not len(x) or np.isinf(x).any():
        raise ValueError("Invalid training features")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(x, axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    filled = np.where(np.isnan(x), median, x)
    augmented = np.column_stack([filled, np.isnan(x).astype(float)])
    scaler = StandardScaler().fit(augmented)
    return median, scaler


def apply_preprocessor(x, fitted):
    median, scaler = fitted
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(median) or np.isinf(x).any():
        raise ValueError("Feature shape or numeric values disagree")
    return scaler.transform(np.column_stack([np.where(np.isnan(x), median, x), np.isnan(x).astype(float)]))


def ridge_check(x, y, roles):
    """Fit preprocessing/train, select alpha/validation, score untouched test."""
    x, y, roles = np.asarray(x), np.asarray(y, dtype=float), np.asarray(roles)
    train, validation, test = (roles == k for k in ("train", "validation", "test"))
    if not all(mask.any() for mask in (train, validation, test)) or not np.isfinite(y).all():
        raise ValueError("Missing split or nonfinite targets")
    preprocessing = fit_preprocessor(x[train])
    transformed = apply_preprocessor(x, preprocessing)
    candidates = []
    with threadpool_limits(limits=2):
        for alpha in ALPHAS:
            reg = Ridge(alpha=alpha).fit(transformed[train], y[train])
            candidates.append((float(np.mean(np.abs(reg.predict(transformed[validation]) - y[validation]))), alpha, reg))
    val_mae, alpha, selected = min(candidates, key=lambda item: (item[0], item[1]))
    prediction = selected.predict(transformed[test])
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite held-out predictions")
    return {"alpha": alpha, "validation_mae": val_mae,
            "test_mae": float(np.mean(np.abs(prediction - y[test]))),
            "test_rmse": float(np.sqrt(np.mean((prediction - y[test]) ** 2))),
            "test_bias": float(np.mean(prediction - y[test])),
            "test_prediction": prediction, "fitted": preprocessing,
            "coefficients": selected.coef_, "intercept": selected.intercept_,
            "features_before_missing_indicators": x.shape[1]}


def ppgbp_data():
    raw_root = ROOT / "datasets/ppg_bp/Data File"
    workbook = raw_root / "PPG-BP dataset.xlsx"
    rows = workbook_rows(workbook)
    header = {value: column for column, value in rows[1].items()}
    by_id = {str(int(float(r[header["subject_ID"]]))): r for r in rows[2:]}
    people = sorted(set(by_id) - {"231"})
    numeric_names = ("Age(year)", "BMI(kg/m^2)", "Heart Rate(b/m)", "Systolic Blood Pressure(mmHg)", "Diastolic Blood Pressure(mmHg)")
    if any(not np.isfinite(float(by_id[s][header[k]])) for s in people for k in numeric_names):
        raise ValueError("Nonfinite clinical label/covariate in PPG-BP inventory")
    if any(by_id[s][header["Sex(M/F)"]] not in ("Male", "Female") for s in people):
        raise ValueError("Missing sex in PPG-BP inventory")
    split = deterministic_split(people, (40, 10, 10))
    subjects = sum(split.values(), [])
    roles = [role for role, ids in split.items() for _ in ids]
    covariates, engineered, targets, windows, waveform_subjects = [], [], [], [], []
    hashes = {workbook.relative_to(ROOT).as_posix(): sha256(workbook)}
    for subject in subjects:
        row = by_id[subject]
        age, bmi, recorded_hr = (float(row[header[k]]) for k in ("Age(year)", "BMI(kg/m^2)", "Heart Rate(b/m)"))
        sex = row[header["Sex(M/F)"]]
        if sex not in ("Male", "Female"):
            raise ValueError("Unrecognized sex label")
        covariates.append([age, float(sex == "Male"), bmi])
        targets.append([float(row[header["Systolic Blood Pressure(mmHg)"]]), float(row[header["Diastolic Blood Pressure(mmHg)"]])])
        shape = []
        for j in (1, 2, 3):
            path = raw_root / "0_subject" / f"{subject}_{j}.txt"
            raw = np.fromstring(path.read_text(), sep="\t")
            if raw.shape != (2100,) or not np.isfinite(raw).all():
                raise ValueError("PPG-BP raw segment differs from 2.1-second protocol")
            hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
            wave = resample_poly(raw, 1, 8).astype(np.float32)
            windows.append(wave)
            waveform_subjects.append(subject)
            shape.append(signal_features(raw, 1000))
        engineered.append([recorded_hr, *np.mean(shape, axis=0)])
    available = [by_id[s] for s in people]
    inventory = {"raw_people": len(by_id), "eligible_people": len(people), "excluded_ids": ["231"],
                 "continuous_labels_complete": True, "disease_nonblank_counts": {k: sum(bool(r.get(header[k], "").strip()) for r in available) for k in ("Diabetes", "cerebral infarction", "cerebrovascular disease")},
                 "disease_blank_interpretation": "Blank cells are not independently adjudicated disease-negative controls",
                 "source": "https://doi.org/10.1038/sdata.2018.20"}
    return {"dataset": "PPG-BP", "subjects": subjects, "roles": roles, "split": split,
            "covariates": np.asarray(covariates), "traditional": np.asarray(engineered),
            "targets": np.asarray(targets), "target_names": ["cuff_sbp_mmhg", "cuff_dbp_mmhg"],
            "windows": np.asarray(windows), "waveform_subjects": waveform_subjects, "fs": 125,
            "source_hashes": hashes, "inventory": inventory,
            "covariate_names": ["age_years", "male_indicator", "bmi_kg_m2"],
            "traditional_description": "Recorded table HR plus mean of 10 fixed PPG spectral/shape descriptors over three separate 2.1s clips; no PRV",
            "observation": "Three independent 2.1-second fingertip PPG clips; 6.3 seconds total, not concatenated",
            "limitations": ["Finger auxiliary domain, not wrist validation", "One cuff label per person; not beat-to-beat BP", "2.1-second clips cannot support reliable PRV", "No causal or clinical validation", "Pretraining person overlap not fully certifiable; no conclusion of pretraining-unseen generalization"]}


def ecsmp_data():
    raw_root = ROOT / "datasets/ppg_bench_raw/ecsmp"
    processed = ROOT / "datasets/ppg_bench_processed/fs_125/ECSMP"
    workbook = raw_root / "scale.xlsx"
    source = workbook_rows(workbook, "SDS")
    labels = {str(int(float(row[0]))).zfill(3): row for row in source[1:] if row.get(0)}
    people = sorted(path.name for path in processed.iterdir() if path.is_dir())
    all_scores, ages, sexes, all_neutral_counts = [], [], [], []
    for subject in people:
        row = labels[subject]
        score, age = float(row[25]), float(row[2])
        if not np.isfinite([score, age]).all() or row[1] not in ("Female", "Male"):
            raise ValueError("Missing ECSMP demographic covariate or SDS score")
        stats = json.loads((processed / subject / "stats.json").read_text())
        if score != float(stats["sds_score"]):
            raise ValueError("Raw SDS workbook differs from processed subject score")
        all_scores.append(score); ages.append(age); sexes.append(row[1])
        with (processed / subject / "metadata.csv").open(newline="") as stream:
            metadata = list(csv.DictReader(stream))
        all_neutral_counts.append(sum(r["stimulus"] == "0" and r["has_video_label"] == "1" for r in metadata))
    split = deterministic_split(people, (18, 6, 6))
    subjects = sum(split.values(), [])
    roles = [role for role, ids in split.items() for _ in ids]
    covariates, engineered, targets, windows, waveform_subjects, intervals = [], [], [], [], [], []
    hashes = {workbook.relative_to(ROOT).as_posix(): sha256(workbook)}
    for subject in subjects:
        row = labels[subject]
        if row[1] not in ("Female", "Male"):
            raise ValueError("Unrecognized sex label")
        covariates.append([float(row[2]), float(row[1] == "Male")]); targets.append([float(row[25])])
        metadata_path = processed / subject / "metadata.csv"
        with metadata_path.open(newline="") as stream:
            metadata = list(csv.DictReader(stream))
        neutral = [r for r in metadata if r["stimulus"] == "0" and r["has_video_label"] == "1"]
        if not neutral:
            raise ValueError("Neutral stimulus missing")
        # Start at the earliest fully neutral 10-second interval, then read one
        # continuous raw slice. Processed-window inclusion cannot create gaps.
        start = min(float(r["start_sec"]) for r in neutral)
        end = min(float(r["video_end_sec"]) for r in neutral)
        if start + 120 > end:
            raise ValueError("Neutral stimulus shorter than fixed observation")
        bvp_path = raw_root / "E4" / subject / "BVP.csv"
        with bvp_path.open() as stream:
            utc_start, fs = float(stream.readline()), int(float(stream.readline()))
        if fs != 64:
            raise ValueError("Unexpected ECSMP raw sampling rate")
        existing_stats = json.loads((processed / subject / "stats.json").read_text())
        if float(existing_stats["bvp_start_utc"]) != utc_start:
            raise ValueError("Raw BVP time origin differs from stimulus metadata")
        raw = np.loadtxt(bvp_path, delimiter=",", skiprows=2)
        core, extended, interval = continuous_core(raw, fs, start)
        interval.update(subject=subject, start_seconds=start, duration_seconds=120., bvp_start_utc=utc_start)
        intervals.append(interval)
        hashes[bvp_path.relative_to(ROOT).as_posix()] = sha256(bvp_path)
        hashes[metadata_path.relative_to(ROOT).as_posix()] = sha256(metadata_path)
        # Continuous resampling before segmentation avoids per-window boundary
        # resampling changes. Native model filters still process each 10s clip.
        resampled = resample_poly(core, 125, 64).astype(np.float32)
        pieces = resampled.reshape(12, 1250)
        windows.extend(pieces); waveform_subjects.extend([subject] * len(pieces))
        shapes = np.asarray([signal_features(x, 125) for x in pieces])
        engineered.append([*np.mean(shapes, axis=0), *pulse_features(extended, fs)])
    inventory = {"eligible_people": len(people), "age_range_years": [min(ages), max(ages)],
                 "sex_counts": dict(Counter(sexes)), "sds_range": [min(all_scores), max(all_scores)],
                 "sds_at_least_50": sum(x >= 50 for x in all_scores), "sds_at_least_53": sum(x >= 53 for x in all_scores),
                 "all_age_sex_sds_complete": True, "raw_sds_matches_processed_people": len(people),
                 "neutral_10s_windows_per_person_range": [min(all_neutral_counts), max(all_neutral_counts)],
                 "source": "https://data.mendeley.com/datasets/vn5nknh3mn/2"}
    return {"dataset": "ECSMP", "subjects": subjects, "roles": roles, "split": split,
            "covariates": np.asarray(covariates), "traditional": np.asarray(engineered),
            "targets": np.asarray(targets), "target_names": ["sds_self_report_score"],
            "windows": np.asarray(windows), "waveform_subjects": waveform_subjects, "fs": 125,
            "source_hashes": hashes, "intervals": intervals, "inventory": inventory,
            "covariate_names": ["age_years", "male_indicator"],
            "traditional_description": "10 mean spectral/shape descriptors plus HR mean/SD, PP interval SD, RMSSD, pulse rate from raw continuous 120s; no correction or ECG reference",
            "observation": "First 120s beginning at earliest fully neutral 10s window; continuous original wrist BVP with 2s padding for peak extraction",
            "limitations": ["SDS is a self-report symptom scale, not clinician diagnosis", "60 total available people aged 18-28; limited population range", "Current fixed single split uses only 30 people", "PRV descriptors are unvalidated algorithm estimates without ECG reference", "Source alignment follows existing EEG/video metadata, not independently reannotated stimulus onsets", "Pretraining overlap not fully certifiable; official listed sources do not establish individual non-overlap"]}


def feature_groups(covariates, traditional, embedding):
    return {"demographics": covariates,
            "demographics_traditional": np.column_stack([covariates, traditional]),
            "embedding": embedding,
            "demographics_embedding": np.column_stack([covariates, embedding]),
            "demographics_traditional_embedding": np.column_stack([covariates, traditional, embedding])}


def independent_prediction_audit(directory, data, rows):
    """Recompute saved head predictions and closed-form Ridge without sklearn."""
    roles = np.asarray(data["roles"])
    train, test = roles == "train", roles == "test"
    max_saved_delta, max_refit_delta, max_metric_delta = 0., 0., 0.
    for row in rows:
        model = MODELS[0] if row["model"] == "shared_control" else row["model"]
        with np.load(directory / f"{model}-features.npz", allow_pickle=False) as feature:
            np.testing.assert_allclose(feature["scale_mean"], feature["raw"][train].mean(axis=0), rtol=1e-5, atol=1e-6)
            raw_standard = (feature["raw"] - feature["scale_mean"]) / feature["scale_scale"]
            rebuilt = (raw_standard - feature["pca_mean"]) @ feature["pca_components"].T
            np.testing.assert_allclose(rebuilt, feature["reduced"], rtol=1e-4, atol=1e-4)
            x = feature_groups(data["covariates"], data["traditional"], feature["reduced"])[row["inputs"]]
        identity = f"{model}__{row['target']}__{row['inputs']}"
        y = data["targets"][:, data["target_names"].index(row["target"])]
        with np.load(directory / (identity + ".npz"), allow_pickle=False) as saved:
            if not np.array_equal(saved["subjects"], np.asarray(data["subjects"])[test]):
                raise ValueError("Saved test participant identity differs")
            np.testing.assert_array_equal(saved["y_true"], y[test])
            filled = np.where(np.isnan(x), saved["imputation_median"], x)
            augmented = np.column_stack([filled, np.isnan(x).astype(float)])
            z = (augmented - saved["standard_mean"]) / saved["standard_scale"]
            direct = z[test] @ saved["coef"] + saved["intercept"]
            centered_x = z[train] - z[train].mean(axis=0)
            centered_y = y[train] - y[train].mean()
            with threadpool_limits(limits=2):
                beta = np.linalg.solve(centered_x.T @ centered_x + row["alpha"] * np.eye(z.shape[1]), centered_x.T @ centered_y)
            refit = (z[test] - z[train].mean(axis=0)) @ beta + y[train].mean()
            max_saved_delta = max(max_saved_delta, float(np.max(np.abs(direct - saved["y_pred"]))))
            max_refit_delta = max(max_refit_delta, float(np.max(np.abs(refit - saved["y_pred"]))))
            metric = np.mean(np.abs(saved["y_pred"] - saved["y_true"]))
            max_metric_delta = max(max_metric_delta, abs(float(metric) - row["test_mae"]))
    if max(max_saved_delta, max_refit_delta, max_metric_delta) > 1e-7:
        raise ValueError("Independent health head reconstruction failed")
    return {"heads_checked": len(rows), "saved_prediction_max_difference": max_saved_delta,
            "independent_closed_form_ridge_max_difference": max_refit_delta,
            "mae_recalculation_max_difference": max_metric_delta,
            "passed": True, "encoder_reforward_performed_by_this_audit": False}


def run_dataset(data, output, shared):
    from scripts.ppg_model_expansion_pilot import extract_features
    directory = output / data["dataset"]
    directory.mkdir(parents=True, exist_ok=True)
    subjects, roles = np.asarray(data["subjects"]), np.asarray(data["roles"])
    check_roles(subjects.tolist(), roles.tolist())
    np.savez_compressed(directory / "participants.npz", subjects=subjects, roles=roles,
                        covariates=data["covariates"], traditional=data["traditional"], targets=data["targets"])
    train, test = roles == "train", roles == "test"
    model_evidence, rows, saved_predictions = {}, [], []
    for model in MODELS:
        raw_features, evidence = extract_features(model, data["windows"], data["fs"], device="cpu", batch_size=8, seed=SEED)
        wave_people = np.asarray(data["waveform_subjects"])
        if set(wave_people) != set(subjects):
            raise ValueError("Waveform and participant identities disagree")
        embeddings = np.asarray([raw_features[wave_people == person].mean(axis=0) for person in subjects])
        # Fixed eight-component projection, learned from training people only.
        # Both models share the projection dimension and head search choices.
        with threadpool_limits(limits=2):
            scale = StandardScaler().fit(embeddings[train])
            projection = PCA(n_components=8, svd_solver="full").fit(scale.transform(embeddings[train]))
            reduced = projection.transform(scale.transform(embeddings))
        evidence.update(projection_components=8, projection_fitted_people=int(train.sum()),
                        person_aggregation="Arithmetic mean over identical fixed observation windows",
                        input_waveforms_sha256=hashlib.sha256(data["windows"].tobytes()).hexdigest())
        model_evidence[model] = evidence
        np.savez_compressed(directory / f"{model}-features.npz", subjects=subjects, roles=roles,
                            raw=embeddings, reduced=reduced, scale_mean=scale.mean_, scale_scale=scale.scale_,
                            pca_mean=projection.mean_, pca_components=projection.components_)
        groups = feature_groups(data["covariates"], data["traditional"], reduced)
        for target_idx, target in enumerate(data["target_names"]):
            y = data["targets"][:, target_idx]
            for name, x in groups.items():
                if model != MODELS[0] and name in ("demographics", "demographics_traditional"):
                    continue
                result = ridge_check(x, y, roles)
                prediction = result.pop("test_prediction")
                preprocess = result.pop("fitted")
                coef, intercept = result.pop("coefficients"), result.pop("intercept")
                entry = {"dataset": data["dataset"], "target": target,
                         "model": model if "embedding" in name else "shared_control", "inputs": name,
                         "train_people": int(train.sum()), "validation_people": int((roles == "validation").sum()),
                         "test_people": int(test.sum()), "scientific_effectiveness_conclusion": False, **result}
                rows.append(entry)
                identity = f"{model}__{target}__{name}"
                np.savez_compressed(directory / (identity + ".npz"), subjects=subjects[test], y_true=y[test], y_pred=prediction,
                                    coef=coef, intercept=intercept, imputation_median=preprocess[0],
                                    standard_mean=preprocess[1].mean_, standard_scale=preprocess[1].scale_)
                saved_predictions.append(identity + ".npz")
    audit = independent_prediction_audit(directory, data, rows)
    evidence = {"status": "passed_pipeline_check_only", "dataset": data["dataset"], "seed": SEED,
                "split_counts": {k: len(v) for k, v in data["split"].items()},
                "distinct_people": len(subjects), "raw_windows": len(data["windows"]),
                "input_shape": list(data["windows"].shape), "observation": data["observation"],
                "inventory": data["inventory"], "covariate_names": data["covariate_names"],
                "traditional_description": data["traditional_description"], "models": model_evidence,
                "independent_head_audit": audit,
                "checks": ["Distinct people across train/validation/test", "Raw SDS scores match processed subject scores" if data["dataset"] == "ECSMP" else "PPG-BP three raw clips per person have 2100 finite samples", "No target-derived BP or disease fields used as covariates", "Preprocessing and eight-component PCA fit training people only", "Alpha selection uses validation people only; no refit on test or validation", "All encoder states unchanged", "Same fixed windows and person-level mean for both encoders", "All predictions finite; one prediction per test person"],
                "ridge_alphas": list(ALPHAS), "limitations": data["limitations"],
                "outputs_sha256": {name: sha256(directory / name) for name in saved_predictions},
                "new_training": "Ridge task heads only; encoders frozen", "full_evaluation_started": False}
    local_manifest = {**evidence, "split": data["split"], "source_hashes": data["source_hashes"],
                      "intervals": data.get("intervals", [])}
    (directory / "manifest.json").write_text(json.dumps(local_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    evidence["local_manifest_sha256"] = sha256(directory / "manifest.json")
    (shared / f"health_{data['dataset'].lower()}_audit.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("both", "ppgbp", "ecsmp"), default="both")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/ppg-expansion-pilot/health")
    parser.add_argument("--shared", type=Path, default=ROOT / "docs/results/ppg_expansion_pilot")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); args.shared.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, builder in (("ppgbp", ppgbp_data), ("ecsmp", ecsmp_data)):
        if args.dataset in ("both", name):
            print(json.dumps({"dataset": name, "phase": "prepare_raw_small_sample"}), flush=True)
            data = builder()
            rows.extend(run_dataset(data, args.output, args.shared))
            print(json.dumps({"dataset": name, "phase": "pipeline_check_passed", "people": len(data["subjects"])}), flush=True)
    filename = "health_summary.csv" if args.dataset == "both" else f"health_{args.dataset}_summary.csv"
    with (args.shared / filename).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status": "passed_pipeline_check_only", "head_fits": len(rows), "full_evaluation_started": False}), flush=True)


if __name__ == "__main__":
    main()
