"""Small, subject-independent BUT-PPG signal-quality evaluation.

The target is expert consensus that HR is readable from a smartphone red PPG
channel, not wrist morphology quality. Full public v2 data are kept separate
from the previous quality-filtered benchmark. No foundation encoder is trained.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request
import zipfile

import numpy as np
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt, welch
from scipy.special import expit
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_RAW = ROOT / "datasets/ppg_bench_raw/butppg-full-v2"
DEFAULT_OUTPUT = ROOT / ".local/ppg-phase2/sqi"
ARCHIVE = ROOT / ".local/ppg-phase2/research/butppg-2.0.0.zip"
SOURCE = "https://physionet.org/content/butppg/2.0.0/"
ZIP_URL = "https://physionet-open.s3.amazonaws.com/butppg/butppg-2.0.0.zip"
METHODS = ("AnyPPG", "SmallCNN", "HandSQI", "TrainPrior")
SEEDS = (42, 43, 44)
FEATURE_NAMES = (
    "flat", "skewness", "kurtosis", "crest", "diff_std_ratio", "zero_crossings",
    "flat_step_fraction", "extreme_fraction", "amplitude_entropy", "cardiac_power_fraction",
    "dominant_power_fraction", "spectral_entropy", "dominant_frequency", "autocorrelation_peak",
    "autocorrelation_lag", "spectral_peak_to_median", "peak_count", "interval_cv",
    "median_interval", "positive_area_fraction",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_csv(path, records):
    if not records:
        raise ValueError("Cannot publish an empty table")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def unpack_archive(archive, destination):
    """Validate zip CRC and extract without replacing existing differing files."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Archive failed CRC")
        for info in z.infolist():
            parts = Path(info.filename).parts
            if not parts or parts[0] != "brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0":
                raise ValueError(f"Unexpected archive prefix: {info.filename}")
            target = (destination / Path(*parts[1:])).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Unsafe archive path")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            payload = z.read(info)
            if target.exists():
                if target.read_bytes() != payload:
                    raise ValueError(f"Existing data differs: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)


def download(raw):
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.is_file():
        partial = ARCHIVE.with_suffix(".partial")
        with urllib.request.urlopen(ZIP_URL, timeout=60) as response, partial.open("wb") as f:
            expected = int(response.headers["Content-Length"])
            while data := response.read(1024 * 1024):
                f.write(data)
        if partial.stat().st_size != expected:
            raise ValueError("Incomplete archive download")
        partial.replace(ARCHIVE)
    unpack_archive(ARCHIVE, raw)
    print(json.dumps({"archive_bytes": ARCHIVE.stat().st_size, "sha256": sha(ARCHIVE)}), flush=True)


def decode_ppg(raw, record_id):
    """Decode the two released WFDB layouts into 300 physical red-channel samples.

    Early records store 300 time points as 300 one-sample channels with a
    separate ADC gain per sample; their digital values can all be zero. Later
    records store red/green/blue as three genuine channels. Quality annotation
    used red (publisher Methods), so the green-channel benchmark cache is not
    reused. No extra sign inversion is applied to the released physical data.
    """
    header = raw / record_id / f"{record_id}_PPG.hea"
    data = header.with_suffix(".dat")
    lines = [line.strip() for line in header.read_text().splitlines() if line.strip()]
    first = lines[0].split()
    channels, fs, samples = int(first[1]), float(first[2]), int(first[3])
    if fs != 30 or len(lines) < channels + 1:
        raise ValueError("Unexpected BUT PPG sampling or incomplete header")
    gains, baselines, names = [], [], []
    for line in lines[1:1 + channels]:
        fields = line.split()
        match = re.fullmatch(r"([-+0-9.eE]+)\(([-+0-9.eE]+)\)/.+", fields[2])
        if fields[0] != data.name or fields[1] != "16" or match is None:
            raise ValueError("Unexpected WFDB encoding")
        gains.append(float(match[1]))
        baselines.append(float(match[2]))
        names.append(fields[-1])
    digital = np.fromfile(data, dtype="<i2")
    if len(digital) != channels * samples or np.any(np.asarray(gains) == 0):
        raise ValueError("Invalid digital dimensions or zero gain")
    physical = (digital.reshape(samples, channels).astype(float) - baselines) / gains
    if (channels, samples) == (300, 1):
        x, layout, channel = physical[0], "sample_channels", "legacy_red"
    elif samples == 300 and channels == 3 and "PPG_R" in names:
        channel = "PPG_R"
        x, layout = physical[:, names.index(channel)], "rgb_channels"
    else:
        raise ValueError(f"Unsupported dimensions: {channels} x {samples}")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite physical waveform")
    return x, {"layout": layout, "channel": channel, "digital_constant": bool(np.ptp(digital) == 0),
               "physical_constant": bool(np.ptp(x) == 0)}


def handcrafted(x, fs=30):
    """Fixed generic SQI features; no labels, identity, or ECG enters here."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite [record,time] physical waveforms")
    result = []
    bandpass = butter(2, [0.5, 8], btype="bandpass", fs=fs, output="sos")
    for raw in x:
        centered = raw - raw.mean()
        scale = centered.std()
        if scale < 1e-12:
            result.append([1.] + [0.] * (len(FEATURE_NAMES) - 1))
            continue
        normalized = centered / scale
        filtered = sosfiltfilt(bandpass, normalized)
        frequencies, power = welch(normalized, fs=fs, nperseg=min(len(raw), 4 * fs), nfft=512)
        band = (frequencies >= 0.5) & (frequencies <= 3.5)
        psd = power[band]
        dominant = int(np.argmax(psd))
        total = max(float(power.sum()), 1e-12)
        prob = np.maximum(power / total, 1e-15)
        local = np.abs(frequencies - frequencies[band][dominant]) <= 0.15
        autocorr = np.correlate(filtered, filtered, mode="full")[len(raw)-1:]
        autocorr /= max(float(autocorr[0]), 1e-12)
        lags = np.arange(int(np.ceil(fs / 3.5)), min(len(raw), int(2 * fs) + 1))
        lag = int(lags[np.argmax(autocorr[lags])])
        hist = np.histogram(normalized, bins=20)[0].astype(float) / len(raw)
        hist = hist[hist > 0]
        # This fixed peak statistic is a feature, not a reference/quality gate.
        peaks = find_peaks(filtered, distance=max(1, round(fs / 3.5)), prominence=0.2)[0]
        ibi = np.diff(peaks) / fs
        result.append([
            0., float(skew(normalized)), float(kurtosis(normalized)), float(np.max(np.abs(normalized))),
            float(np.std(np.diff(normalized))), float(np.mean(normalized[1:] * normalized[:-1] < 0)),
            float(np.mean(np.diff(raw) == 0)), float(np.mean((raw == raw.min()) | (raw == raw.max()))),
            float(-(hist * np.log(hist)).sum() / np.log(20)), float(psd.sum() / total),
            float(power[local].sum() / total), float(-(prob * np.log(prob)).sum() / np.log(len(prob))),
            float(frequencies[band][dominant]), float(autocorr[lag]), lag / fs,
            float(psd.max() / max(float(np.median(psd)), 1e-12)), float(len(peaks)),
            float(ibi.std() / ibi.mean()) if len(ibi) else 0., float(np.median(ibi)) if len(ibi) else 0.,
            float(np.maximum(filtered, 0).sum() / max(float(np.abs(filtered).sum()), 1e-12)),
        ])
    features = np.asarray(result, dtype=np.float32)
    if features.shape != (len(x), len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise ValueError("Invalid handcrafted features")
    return features


def prepare(raw, output):
    import wfdb  # Available in the project venv; model inference uses the benchmark venv.
    output.mkdir(parents=True, exist_ok=True)
    labels = sorted(read_csv(raw / "quality-hr-ann.csv"), key=lambda r: r["ID"])
    info = {r["ID"]: r for r in read_csv(raw / "subject-info.csv")}
    if len(labels) != 3888 or len({r["ID"] for r in labels}) != 3888:
        raise ValueError("Expected the complete v2.0 quality table")
    waveforms, rows, source_hashes, layouts = [], [], {}, []
    maximum_difference = 0.
    for label in labels:
        rid = label["ID"]
        x, layout = decode_ppg(raw, rid)
        independent = wfdb.rdrecord(str(raw / rid / f"{rid}_PPG")).p_signal
        independent = independent[0] if layout["layout"] == "sample_channels" else independent[:, 0]
        difference = float(np.max(np.abs(x - independent)))
        maximum_difference = max(maximum_difference, difference)
        if difference > 1e-10:
            raise ValueError(f"WFDB decoder mismatch: {rid}")
        waveforms.append(x)
        row = {"record_id": rid, "subject": rid[:3], "quality": int(label["Quality"]),
               "hr_reference": float(label["HR"]), "site": int(info[rid]["Ear/finger"]),
               "motion": info[rid]["Motion"], **layout}
        rows.append(row)
        layouts.append({"record_id": rid, "quality": row["quality"], **layout})
        for ext in ("dat", "hea"):
            relative = f"{rid}/{rid}_PPG.{ext}"
            source_hashes[relative] = sha(raw / relative)
    x = np.asarray(waveforms, dtype=np.float64)
    np.save(output / "x30.npy", x)
    np.save(output / "x64.npy", resample_poly(x, 32, 15, axis=1).astype(np.float32))
    np.save(output / "x125.npy", resample_poly(x, 25, 6, axis=1).astype(np.float32))
    np.save(output / "hand_features.npy", handcrafted(x))
    save_csv(output / "rows.csv", rows)
    for name in ("quality-hr-ann.csv", "subject-info.csv"):
        source_hashes[name] = sha(raw / name)
    published = {}
    for line in (raw / "SHA256SUMS.txt").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        published[relative.lstrip("*")] = digest
    for relative, digest in source_hashes.items():
        if published.get(relative) != digest:
            raise ValueError(f"Publisher checksum mismatch: {relative}")
    legacy = [r for r in layouts if r["layout"] == "sample_channels"]
    old = ROOT / "datasets/ppg_bench_raw/butppg"
    audit = {"full_records": len(rows), "layouts": dict(Counter(r["layout"] for r in rows)),
             "legacy_total": len(legacy), "legacy_quality_good": sum(r["quality"] for r in legacy),
             "legacy_digital_constant": sum(r["digital_constant"] for r in legacy),
             "legacy_physical_constant": sum(r["physical_constant"] for r in legacy),
             "old_good_subset_present": sum((old/r["record_id"]/f"{r['record_id']}_PPG.dat").is_file() for r in rows if r["quality"]),
             "old_good_rgb_green_selected": sum(r["quality"] == 1 and r["layout"] == "rgb_channels" for r in rows),
             "old_upstream_issue": "sample_channels decoded as constant raw ADC values; RGB channels selected green while quality annotations used red",
             "old_files_modified": False, "wfdb_independent_checked_records": len(rows),
             "wfdb_maximum_absolute_difference": maximum_difference, "wfdb_version": wfdb.__version__}
    save_json(output / "legacy_input_audit.json", audit)
    manifest = {"source": SOURCE, "zip_url": ZIP_URL, "archive_sha256": sha(ARCHIVE),
                "records": len(rows), "subjects": sorted({r["subject"] for r in rows}),
                "quality_counts": dict(Counter(str(r["quality"]) for r in rows)),
                "channel": "physical red; early sample-channel layout transposed; no additional sign inversion",
                "reference_scope": "Expert HR-readability consensus; smartphone finger/ear; not wrist morphology quality",
                "source_files_sha256": source_hashes, "preparation_code_sha256": sha(Path(__file__)),
                "publisher_checksum_file_sha256": sha(raw / "SHA256SUMS.txt"), "publisher_checksums_verified": len(source_hashes),
                "files_sha256": {name: sha(output / name) for name in ("rows.csv", "x30.npy", "x64.npy", "x125.npy", "hand_features.npy")},
                "hand_feature_names": FEATURE_NAMES, "legacy_input_audit": audit}
    save_json(output / "data_manifest.json", manifest)
    print(json.dumps({"prepared": len(rows), "classes": manifest["quality_counts"], "legacy": audit}), flush=True)


def make_jobs(subjects):
    subjects = np.asarray(sorted(subjects))
    jobs = []
    for fold, (available, test) in enumerate(KFold(5, shuffle=True, random_state=20261104).split(subjects)):
        for seed in SEEDS:
            order = np.random.default_rng(seed + 1000 * fold).permutation(subjects[available])
            jobs.append({"id": f"fold{fold}-seed{seed}", "fold": fold, "seed": seed,
                         "train": order[8:].tolist(), "validation": order[:8].tolist(), "test": subjects[test].tolist()})
    return jobs


def register(output):
    manifest = json.loads((output / "data_manifest.json").read_text())
    protocol = {"version": 1, "dataset": "BUT-PPG-v2-complete-red-physical", "target": "quality_good",
                "data_manifest_sha256": sha(output / "data_manifest.json"), "code_sha256": sha(Path(__file__)),
                "cnn_source_sha256": sha(ROOT / "scripts/ppg_hr_cnn.py"),
                "model_config_sha256": sha(ROOT / "watch_lsm/configs/ppg_bench_models.yaml"),
                "methods": METHODS, "jobs": make_jobs(manifest["subjects"]),
                "input": "Entire 10 seconds, same records; AnyPPG 125Hz native filtering; CNN 64Hz local zscore; handcrafted 30Hz physical red",
                "weighting": "Each participant has equal total weight; no class reweighting",
                "logistic": {"C": [0.01, 0.1, 1, 10, 100], "scaler": "train-only subject-weighted StandardScaler", "max_iter": 3000, "select": "validation subject-weighted pooled AUROC"},
                "cnn": {"architecture": "SmallHRCNN raw scalar used as logit", "parameters": 27065, "epochs": 20, "patience": 5, "batch_size": 256, "lr": 0.001, "loss": "subject-weighted BCE", "select": "validation subject-weighted pooled AUROC"},
                "decision_threshold": "Max validation subject-weighted balanced accuracy on fixed 0.05:0.05:0.95; ties nearest 0.5 then smaller",
                "primary_metric": "Mean across seeds of subject-weighted pooled out-of-fold AUROC; not mean of individual subject AUROCs",
                "secondary_metrics": ["balanced_accuracy", "good_F1", "sensitivity", "specificity", "coverage"],
                "uncertainty": "2000 paired participant bootstrap replicates; average each metric over seeds; exploratory no multiplicity correction",
                "no_refit": True, "no_pretraining_or_encoder_finetuning": True}
    path = output / "protocol.json"
    if path.is_file():
        prior = json.loads(path.read_text())
        if prior != json.loads(json.dumps(protocol)):
            raise ValueError("Registered protocol identity changed; use a new output directory")
    else:
        save_json(path, protocol)
    print(json.dumps({"registered": len(protocol["jobs"]), "fits": len(protocol["jobs"]) * len(METHODS)}), flush=True)


def weights(subjects):
    _, inv, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    return len(subjects) / (len(counts) * counts[inv])


def metrics(y, probability, threshold=0.5, sample_weight=None):
    y = np.asarray(y)
    p = np.asarray(probability)
    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("All classification outputs must be finite probabilities")
    prediction = p >= threshold
    tp, tn = float(w[(y == 1) & prediction].sum()), float(w[(y == 0) & ~prediction].sum())
    fp, fn = float(w[(y == 0) & prediction].sum()), float(w[(y == 1) & ~prediction].sum())
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    both = sensitivity is not None and specificity is not None
    return {"auroc": float(roc_auc_score(y, p, sample_weight=w)) if both else None,
            "balanced_accuracy": (sensitivity + specificity) / 2 if both else None,
            "good_F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "sensitivity": sensitivity, "specificity": specificity, "coverage": 1.}


def choose_threshold(y, p, w):
    candidates = [round(i / 20, 2) for i in range(1, 20)]
    scored = [(metrics(y, p, t, w)["balanced_accuracy"], -abs(t - .5), -t, t) for t in candidates]
    if any(v[0] is None for v in scored):
        raise ValueError("Validation fold must contain both classes")
    return max(scored)[-1]


def features(output, device):
    import torch
    import yaml
    from ppg_bench.models import get_port
    cfg_path = ROOT / "watch_lsm/configs/ppg_bench_models.yaml"
    cfg = next(c for c in yaml.safe_load(cfg_path.read_text()) if c["name"] == "AnyPPG")
    cfg["init"] = {k: str((cfg_path.parent / v).resolve()) for k, v in cfg["init"].items()}
    cfg["init"]["target_fs"] = cfg["fs"]
    checkpoint = Path(cfg["init"]["ckpt_path"])
    identity = {"data_manifest_sha256": sha(output / "data_manifest.json"), "weights_sha256": sha(checkpoint),
                "config_sha256": sha(cfg_path), "source_files_sha256": {
                    str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted((ROOT / "repos/ppg-bench/ppg_bench").rglob("*.py"))},
                "anyppg_source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted((ROOT / "repos/anyppg/load_anyppg").rglob("*.py"))},
                "precision": "FP32", "device": device, "code_sha256": sha(Path(__file__))}
    destination = output / "anyppg_features.npy"
    metadata = output / "feature_manifest.json"
    if metadata.exists():
        old = json.loads(metadata.read_text())
        if old["identity"] != identity or old["features_sha256"] != sha(destination):
            raise ValueError("Existing feature identity differs")
        return
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    x = np.load(output / "x125.npy")
    port = get_port(cfg)
    encoder = port.load_encoder().float().to(device).eval().requires_grad_(False)
    out = []
    with torch.inference_mode():
        for start in range(0, len(x), 64):
            batch = {"x": torch.from_numpy(x[start:start+64].copy())}
            out.append(encoder(port.prepare_batch(batch, cfg, device)).cpu().numpy())
    out = np.concatenate(out)
    if out.shape != (len(x), 512) or not np.isfinite(out).all():
        raise ValueError("Invalid frozen features")
    if sha(checkpoint) != identity["weights_sha256"]:
        raise ValueError("Foundation weights changed")
    np.save(destination, out)
    save_json(metadata, {"identity": identity, "features_sha256": sha(destination), "shape": list(out.shape)})
    print(json.dumps({"features": list(out.shape)}), flush=True)


def fit_logistic(x, y, tr, va, subjects):
    scaler = StandardScaler().fit(x[tr], sample_weight=weights(subjects[tr]))
    tx, vx = scaler.transform(x[tr]), scaler.transform(x[va])
    best = None
    candidates = []
    for c in (0.01, 0.1, 1, 10, 100):
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=3000, random_state=0)
        model.fit(tx, y[tr], sample_weight=weights(subjects[tr]))
        if int(model.n_iter_[0]) >= 3000:
            raise RuntimeError("Logistic regression did not converge")
        vp = model.predict_proba(vx)[:, 1]
        score = metrics(y[va], vp, sample_weight=weights(subjects[va]))["auroc"]
        candidates.append({"C": c, "validation_auroc": score})
        if best is None or score > best[0]:
            best = (score, model, vp)
    _, model, vp = best
    return model.predict_proba(scaler.transform(x))[:, 1], {
        "selected_C": model.C, "candidates": candidates,
        "threshold": choose_threshold(y[va], vp, weights(subjects[va])),
        "state": {"mean": scaler.mean_, "scale": scaler.scale_, "coefficient": model.coef_, "intercept": model.intercept_}}


def fit_classifier_cnn(x, y, tr, va, subjects, seed, device):
    import torch
    from torch import nn
    from scripts.ppg_hr_cnn import SmallHRCNN, normalize_windows, cpu_state_dict
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = SmallHRCNN().to(device)
    tensor = normalize_windows(torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(1))
    labels = torch.tensor(y, dtype=torch.float32, device=device)
    wt = torch.tensor(weights(subjects[tr]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(seed)
    best_score, best_epoch, best_state, history = -np.inf, 0, None, []
    for epoch in range(1, 21):
        model.train()
        order = rng.permutation(len(tr))
        loss_sum = 0.
        for begin in range(0, len(order), 256):
            local = order[begin:begin+256]
            idx = tr[local]
            optimizer.zero_grad(set_to_none=True)
            loss = (loss_fn(model(tensor[idx]), labels[idx]) * wt[local]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("CNN loss non-finite")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(local)
        model.eval()
        with torch.inference_mode():
            vp = torch.sigmoid(model(tensor[va])).cpu().numpy()
        score = metrics(y[va], vp, sample_weight=weights(subjects[va]))["auroc"]
        history.append({"epoch": epoch, "loss": loss_sum / len(tr), "validation_auroc": score})
        if score > best_score:
            best_score, best_epoch, best_state = score, epoch, cpu_state_dict(model)
        if epoch - best_epoch >= 5:
            break
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for begin in range(0, len(x), 256):
            predictions.append(torch.sigmoid(model(tensor[begin:begin+256])).cpu().numpy())
    prediction = np.concatenate(predictions)
    return prediction, {"best_epoch": best_epoch, "validation_auroc": best_score, "history": history,
                        "threshold": choose_threshold(y[va], prediction[va], weights(subjects[va])), "state": best_state,
                        "parameters": sum(p.numel() for p in model.parameters())}


def validate_protocol(output):
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["code_sha256"] != sha(Path(__file__)) or protocol["cnn_source_sha256"] != sha(ROOT / "scripts/ppg_hr_cnn.py"):
        raise ValueError("Registered source changed")
    if protocol["data_manifest_sha256"] != sha(output / "data_manifest.json"):
        raise ValueError("Registered data changed")
    manifest = json.loads((output / "data_manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        if sha(output / name) != digest:
            raise ValueError(f"Prepared file changed: {name}")
    return protocol


def run(output, device, selected_methods):
    import torch
    protocol = validate_protocol(output)
    rows = read_csv(output / "rows.csv")
    subjects = np.asarray([r["subject"] for r in rows])
    y = np.asarray([int(r["quality"]) for r in rows])
    data = {"HandSQI": np.load(output / "hand_features.npy"), "SmallCNN": np.load(output / "x64.npy")}
    if "AnyPPG" in selected_methods:
        feature_manifest = json.loads((output / "feature_manifest.json").read_text())
        if feature_manifest["features_sha256"] != sha(output / "anyppg_features.npy"):
            raise ValueError("Feature cache changed")
        data["AnyPPG"] = np.load(output / "anyppg_features.npy")
    for job in protocol["jobs"]:
        tr, va, te = (np.flatnonzero(np.isin(subjects, job[k])) for k in ("train", "validation", "test"))
        if any(set(job[a]) & set(job[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError("Participant leakage")
        for method in selected_methods:
            dest = output / "runs" / job["id"] / method
            dest.mkdir(parents=True, exist_ok=True)
            identity = {"protocol_sha256": sha(output / "protocol.json"), "job": job, "method": method}
            if method == "AnyPPG":
                identity["feature_manifest_sha256"] = sha(output / "feature_manifest.json")
            complete = dest / "complete.json"
            if complete.exists():
                metadata = json.loads(complete.read_text())
                if metadata["identity"] != identity or metadata["predictions_sha256"] != sha(dest / "predictions.npz"):
                    raise ValueError("Existing output changed")
                continue
            started = time.monotonic()
            if method == "SmallCNN":
                prediction, info = fit_classifier_cnn(data[method], y, tr, va, subjects, job["seed"], device)
                torch.save(info.pop("state"), dest / "model.pt")
            elif method in ("AnyPPG", "HandSQI"):
                prediction, info = fit_logistic(data[method], y, tr, va, subjects)
                np.savez(dest / "model.npz", **info.pop("state"))
            else:
                prior = float(np.average(y[tr], weights=weights(subjects[tr])))
                prediction = np.full(len(y), prior)
                info = {"prior": prior, "threshold": choose_threshold(y[va], prediction[va], weights(subjects[va]))}
            np.savez(dest / "predictions.npz", row=te, record_id=np.asarray([rows[i]["record_id"] for i in te]),
                     subject=subjects[te], reference=y[te], probability=prediction[te], decision=prediction[te] >= info["threshold"])
            score = metrics(y[te], prediction[te], info["threshold"], weights(subjects[te]))
            save_json(complete, {"identity": identity, "selection": info, "test_metrics": score,
                                 "predictions_sha256": sha(dest / "predictions.npz"), "elapsed_seconds": time.monotonic() - started,
                                 "train_records": len(tr), "validation_records": len(va), "test_records": len(te)})
            print(json.dumps({"job": job["id"], "method": method, "seconds": round(time.monotonic()-started, 2)}), flush=True)


def summarize(output, bootstrap=2000):
    protocol = validate_protocol(output)
    rows = read_csv(output / "rows.csv")
    subjects = np.asarray([r["subject"] for r in rows])
    y = np.asarray([int(r["quality"]) for r in rows])
    predictions, decisions = {}, {}
    per_subject, flat_predictions, seed_scores = [], [], []
    verified = 0
    for method in METHODS:
        predictions[method], decisions[method] = {}, {}
        for seed in SEEDS:
            p, decision, counts = np.full(len(rows), np.nan), np.zeros(len(rows), bool), np.zeros(len(rows), int)
            for job in (j for j in protocol["jobs"] if j["seed"] == seed):
                directory = output / "runs" / job["id"] / method
                meta = json.loads((directory / "complete.json").read_text())
                if meta["identity"]["protocol_sha256"] != sha(output / "protocol.json") or meta["predictions_sha256"] != sha(directory / "predictions.npz"):
                    raise ValueError("Result digest mismatch")
                with np.load(directory / "predictions.npz") as z:
                    idx = z["row"]
                    expected = np.flatnonzero(np.isin(subjects, job["test"]))
                    if not np.array_equal(idx, expected) or not np.array_equal(z["reference"], y[idx]) or not np.array_equal(z["subject"], subjects[idx]):
                        raise ValueError("Prediction identity mismatch")
                    if not np.array_equal(z["record_id"], [rows[i]["record_id"] for i in idx]):
                        raise ValueError("Record identity mismatch")
                    p[idx], decision[idx] = z["probability"], z["decision"]
                    if not np.array_equal(decision[idx], p[idx] >= meta["selection"]["threshold"]):
                        raise ValueError("Decision threshold mismatch")
                    observed = metrics(y[idx], p[idx], meta["selection"]["threshold"], weights(subjects[idx]))
                    if any(not np.isclose(observed[k], meta["test_metrics"][k]) for k in observed):
                        raise ValueError("Recomputed metric mismatch")
                    counts[idx] += 1
                verified += 1
            if not np.all(counts == 1):
                raise ValueError("Incomplete or overlapping held-out coverage")
            predictions[method][seed], decisions[method][seed] = p, decision
            scores = metrics(y, p, sample_weight=weights(subjects))
            classified = metrics(y, decision.astype(float), sample_weight=weights(subjects))
            scores.update({k: v for k, v in classified.items() if k != "auroc"})
            seed_scores.append({"method": method, "seed": seed, **scores})
            for subject in np.unique(subjects):
                mask = subjects == subject
                score = metrics(y[mask], p[mask])
                classified = metrics(y[mask], decision[mask].astype(float))
                score.update({k: v for k, v in classified.items() if k != "auroc"})
                per_subject.append({"method": method, "seed": seed, "subject": subject,
                                    "records": int(mask.sum()), "good": int(y[mask].sum()), **score})
            flat_predictions.extend({"method": method, "seed": seed, "record_id": rows[i]["record_id"],
                                     "subject": subjects[i], "reference": int(y[i]), "probability": float(p[i]),
                                     "decision": int(decision[i])} for i in range(len(rows)))
    # Resample people, preserving all their records and per-fold decision rules.
    people, inv = np.unique(subjects, return_inverse=True)
    base = weights(subjects)
    rng = np.random.default_rng(20261104)
    replicates = {method: {key: [] for key in ("auroc", "balanced_accuracy", "good_F1")} for method in METHODS}
    for _ in range(bootstrap):
        multiplicity = np.bincount(rng.integers(len(people), size=len(people)), minlength=len(people))
        w = base * multiplicity[inv]
        for method in METHODS:
            values = {key: [] for key in replicates[method]}
            for seed in SEEDS:
                scores = metrics(y, predictions[method][seed], sample_weight=w)
                classified = metrics(y, decisions[method][seed].astype(float), sample_weight=w)
                values["auroc"].append(scores["auroc"])
                for metric in ("balanced_accuracy", "good_F1"):
                    values[metric].append(classified[metric])
            for metric in replicates[method]:
                replicates[method][metric].append(float(np.mean(values[metric])))
    summary, pairs = [], []
    for method in METHODS:
        for metric in replicates[method]:
            estimate = float(np.mean([s[metric] for s in seed_scores if s["method"] == method]))
            lo, hi = np.percentile(replicates[method][metric], [2.5, 97.5])
            summary.append({"dataset": "BUT-PPG", "task": "HR readability SQI", "method": method, "metric": metric,
                            "estimate": estimate, "ci95_low": float(lo), "ci95_high": float(hi), "subjects": len(people),
                            "records": len(rows), "coverage": 1., "seeds": len(SEEDS)})
    for comparator in METHODS[1:]:
        for metric in replicates["AnyPPG"]:
            difference = np.asarray(replicates["AnyPPG"][metric]) - replicates[comparator][metric]
            lo, hi = np.percentile(difference, [2.5, 97.5])
            estimates = {s["method"]: s["estimate"] for s in summary if s["metric"] == metric}
            pairs.append({"model": "AnyPPG", "comparator": comparator, "metric": metric,
                          "difference": estimates["AnyPPG"] - estimates[comparator], "ci95_low": float(lo), "ci95_high": float(hi)})
    for name, content in (("summary.csv", summary), ("paired.csv", pairs), ("seed_scores.csv", seed_scores),
                          ("per_subject.csv", per_subject), ("predictions.csv", flat_predictions)):
        save_csv(output / name, content)
    save_json(output / "audit.json", {"status": "complete", "verified_results": verified, "expected_results": 60,
                                      "complete_held_out_predictions_per_seed_per_method": len(rows), "participants": len(people),
                                      "bootstrap_replicates": bootstrap, "protocol_sha256": sha(output / "protocol.json"),
                                      "summary_files_sha256": {n: sha(output/n) for n in ("summary.csv", "paired.csv", "seed_scores.csv", "per_subject.csv", "predictions.csv")}})
    print(json.dumps({"audit": "complete", "results": verified, "summary": summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("download", "prepare", "register", "features", "run", "summarize"), required=True)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    if args.stage == "download": download(args.raw)
    elif args.stage == "prepare": prepare(args.raw, args.output)
    elif args.stage == "register": register(args.output)
    elif args.stage == "features": features(args.output, args.device)
    elif args.stage == "run": run(args.output, args.device, args.methods)
    elif args.stage == "summarize": summarize(args.output)


if __name__ == "__main__":
    main()
