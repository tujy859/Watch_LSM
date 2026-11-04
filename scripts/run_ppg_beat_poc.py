"""Frozen AnyPPG event head, small CNN, and classical wrist beat/PRV PoC.

Targets are ECG-referenced cardiac events, not annotated PPG systolic peaks.
The fixed study uses DaLiA's manually corrected reference for fitting; WildPPG
is an exploratory external test with separately audited automatic ECG labels.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import find_peaks, resample_poly
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ppg_beat_models import (AnyPPGEventHead, SmallEventCNN, anyppg_input,
                             frozen_stages, soft_events, parameter_counts, THRESHOLDS)
from ppg_beat_metrics import match_events, elgendi_peaks, select_global_pat


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_reference(root, dataset, subject):
    directory = root / dataset / subject
    with np.load(directory / "beats.npz") as z:
        beats = {k: z[k] for k in z.files}
    x = np.load(directory / "ppg_native.npy", mmap_mode="r")
    return np.asarray(x, np.float32), float(beats["ppg_fs"]), beats["r_peak_times_s"]


def resample(x, fs, target):
    from math import gcd
    if fs != int(fs):
        raise ValueError("Only integer native rates are supported")
    common = gcd(int(fs), target)
    return resample_poly(x, target // common, int(fs) // common).astype(np.float32)


def load_encoder(device):
    sys.path.insert(0, str(ROOT / "repos/ppg-bench"))
    from ppg_bench.models import load_models, get_port
    config = next(c for c in load_models(ROOT / "watch_lsm/configs/ppg_bench_models.yaml")
                  if c["name"] == "AnyPPG")
    encoder = get_port(config).load_encoder().float().to(device).eval()
    encoder.requires_grad_(False)
    return encoder


def make_protocol(reference):
    phase1 = json.loads((ROOT / ".local/ppg-hr-poc/protocol.json").read_text())
    jobs = [j for j in phase1["jobs"] if j["budget"] == "full" and j["seed"] == 42
            and j["source"] == "Dalia"]
    refs = {p.relative_to(reference).as_posix(): sha256(p)
            for p in sorted(reference.glob("*/*/manifest.json"))}
    if len(refs) != 31 or len(jobs) != 6:
        raise ValueError("Require all 31 reference manifests and six fixed source-only jobs")
    subjects = {d: sorted(p.name for p in (reference / d).iterdir() if (p / "manifest.json").is_file())
                for d in ("Dalia", "WildPPG")}
    observed_test = []
    for job in jobs:
        train, val, test = (set(job[key]) for key in ("train", "validation", "test"))
        if not train or not val or not test or train & val or train | val != set(subjects["Dalia"]) - (test if job["scenario"] == "within" else set()):
            raise ValueError("Wrong source split or training/validation overlap")
        if job["scenario"] == "within":
            if (train | val) & test:
                raise ValueError("Held-out subject leakage")
            observed_test.extend(job["test"])
        elif test != set(subjects["WildPPG"]):
            raise ValueError("Incomplete external test")
    if sorted(observed_test) != subjects["Dalia"]:
        raise ValueError("Outer test people must appear exactly once")
    for relative in refs:
        path = reference / relative
        manifest = json.loads(path.read_text())
        for name, expected in manifest["outputs_sha256"].items():
            if sha256(path.parent / name) != expected:
                raise ValueError(f"Reference/PPG cache modified: {relative}/{name}")
    scripts = ["run_ppg_beat_poc.py", "ppg_beat_models.py", "ppg_beat_metrics.py"]
    return {
        "version": "cardiac-event-poc-v1", "jobs": jobs,
        "declared_before_model_results": True, "models": ["AnyPPG", "SmallCNN", "Elgendi"],
        "target": "ECG-referenced cardiac events from PPG, not PPG systolic-peak ground truth",
        "reference_manifests": refs,
        "reference_policy": "DaLiA manually corrected R peaks for training; WildPPG external test only",
        "windows": "8 s input, center 6 s scored; 6 s stride; no synthetic joins",
        "sampling": "Up to 256 uniformly spaced frames per train person; every other cached frame for validation (up to 128); full test duration. Frames containing duplicate ECG annotation timestamps are excluded before sampling.",
        "ecg_target": {"gaussian_sigma_s": .04, "output_fs": 64, "positive_weight": 5.},
        "architecture": "AnyPPG six frozen temporal stages + 8-channel per-stage projections + small decoder; dense small CNN",
        "parameters": parameter_counts(),
        "fit": {"seed": 42, "epochs": 20, "patience": 5, "batch": 64, "adam_lr": .001,
                "selection": "validation person-mean event F1 at 100 ms, epoch and threshold; no refit",
                "thresholds": list(THRESHOLDS), "nms_distance_s": .28, "precision": "FP32"},
        "traditional": "Elgendi-style PPG peaks; source-training-only fixed global PAT, 0 to 0.6 s step1/64; no test alignment",
        "metrics": "F1 at 50/100/150ms; adjacent matched RR errors; continuous 300s RR SDNN/RMSSD agreement, 60s exploratory",
        "limitation": "No arrhythmia annotations: RR, not NN; PRV/HRV physiological differences remain. Single training seed, exploratory subject bootstrap.",
        "code_sha256": {name: sha256(ROOT / "scripts" / name) for name in scripts},
        "weight_sha256": sha256(ROOT / "models/weights/anyppg/anyppg_ckpt.pth"),
        "encoder_source_sha256": sha256(ROOT / "repos/anyppg/load_anyppg/resnet1d.py"),
        "phase1_protocol_sha256": sha256(ROOT / ".local/ppg-hr-poc/protocol.json"),
    }


def register(root, reference):
    current = make_protocol(reference)
    path = root / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != current:
        raise ValueError("Registered protocol, reference or code changed; use a new output directory")
    write_json(path, current)
    return current


def prepare_frames(root, reference, encoder, device):
    directory = root / "frames"
    directory.mkdir(parents=True, exist_ok=True)
    done = directory / "complete.json"
    protocol_hash = sha256(root / "protocol.json")
    if done.exists():
        metadata = json.loads(done.read_text())
        if metadata["protocol_sha256"] != protocol_hash or any(
                sha256(directory / k) != v for k, v in metadata["sha256"].items()):
            raise ValueError("Frame cache changed")
        return
    subjects, starts, xs64, xs125, targets = [], [], [], [], []
    for subject in sorted(p.name for p in (reference / "Dalia").iterdir() if p.is_dir()):
        x, fs, r = load_reference(reference, "Dalia", subject)
        x64, x125 = resample(x, fs, 64), resample(x, fs, 125)
        full_starts = 6. * np.arange((len(x64) - 512) // 384 + 1)
        with np.load(reference / "Dalia" / subject / "beats.npz") as b:
            duplicate_times = b["duplicate_event_times_s"] if "duplicate_event_times_s" in b else np.array([])
        for duplicate in duplicate_times:
            full_starts = full_starts[~((duplicate >= full_starts) & (duplicate < full_starts + 8))]
        selected = np.unique(np.linspace(0, len(full_starts) - 1, min(256, len(full_starts))).astype(int))
        selected_starts = full_starts[selected]
        subjects += [subject] * len(selected)
        starts.extend(selected_starts)
        xs64.extend([x64[int(s * 64):int(s * 64) + 512] for s in selected_starts])
        xs125.extend([x125[int(s * 125):int(s * 125) + 1000] for s in selected_starts])
        targets.extend(soft_events(r, selected_starts))
    x64, x125 = np.asarray(xs64, np.float32), np.asarray(xs125, np.float32)
    np.savez(directory / "rows.npz", subject=np.asarray(subjects), start_s=np.asarray(starts),
             target=np.asarray(targets, np.float32))
    np.save(directory / "x64.npy", x64)
    np.save(directory / "x125.npy", x125)
    # Frozen features are extracted once. FP32 caches preserve the forward path.
    outputs = None
    for start in range(0, len(x125), 64):
        value = torch.from_numpy(anyppg_input(x125[start:start + 64])).to(device).unsqueeze(1)
        stages = frozen_stages(encoder, value)
        if outputs is None:
            outputs = [np.lib.format.open_memmap(directory / f"stage{i}.npy", mode="w+", dtype=np.float32,
                        shape=(len(x125), *z.shape[1:])) for i, z in enumerate(stages)]
        for target, value in zip(outputs, stages):
            target[start:start + len(value)] = value.cpu().numpy()
    for out in outputs:
        if not np.isfinite(out).all():
            raise ValueError("Frozen stage cache contains nonfinite values")
        out.flush()
    write_json(done, {"protocol_sha256": protocol_hash, "windows": len(x64),
                     "sha256": {p.name: sha256(p) for p in directory.iterdir() if p.suffix in (".npy", ".npz")}})
    print(f"Frozen temporal cache complete: {len(x64)} frames", flush=True)


def probability_peaks(probability, threshold, offset=0.):
    peaks = find_peaks(np.asarray(probability), height=threshold, distance=int(round(.28 * 64)))[0]
    return peaks / 64. + offset


def frame_f1(probabilities, rows, selected, reference, threshold):
    scores = []
    for subject in np.unique(rows["subject"][selected]):
        _, _, r = load_reference(reference, "Dalia", subject)
        counts = np.zeros(3)
        for index, prob in zip(selected, probabilities):
            if rows["subject"][index] != subject:
                continue
            start = rows["start_s"][index]
            ref = r[(r >= start + 1) & (r < start + 7)]
            pred = probability_peaks(prob, threshold, start)
            pred = pred[(pred >= start + 1) & (pred < start + 7)]
            matched = match_events(ref, pred, tolerance_s=.1)
            counts += [matched["tp"], matched["fp"], matched["fn"]]
        denominator = 2 * counts[0] + counts[1] + counts[2]
        scores.append(2 * counts[0] / denominator if denominator else 0.)
    return float(np.mean(scores))


def fit_model(root, reference, job, method, device):
    directory = root / "runs" / job["id"] / method
    directory.mkdir(parents=True, exist_ok=True)
    fit_path = directory / "fit.json"
    model = AnyPPGEventHead() if method == "AnyPPG" else SmallEventCNN()
    model.to(device)
    identity = {"protocol_sha256": sha256(root / "protocol.json"), "job": job, "method": method}
    if fit_path.exists():
        fit = json.loads(fit_path.read_text())
        if fit["identity"] != identity or sha256(directory / "head.pt") != fit["head_sha256"]:
            raise ValueError("Completed fitted head changed")
        model.load_state_dict(torch.load(directory / "head.pt", map_location=device, weights_only=True))
        return model.eval(), fit
    torch.manual_seed(job["seed"])
    model = (AnyPPGEventHead() if method == "AnyPPG" else SmallEventCNN()).to(device)
    with np.load(root / "frames/rows.npz") as data:
        rows = {key: data[key] for key in data.files}
    train = np.flatnonzero(np.isin(rows["subject"], job["train"]))
    validation = np.concatenate([np.flatnonzero(rows["subject"] == s)[::2] for s in job["validation"]])
    values = ([np.load(root / f"frames/stage{i}.npy", mmap_mode="r") for i in range(6)]
              if method == "AnyPPG" else [np.load(root / "frames/x64.npy", mmap_mode="r")])
    def batch(indices):
        result = [torch.tensor(np.asarray(v[indices]), dtype=torch.float32, device=device) for v in values]
        return result if method == "AnyPPG" else result[0].unsqueeze(1)
    n_by_subject = {s: int(np.count_nonzero(rows["subject"][train] == s)) for s in job["train"]}
    train_weights = np.asarray([len(train) / (len(n_by_subject) * n_by_subject[rows["subject"][i]]) for i in train])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    best, best_state, best_threshold, best_epoch, stale = -1., None, None, None, 0
    history, start_time = [], time.monotonic()
    for epoch in range(1, 21):
        model.train()
        permutation = np.random.default_rng(job["seed"] + epoch).permutation(len(train))
        total = 0.
        for offset in range(0, len(train), 64):
            pos = permutation[offset:offset + 64]
            indices = train[pos]
            y = torch.tensor(rows["target"][indices, 64:448], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch(indices))[:, 64:448]
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(5., device=device),
                                                      reduction="none").mean(dim=1)
            loss = (loss * torch.tensor(train_weights[pos], dtype=torch.float32, device=device)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite event loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        probs = []
        with torch.no_grad():
            for offset in range(0, len(validation), 64):
                probs.extend(model(batch(validation[offset:offset + 64])).sigmoid().cpu().numpy())
        scores = [frame_f1(probs, rows, validation, reference, threshold) for threshold in THRESHOLDS]
        chosen = int(np.argmax(scores))
        record = {"epoch": epoch, "loss": total / len(train), "validation_f1": scores[chosen],
                  "threshold": THRESHOLDS[chosen], "threshold_scores": dict(zip(map(str, THRESHOLDS), scores))}
        history.append(record)
        if scores[chosen] > best:
            best, best_threshold, best_epoch, stale = scores[chosen], THRESHOLDS[chosen], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        print(f"{job['id']} {method} epoch{epoch} validationF1={scores[chosen]:.4f}", flush=True)
        if stale >= 5:
            break
    torch.save(best_state, directory / "head.pt")
    model.load_state_dict(best_state)
    fit = {"identity": identity, "best_epoch": best_epoch, "threshold": best_threshold,
           "validation_f1": best, "history": history, "train_frames": len(train),
           "validation_frames": len(validation), "elapsed_seconds": time.monotonic() - start_time,
           "head_sha256": sha256(directory / "head.pt")}
    write_json(fit_path, fit)
    return model.eval(), fit


@torch.no_grad()
def infer_subject(root, reference, job, subject, method, model, fit, encoder, device):
    directory = root / "runs" / job["id"] / method
    path, meta = directory / f"{subject}.npz", directory / f"{subject}.json"
    identity = {"fit_sha256": sha256(directory / "fit.json"), "subject": subject, "target": job["target"]}
    if meta.exists():
        previous = json.loads(meta.read_text())
        if previous["identity"] != identity or previous["sha256"] != sha256(path):
            raise ValueError("Prediction identity changed")
        return
    x, fs, _ = load_reference(reference, job["target"], subject)
    target_fs = 125 if method == "AnyPPG" else 64
    x = resample(x, fs, target_fs)
    starts = 6 * np.arange((len(x) - 8 * target_fs) // (6 * target_fs) + 1)
    output = []
    for offset in range(0, len(starts), 64):
        frames = np.asarray([x[int(s * target_fs):int((s + 8) * target_fs)] for s in starts[offset:offset + 64]])
        if method == "AnyPPG":
            inp = torch.tensor(anyppg_input(frames), device=device).unsqueeze(1)
            prob = model(frozen_stages(encoder, inp)).sigmoid()
        else:
            prob = model(torch.tensor(frames, device=device).unsqueeze(1)).sigmoid()
        output.append(prob[:, 64:448].cpu().numpy().reshape(-1))
    probability = np.concatenate(output)
    if not np.isfinite(probability).all():
        raise FloatingPointError("Nonfinite event probability")
    predicted = probability_peaks(probability, fit["threshold"], 1.)
    np.savez_compressed(path, event_times_s=predicted, start_s=1., end_s=1. + len(probability) / 64.,
                        probability=probability.astype(np.float32), probability_fs=64)
    write_json(meta, {"identity": identity, "sha256": sha256(path), "events": len(predicted),
                      "duration_s": len(probability) / 64.})
    print(f"Prediction {job['id']} {method} {subject} complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / ".local/ppg-phase2/beats")
    parser.add_argument("--reference", type=Path, default=ROOT / ".local/ppg-phase2/reference-v3")
    parser.add_argument("--stage", choices=("register", "prepare", "learned", "traditional"), required=True)
    parser.add_argument("--method", choices=("AnyPPG", "SmallCNN", "all"), default="all")
    parser.add_argument("--job")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    protocol = register(args.root, args.reference)
    if args.stage == "register":
        return
    encoder = load_encoder(args.device) if args.stage == "prepare" or (args.stage == "learned" and args.method != "SmallCNN") else None
    if args.stage == "prepare":
        prepare_frames(args.root, args.reference, encoder, args.device)
    elif args.stage == "learned":
        for job in protocol["jobs"]:
            if args.job and job["id"] != args.job:
                continue
            for method in (["AnyPPG", "SmallCNN"] if args.method == "all" else [args.method]):
                model, fit = fit_model(args.root, args.reference, job, method, args.device)
                for subject in job["test"]:
                    infer_subject(args.root, args.reference, job, subject, method, model, fit, encoder, args.device)
                del model
                torch.cuda.empty_cache()
    else:
        run_traditional(args.root, args.reference, protocol)


def run_traditional(root, reference, protocol):
    """The classical method has no fitted waveform parameters; PAT uses source train only."""
    with np.load(root / "frames/rows.npz") as z:
        rows = {k: z[k] for k in z.files}
    x64 = np.load(root / "frames/x64.npy", mmap_mode="r")
    calibration = []
    for subject in np.unique(rows["subject"]):
        _, _, r = load_reference(reference, "Dalia", subject)
        for index in np.flatnonzero(rows["subject"] == subject)[::8]:
            start = float(rows["start_s"][index])
            calibration.append({"subject": subject, "reference_s": r[(r >= start) & (r < start + 8)],
                                "ppg_s": elgendi_peaks(x64[index], 64) + start,
                                "start_s": start + 1, "end_s": start + 7})
    raw_predictions = {}
    for dataset in ("Dalia", "WildPPG"):
        for subject in sorted(p.name for p in (reference / dataset).iterdir() if p.is_dir()):
            x, fs, _ = load_reference(reference, dataset, subject)
            x = resample(x, fs, 64)
            # Fixed 60 s processing blocks with 2 s real context on each side.
            events = []
            for start in range(0, len(x), 60 * 64):
                stop = min(start + 60 * 64, len(x))
                lo, hi = max(0, start - 128), min(len(x), stop + 128)
                p = elgendi_peaks(x[lo:hi], 64) + lo / 64.
                events.extend(p[(p >= start / 64.) & (p < stop / 64.)])
            raw_predictions[(dataset, subject)] = (np.asarray(events), len(x) / 64.)
    for job in protocol["jobs"]:
        directory = root / "runs" / job["id"] / "Elgendi"
        directory.mkdir(parents=True, exist_ok=True)
        frames = [frame for frame in calibration if frame["subject"] in job["train"]]
        pat = select_global_pat(frames, job["train"])
        delay = float(pat["delay_s"])
        write_json(directory / "fit.json", {"identity": {"protocol_sha256": sha256(root / "protocol.json"),
                                                         "job": job, "method": "Elgendi"}, "pat": pat})
        for subject in job["test"]:
            p, duration = raw_predictions[(job["target"], subject)]
            end = 1. + 6 * (int((duration - 8) // 6) + 1)
            pred = p - delay
            pred = pred[(pred >= 1.) & (pred < end)]
            path = directory / f"{subject}.npz"
            np.savez_compressed(path, event_times_s=pred, start_s=1., end_s=end)
            write_json(directory / f"{subject}.json", {"identity": {"fit_sha256": sha256(directory / "fit.json"),
                         "subject": subject, "target": job["target"]}, "sha256": sha256(path), "events": len(pred)})
    print("Traditional event outputs complete", flush=True)


if __name__ == "__main__":
    main()
