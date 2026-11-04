"""Small, subject-isolated PoC for manually annotated wrist PPG onsets.

This is a separate experiment from the ECG-referenced diagnostic pilot. The
encoder is frozen; only a temporal task head or a small CNN is fitted. Missing
annotation regions are never training negatives, and recordings are not joined.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ppg_beat_models import (AnyPPGEventHead, SmallEventCNN, anyppg_input,
                             frozen_stages, soft_events, parameter_counts, THRESHOLDS)
from ppg_beat_metrics import match_events
from ppg_onset_baseline import detect_onsets, ONSET_BASELINE_PROTOCOL
from run_ppg_beat_poc import load_encoder, probability_peaks, resample, sha256, write_json

METHODS = ("AnyPPG", "SmallCNN", "MSPTDfast")


def array_digest(value):
    """Hash an array's C-order bytes, with its dtype and shape in the identity."""
    value = np.ascontiguousarray(value)
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def record_identity(record):
    return {"record_id": str(record["record_id"]), "subject": str(record["subject"]),
            "activity": str(record["activity"]), "fs": float(record["fs"]),
            "duration_s": float(record["duration_s"]),
            "arrays": {k: array_digest(record[k]) for k in
                       ("raw_ppg", "input_ppg", "onset_times_s", "annotated_intervals_s")}}


def record_digest(record):
    """Canonical JSON (sorted keys, compact separators) of record_identity."""
    value = json.dumps(record_identity(record), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_records(records):
    if not records:
        raise ValueError("No acquisition records")
    identities = [str(r["record_id"]) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate acquisition record identity")
    for r in records:
        raw, ppg, events = np.asarray(r["raw_ppg"]), np.asarray(r["input_ppg"]), np.asarray(r["onset_times_s"])
        intervals = np.asarray(r["annotated_intervals_s"])
        duration, fs = float(r["duration_s"]), float(r["fs"])
        if ppg.ndim != 1 or not len(ppg) or not np.isfinite(ppg).all() or fs <= 16 or fs != int(fs):
            raise ValueError("Invalid native waveform or sampling rate")
        if raw.shape != ppg.shape or not np.array_equal(raw[np.isfinite(raw)], ppg[np.isfinite(raw)]):
            raise ValueError("Finite native samples must remain unchanged in the input copy")
        if abs(len(ppg) / fs - duration) > 1 / fs + 1e-9:
            raise ValueError("Waveform duration and sample count disagree")
        if events.ndim != 1 or not np.isfinite(events).all() or (np.diff(events) <= 0).any():
            raise ValueError("Onsets must be finite and strictly increasing")
        if intervals.ndim != 2 or intervals.shape[1] != 2 or not np.isfinite(intervals).all():
            raise ValueError("Annotation intervals must have shape (N,2)")
        if len(intervals) and ((intervals[:, 0] < 0).any() or (intervals[:, 1] > duration + 1e-9).any()
                               or (intervals[:, 1] <= intervals[:, 0]).any()
                               or (intervals[1:, 0] < intervals[:-1, 1]).any()):
            raise ValueError("Invalid or overlapping annotation intervals")
        if len(events) and ((events < 0).any() or (events >= duration).any()):
            raise ValueError("Annotation event outside acquisition")
        covered = np.zeros(len(events), dtype=bool)
        for left, right in intervals:
            covered |= (events >= left) & (events < right)
            native_low, native_high = int(round(left * fs)), int(round(right * fs))
            if not np.isfinite(raw[native_low:native_high]).all():
                raise ValueError("A labeled interval crosses missing native samples")
        if not covered.all():
            raise ValueError("Valid onset label outside annotation coverage")


def make_jobs(subjects):
    subjects = sorted(map(str, subjects))
    if len(subjects) != 7 or len(set(subjects)) != 7:
        raise ValueError("Manual onset study requires seven distinct people")
    jobs = []
    for fold, test in enumerate(subjects):
        remaining = [s for s in subjects if s != test]
        validation = str(np.random.default_rng(42 + fold).choice(remaining))
        jobs.append({"id": f"EMBC21-fold{fold}-seed42", "fold": fold, "seed": 42,
                     "scenario": "within", "source": "EMBC21", "target": "EMBC21",
                     "train": [s for s in remaining if s != validation],
                     "validation": [validation], "test": [test]})
    return jobs


def full_frame_starts(record):
    """Native clock, 8-second frames every six seconds, no partial last frame."""
    count = max(0, int(np.floor((len(record["raw_ppg"]) / float(record["fs"]) - 8) / 6)) + 1)
    return 6. * np.arange(count)


def annotated_frame_starts(record):
    """Require the ENTIRE context to be inside one contiguous labeled region."""
    starts = full_frame_starts(record)
    valid = np.zeros(len(starts), dtype=bool)
    for left, right in np.asarray(record["annotated_intervals_s"]):
        valid |= (starts >= left) & (starts + 8 <= right)
    return starts[valid]


def sample_person_indices(rows, subjects, limit):
    """Uniform deterministic subsampling within each person's ordered records."""
    selected = []
    for subject in subjects:
        candidates = np.flatnonzero(rows["subject"] == subject)
        if not len(candidates):
            raise ValueError(f"No fully annotated training context for {subject}")
        positions = np.unique(np.linspace(0, len(candidates) - 1, min(limit, len(candidates))).astype(int))
        selected.extend(candidates[positions])
    return np.asarray(selected, dtype=int)


def make_protocol(records, data_root):
    validate_records(records)
    identities = {str(r["record_id"]): {**record_identity(r), "record_sha256": record_digest(r)} for r in records}
    scripts = ("run_ppg_onset_poc.py", "ppg_embc21_data.py", "ppg_onset_baseline.py",
               "ppg_beat_models.py", "ppg_beat_metrics.py", "run_ppg_beat_poc.py")
    return {
        "version": "manual-wrist-onset-poc-v1", "declared_before_model_results": True,
        "dataset": "EMBC21/Biagetti", "models": list(METHODS),
        "jobs": make_jobs({str(r["subject"]) for r in records}), "records": identities,
        "target": "Single-expert PPG diastolic onsets on the original PPG clock; no ECG-derived timing target",
        "input_polarity": "Neural models: negative reflected-light intensity = blood-volume convention. Traditional: intensity maxima. Fixed from acquisition convention, no test polarity search.",
        "missing_samples": "All methods use the adapter's input_ppg copy with only isolated native NaNs linearly filled for whole-record inference. Original raw_ppg is retained and hashed; missing sample positions remain excluded from labeled intervals. No complete training or PRV evaluation window crosses a missing sample.",
        "frames": "Continuous 8 s contexts, 6 s stride, 64 Hz center 6 s outputs; record time starts at zero; inference [1,1+6*N) seconds",
        "annotation_policy": "Training/validation entire 8 s context must fit within a single annotated interval. Test predictions cover each acquisition; score only annotation/inference intersections. Never concatenate recordings or gaps.",
        "sampling": {"train_frames_per_person": 256, "validation_frames_per_person": 128,
                     "selection": "Uniform deterministic indices across sorted record_id then start_s within each person; training and validation sampled independently. Full test acquisition inference."},
        "split": "7-fold LOSO; validation person sampled from other six with numpy.default_rng(42+fold); remaining five train; no refit",
        "fit": {"seed": 42, "epochs": 20, "patience": 5, "batch": 64, "adam_lr": .001,
                "gaussian_sigma_s": .04, "positive_weight": 5., "output_fs": 64,
                "nms_distance_s": .28, "precision": "FP32", "thresholds": list(THRESHOLDS),
                "selection": "Validation person-mean F1 at +/-50 ms; select epoch and threshold; first maximum wins",
                "loss_weighting": "Equal total weight per training person; BCE over center six seconds"},
        "architecture": "Frozen AnyPPG six temporal stages with learned 8-channel projections and small decoder; randomly initialized dense SmallCNN; reused fixed architectures without test-based redesign",
        "parameters": parameter_counts(), "traditional": ONSET_BASELINE_PROTOCOL,
        "metrics": "Primary event F1 +/-50 ms; sensitivity +/-25/100 ms. Consecutive matched PP errors and 60s PRV agreement on complete continuous annotated windows; no 5min claim if unavailable.",
        "limitations": "Seven people, single expert, annotated regions only, single training seed, no disease/NN annotations. PRV algorithm agreement is not ECG HRV equivalence.",
        "data_root": Path(data_root).relative_to(ROOT).as_posix() if Path(data_root).is_relative_to(ROOT) else str(data_root),
        "data_manifest_sha256": sha256(Path(data_root) / "manifest.json"),
        "code_sha256": {name: sha256(ROOT / "scripts" / name) for name in scripts},
        "weight_sha256": sha256(ROOT / "models/weights/anyppg/anyppg_ckpt.pth"),
        "encoder_source_sha256": sha256(ROOT / "repos/anyppg/load_anyppg/resnet1d.py"),
    }


def register(root, records, data_root):
    protocol = make_protocol(records, data_root)
    path = root / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Registered onset protocol, inputs or code changed; use a new output directory")
    else:
        write_json(path, protocol)
    return protocol


def prepare_frames(root, records, encoder, device):
    directory = root / "frames"
    directory.mkdir(parents=True, exist_ok=True)
    done = directory / "complete.json"
    protocol_hash = sha256(root / "protocol.json")
    if done.exists():
        meta = json.loads(done.read_text())
        if meta["protocol_sha256"] != protocol_hash or any(sha256(directory / k) != v for k, v in meta["sha256"].items()):
            raise ValueError("Onset frame cache changed")
        return
    subject, record_ids, starts, xs64, xs125, targets = [], [], [], [], [], []
    for record in sorted(records, key=lambda r: str(r["record_id"])):
        selected = annotated_frame_starts(record)
        if not len(selected):
            continue
        volume = -np.asarray(record["input_ppg"], np.float32)
        x64, x125 = resample(volume, record["fs"], 64), resample(volume, record["fs"], 125)
        subject.extend([str(record["subject"])] * len(selected))
        record_ids.extend([str(record["record_id"])] * len(selected))
        starts.extend(selected)
        xs64.extend([x64[int(s * 64):int(s * 64) + 512] for s in selected])
        xs125.extend([x125[int(s * 125):int(s * 125) + 1000] for s in selected])
        targets.extend(soft_events(record["onset_times_s"], selected))
    if not starts:
        raise ValueError("No fully annotated frame")
    x64, x125 = np.asarray(xs64, np.float32), np.asarray(xs125, np.float32)
    np.savez(directory / "rows.npz", subject=np.asarray(subject), record_id=np.asarray(record_ids),
             start_s=np.asarray(starts), target=np.asarray(targets, np.float32))
    np.save(directory / "x64.npy", x64)
    np.save(directory / "x125.npy", x125)
    outputs = None
    for start in range(0, len(x125), 64):
        inp = torch.from_numpy(anyppg_input(x125[start:start + 64])).unsqueeze(1).to(device)
        stages = frozen_stages(encoder, inp)
        if outputs is None:
            outputs = [np.lib.format.open_memmap(directory / f"stage{i}.npy", mode="w+", dtype=np.float32,
                       shape=(len(x125), *value.shape[1:])) for i, value in enumerate(stages)]
        for target, value in zip(outputs, stages):
            target[start:start + len(value)] = value.cpu().numpy()
    for output in outputs:
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite frozen features")
        output.flush()
    write_json(done, {"protocol_sha256": protocol_hash, "frames": len(starts),
                     "sha256": {p.name: sha256(p) for p in directory.iterdir() if p.suffix in (".npy", ".npz")}})
    print(f"Manual-onset frozen cache complete: {len(starts)} contexts", flush=True)


def frame_f1(probabilities, rows, indices, records_by_id, threshold):
    """Validation labels only; fixed 50 ms one-to-one matching on each core."""
    totals = {}
    for index, probability in zip(indices, probabilities):
        subject = str(rows["subject"][index])
        record = records_by_id[str(rows["record_id"][index])]
        start = float(rows["start_s"][index])
        onsets = np.asarray(record["onset_times_s"])
        reference = onsets[(onsets >= start + 1) & (onsets < start + 7)]
        predicted = probability_peaks(probability, threshold, start)
        predicted = predicted[(predicted >= start + 1) & (predicted < start + 7)]
        match = match_events(reference, predicted, .05)
        totals.setdefault(subject, np.zeros(3))[:] += [match["tp"], match["fp"], match["fn"]]
    scores = [2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0. for tp, fp, fn in totals.values()]
    return float(np.mean(scores))


def fit_model(root, records_by_id, job, method, device):
    directory = root / "runs" / job["id"] / method
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"protocol_sha256": sha256(root / "protocol.json"), "job": job, "method": method}
    torch.manual_seed(job["seed"])
    model = (AnyPPGEventHead() if method == "AnyPPG" else SmallEventCNN()).to(device)
    if (directory / "fit.json").exists():
        fit = json.loads((directory / "fit.json").read_text())
        if fit["identity"] != identity or sha256(directory / "head.pt") != fit["head_sha256"]:
            raise ValueError("Fitted onset head changed")
        model.load_state_dict(torch.load(directory / "head.pt", map_location=device, weights_only=True))
        return model.eval(), fit
    with np.load(root / "frames/rows.npz") as z:
        rows = {key: z[key] for key in z.files}
    train = sample_person_indices(rows, job["train"], 256)
    validation = sample_person_indices(rows, job["validation"], 128)
    values = ([np.load(root / f"frames/stage{i}.npy", mmap_mode="r") for i in range(6)] if method == "AnyPPG"
              else [np.load(root / "frames/x64.npy", mmap_mode="r")])
    def batch(indices):
        value = [torch.tensor(np.asarray(v[indices]), dtype=torch.float32, device=device) for v in values]
        return value if method == "AnyPPG" else value[0].unsqueeze(1)
    counts = {s: int(np.count_nonzero(rows["subject"][train] == s)) for s in job["train"]}
    weights = np.asarray([len(train) / (len(counts) * counts[rows["subject"][i]]) for i in train])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    best, state, threshold, best_epoch, stale = -1., None, None, None, 0
    history, started = [], time.monotonic()
    for epoch in range(1, 21):
        model.train()
        order = np.random.default_rng(job["seed"] + epoch).permutation(len(train))
        total = 0.
        for offset in range(0, len(train), 64):
            positions = order[offset:offset + 64]
            indices = train[positions]
            target = torch.tensor(rows["target"][indices, 64:448], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch(indices))[:, 64:448]
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(5., device=device), reduction="none").mean(1)
            loss = (loss * torch.tensor(weights[positions], dtype=torch.float32, device=device)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite onset loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        probabilities = []
        with torch.no_grad():
            for offset in range(0, len(validation), 64):
                probabilities.extend(model(batch(validation[offset:offset + 64])).sigmoid().cpu().numpy())
        scores = [frame_f1(probabilities, rows, validation, records_by_id, t) for t in THRESHOLDS]
        chosen = int(np.argmax(scores))
        history.append({"epoch": epoch, "loss": total / len(train), "validation_f1_50ms": scores[chosen],
                        "threshold": THRESHOLDS[chosen], "threshold_scores": dict(zip(map(str, THRESHOLDS), scores))})
        if scores[chosen] > best:
            best, threshold, best_epoch, stale = scores[chosen], THRESHOLDS[chosen], epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        print(f"{job['id']} {method} epoch{epoch} validationF1_50ms={scores[chosen]:.4f}", flush=True)
        if stale >= 5:
            break
    torch.save(state, directory / "head.pt")
    model.load_state_dict(state)
    fit = {"identity": identity, "best_epoch": best_epoch, "threshold": threshold,
           "validation_f1_50ms": best, "history": history, "head_sha256": sha256(directory / "head.pt"),
           "train_frames": len(train), "validation_frames": len(validation), "train_frames_by_person": counts,
           "train_frame_indices": train.tolist(), "validation_frame_indices": validation.tolist(),
           "elapsed_seconds": time.monotonic() - started}
    write_json(directory / "fit.json", fit)
    return model.eval(), fit


@torch.no_grad()
def infer_record(root, job, record, method, model, fit, encoder, device):
    directory = root / "runs" / job["id"] / method
    name = str(record["record_id"])
    path, meta = directory / f"{name}.npz", directory / f"{name}.json"
    identity = {"fit_sha256": sha256(directory / "fit.json"), "subject": str(record["subject"]),
                "record_id": name, "target": "EMBC21", "record_sha256": record_digest(record)}
    if meta.exists():
        previous = json.loads(meta.read_text())
        if previous["identity"] != identity or previous["sha256"] != sha256(path):
            raise ValueError("Onset prediction identity changed")
        return
    starts = full_frame_starts(record)
    if not len(starts):
        raise ValueError("Acquisition shorter than 8 seconds")
    end = 1. + 6 * len(starts)
    if method == "MSPTDfast":
        raw64 = resample(np.asarray(record["input_ppg"], np.float32), record["fs"], 64)
        events = detect_onsets(raw64, 64, polarity="raw_intensity")
        events = events[(events >= 1.) & (events < end)]
        payload = {}
    else:
        fs = 125 if method == "AnyPPG" else 64
        volume = resample(-np.asarray(record["input_ppg"], np.float32), record["fs"], fs)
        output = []
        for offset in range(0, len(starts), 64):
            frames = np.asarray([volume[int(s * fs):int((s + 8) * fs)] for s in starts[offset:offset + 64]])
            if method == "AnyPPG":
                inp = torch.from_numpy(anyppg_input(frames)).unsqueeze(1).to(device)
                probability = model(frozen_stages(encoder, inp)).sigmoid()
            else:
                probability = model(torch.from_numpy(frames).unsqueeze(1).to(device)).sigmoid()
            output.append(probability[:, 64:448].cpu().numpy().reshape(-1))
        probability = np.concatenate(output).astype(np.float32)
        if not np.isfinite(probability).all():
            raise FloatingPointError("Nonfinite onset probability")
        events = probability_peaks(probability, fit["threshold"], 1.)
        payload = {"probability": probability, "probability_fs": 64}
    np.savez_compressed(path, event_times_s=events, start_s=1., end_s=end, **payload)
    write_json(meta, {"identity": identity, "sha256": sha256(path), "events": len(events),
                      "inference_duration_s": end - 1., "native_duration_s": float(record["duration_s"]),
                      "annotation_mask_applied_at_scoring_only": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / ".local/ppg-phase2/onsets")
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/embc21")
    parser.add_argument("--stage", choices=("register", "prepare", "traditional", "learned", "all"), required=True)
    parser.add_argument("--method", choices=("AnyPPG", "SmallCNN", "all"), default="all")
    parser.add_argument("--job")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from ppg_embc21_data import load_records
    records = load_records(args.data_root)
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    protocol = register(args.root, records, args.data_root)
    if args.stage == "register":
        print(f"Registered {len(records)} acquisitions, seven person-isolated folds", flush=True)
        return
    encoder = load_encoder(args.device) if args.stage in ("prepare", "all") or (args.stage == "learned" and args.method != "SmallCNN") else None
    if args.stage in ("prepare", "all"):
        prepare_frames(args.root, records, encoder, args.device)
    by_id = {str(r["record_id"]): r for r in records}
    for job in protocol["jobs"]:
        if args.job and job["id"] != args.job:
            continue
        test_records = [r for r in records if str(r["subject"]) in job["test"]]
        if args.stage in ("traditional", "all"):
            directory = args.root / "runs" / job["id"] / "MSPTDfast"
            fit = {"identity": {"protocol_sha256": sha256(args.root / "protocol.json"), "job": job, "method": "MSPTDfast"},
                   "fit": "No parameters fitted; fixed algorithm and acquisition polarity", "traditional": ONSET_BASELINE_PROTOCOL}
            if (directory / "fit.json").exists() and json.loads((directory / "fit.json").read_text()) != fit:
                raise ValueError("Traditional method identity changed")
            if not (directory / "fit.json").exists():
                write_json(directory / "fit.json", fit)
            for record in test_records:
                infer_record(args.root, job, record, "MSPTDfast", None, fit, None, "cpu")
        if args.stage in ("learned", "all"):
            for method in (("AnyPPG", "SmallCNN") if args.method == "all" else (args.method,)):
                model, fit = fit_model(args.root, by_id, job, method, args.device)
                for record in test_records:
                    infer_record(args.root, job, record, method, model, fit, encoder, args.device)
                del model
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(f"Completed {job['id']} {method}: {len(test_records)} acquisitions", flush=True)


if __name__ == "__main__":
    main()
