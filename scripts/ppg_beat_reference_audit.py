"""Prepare ECG beat references and continuous variability windows for the PoC.

DaLiA supplies the authors' manually corrected ECG R peaks. WildPPG supplies
automatic ECG peaks only; agreement with WFDB XQRS is a quality proxy, never an
independent human truth. No PPG waveform or model prediction selects references.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".local/ppg-phase2/reference-v3"
AGREEMENT_TOLERANCE_S = 0.050
MIN_AGREEMENT_F1 = 0.95
RR_BOUNDS_S = (0.3, 2.0)
WINDOW_DURATIONS = (60, 300)
DALIA_AUDIT_WINDOWS = 15
XQRS_BLOCK_S = 300
XQRS_CONTEXT_S = 5


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate_peaks(peaks, duration_s):
    peaks = np.asarray(peaks, dtype=np.float64)
    if peaks.ndim != 1 or not np.isfinite(peaks).all():
        raise ValueError("Beat times must be a finite vector")
    if np.any(np.diff(peaks) <= 0) or np.any(peaks < 0) or np.any(peaks >= duration_s):
        raise ValueError("Beat times must be unique, ordered, and within the acquisition")
    return peaks


def canonical_reference(peaks, duration_s):
    """Preserve raw annotations, collapse exact duplicate event entries only.

    Duplicate timestamps cannot denote two physical events. Their locations are
    nevertheless flagged so every affected evaluation window remains excluded.
    Reversed/out-of-range timestamps are errors, not silently sorted or clipped.
    """
    raw = np.asarray(peaks, dtype=np.float64)
    if raw.ndim != 1 or not np.isfinite(raw).all() or np.any(np.diff(raw) < 0):
        raise ValueError("Raw beat annotations are nonfinite or reversed")
    if np.any(raw < 0) or np.any(raw >= duration_s):
        raise ValueError("Raw beat annotations extend beyond the acquisition")
    duplicate = raw[1:][np.diff(raw) == 0]
    unique = raw[np.r_[True, np.diff(raw) > 0]] if len(raw) else raw.copy()
    return unique, duplicate


def match_beats(reference, detected, tolerance_s=AGREEMENT_TOLERANCE_S):
    """Maximum-cardinality ordered one-to-one matching, with no learned lag."""
    reference, detected = np.asarray(reference), np.asarray(detected)
    if tolerance_s < 0:
        raise ValueError("Matching tolerance must be nonnegative")
    i = j = 0
    pairs = []
    while i < len(reference) and j < len(detected):
        delta = detected[j] - reference[i]
        if abs(delta) <= tolerance_s + 1e-12:
            pairs.append((i, j))
            i += 1
            j += 1
        elif delta < 0:
            j += 1
        else:
            i += 1
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def beat_agreement(reference, detected):
    pairs = match_beats(reference, detected)
    n = len(pairs)
    denominator = len(reference) + len(detected)
    timing = np.abs(np.asarray(reference)[pairs[:, 0]] - np.asarray(detected)[pairs[:, 1]]) * 1000
    return {
        "reference_beats": len(reference), "xqrs_beats": len(detected), "matched_beats": n,
        "agreement_f1": 2 * n / denominator if denominator else 0.0,
        "matched_abs_timing_mae_ms": float(timing.mean()) if n else None,
        "matched_abs_timing_p95_ms": float(np.quantile(timing, .95)) if n else None,
    }


def variability(peaks_s):
    """Unedited consecutive intervals; no normal-beat claim or gap bridging."""
    rr_ms = np.diff(np.asarray(peaks_s, dtype=np.float64)) * 1000
    return {
        "n_beats": len(peaks_s), "n_rr": len(rr_ms),
        "mean_rr_ms": float(rr_ms.mean()) if len(rr_ms) else None,
        "sdnn_ms": float(np.std(rr_ms, ddof=1)) if len(rr_ms) >= 2 else None,
        "rmssd_ms": float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) >= 2 else None,
        "hr_bpm": float(60000 / rr_ms.mean()) if len(rr_ms) and rr_ms.mean() > 0 else None,
        "rr_outside_bounds": int(np.count_nonzero((rr_ms < 1000 * RR_BOUNDS_S[0]) | (rr_ms > 1000 * RR_BOUNDS_S[1]))),
    }


def make_windows(dataset, subject, reference, duration_s, xqrs=None, duplicate_times=None):
    """Complete nonoverlapping windows; retain invalid rows with explicit flags."""
    reference = validate_peaks(reference, duration_s)
    if xqrs is not None:
        xqrs = validate_peaks(xqrs, duration_s)
    duplicate_times = np.asarray([] if duplicate_times is None else duplicate_times)
    rows = []
    for duration in WINDOW_DURATIONS:
        for start in np.arange(int(duration_s // duration), dtype=np.int64) * duration:
            end = start + duration
            lo, hi = np.searchsorted(reference, [start, end], side="left")
            selected = reference[lo:hi]
            stats = variability(selected)
            gap_ok = len(selected) >= 2 and selected[0] - start <= RR_BOUNDS_S[1] and end - selected[-1] <= RR_BOUNDS_S[1]
            physiologic = stats["n_rr"] >= 20 and stats["rr_outside_bounds"] == 0 and gap_ok
            agreement = None
            if xqrs is not None:
                a, b = np.searchsorted(xqrs, [start, end], side="left")
                agreement = beat_agreement(selected, xqrs[a:b])["agreement_f1"]
            duplicates = int(np.count_nonzero((duplicate_times >= start) & (duplicate_times < end)))
            accepted = bool(physiologic and not duplicates and (dataset == "Dalia" or (agreement is not None and agreement >= MIN_AGREEMENT_F1)))
            reasons = []
            if duplicates:
                reasons.append("duplicate_reference_annotation")
            if stats["n_rr"] < 20:
                reasons.append("fewer_than_20_rr")
            if stats["rr_outside_bounds"]:
                reasons.append("rr_outside_0.3_to_2.0_s")
            if not gap_ok:
                reasons.append("boundary_gap_over_2_s")
            if dataset == "WildPPG" and (agreement is None or agreement < MIN_AGREEMENT_F1):
                reasons.append("ecg_detector_disagreement")
            rows.append({
                "dataset": dataset, "subject": subject, "start_s": int(start), "duration_s": duration,
                "reference_quality": "manual_corrected" if dataset == "Dalia" else "automatic_ecg_consensus",
                "reference_valid": accepted, "exclusion_reason": ";".join(reasons),
                "r_peak_start_index": int(lo), "r_peak_stop_index": int(hi),
                "xqrs_agreement_f1": agreement, "duplicate_annotations": duplicates, **stats,
            })
    return rows


def sampled_xqrs(ecg, fs, reference, n_windows=DALIA_AUDIT_WINDOWS):
    """Uniformly sample full 60-second blocks, with 5 seconds of ECG-only context."""
    from wfdb.processing import xqrs_detect
    duration_s = len(ecg) / fs
    n_blocks = int(duration_s // 60)
    starts = 60 * np.unique(np.rint(np.linspace(0, n_blocks - 1, min(n_blocks, n_windows))).astype(int))
    rows, times = [], []
    for start in starts:
        lo, hi = max(0, int((start - 5) * fs)), min(len(ecg), int((start + 65) * fs))
        detected = xqrs_detect(ecg[lo:hi], fs=fs, verbose=False) / fs + lo / fs
        detected = detected[(detected >= start) & (detected < start + 60)]
        selected = reference[(reference >= start) & (reference < start + 60)]
        rows.append({"start_s": int(start), "duration_s": 60, **beat_agreement(selected, detected)})
        times.extend(detected)
    return rows, np.asarray(times, dtype=np.float64)


def blockwise_xqrs(ecg, fs, detector=None):
    """Fixed five-minute cores with real five-second ECG context on both sides.

    Detector state resets independently in each block. Only the core contributes
    peaks, preserving original sample coordinates without duplicate boundaries.
    This bounds adaptive state propagation; it does not guarantee correct peaks.
    """
    if detector is None:
        from wfdb.processing import xqrs_detect
        detector = xqrs_detect
    block, context = int(XQRS_BLOCK_S * fs), int(XQRS_CONTEXT_S * fs)
    output = []
    for core_start in range(0, len(ecg), block):
        core_end = min(core_start + block, len(ecg))
        lo, hi = max(0, core_start - context), min(len(ecg), core_end + context)
        detected = np.asarray(detector(ecg[lo:hi], fs=fs, verbose=False), dtype=np.int64) + lo
        output.extend(detected[(detected >= core_start) & (detected < core_end)])
    return np.asarray(output, dtype=np.float64) / fs


def prepare_subject(dataset, subject, output):
    from wfdb.processing import xqrs_detect
    import wfdb
    destination = output / dataset / subject
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    source = (ROOT / "datasets/ppg_bench_raw/dalia/PPG_FieldStudy" / subject / f"{subject}.pkl" if dataset == "Dalia"
              else ROOT / "datasets/ppg_bench_raw/wildppg/WildPPG/data" / f"WildPPG_Part_{subject}.mat")
    identity = {"source": source.relative_to(ROOT).as_posix(), "source_sha256": sha256(source),
                "script_sha256": sha256(__file__), "wfdb_version": wfdb.__version__}
    if dataset == "WildPPG":
        phase1 = ROOT / ".local/ppg-hr-poc/wild-reference" / f"{subject}.npz"
        phase1_metadata = json.loads(phase1.with_suffix(".json").read_text())
        identity["phase1_reference_sha256"] = sha256(phase1)
        if identity["phase1_reference_sha256"] != phase1_metadata["arrays_sha256"]:
            raise ValueError("Existing official reference cache failed its digest")
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior["identity"] != identity or any(sha256(destination / name) != digest for name, digest in prior["outputs_sha256"].items()):
            raise ValueError(f"Reference cache changed: {dataset}/{subject}; use a fresh destination")
        return prior
    started = time.monotonic()
    if dataset == "Dalia":
        # Only the locally provisioned official dataset pickle is loaded.
        with source.open("rb") as stream:
            raw = pickle.load(stream, encoding="latin1")
        ecg = np.asarray(raw["signal"]["chest"]["ECG"]).squeeze()
        ppg = np.asarray(raw["signal"]["wrist"]["BVP"]).squeeze()
        fs, ppg_fs = 700, 64
        original_r_samples = np.asarray(raw["rpeaks"], dtype=np.int64)
        duration_s = len(ppg) / ppg_fs
        if len(ecg) / fs != duration_s or str(raw["subject"]) != subject:
            raise ValueError("DaLiA aligned signal durations or participant identity differ")
        reference, duplicate_times = canonical_reference(original_r_samples / fs, duration_s)
        r_samples = np.rint(reference * fs).astype(np.int64)
        audits, xqrs = sampled_xqrs(ecg, fs, reference)
        xqrs_scope = "15 uniformly spaced complete 60-second windows; 5-second ECG-only context"
        rows = make_windows(dataset, subject, reference, duration_s, duplicate_times=duplicate_times)
    else:
        raw = loadmat(source, variable_names=["sternum", "wrist"], struct_as_record=False, squeeze_me=True)
        ecg = np.asarray(raw["sternum"].ecg.v).squeeze()
        ppg = np.asarray(raw["wrist"].ppg_g.v).squeeze()
        fs, ppg_fs = int(raw["sternum"].ecg.fs), int(raw["wrist"].ppg_g.fs)
        duration_s = min(len(ppg) / ppg_fs, len(ecg) / fs)
        if len(ecg) / fs != len(ppg) / ppg_fs:
            raise ValueError("WildPPG aligned PPG/ECG durations differ")
        with np.load(phase1) as cached:
            original_r_samples = cached["r_peak_samples"].copy()
            original_reference = cached["r_peak_times_s"].copy()
        if not np.array_equal(original_r_samples / fs, original_reference):
            raise ValueError("Existing official reference samples/seconds differ")
        reference, duplicate_times = canonical_reference(original_reference, duration_s)
        r_samples = np.rint(reference * fs).astype(np.int64)
        xqrs = blockwise_xqrs(ecg, fs)
        audits = [{"start_s": 0, "duration_s": duration_s, **beat_agreement(reference, xqrs)}]
        xqrs_scope = "entire native ECG; fixed 300-second cores, 5-second real ECG context each side, WFDB XQRS defaults reset each core"
        rows = make_windows(dataset, subject, reference, duration_s, xqrs, duplicate_times)
    reference = validate_peaks(reference, duration_s)
    if not np.isfinite(ecg).all() or not np.isfinite(ppg).all():
        raise ValueError("Nonfinite acquisition values; do not silently fill")
    # Preserve native PPG values and sampling rate. Consumers may resample once.
    np.save(destination / "ppg_native.npy", ppg)
    np.savez_compressed(destination / "beats.npz", r_peak_samples=r_samples,
                        r_peak_times_s=reference, xqrs_peak_times_s=xqrs,
                        original_r_peak_samples=original_r_samples,
                        original_r_peak_times_s=original_r_samples / fs,
                        original_rr_ms=np.diff(original_r_samples / fs) * 1000,
                        duplicate_event_times_s=duplicate_times,
                        rr_ms=np.diff(reference) * 1000, rr_midpoint_s=(reference[1:] + reference[:-1]) / 2,
                        ecg_fs=np.asarray(fs), ppg_fs=np.asarray(ppg_fs), duration_s=np.asarray(duration_s))
    with (destination / "windows.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(destination / "detector_audit.json", audits)
    result = {
        "dataset": dataset, "subject": subject, "identity": identity, "duration_s": duration_s,
        "ecg_fs": fs, "ppg_fs": ppg_fs, "ppg_dtype": str(ppg.dtype), "ecg_dtype": str(ecg.dtype),
        "n_reference_beats": len(reference), "n_original_annotations": len(original_r_samples),
        "duplicate_annotation_count": len(duplicate_times), "xqrs_scope": xqrs_scope,
        "detector_agreement": {"reference_beats": sum(a["reference_beats"] for a in audits),
                               "xqrs_beats": sum(a["xqrs_beats"] for a in audits),
                               "matched_beats": sum(a["matched_beats"] for a in audits)},
        "windows": {str(d): {"total": sum(r["duration_s"] == d for r in rows),
                             "accepted": sum(r["duration_s"] == d and r["reference_valid"] for r in rows)} for d in WINDOW_DURATIONS},
        "outputs_sha256": {name: sha256(destination / name) for name in ("beats.npz", "ppg_native.npy", "windows.csv", "detector_audit.json")},
        "elapsed_seconds": time.monotonic() - started,
    }
    save_json(manifest_path, result)
    print(f"{dataset}/{subject}: {result['windows']}; {result['elapsed_seconds']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", choices=["Dalia", "WildPPG", "all"], default="all")
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()
    datasets = ["Dalia", "WildPPG"] if args.dataset == "all" else [args.dataset]
    manifests = []
    for dataset in datasets:
        subjects = sorted(p.name for p in (ROOT / "datasets/ppg_bench_processed/fs_64" / dataset).iterdir() if (p / "ppg.npy").is_file())
        if args.subjects:
            if set(args.subjects) - set(subjects):
                raise ValueError("Unknown participant requested")
            subjects = [s for s in subjects if s in args.subjects]
        for subject in subjects:
            manifests.append(prepare_subject(dataset, subject, args.output))
    summary = {
        "complete_requested_scope": True, "datasets": datasets, "subject_count": len(manifests), "by_subject": manifests,
        "protocol": {"durations_s": list(WINDOW_DURATIONS), "main_prv_duration_s": 300,
                     "wild_xqrs_block_s": XQRS_BLOCK_S, "wild_xqrs_context_s": XQRS_CONTEXT_S,
                     "reference_peak_source": {"Dalia": "Released manually corrected ECG rpeaks, native 700 Hz indices",
                                                "WildPPG": "Unfiltered original official ECG Pan-Tompkins peaks, native 128 Hz indices; NOT human labels"},
                     "interval_definition": "Only consecutive peaks both inside each complete [start,end) interval; never concatenate filtered peaks or bridge removed intervals",
                     "quality_selection": "ECG only. At least 20 intervals, all 0.3 to 2.0 s, first/last peak within 2 s of boundaries. Wild additionally requires XQRS agreement F1 >=0.95 at 50 ms. DaLiA manual annotations remain primary regardless of sampled detector agreement.",
                     "no_editing": "All raw R peaks and intervals are retained. Only exact duplicate annotation timestamps are collapsed into a canonical event view; affected windows are excluded even after collapse. No detector disagreement is corrected or replaced.",
                     "limitations": ["No PPG systolic-peak manual ground truth in either dataset.",
                                     "ECG-to-PPG pulse arrival delay makes absolute PPG-peak error against R times invalid without an explicitly declared mapping.",
                                     "Wild cross-device synchronization accuracy is approximately 33 ms in its paper; automatic detector consensus is not verified accuracy.",
                                     "XQRS can miss clear ECG beats after large artifacts even with block resets. Rejection means detector disagreement, not proven corrupt ECG; reference selection is conservative.",
                                     "No normal/ectopic rhythm annotations: these are RR variability statistics, not clinically verified NN HRV.",
                                     "Quality criteria select reference availability and may exclude physiological irregularity; report coverage.",
                                     "SDNN and RMSSD from 60 s are exploratory short-window estimates; primary comparison uses 300 s."]},
        "sources": ["https://pmc.ncbi.nlm.nih.gov/articles/PMC6679242/", "https://ppg-beats.readthedocs.io/en/latest/datasets/ppg_dalia/",
                    "https://arxiv.org/html/2412.17540v1", "https://wfdb.readthedocs.io/en/latest/processing.html#wfdb.processing.xqrs_detect"],
    }
    save_json(args.output / f"summary_{args.dataset}.json", summary)


if __name__ == "__main__":
    main()
