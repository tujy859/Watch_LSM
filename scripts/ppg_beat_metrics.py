"""Small, explicit ECG-event/PPG-peak metrics for the phase-two PoC.

PPG peaks follow ECG R events by pulse arrival time (PAT); a training-selected
global delay is an evaluation alignment convention, not per-test optimization.
RR variability below uses observed consecutive intervals without ectopic-beat
adjudication and must not be called NN variability or clinical HRV validation.
The Elgendi-style detector is independent of the old eight-second HR baseline.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, sosfiltfilt

TOLERANCES_S = (.05, .1, .15)


def _events(value, name):
    times = np.asarray(value, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError(f"{name} must be finite strictly increasing one-dimensional times")
    return times


def match_events(reference_s, predicted_s, tolerance_s=.1):
    """Maximum-cardinality one-to-one assignment, then minimum timing error.

Events exactly on the tolerance boundary are eligible. A private dummy column
for every reference makes all unmatched references feasible; a dummy penalty
larger than all possible timing costs prioritizes match count over closeness.
Inputs may be empty. Both-empty F1 and any undefined denominator are NaN.
"""
    reference = _events(reference_s, "reference_s")
    predicted = _events(predicted_s, "predicted_s")
    if not np.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be finite and nonnegative")
    nr, npred = len(reference), len(predicted)
    ri, pi = np.array([], dtype=int), np.array([], dtype=int)
    if nr and npred:
        delta = np.abs(reference[:, None] - predicted[None, :])
        eligible = delta <= tolerance_s + 1e-12
        penalty = (min(nr, npred) + 1) * (tolerance_s + 1.)
        cost = np.full((nr, npred + nr), penalty)
        cost[:, :npred] = np.where(eligible, delta, 2. * penalty)
        rows, columns = linear_sum_assignment(cost)
        real = columns < npred
        ri, pi = rows[real], columns[real]
        if not eligible[ri, pi].all():
            raise AssertionError("Invalid real assignment despite available dummies")
        # Absolute distance on a line admits an order-preserving optimum. The
        # solver can choose a crossing tie, so restore its equivalent ordered
        # pairing; this makes consecutive-interval eligibility deterministic.
        pi = np.sort(pi)
        if not (np.abs(reference[ri] - predicted[pi]) <= tolerance_s + 1e-12).all():
            raise AssertionError("Order-preserving matching feasibility failed")
    tp = len(ri)
    errors = predicted[pi] - reference[ri]
    return {"reference_index": ri, "prediction_index": pi, "timing_error_s": errors,
            "tp": tp, "fp": npred - tp, "fn": nr - tp,
            "precision": tp / npred if npred else float("nan"),
            "recall": tp / nr if nr else float("nan"),
            "f1": 2. * tp / (nr + npred) if nr + npred else float("nan"),
            "timing_mae_ms": float(np.mean(np.abs(errors)) * 1000.) if tp else float("nan"),
            "timing_bias_ms": float(np.mean(errors) * 1000.) if tp else float("nan")}


def matched_rr_errors(reference_s, predicted_s, match):
    """Only differences between pairs consecutive in BOTH original sequences.

No interval crosses an unmatched reference beat or an unmatched prediction.
The denominator for reference-pair coverage is all consecutive reference pairs,
not just successfully matched beats. It is not a full-record tracking coverage.
"""
    reference, predicted = _events(reference_s, "reference_s"), _events(predicted_s, "predicted_s")
    ri = np.asarray(match["reference_index"], dtype=int)
    pi = np.asarray(match["prediction_index"], dtype=int)
    if ri.shape != pi.shape or (np.diff(ri) <= 0).any() or (np.diff(pi) <= 0).any():
        raise ValueError("Matched indices must be equally sized and ordered")
    if (ri < 0).any() or (ri >= len(reference)).any() or (pi < 0).any() or (pi >= len(predicted)).any():
        raise ValueError("Matched indices out of bounds")
    consecutive = (np.diff(ri) == 1) & (np.diff(pi) == 1)
    rri = np.diff(reference[ri])[consecutive] * 1000.
    ppi = np.diff(predicted[pi])[consecutive] * 1000.
    error = ppi - rri
    return {"reference_rr_ms": rri, "prediction_rr_ms": ppi, "rr_error_ms": error,
            "reference_pair_start_index": ri[:-1][consecutive],
            "prediction_pair_start_index": pi[:-1][consecutive],
            "n_rr_errors": len(error),
            "rr_mae_ms": float(np.mean(np.abs(error))) if len(error) else float("nan"),
            "rr_bias_ms": float(np.mean(error)) if len(error) else float("nan"),
            "reference_pair_coverage": len(error) / (len(reference) - 1) if len(reference) > 1 else float("nan")}


def rr_features(event_times_s, plausible_rr_s=(.3, 2.)):
    """Raw interval features; flag implausible intervals without deleting them.

SDNN here is the sample SD of all RR/PP intervals (ddof=1); RMSSD is the root
mean square of successive interval differences. A minimum of three events is
needed for both. Call independently on genuinely continuous segments, never on
concatenated disjoint snippets or filtered peak lists that bridge missing beats.
"""
    events = _events(event_times_s, "event_times_s")
    lower, upper = plausible_rr_s
    if not 0 < lower < upper:
        raise ValueError("Invalid plausible RR range")
    intervals = np.diff(events)
    n = len(intervals)
    return {"n_events": len(events), "n_intervals": n,
            "mean_rr_ms": float(intervals.mean() * 1000.) if n else float("nan"),
            "sdnn_rr_ms": float(intervals.std(ddof=1) * 1000.) if n >= 2 else float("nan"),
            "rmssd_rr_ms": float(np.sqrt(np.mean(np.diff(intervals) ** 2)) * 1000.) if n >= 2 else float("nan"),
            "plausible_rr_fraction": float(np.mean((intervals >= lower) & (intervals <= upper))) if n else float("nan")}


def crop_events(event_times_s, start_s, end_s):
    """Half-open [start, end), used consistently for both event sequences."""
    events = _events(event_times_s, "event_times_s")
    if not np.isfinite([start_s, end_s]).all() or start_s >= end_s:
        raise ValueError("Invalid frame boundaries")
    return events[(events >= start_s) & (events < end_s)]


def evaluate_event_frame(reference_s, predicted_s, start_s, end_s, prediction_delay_s=0., return_rr_detail=False):
    """Scalar frame output; delay is subtracted BEFORE half-open cropping.

Prediction context must include any delayed peaks outside the unshifted core.
No unmatched beat is silently dropped from F1. RR/PP summaries use each full
cropped sequence; matched-interval errors use only consecutive matched pairs.
Reference quality/eligibility must be established independently by the caller.
"""
    if not np.isfinite(prediction_delay_s):
        raise ValueError("Delay must be finite")
    reference = crop_events(reference_s, start_s, end_s)
    predicted = crop_events(_events(predicted_s, "predicted_s") - prediction_delay_s, start_s, end_s)
    output = {"duration_s": float(end_s - start_s), "n_reference_events": len(reference),
              "n_prediction_events": len(predicted)}
    primary = None
    for tolerance in TOLERANCES_S:
        matched = match_events(reference, predicted, tolerance)
        suffix = f"{int(round(tolerance * 1000))}ms"
        for key in ("tp", "fp", "fn", "precision", "recall", "f1", "timing_mae_ms", "timing_bias_ms"):
            output[f"{key}_{suffix}"] = matched[key]
        if tolerance == .1:
            primary = matched
    rr = matched_rr_errors(reference, predicted, primary)
    for key in ("n_rr_errors", "rr_mae_ms", "rr_bias_ms", "reference_pair_coverage"):
        output[key] = rr[key]
    for prefix, events in (("reference", reference), ("prediction", predicted)):
        for key, value in rr_features(events).items():
            output[f"{prefix}_{key}"] = value
    for key in ("mean_rr_ms", "sdnn_rr_ms", "rmssd_rr_ms"):
        output[f"{key}_absolute_error"] = abs(output[f"prediction_{key}"] - output[f"reference_{key}"])
    if return_rr_detail:
        output["_rr_reference_pair_index"] = rr["reference_pair_start_index"]
        output["_rr_abs_error_ms"] = np.abs(rr["rr_error_ms"])
    return output


def elgendi_peaks(signal, fs=64.):
    """Return positive pulse-peak seconds on a continuous PPG segment.

Independent paper-informed Algorithm IV: 0.5–8 Hz order-2 zero-phase Butterworth,
111/667 ms odd moving averages, offset .02, no refractory/outlier removal. Caller
should supply fixed padding around each evaluation core (e.g. >=2 seconds) and
then crop returned times. Nonfinite input is rejected; flat/short inputs return
an empty detection. No reference information, polarity search or HR is used.
"""
    x = np.asarray(signal)
    if x.ndim != 1 or not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError("PPG must be a real one-dimensional signal")
    if not np.isfinite(fs) or fs <= 16:
        raise ValueError("Sampling rate must exceed 16 Hz")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite PPG segment: report failed/invalid coverage")
    if len(x) < np.ceil(2 * fs):
        return np.array([], dtype=float)
    x = np.asarray(x, dtype=float).copy()
    scale = np.max(np.abs(x))
    if scale:
        x /= scale
    x -= x.mean()
    deviation = x.std()
    if deviation == 0:
        return np.array([], dtype=float)
    x /= deviation
    filtered = sosfiltfilt(butter(2, [.5, 8.], btype="bandpass", fs=fs, output="sos"), x)
    squared = np.maximum(filtered, 0.) ** 2
    short = max(1, 2 * int(np.floor(.111 * fs / 2)) + 1)
    long = max(1, 2 * int(np.floor(.667 * fs / 2)) + 1)
    mask = uniform_filter1d(squared, short, mode="constant") > uniform_filter1d(squared, long, mode="constant") + .02 * squared.mean()
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    indices = [start + int(np.argmax(filtered[start:stop])) for start, stop in zip(starts, stops) if stop - start >= short]
    return np.asarray(indices, dtype=float) / fs


def select_global_pat(frames, training_subjects, candidates_s=None, tolerance_s=.1):
    """Estimate one fixed PAT using only declared training subjects.

Each frame is a dict with subject, reference_s, ppg_s, start_s, end_s. PPG peaks
include padded context. Candidate scores are frame-F1 means within participant,
then participant means. Ties use lower mean matched timing error, then smaller
PAT. Undefined reference-empty frames are rejected, not scored as perfect.
"""
    frames = list(frames)
    allowed = set(training_subjects)
    if not frames or not allowed:
        raise ValueError("Need training frames and declared training subjects")
    if any(frame["subject"] not in allowed for frame in frames):
        raise ValueError("PAT estimation received a non-training subject")
    if candidates_s is None:
        candidates_s = np.arange(0, .6 + 1e-12, 1. / 64.)
    candidates = np.asarray(candidates_s, dtype=float)
    if candidates.ndim != 1 or not len(candidates) or not np.isfinite(candidates).all() or (candidates < 0).any() or (np.diff(candidates) <= 0).any():
        raise ValueError("PAT candidates must be finite nonnegative sorted unique values")
    prepared = []
    for frame in frames:
        ref = crop_events(frame["reference_s"], frame["start_s"], frame["end_s"])
        ppg = _events(frame["ppg_s"], "ppg_s")
        if not len(ref):
            raise ValueError("PAT training frames require reference events")
        prepared.append((frame, ref, ppg))
    scores, timing = [], []
    for delay in candidates:
        by_person, by_person_timing = defaultdict(list), defaultdict(list)
        for frame, reference, ppg in prepared:
            prediction = crop_events(ppg - delay, frame["start_s"], frame["end_s"])
            result = match_events(reference, prediction, tolerance_s)
            by_person[frame["subject"]].append(result["f1"])
            if result["tp"]:
                by_person_timing[frame["subject"]].append(result["timing_mae_ms"])
        scores.append(float(np.mean([np.mean(v) for v in by_person.values()])))
        timing.append(float(np.mean([np.mean(v) for v in by_person_timing.values()])) if by_person_timing else float("inf"))
    chosen = min(range(len(candidates)), key=lambda i: (-scores[i], timing[i], candidates[i]))
    if scores[chosen] <= 0:
        raise ValueError("No training peak matches: global PAT is not estimable")
    return {"delay_s": float(candidates[chosen]), "training_macro_f1": scores[chosen],
            "training_timing_mae_ms": timing[chosen], "n_frames": len(frames),
            "subjects": sorted({f["subject"] for f in frames}), "candidates_s": candidates.tolist(),
            "candidate_macro_f1": scores, "candidate_timing_mae_ms": [v if np.isfinite(v) else None for v in timing],
            "selection": "maximum participant-mean frame F1, then minimum matched timing MAE, then smaller delay"}
