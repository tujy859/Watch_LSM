"""Prepare patient-separated clinical PPG evaluation inputs; no model training.

BIDMC uses human breath marks and PulseDB uses official Train/CalFree rows.
SpO2 is audited only to explain why it is excluded from the formal benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.signal import butter, detrend, find_peaks, periodogram, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ppg_additional_tasks_pilot import load_breaths, manual_rate
from scripts.ppg_hr_poc_features import sha256, write_json
from scripts.pulsedb_pilot import ppg_features

LOCAL = ROOT / ".local/ppg-formal-expansion/clinical"
SHARED = ROOT / "docs/results/ppg_formal_expansion/clinical"
SEED = 20261105


def patient_folds(people, seed=SEED):
    """Five fixed patient folds, with the next fold used only for validation."""
    people = np.asarray(sorted(set(people)))
    if len(people) < 10:
        raise ValueError("Need at least ten patients")
    chunks = np.array_split(np.random.default_rng(seed).permutation(people), 5)
    output = []
    for fold in range(5):
        test, val = chunks[fold].tolist(), chunks[(fold + 1) % 5].tolist()
        train = sorted(set(people) - set(test) - set(val))
        if set(train) & set(test) or set(train) & set(val) or set(test) & set(val):
            raise AssertionError("Patient overlap")
        output.append({"fold": fold, "train": train, "validation": val, "test": test})
    return output


def bidmc_records():
    raw = ROOT / "datasets/ppg_bench_raw/bidmc/bidmc_csv"
    records = []
    for fix in sorted(raw.glob("*_Fix.txt")):
        match = re.search(r"MIMIC II matched wdb ID:\s*(\S+)", fix.read_text())
        if not match:
            raise ValueError("Missing original patient identifier")
        records.append({"subject": match[1], "record": fix.name.split("_")[1], "fix": fix})
    if len(records) != 53 or len({r["subject"] for r in records}) != 46:
        raise ValueError("Unexpected BIDMC cohort")
    return raw, records


def audit_spo2():
    raw, records = bidmc_records()
    values, patients, record_summary, hashes = [], [], [], {}
    invalid = Counter()
    for item in records:
        path = raw / f"bidmc_{item['record']}_Numerics.csv"
        numeric = np.genfromtxt(path, delimiter=",", names=True)
        y = numeric["SpO2"]
        good = np.isfinite(y) & (y > 0) & (y <= 100)
        invalid["nonfinite"] += int((~np.isfinite(y)).sum())
        invalid["out_of_0_to_100"] += int((np.isfinite(y) & ~good).sum())
        values.extend(y[good]); patients.extend([item["subject"]] * good.sum())
        record_summary.append({"record": item["record"], "patient": item["subject"],
                               "total": len(y), "valid": int(good.sum()),
                               "median": float(np.median(y[good])), "min": float(y[good].min()),
                               "max": float(y[good].max())})
        for source in (path, item["fix"]):
            hashes[source.relative_to(ROOT).as_posix()] = sha256(source)
    y, people = np.asarray(values), np.asarray(patients)
    predictions, constants = np.zeros(len(y)), []
    for split in patient_folds(people):
        # This diagnostic constant uses all non-test patients, not the formal
        # train/validation protocol; it has no PPG input or selected parameters.
        test = np.isin(people, split["test"])
        train_people = sorted(set(people[~test]))
        constant = np.mean([y[people == person].mean() for person in train_people])
        predictions[test] = constant; constants.append(float(constant))
    report = {"source": "BIDMC monitor SpO2 numerics (1 Hz), no PPG used",
              "scope": "Descriptive audit only; excluded from new formal training and performance plots",
              "records": len(records), "patients": len(set(people)), "finite_0_to_100_values": len(y),
              "invalid": dict(invalid), "quantiles": dict(zip(
                  ["min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max"],
                  np.quantile(y, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist())),
              "fraction_below90": float(np.mean(y < 90)), "fraction_below94": float(np.mean(y < 94)),
              "fraction_95_to100": float(np.mean(y >= 95)),
              "patients_with_any_below90": len(set(people[y < 90])),
              "five_fold_train_patient_mean_constant_mae_patient": float(np.mean([
                  np.mean(np.abs(y[people == p] - predictions[people == p])) for p in sorted(set(people))])),
              "constant_values": constants, "per_record": record_summary,
              "source_files_sha256": hashes, "seed": SEED, "code_sha256": sha256(Path(__file__)),
              "limits": ["Not directly comparable to older 10-second or new 32-second model metrics",
                         "Monitor readings are not arterial co-oximetry reference measurements",
                         "A single exported Pleth waveform is not documented as calibrated red/infrared raw optics"]}
    write_json(SHARED / "spo2_scope_audit.json", report)
    return report


def spectral_respiration(values, fs):
    """Fixed 6-42 breaths/min spectral maximum, no label-based tuning."""
    values = np.asarray(values, float)
    if len(values) < 16 or not np.isfinite(values).all() or np.std(values) < 1e-10:
        return float("nan")
    f, power = periodogram(detrend(values), fs=fs, window="hann", nfft=4096)
    selected = (f >= .1) & (f <= .7)
    return float(60 * f[selected][np.argmax(power[selected])])


def respiration_modulation(ppg, fs=125):
    """Simple baseline/amplitude/interval modulation comparator, not RRest reproduction.

    Peaks are found from PPG alone. AM and FM are interpolated only between
    valid event centers; no extrapolated padding enters their spectra.
    """
    x = np.asarray(ppg, float)
    if x.ndim != 1 or len(x) < fs * 16 or not np.isfinite(x).all():
        raise ValueError("Expected finite PPG of at least 16 seconds")
    baseline = spectral_respiration(x, fs)
    filtered = sosfiltfilt(butter(2, [.5, 8], fs=fs, btype="bandpass", output="sos"), x)
    peaks, _ = find_peaks(filtered, distance=round(.3 * fs), prominence=max(.1 * np.std(filtered), 1e-9))
    amplitude_time, amplitude = [], []
    for prev, peak in zip(peaks[:-1], peaks[1:]):
        valley = prev + int(np.argmin(filtered[prev:peak]))
        amplitude_time.append(peak / fs)
        amplitude.append(filtered[peak] - filtered[valley])

    def irregular_rate(times, values):
        times, values = np.asarray(times), np.asarray(values)
        if len(times) < 6 or times[-1] - times[0] < 16:
            return float("nan")
        grid = np.arange(times[0], times[-1] + 1e-9, .25)
        return spectral_respiration(np.interp(grid, times, values), 4)

    am = irregular_rate(amplitude_time, amplitude)
    intervals = np.diff(peaks / fs)
    centers = (peaks[1:] + peaks[:-1]) / (2 * fs)
    eligible = (intervals >= .3) & (intervals <= 2.)
    fm = irregular_rate(centers[eligible], intervals[eligible])
    candidates = np.array([baseline, am, fm])
    fusion = float(np.median(candidates[np.isfinite(candidates)])) if np.isfinite(candidates).any() else float("nan")
    return np.array([baseline, am, fm, fusion])


def prepare_bidmc():
    raw, records = bidmc_records()
    output = LOCAL / "BIDMC"; output.mkdir(parents=True, exist_ok=True)
    waves, rows, stats, hashes, rejected, traditionals = [], [], [], {}, Counter(), []
    for item in records:
        rid = item["record"]
        signal_path, breath_path = [raw / f"bidmc_{rid}_{name}" for name in ("Signals.csv", "Breaths.csv")]
        signal = np.genfromtxt(signal_path, delimiter=",", names=True)
        breaths = load_breaths(breath_path)
        full = np.asarray(signal["PLETH"], float)
        record_stats = [float(np.mean(full)), float(np.std(full))]
        for path in (signal_path, breath_path, item["fix"]):
            hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
        for start in range(0, len(full) // 125 - 31, 32):
            rates = np.asarray([manual_rate(b, start, start + 32) for b in breaths])
            if not np.isfinite(rates).all():
                rejected["insufficient_or_invalid_breaths"] += 1; continue
            if abs(rates[0] - rates[1]) > 2:
                rejected["annotator_disagreement_above2"] += 1; continue
            x = full[start * 125:(start + 32) * 125]
            if len(x) != 4000 or not np.isfinite(x).all():
                rejected["invalid_ppg"] += 1; continue
            waves.append(x.astype(np.float32)); stats.append(record_stats)
            rows.append({"subject": item["subject"], "record": rid, "start_s": start,
                         "duration_s": 32, "reference_rr": float(rates.mean()),
                         "annotator1_rr": float(rates[0]), "annotator2_rr": float(rates[1]),
                         "ppg_sha256": hashlib.sha256(x.astype(np.float32).tobytes()).hexdigest()})
            traditionals.append(respiration_modulation(x))
    x = np.asarray(waves)
    subjects = np.array([r["subject"] for r in rows])
    np.savez_compressed(output / "data.npz", x=x, y=np.array([r["reference_rr"] for r in rows]),
                        subject=subjects, recording_stats=np.asarray(stats), traditional=np.asarray(traditionals),
                        fs=np.asarray(125))
    write_json(output / "rows.json", rows)
    splits = patient_folds(subjects); write_json(output / "splits.json", splits)
    # PPG-only traditional results are now fixed; formal learned comparisons use
    # these same eligible windows and retain failed-output coverage explicitly.
    traditional_summary = []
    y = np.array([r["reference_rr"] for r in rows])
    for j, method in enumerate(["Baseline_modulation", "Amplitude_modulation", "Interval_modulation", "Median_fusion"]):
        p = np.asarray(traditionals)[:, j]; valid = np.isfinite(p)
        person_losses = [float(np.mean(np.abs(y[(subjects == s) & valid] - p[(subjects == s) & valid])))
                         for s in sorted(set(subjects[valid]))]
        traditional_summary.append({"method": method, "patient_mae_breaths_per_min": float(np.mean(person_losses)),
                                    "output_coverage": float(valid.mean()), "patients_with_output": len(person_losses)})
    report = {"dataset": "BIDMC", "status": "prepared_for_formal_evaluation", "windows": len(rows),
              "patients": len(set(subjects)), "records": len(set(r["record"] for r in rows)),
              "window_seconds": 32, "sampling_hz": 125, "rejected": dict(rejected),
              "reference": "Two independent human breath marks: mean of 60/mean inter-breath intervals; disagreement <=2 bpm",
              "boundary": "MATLAB sample indices minus one; half-open 32-second windows",
              "SpO2_used_for_selection_or_training": False, "split_seed": SEED, "splits": splits,
              "traditional": traditional_summary, "source_files_sha256": hashes,
              "rows_sha256": sha256(output / "rows.json"), "data_sha256": sha256(output / "data.npz"),
              "code_sha256": sha256(Path(__file__)),
              "limits": ["Manual breaths annotate impedance respiration, not directly measured airflow",
                         "Reference-agreement filtering is disclosed and shared by all methods",
                         "Simple PPG modulation methods are not claimed to reproduce RRest or be state of the art"]}
    write_json(output / "report.json", report); write_json(SHARED / "bidmc_preparation.json", report)
    print(json.dumps({"dataset": "BIDMC", "windows": len(rows), "patients": len(set(subjects)), "rejected": dict(rejected)}), flush=True)
    return report


def select_pulsedb(root, filename, count, excluded, seed):
    from watch_lsm.data.pulsedb import PulseDBIndex, segment_file, subject_identity
    selected, seen, candidates, unavailable = [], set(), 0, set()
    with PulseDBIndex(root / "Info_Files" / filename) as index:
        positions = np.random.default_rng(seed).choice(index.length, min(50000, index.length), replace=False)
        for position in positions:
            candidates += 1; name = index.subject(int(position))
            source, patient = subject_identity(name)
            if patient in seen or patient in excluded or patient in unavailable:
                continue
            if not segment_file(root, name).is_file():
                unavailable.add(patient); continue
            neighbours = [int(position)]
            for step in range(1, 100):
                for offset in (step, -step):
                    trial = int(position) + offset
                    if 0 <= trial < index.length and index.subject(trial) == name:
                        neighbours.append(trial)
                    if len(neighbours) == 50:
                        break
                if len(neighbours) == 50:
                    break
            selected.extend(index.row(i) for i in sorted(neighbours)); seen.add(patient)
            if len(seen) == count:
                break
        total = index.length
    return selected, {"official_rows": total, "candidate_rows_examined": candidates,
                      "selected_patients": len(seen), "requested_patients": count,
                      "unavailable_patient_files_encountered": len(unavailable)}


def prepare_pulsedb():
    import h5py
    from watch_lsm.data.pulsedb import PulseDBIndex, _value, assert_patient_disjoint, read_segment, subject_identity, UPSTREAM_COMMIT
    root = ROOT / "datasets/pulsedb"
    output = LOCAL / "PulseDB"; output.mkdir(parents=True, exist_ok=True)
    with PulseDBIndex(root / "Info_Files/AAMI_Test_Info.mat") as index:
        aami_people = {subject_identity(index.subject(i))[1] for i in range(index.length)}
    pilot = json.loads((ROOT / "docs/results/ppg_expansion_pilot/pulsedb_pilot.json").read_text())
    exposed = set(pilot["split_patients"]["test"])
    training, search_train = select_pulsedb(root, "Train_Info.mat", 120, aami_people, SEED)
    names = list(dict.fromkeys(r.patient_id for r in training))
    if len(names) < 30:
        raise RuntimeError("Too few training patients in bounded local search")
    shuffled = np.random.default_rng(SEED + 2).permutation(names)
    validation_names = set(shuffled[:min(20, len(names) // 5)])
    train_names = set(shuffled) - validation_names
    testing, search_test = select_pulsedb(root, "CalFree_Test_Info.mat", 50,
                                        aami_people | exposed | set(names), SEED + 1)
    if len({r.patient_id for r in testing}) < 10:
        raise RuntimeError("Too few independent CalFree patients in bounded local search")
    parts = {"train": [r for r in training if r.patient_id in train_names],
             "validation": [r for r in training if r.patient_id in validation_names], "test": testing}
    identities = assert_patient_disjoint(parts)
    all_selected = set(sum(identities.values(), []))
    if all_selected & aami_people or set(identities["test"]) & exposed:
        raise AssertionError("AAMI or exposed pilot patient entered formal evaluation")
    manifest, signals, features, labels, roles, people, stats = [], [], [], [], [], [], []
    max_reference_error, mapped_differences = 0., 0
    for role, part in parts.items():
        for k, row in enumerate(part):
            segment = read_segment(root, row, include_reference_signals=True)
            with h5py.File(segment["file"], "r") as h:
                group = h["Subj_Wins"]; position = row.segment_position_one_based - 1
                peaks = _value(h, group["ABP_SPeaks"], position).ravel().astype(int) - 1
                troughs = _value(h, group["ABP_Turns"], position).ravel().astype(int) - 1
                if not len(peaks) or not len(troughs) or min(peaks.min(), troughs.min()) < 0 or max(peaks.max(), troughs.max()) >= 1250:
                    raise ValueError("Invalid official ABP fiducial indices")
                reference = [segment["ABP_Raw"][peaks].mean(), segment["ABP_Raw"][troughs].mean()]
            error = float(np.max(np.abs(np.asarray(reference) - [row.sbp, row.dbp])))
            max_reference_error = max(error, max_reference_error)
            if error > 1e-6:
                raise ValueError("ABP fiducial means disagree with official BP labels")
            x = segment["PPG_F"].astype(np.float32)
            signals.append(x); features.append(ppg_features(x)); labels.append([row.sbp, row.dbp])
            roles.append(role); people.append(row.patient_id); stats.append([float(x.mean()), float(x.std())])
            mapped_differences += int(segment["segment_id"] != row.segment_position_one_based)
            manifest.append({"split": role, **row.to_dict(), "original_segment_id": segment["segment_id"],
                             "time_first": float(segment["T"][0]), "time_last": float(segment["T"][-1]),
                             "source_file": segment["file"].relative_to(ROOT).as_posix(),
                             "source_file_bytes": segment["file"].stat().st_size,
                             "abp_reference_max_error_mmhg": error,
                             "ppg_sha256": hashlib.sha256(x.tobytes()).hexdigest()})
            if (k + 1) % 500 == 0:
                print(json.dumps({"dataset": "PulseDB", "role": role, "loaded": k + 1}), flush=True)
    np.savez_compressed(output / "data.npz", x=np.asarray(signals), y=np.asarray(labels), role=np.asarray(roles),
                        subject=np.asarray(people), recording_stats=np.asarray(stats), hand_features=np.asarray(features), fs=np.asarray(125))
    write_json(output / "rows.json", manifest)
    report = {"dataset": "PulseDB", "status": "prepared_for_formal_evaluation", "seed": SEED,
              "upstream_commit": UPSTREAM_COMMIT, "waveform_field": "PPG_F", "sampling_hz": 125, "window_seconds": 10,
              "selection": "Fixed-seed sample of official rows, local file availability, no BP or quality based selection; patient chance can depend on official row count",
              "search": {"Train_Info": search_train, "CalFree_Test_Info": search_test}, "split_patients": identities,
              "split_sizes": {role: {"patients": len(identities[role]), "segments": len(part),
                                      "sources": dict(Counter(r.source for r in part))} for role, part in parts.items()},
              "aami_patient_exclusion": {"count": len(aami_people), "patients": sorted(aami_people), "overlap": []},
              "pilot_exposed_test_excluded": sorted(exposed), "selected_patient_overlap": [],
              "rows_verified": len(manifest), "ABP_fiducial_reference_max_error_mmhg": max_reference_error,
              "original_segment_id_not_equal_to_position": mapped_differences,
              "rows_sha256": sha256(output / "rows.json"), "data_sha256": sha256(output / "data.npz"),
              "code_sha256": sha256(Path(__file__)),
              "limits": ["Bounded local-availability cohort, not all PulseDB or clinical device validation",
                         "AnyPPG pretraining includes PulseDB; source overlap in other encoders remains possible",
                         "PPG_F is already filtered/normalized and does not provide clean/noisy waveform truth",
                         "Clinical finger PPG does not establish wrist blood-pressure measurement",
                         "Normalization statistics here refer to each provided ten-second segment, not unavailable parent recording optics"]}
    write_json(output / "report.json", report); write_json(SHARED / "pulsedb_preparation.json", report)
    print(json.dumps({"dataset": "PulseDB", "split_sizes": report["split_sizes"], "reference_error": max_reference_error}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", nargs="*", choices=["bidmc", "pulsedb"], default=[])
    parser.add_argument("--audit-spo2", action="store_true")
    args = parser.parse_args()
    if args.audit_spo2:
        audit_spo2()
    for name in args.prepare:
        (prepare_bidmc if name == "bidmc" else prepare_pulsedb)()


if __name__ == "__main__":
    main()
