"""Independently reread formal clinical references and selected waveforms."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ppg_hr_poc_features import sha256, write_json

LOCAL = ROOT / ".local/ppg-formal-expansion/clinical"
SHARED = ROOT / "docs/results/ppg_formal_expansion/clinical"


def audit_bidmc():
    folder = LOCAL / "BIDMC"
    rows = json.loads((folder / "rows.json").read_text())
    with np.load(folder / "data.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["record"]].append((i, row))
    largest_error = 0.
    for record, items in groups.items():
        prefix = ROOT / f"datasets/ppg_bench_raw/bidmc/bidmc_csv/bidmc_{record}"
        text = Path(str(prefix) + "_Fix.txt").read_text()
        subject = re.search(r"MIMIC II matched wdb ID:\s*(\S+)", text).group(1)
        signal = np.genfromtxt(str(prefix) + "_Signals.csv", delimiter=",", names=True)["PLETH"]
        with open(str(prefix) + "_Breaths.csv", newline="") as stream:
            reader = csv.reader(stream); next(reader)
            annotations = list(reader)
        marks = [np.asarray([float(r[j]) for r in annotations if r[j].strip()]) for j in (0, 1)]
        for i, row in items:
            if row["subject"] != subject or data["subject"][i] != subject:
                raise AssertionError("BIDMC patient mismatch")
            first, last = row["start_s"] * 125, (row["start_s"] + 32) * 125
            np.testing.assert_array_equal(signal[first:last].astype(np.float32), data["x"][i])
            estimates = []
            for samples in marks:
                positions = samples - 1
                selected = positions[(positions >= first) & (positions < last)]
                if len(selected) < 3:
                    raise AssertionError("Insufficient reference marks")
                estimates.append(60 * 125 * (len(selected) - 1) / (selected[-1] - selected[0]))
            if abs(estimates[0] - estimates[1]) > 2 + 1e-10:
                raise AssertionError("Reference disagreement")
            largest_error = max(largest_error, abs(float(np.mean(estimates)) - data["y"][i]))
            np.testing.assert_allclose(data["recording_stats"][i], [signal.mean(), signal.std()], rtol=0, atol=1e-12)
    if largest_error > 1e-10:
        raise AssertionError("Manual reference mismatch")
    splits = json.loads((folder / "splits.json").read_text())
    all_patients = set(data["subject"])
    test_visits = []
    for split in splits:
        sets = [set(split[name]) for name in ("train", "validation", "test")]
        if set.union(*sets) != all_patients or any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise AssertionError("Invalid patient fold")
        test_visits.extend(split["test"])
    if len(test_visits) != len(all_patients) or set(test_visits) != all_patients:
        raise AssertionError("Test folds do not partition patients")
    return {"dataset": "BIDMC", "rows_independently_reread": len(rows), "records": len(groups),
            "patients": len(all_patients), "max_manual_reference_error": largest_error,
            "waveforms_exact_float32_match": True, "five_fold_patient_separation": True,
            "data_sha256": sha256(folder / "data.npz"), "rows_sha256": sha256(folder / "rows.json")}


def cell(handle, group, field, offset):
    dataset = group[field]
    ref = dataset[0, offset] if dataset.shape[0] == 1 else dataset[offset, 0]
    return np.asarray(handle[ref][()]).ravel(order="F")


def decode(array):
    return "".join(map(chr, map(int, array)))


def audit_pulsedb():
    import h5py
    folder = LOCAL / "PulseDB"
    rows = json.loads((folder / "rows.json").read_text())
    with np.load(folder / "data.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    file_groups, index_groups = defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        file_groups[row["source_file"]].append((i, row)); index_groups[row["info_file"]].append((i, row))
    max_abp_error = 0.
    for source, items in file_groups.items():
        with h5py.File(ROOT / source, "r") as handle:
            group = handle["Subj_Wins"]
            for i, row in items:
                position = int(row["segment_position_one_based"]) - 1
                patient = decode(cell(handle, group, "SubjectID", position))
                if patient != row["subject_name"][:7] or row["patient_id"] != data["subject"][i]:
                    raise AssertionError("PulseDB patient mismatch")
                x = cell(handle, group, "PPG_F", position)
                np.testing.assert_array_equal(x.astype(np.float32), data["x"][i])
                t = cell(handle, group, "T", position)
                if len(t) != 1250 or abs(t[0] - row["time_first"]) > 1e-10 or abs(t[-1] - row["time_last"]) > 1e-10:
                    raise AssertionError("Time metadata mismatch")
                np.testing.assert_allclose(np.diff(t), .008, atol=1e-8, rtol=0)
                original = float(cell(handle, group, "SegmentID", position)[0])
                if original != row["original_segment_id"]:
                    raise AssertionError("Original segment identity mismatch")
                abp = cell(handle, group, "ABP_Raw", position)
                peaks = cell(handle, group, "ABP_SPeaks", position).astype(int) - 1
                troughs = cell(handle, group, "ABP_Turns", position).astype(int) - 1
                reference = np.array([np.mean(abp[peaks]), np.mean(abp[troughs])])
                maximum = float(np.abs(reference - data["y"][i]).max()); max_abp_error = max(max_abp_error, maximum)
                if maximum > 1e-6:
                    raise AssertionError("Raw ABP does not reproduce labels")
                if hashlib.sha256(data["x"][i].tobytes()).hexdigest() != row["ppg_sha256"]:
                    raise AssertionError("Waveform row hash mismatch")
    mapping = {"Train_Info.mat": "Train_Subset", "CalFree_Test_Info.mat": "CalFree_Test_Subset"}
    for source, items in index_groups.items():
        with h5py.File(ROOT / "datasets/pulsedb/Info_Files" / source, "r") as handle:
            group = handle[mapping[source]]
            for i, row in items:
                offset = row["info_row_zero_based"]
                if decode(cell(handle, group, "Subj_Name", offset)) != row["subject_name"]:
                    raise AssertionError("Official index patient mismatch")
                if int(cell(handle, group, "Subj_SegIDX", offset)[0]) != row["segment_position_one_based"]:
                    raise AssertionError("Official index position mismatch")
                ref = [cell(handle, group, name, offset)[0] for name in ("Seg_SBP", "Seg_DBP")]
                np.testing.assert_array_equal(ref, data["y"][i])
    sets = {role: set(data["subject"][data["role"] == role]) for role in ("train", "validation", "test")}
    if any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("PulseDB patient overlap")
    with h5py.File(ROOT / "datasets/pulsedb/Info_Files/AAMI_Test_Info.mat", "r") as handle:
        group = handle["AAMI_Test_Subset"]
        aami = set()
        for i in range(group["Subj_Name"].size):
            name = decode(cell(handle, group, "Subj_Name", i))
            aami.add(("MIMIC:" if name[-1] == "0" else "VitalDB:") + name[:7])
    if aami & set(data["subject"]):
        raise AssertionError("AAMI patients exposed")
    return {"dataset": "PulseDB", "rows_independently_reread": len(rows), "patient_files": len(file_groups),
            "max_raw_ABP_reference_error_mmhg": max_abp_error, "official_index_checks": len(rows),
            "waveforms_exact_float32_match": True, "time_axes_checked": len(rows),
            "aami_patient_exclusion_checked": len(aami), "split_patient_overlap": [],
            "data_sha256": sha256(folder / "data.npz"), "rows_sha256": sha256(folder / "rows.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["bidmc", "pulsedb"], default=["bidmc", "pulsedb"])
    args = parser.parse_args()
    results = [(audit_bidmc if name == "bidmc" else audit_pulsedb)() for name in args.datasets]
    report = {"status": "passed", "independent_implementation": True, "checks": results, "code_sha256": sha256(Path(__file__))}
    write_json(SHARED / "clinical_reference_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
