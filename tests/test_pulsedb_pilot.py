"""Tests for consequential PulseDB indexing and split-leakage failure modes."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from scripts.pulsedb_pilot import fit_heads, validate_prepared_arrays
from watch_lsm.data.pulsedb import (
    PulseDBIndex, assert_patient_disjoint, read_segment, subject_identity,
)


def mat_field(handle, group, name, values):
    references = group.create_dataset(name, (1, len(values)), dtype=h5py.ref_dtype)
    for i, value in enumerate(values):
        if isinstance(value, str):
            array = np.array([ord(c) for c in value], dtype=np.uint16)[:, None]
        else:
            array = np.atleast_2d(value)
        target = handle.create_dataset(f"refs/{name}_{i}", data=array)
        references[0, i] = target.ref


class PulseDBIndexTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.info = self.root / "Info_Files/Train_Info.mat"
        self.info.parent.mkdir(parents=True)
        with h5py.File(self.info, "w") as handle:
            group = handle.create_group("Train_Subset")
            # Position 2 corresponds to original SegmentID 18, not SegmentID 2.
            for name, values in {
                "Subj_Name": ["p000001_0"], "Source": ["MIMIC"],
                "Subj_SegIDX": [2], "Seg_SBP": [121.5], "Seg_DBP": [72.5],
            }.items():
                mat_field(handle, group, name, values)
        self.wave = self.root / "Segment_Files/PulseDB_MIMIC/p000001.mat"
        self.wave.parent.mkdir(parents=True)
        with h5py.File(self.wave, "w") as handle:
            group = handle.create_group("Subj_Wins")
            time = np.arange(1250) / 125
            for name, values in {
                "SubjectID": ["p000001", "p000001"], "SegmentID": [7, 18],
                "SegSBP": [110.5, 121.5], "SegDBP": [65.5, 72.5],
                "PPG_F": [np.sin(time), np.cos(time)],
                "ECG_F": [time * 0, time * 0],
                "ABP_Raw": [time * 0 + 80, time * 0 + 90],
                "T": [time, time + 100],
            }.items():
                mat_field(handle, group, name, values)
        with PulseDBIndex(self.info) as index:
            self.row = index.row(0)

    def test_one_based_position_is_not_original_segment_id(self):
        segment = read_segment(self.root, self.row)
        self.assertEqual(segment["segment_id"], 18)
        self.assertEqual(segment["sbp"], 121.5)
        np.testing.assert_allclose(segment["PPG_F"], np.cos(np.arange(1250) / 125))
        self.assertNotIn("ABP_Raw", segment)

    def test_reference_signals_are_explicitly_opt_in(self):
        segment = read_segment(self.root, self.row, include_reference_signals=True)
        self.assertEqual(segment["ABP_Raw"].shape, (1250,))
        self.assertEqual(segment["ECG_F"].shape, (1250,))

    def test_zero_or_out_of_range_index_fails(self):
        with self.assertRaises(ValueError):
            read_segment(self.root, replace(self.row, segment_position_one_based=0))
        with self.assertRaises(IndexError):
            read_segment(self.root, replace(self.row, segment_position_one_based=3))

    def test_mismatched_labels_fail_instead_of_silent_misalignment(self):
        with self.assertRaisesRegex(ValueError, "labels"):
            read_segment(self.root, replace(self.row, sbp=120))

    def test_waveform_patient_must_match_metadata(self):
        with h5py.File(self.wave, "r+") as handle:
            handle["refs/SubjectID_1"][...] = np.array([ord(c) for c in "p000002"])[:, None]
        with self.assertRaisesRegex(ValueError, "patient mismatch"):
            read_segment(self.root, self.row)

    def test_source_suffix_and_declared_source_must_agree(self):
        with h5py.File(self.info, "r+") as handle:
            handle["refs/Subj_Name_0"][...] = np.array([ord(c) for c in "p000001_1"])[:, None]
        with PulseDBIndex(self.info) as index, self.assertRaisesRegex(ValueError, "Source mismatch"):
            index.row(0)

    def test_irregular_time_grid_fails(self):
        with h5py.File(self.wave, "r+") as handle:
            handle["refs/T_1"][0, 100] += 0.001
        with self.assertRaisesRegex(ValueError, "125 Hz"):
            read_segment(self.root, self.row)

    def test_patient_overlap_is_rejected_even_for_different_segments(self):
        with self.assertRaisesRegex(ValueError, "Patient leakage"):
            assert_patient_disjoint({"train": [self.row], "test": [
                replace(self.row, segment_position_one_based=1, info_file="CalFree_Test_Info.mat")]})
        self.assertNotEqual(subject_identity("p000001_0")[1], subject_identity("p000001_1")[1])

    def test_prepared_waveform_corruption_is_rejected_before_encoder(self):
        waveform = np.arange(1250, dtype=np.float64)[None, :]
        manifest = [{"patient_id": self.row.patient_id, "split": "train",
                     "sbp": self.row.sbp, "dbp": self.row.dbp,
                     "ppg_sha256": hashlib.sha256(waveform[0].tobytes()).hexdigest()}]
        path = self.root / "selected_rows.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        report = {"selected_rows_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        arrays = {"ppg": waveform, "patient_id": np.array([self.row.patient_id]),
                  "split": np.array(["train"]), "labels": np.array([[self.row.sbp, self.row.dbp]])}
        self.assertEqual(validate_prepared_arrays(self.root, report, arrays), 1)
        waveform[0, 123] += 1
        with self.assertRaisesRegex(ValueError, "differs from official-row"):
            validate_prepared_arrays(self.root, report, arrays)


class PulseDBRegressionTests(unittest.TestCase):
    def test_test_labels_cannot_change_selected_alpha_or_predictions(self):
        rng = np.random.default_rng(100)
        features = rng.normal(size=(20, 4))
        labels = rng.normal(size=(20, 2)) * 10 + np.array([120, 80])
        splits = np.array(["train"] * 10 + ["validation"] * 5 + ["test"] * 5)
        people = np.array(["a"] * 5 + ["b"] * 5 + ["c"] * 5 + ["d"] * 5)
        original, predicted = fit_heads(features, labels, splits, people)
        changed = labels.copy(); changed[splits == "test"] += 1000
        perturbed, predicted_again = fit_heads(features, changed, splits, people)
        for before, after in zip(original, perturbed):
            self.assertEqual(before["alpha"], after["alpha"])
        for target in predicted:
            np.testing.assert_array_equal(predicted[target], predicted_again[target])


if __name__ == "__main__":
    unittest.main()
