"""Coverage and coordinate checks for the native PPG annotation adapter."""

import io
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ppg_embc21_data import (  # noqa: E402
    _BuiltinsOnlyUnpickler,
    eligible_windows,
    load_records,
    mask_intervals_s,
    valid_mask_from_artifacts,
)


class EMBC21MaskTests(unittest.TestCase):
    def test_artifact_endpoints_both_excluded(self):
        mask = valid_mask_from_artifacts(10, np.array([[2, 4], [8, 9]]))
        np.testing.assert_array_equal(mask, [1, 1, 0, 0, 0, 1, 1, 1, 0, 0])
        np.testing.assert_array_equal(mask_intervals_s(mask, 2), [[0, 1], [2.5, 4]])

    def test_complete_rejection_and_empty_intervals(self):
        mask = valid_mask_from_artifacts(4, np.array([[0, 3]]))
        self.assertEqual(mask_intervals_s(mask).shape, (0, 2))

    def test_invalid_or_overlapping_ranges_rejected(self):
        for ranges in [[[2, 4]], [[1, 2], [2, 3]], [[2, 1]], [[-1, 2]]]:
            with self.assertRaises(ValueError):
                valid_mask_from_artifacts(4, np.array(ranges))

    def test_windows_cannot_cross_single_missing_sample(self):
        mask = np.ones(20, dtype=bool)
        mask[8] = False
        record = {"fs": 2, "valid_mask": mask}
        np.testing.assert_array_equal(eligible_windows(record, 2, 2), [[0, 2], [2, 4], [6, 8], [8, 10]])
        with self.assertRaises(ValueError):
            eligible_windows(record, 0.3, 2)

    def test_pickle_audit_rejects_global_construction(self):
        data = pickle.dumps(Path("not-opened"))
        with self.assertRaises(pickle.UnpicklingError):
            _BuiltinsOnlyUnpickler(io.BytesIO(data)).load()


class EMBC21LocalDataTests(unittest.TestCase):
    @unittest.skipUnless(Path("datasets/embc21/manifest.json").exists(), "Optional public data not downloaded")
    def test_official_full_data_and_known_source_exceptions(self):
        records = load_records("datasets/embc21")
        by_id = {r["record_id"]: r for r in records}
        self.assertEqual(len(records), 105)
        self.assertEqual(len({r["subject"] for r in records}), 7)
        self.assertEqual(sum(len(r["all_onset_times_s"]) for r in records), 21806)
        self.assertEqual(sum(len(r["onset_times_s"]) for r in records), 21508)
        self.assertFalse(by_id["S5_squat3"]["valid_mask"].any())
        missing = by_id["S2_squat3"]
        self.assertEqual(int(np.isnan(missing["raw_ppg"]).sum()), 39)
        self.assertTrue(np.isfinite(missing["input_ppg"]).all())
        self.assertEqual(len(eligible_windows(missing, 8, 6)), 0)
        self.assertEqual(by_id["S4_rest1"]["source_mat_columns"], 3)
        self.assertEqual(by_id["S5_step1"]["source_time_origin_s"], 0.045)
        for r in records:
            np.testing.assert_allclose(r["all_onset_times_s"], r["onset_samples"] / 400, atol=1e-10)
            for start, end in r["annotated_intervals_s"]:
                self.assertTrue(r["valid_mask"][round(start * 400):round(end * 400)].all())
        self.assertEqual(sum(len(eligible_windows(r, 8, 6)) for r in records), 2577)
        self.assertEqual(sum(len(eligible_windows(r, 60, 60)) for r in records), 158)
        self.assertEqual(sum(len(eligible_windows(r, 300, 300)) for r in records), 3)


if __name__ == "__main__":
    unittest.main()
