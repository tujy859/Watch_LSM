"""Saved-artifact audit must reject leakage, misalignment and corrupt metrics."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.audit_ppg_bench_wrist_hr import audit_result, expected_split, load_cohort


class WristHRAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.cohort = {"A": {"hr": np.array([60, 80], dtype=np.float32)},
                       "B": {"hr": np.array([90, 100], dtype=np.float32)}}
        self.record = {"model": "AnyPPG", "method": "linear", "task": "dalia-hr", "seed": 42, "fold": 0,
                       "train_groups": ["A"], "test_groups": ["B"],
                       "results": [{"repeat": 0, "predictions": "predictions.npz", "test": {"mae": 3.0}}],
                       "baseline": {"mae": 25.0}}
        self.arrays = {"subjects": np.array(["B", "B"]), "y_true": np.array([90, 100], dtype=np.float32),
                       "y_pred": np.array([92, 96], dtype=np.float32)}

    def check(self):
        (self.directory / "results.json").write_text(json.dumps(self.record), encoding="utf-8")
        np.savez(self.directory / "predictions.npz", **self.arrays)
        return audit_result(self.directory, "AnyPPG", "dalia", self.cohort, enforce_official_split=False)

    def test_recomputes_mae_and_per_subject_metrics(self):
        result = self.check()
        self.assertEqual(result["mae"], 3)
        self.assertEqual(result["baseline"], 25)
        self.assertEqual(result["subject_rows"][0]["windows"], 2)

    def test_rejects_subject_leakage(self):
        self.record["test_groups"] = ["A", "B"]
        with self.assertRaisesRegex(ValueError, "subject overlap"):
            self.check()

    def test_rejects_reordered_source_labels(self):
        self.arrays["y_true"] = self.arrays["y_true"][::-1]
        with self.assertRaisesRegex(ValueError, "ordered source labels"):
            self.check()

    def test_rejects_wrong_subject_identity(self):
        self.arrays["subjects"][0] = "A"
        with self.assertRaisesRegex(ValueError, "subject order"):
            self.check()

    def test_rejects_nonfinite_prediction(self):
        self.arrays["y_pred"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.check()

    def test_rejects_stale_metric(self):
        self.record["results"][0]["test"]["mae"] = 1.0
        with self.assertRaisesRegex(ValueError, "Saved MAE"):
            self.check()

    def test_rejects_test_mean_constant_baseline(self):
        self.record["baseline"]["mae"] = 5.0
        with self.assertRaisesRegex(ValueError, "training-mean baseline"):
            self.check()

    def test_rejects_prediction_path_escape(self):
        self.record["results"][0]["predictions"] = "../predictions.npz"
        with self.assertRaisesRegex(ValueError, "escaped"):
            self.check()

    def test_split_partition_is_subject_disjoint_and_exhaustive(self):
        subjects = [f"S{x}" for x in range(1, 17)]
        tests = []
        for fold in range(5):
            train, test = expected_split(subjects, "wildppg", 42, fold)
            self.assertFalse(set(train) & set(test))
            self.assertEqual(set(train) | set(test), set(subjects))
            tests.extend(test)
        self.assertEqual(len(tests), len(set(tests)))
        self.assertEqual(set(tests), set(subjects))

    def test_cohort_retains_original_row_after_finite_label_filter(self):
        subject = self.directory / "S1"
        subject.mkdir()
        np.save(subject / "hr.npy", np.array([70, np.nan, 80]))
        np.save(subject / "ppg.npy", np.ones((3, 10)))
        cohort = load_cohort(self.directory)
        np.testing.assert_array_equal(cohort["S1"]["source_rows"], [0, 2])
        np.testing.assert_array_equal(cohort["S1"]["hr"], [70, 80])


if __name__ == "__main__":
    unittest.main()
