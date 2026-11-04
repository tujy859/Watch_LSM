"""Statistical invariants for descriptive transfer diagnostics."""
import unittest

import numpy as np

from scripts.analyze_ppg_hr_transfer import (
    aggregate_seed_rows, calibration, describe_distribution, feature_distance,
    paired_bootstrap, stratified_pair, subject_weights, weighted_quantile,
)


class TestTransferDiagnosis(unittest.TestCase):
    def test_participant_weights_avoid_window_count_bias(self):
        subjects = np.array(["a"] * 9 + ["b"])
        y = np.array([60.] * 9 + [100.])
        result = describe_distribution(y, subjects)
        self.assertEqual(result["mean"], 80.)
        self.assertAlmostEqual(subject_weights(subjects).sum(), 1.)
        self.assertEqual(result["p50"], 60.)
        self.assertEqual(result["p95"], 100.)

    def test_gap_decomposition_accounts_for_absent_subject_strata(self):
        y = np.zeros(6)
        subjects = np.array(["a", "a", "a", "a", "b", "b"])
        within = np.array([1., 1., 1., 1., 2., 2.])
        transfer = np.array([3., 3., 3., 3., 6., 6.])
        groups = np.array([0, 0, 0, 0, 1, 1])
        result = stratified_pair(y, within, transfer, subjects, groups, [("low", 0), ("high", 1), ("absent", 2)])
        self.assertEqual(result[0]["gap_contribution_bpm"], 1.)
        self.assertEqual(result[1]["gap_contribution_bpm"], 2.)
        self.assertEqual(sum(r["gap_contribution_bpm"] for r in result), 3.)
        self.assertEqual(result[0]["n_subjects"], 1)
        self.assertIsNone(result[2]["within_mae_bpm"])
        self.assertEqual(result[2]["gap_contribution_bpm"], 0.)

    def test_seed_averaging_precedes_participant_bootstrap(self):
        rows = [{"subject": "a", "seed": 42, "gap": 0.},
                {"subject": "a", "seed": 43, "gap": 2.},
                {"subject": "b", "seed": 42, "gap": 7.},
                {"subject": "b", "seed": 43, "gap": 9.}]
        result = aggregate_seed_rows(rows, ("subject",))
        self.assertEqual([r["gap"] for r in result], [1., 8.])
        self.assertEqual(paired_bootstrap([r["gap"] for r in result]), (4.5, 1., 8.))

    def test_calibration_describes_compressed_predictions(self):
        y = np.array([60., 80., 100.])
        result = calibration(y, .5 * y + 50., np.array(["a", "a", "a"]))
        self.assertAlmostEqual(result["prediction_on_reference_slope"], .5)
        self.assertAlmostEqual(result["bias_bpm"], 10.)
        self.assertAlmostEqual(result["correlation"], 1.)
        self.assertIsNone(calibration(y, np.ones(3), np.array(["a"] * 3))["correlation"])

    def test_feature_distance_uses_supplied_source_statistics(self):
        x = np.array([[2., 3.], [4., 5.]])
        result = feature_distance(x, np.array([2., 3.]), np.array([2., 1.]), chunk=1)
        np.testing.assert_allclose(result, [0., np.sqrt(2.5)])
        np.testing.assert_array_equal(x, [[2., 3.], [4., 5.]])

    def test_invalid_distribution_and_bootstrap_rejected(self):
        with self.assertRaises(ValueError):
            weighted_quantile(np.array([1., np.nan]), np.ones(2), [.5])
        with self.assertRaises(ValueError):
            weighted_quantile(np.ones(2), np.array([1., -1.]), [.5])
        with self.assertRaises(ValueError):
            paired_bootstrap([1., np.nan])


if __name__ == "__main__":
    unittest.main()
