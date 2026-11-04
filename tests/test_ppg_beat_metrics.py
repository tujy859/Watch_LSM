"""Independent beat matching, continuous interval and training-only PAT checks."""
import itertools
import unittest

import numpy as np

from scripts.ppg_beat_metrics import (
    crop_events, elgendi_peaks, evaluate_event_frame, match_events,
    matched_rr_errors, rr_features, select_global_pat,
)


class TestBeatMetrics(unittest.TestCase):
    def test_assignment_maximizes_count_before_closeness(self):
        # Nearest-first would give 0.1 to 0.11, losing the reference at 0.
        result = match_events([0., .1], [.09, .2], .1)
        self.assertEqual(result["tp"], 2)
        np.testing.assert_array_equal(result["reference_index"], [0, 1])
        np.testing.assert_array_equal(result["prediction_index"], [0, 1])
        self.assertAlmostEqual(result["f1"], 1.)

    def test_assignment_matches_exhaustive_small_oracle(self):
        rng = np.random.default_rng(20261104)
        for _ in range(30):
            reference = np.sort(rng.uniform(0, 1, 3))
            predicted = np.sort(rng.uniform(0, 1, 4))
            possibilities = []
            for n in range(4):
                for ri in itertools.combinations(range(3), n):
                    for pi in itertools.permutations(range(4), n):
                        error = np.abs(reference[list(ri)] - predicted[list(pi)])
                        if (error <= .2).all():
                            possibilities.append((-n, error.sum()))
            expected = min(possibilities)
            actual = match_events(reference, predicted, .2)
            self.assertEqual(-actual["tp"], expected[0])
            self.assertAlmostEqual(np.abs(actual["timing_error_s"]).sum(), expected[1])

    def test_empty_detection_is_failure_and_empty_pair_not_perfect(self):
        missing = match_events([1., 2.], [])
        self.assertEqual(missing["fn"], 2)
        self.assertEqual(missing["f1"], 0.)
        self.assertTrue(np.isnan(missing["precision"]))
        self.assertTrue(np.isnan(match_events([], [])["f1"]))

    def test_missed_and_extra_peaks_never_bridge_rr_errors(self):
        reference, missing = [0., 1., 2., 3.], [0., 2., 3.]
        result = matched_rr_errors(reference, missing, match_events(reference, missing))
        self.assertEqual(result["n_rr_errors"], 1)
        self.assertEqual(result["reference_pair_coverage"], 1. / 3.)
        np.testing.assert_array_equal(result["reference_rr_ms"], [1000.])
        reference, extra = [0., 1., 2.], [0., .4, 1., 2.]
        result = matched_rr_errors(reference, extra, match_events(reference, extra))
        self.assertEqual(result["n_rr_errors"], 1)
        self.assertEqual(result["reference_pair_coverage"], .5)

    def test_interval_sample_sd_rmssd_and_no_silent_outlier_removal(self):
        result = rr_features([0., 1., 2.1, 3.])
        self.assertAlmostEqual(result["mean_rr_ms"], 1000.)
        self.assertAlmostEqual(result["sdnn_rr_ms"], 100.)
        self.assertAlmostEqual(result["rmssd_rr_ms"], np.sqrt(.025) * 1000.)
        result = rr_features([0., 1., 4.])
        self.assertEqual(result["n_intervals"], 2)
        self.assertEqual(result["plausible_rr_fraction"], .5)
        self.assertEqual(result["mean_rr_ms"], 2000.)
        self.assertTrue(np.isnan(rr_features([0., 1.])["rmssd_rr_ms"]))

    def test_delay_precedes_half_open_cropping(self):
        np.testing.assert_array_equal(crop_events([0., 1., 2., 3.], 1., 3.), [1., 2.])
        result = evaluate_event_frame([0., 1., 2.], [.2, 1.2, 2.2], 0., 2., .2)
        self.assertEqual(result["n_reference_events"], 2)
        self.assertEqual(result["n_prediction_events"], 2)
        self.assertEqual(result["f1_50ms"], 1.)
        self.assertAlmostEqual(result["rr_mae_ms"], 0.)

    def test_pat_is_training_only_and_tie_uses_timing(self):
        frames = [{"subject": "s1", "reference_s": np.array([1., 2., 3.]),
                   "ppg_s": np.array([1.25, 2.25, 3.25]), "start_s": 0., "end_s": 4.}]
        result = select_global_pat(frames, {"s1"}, candidates_s=[.2, .25, .3])
        self.assertEqual(result["delay_s"], .25)
        self.assertEqual(result["training_macro_f1"], 1.)
        with self.assertRaisesRegex(ValueError, "non-training"):
            select_global_pat(frames, {"s2"})
        with self.assertRaisesRegex(ValueError, "not estimable"):
            select_global_pat([{**frames[0], "ppg_s": np.array([])}], {"s1"})

    def test_positive_pulse_detector_continuous_and_scale_invariant(self):
        fs = 64
        t = np.arange(20 * fs) / fs
        # Smooth positive pulses at 1 Hz, with long baseline between pulses.
        signal = np.exp(-((t % 1. - .4) / .07) ** 2)
        original = signal.copy()
        peaks = elgendi_peaks(signal, fs)
        interior = peaks[(peaks > 2.) & (peaks < 18.)]
        self.assertEqual(len(interior), 16)
        np.testing.assert_allclose(np.diff(interior), np.ones(15), atol=1/fs)
        np.testing.assert_allclose(elgendi_peaks(signal * 1e-14 + 3e-14, fs), peaks)
        np.testing.assert_array_equal(signal, original)
        self.assertEqual(len(elgendi_peaks(np.ones(256), fs)), 0)

    def test_invalid_unsorted_events_and_nonfinite_signal_are_rejected(self):
        for bad in ([1., 1.], [2., 1.], [np.nan]):
            with self.assertRaises(ValueError):
                match_events(bad, [1.])
        with self.assertRaises(ValueError):
            elgendi_peaks(np.array([1., np.nan]))
        with self.assertRaises(ValueError):
            crop_events([1.], 2., 1.)


if __name__ == "__main__":
    unittest.main()
