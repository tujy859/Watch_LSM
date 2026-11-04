import unittest

import numpy as np

from scripts.ppg_beat_reference_audit import (
    beat_agreement, blockwise_xqrs, canonical_reference, make_windows, match_beats, validate_peaks, variability,
)


class BeatReferenceTests(unittest.TestCase):
    def test_matching_is_one_to_one_and_has_no_hidden_lag(self):
        pairs = match_beats([1., 2., 3.], [.98, 1.01, 2.02, 3.2], .05)
        np.testing.assert_array_equal(pairs, [[0, 0], [1, 2]])
        self.assertEqual(beat_agreement([1., 2.], [1.2, 2.2])["matched_beats"], 0)
        self.assertEqual(len(match_beats([], [])), 0)
        self.assertEqual(beat_agreement([], [1.])["agreement_f1"], 0)
        self.assertEqual(len(match_beats([1.], [1.05], .05)), 1)
        self.assertEqual(len(match_beats([1.], [1.051], .05)), 0)

    def test_block_context_retains_original_time_without_duplicate_boundaries(self):
        fs = 100
        signal = np.zeros(650 * fs)
        signal[np.arange(fs, len(signal), fs)] = 1
        observed_lengths = []

        def detector(values, fs, verbose):
            self.assertEqual(fs, 100)
            self.assertFalse(verbose)
            observed_lengths.append(len(values))
            return np.flatnonzero(values)

        actual = blockwise_xqrs(signal, fs, detector)
        np.testing.assert_array_equal(actual, np.arange(1., 650.))
        self.assertEqual(observed_lengths, [30500, 31000, 5500])

    def test_invalid_peak_time_is_rejected(self):
        for peaks in ([1., 1.], [2., 1.], [np.nan], [-1.], [60.]):
            with self.assertRaises(ValueError):
                validate_peaks(peaks, 60.)

    def test_duplicate_annotations_are_preserved_as_exclusion_evidence(self):
        raw = np.sort(np.r_[np.arange(.5, 120., 1.), 10.5])
        unique, duplicates = canonical_reference(raw, 120.)
        np.testing.assert_array_equal(unique, np.arange(.5, 120., 1.))
        np.testing.assert_array_equal(duplicates, [10.5])
        self.assertEqual(len(raw), 121)
        rows = make_windows("Dalia", "S14", unique, 120., duplicate_times=duplicates)
        self.assertFalse(rows[0]["reference_valid"])
        self.assertTrue(rows[1]["reference_valid"])
        self.assertIn("duplicate_reference_annotation", rows[0]["exclusion_reason"])
        with self.assertRaises(ValueError):
            canonical_reference([2., 1.], 3.)

    def test_unedited_interval_statistics_and_units(self):
        values = variability([0., 1., 2.1, 3.])
        self.assertAlmostEqual(values["mean_rr_ms"], 1000)
        self.assertAlmostEqual(values["sdnn_ms"], 100)
        self.assertAlmostEqual(values["rmssd_ms"], np.sqrt(25000))
        self.assertEqual(values["rr_outside_bounds"], 0)

    def test_no_window_boundary_bridging_and_full_window_requirement(self):
        peaks = np.arange(.5, 360, 1.)
        rows = make_windows("Dalia", "S1", peaks, 360.)
        self.assertEqual(sum(r["duration_s"] == 60 for r in rows), 6)
        self.assertEqual(sum(r["duration_s"] == 300 for r in rows), 1)
        self.assertEqual(rows[0]["n_rr"], 59)
        self.assertEqual(rows[1]["r_peak_start_index"], 60)
        self.assertTrue(all(r["reference_valid"] for r in rows))

    def test_bad_reference_retained_and_consensus_is_wild_only(self):
        peaks = np.arange(.5, 60., 1.)
        disagreement = np.arange(.7, 60., 1.)
        rows = make_windows("WildPPG", "an0", peaks, 60., disagreement)
        self.assertFalse(rows[0]["reference_valid"])
        self.assertIn("ecg_detector_disagreement", rows[0]["exclusion_reason"])
        self.assertEqual(rows[0]["n_beats"], 60)
        self.assertTrue(make_windows("Dalia", "S1", peaks, 60.)[0]["reference_valid"])
        # Missing beats are not filled, and rejected intervals are not removed.
        bad = np.concatenate([peaks[:10], peaks[20:]])
        result = make_windows("Dalia", "S1", bad, 60.)[0]
        self.assertFalse(result["reference_valid"])
        self.assertEqual(result["rr_outside_bounds"], 1)
        self.assertGreater(result["sdnn_ms"], 1000)


if __name__ == "__main__":
    unittest.main()
