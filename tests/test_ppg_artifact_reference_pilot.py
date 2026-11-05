"""Ensure annotation provenance requires exact waveform identity, not similarity."""
import unittest

import numpy as np
from scipy.signal import butter, sosfilt

from scripts.ppg_artifact_reference_pilot import (
    grid_matches, minmax, raw_affine_matches, validate_annotation_pair,
)


class ArtifactReferencePilotTests(unittest.TestCase):
    def test_exact_raw_affine_match_finds_non_grid_offset(self):
        raw = np.random.default_rng(4).normal(size=4000)
        query = minmax(raw[237:237+1920])
        matches = raw_affine_matches(raw, query)
        self.assertEqual([item["start_sample"] for item in matches], [237])

    def test_high_correlation_with_changed_point_is_not_a_match(self):
        raw = np.random.default_rng(4).normal(size=4000)
        query = minmax(raw[237:237+1920])
        query[500] += 1e-5
        self.assertEqual(raw_affine_matches(raw, query), [])

    def test_duplicate_windows_are_reported_and_not_arbitrarily_resolved(self):
        window = np.random.default_rng(4).normal(size=1920)
        matches = raw_affine_matches(np.tile(window, 2), minmax(window))
        self.assertEqual([item["start_sample"] for item in matches], [0, 1920])

    def test_known_filter_requires_full_length_equality(self):
        raw = np.random.default_rng(4).normal(size=3840)
        sos = butter(2, [0.9/32, 5/32], btype="bandpass", output="sos")
        query = minmax(sosfilt(sos, raw[1920:3840]))
        results = grid_matches(raw, [query])[0]
        self.assertTrue(any(item["start_sample"] == 1920 and "per_30s" in item["method"] for item in results))
        query[500] += 1e-5
        self.assertFalse(grid_matches(raw, [query])[0])

    def test_label_shape_and_binary_values_are_guarded(self):
        validate_annotation_pair(np.ones((2, 1920)), np.zeros((2, 1920)))
        with self.assertRaisesRegex(ValueError, "aligned"):
            validate_annotation_pair(np.ones((2, 1920)), np.zeros((1, 1920)))
        with self.assertRaisesRegex(ValueError, "nonbinary"):
            validate_annotation_pair(np.ones((2, 1920)), np.full((2, 1920), 0.5))


if __name__ == "__main__":
    unittest.main()
