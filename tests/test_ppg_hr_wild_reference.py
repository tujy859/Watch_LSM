"""Protect released-function loading and ECG-reference window semantics."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts.ppg_hr_wild_reference import (
    FUNCTIONS, OFFICIAL_SOURCE, derive_reference, load_official_reference,
)


class WildReferenceTests(unittest.TestCase):
    def test_function_loader_never_runs_top_level_conversion(self):
        definitions = "\n".join(
            f"def {name}(*args, **kwargs):\n    return {i}\n"
            for i, name in enumerate(FUNCTIONS)
        )
        source = (
            "raise RuntimeError('top-level conversion executed')\n"
            "def unrelated():\n    raise RuntimeError('not selected')\n"
            + definitions
        )
        with patch.object(Path, "read_text", return_value=source):
            loaded = load_official_reference("released.py")
        self.assertNotIn("unrelated", loaded)
        for i, name in enumerate(FUNCTIONS):
            self.assertEqual(loaded[name](), i)

    def test_function_loader_requires_each_definition_once(self):
        for names in (FUNCTIONS[:-1], (*FUNCTIONS, FUNCTIONS[0])):
            source = "\n".join(f"def {name}():\n    pass\n" for name in names)
            with patch.object(Path, "read_text", return_value=source):
                with self.assertRaisesRegex(ValueError, "function set changed"):
                    load_official_reference("released.py")

    def test_window_boundaries_minimum_rr_and_hr_units(self):
        functions = {
            "pan_tompkins_detector": lambda ecg, fs: np.array([100, 150, 250]),
            "quotient_filter": lambda peaks, **kwargs: (
                np.array([100, 150, 250]),
                np.array([999, 50, 100, 999, 25, 999]),
                np.array([0, 100, 799, 800, 900, 1600]),
            ),
        }
        result = derive_reference(np.zeros(2400), 100, np.array([0., 8., 16.]), functions)
        # Midpoints exactly at either window boundary are excluded. HR is
        # 60 / mean(RR seconds), not the mean of instantaneous heart rates.
        np.testing.assert_array_equal(result["n_rr_intervals"], [2, 1, 0])
        np.testing.assert_array_equal(result["valid_reference"], [True, False, False])
        self.assertEqual(result["reference_official"][0], 80.)
        self.assertTrue(np.isnan(result["reference_official"][1:]).all())
        np.testing.assert_array_equal(result["r_peak_times_s"], [1., 1.5, 2.5])
        np.testing.assert_array_equal(result["good_peak_times_s"], [1., 1.5, 2.5])

    def test_native_ecg_is_passed_without_precision_change(self):
        original = np.array([16777217, 16777219, 16777221], dtype=np.int32)

        def detector(ecg, fs):
            self.assertIs(ecg, original)
            self.assertEqual(ecg.dtype, np.int32)
            self.assertEqual(fs, 128)
            return np.array([128, 256, 384])

        def quotient(peaks, **kwargs):
            np.testing.assert_array_equal(peaks, [128, 256, 384])
            self.assertEqual(kwargs, {"outlier_over": 5, "sampling_rate": 128, "tol": 0.75})
            return peaks, np.array([128, 128]), np.array([192., 320.])

        result = derive_reference(original, 128, np.array([0.]), {
            "pan_tompkins_detector": detector, "quotient_filter": quotient,
        })
        self.assertEqual(result["reference_official"][0], 60.)

    @unittest.skipUnless(OFFICIAL_SOURCE.is_file(), "Released WildPPG code asset is not installed")
    def test_released_peak_detector_adaptation_regression(self):
        # A small signal-only fixture exercises both previously omitted details:
        # the weight switch after 100 peaks and integer rounding of the missed-RR
        # threshold. Removing either yields 181 or 200 peaks rather than 207.
        rng = np.random.default_rng(2)
        detection = np.zeros(128 * 180)
        locations = np.cumsum(rng.integers(42, 132, size=260))
        detection[locations] = rng.lognormal(0, 1.1, len(locations))
        peaks = load_official_reference()["panPeakDetect"](detection, 128)
        self.assertEqual(len(peaks), 207)
        self.assertEqual(peaks[100], 11796)
        self.assertEqual(peaks[-1], 22739)


if __name__ == "__main__":
    unittest.main()
