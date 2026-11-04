"""Annotation-free checks of fixed polarity, multiscale logic and sample clock."""
import unittest

import numpy as np
from scipy.signal import detrend

from scripts.ppg_onset_baseline import _multiscale_maxima, _window_starts, detect_onsets


class TestOnsetBaseline(unittest.TestCase):
    def test_multiscale_matches_independent_scalar_definition(self):
        rng = np.random.default_rng(10)
        for n in (90, 120, 129):
            fs = 20.
            signal = np.sin(2 * np.pi * 1.3 * np.arange(n) / fs) + .03 * rng.normal(size=n)
            x = detrend(signal)
            largest = int(np.ceil(n / 2)) - 1
            rows = []
            for scale in range(1, largest + 1):
                if (largest / scale) / (n / fs) < .5:
                    break
                row = np.zeros(n, bool)
                for index in range(scale, n - scale):
                    row[index] = x[index] > x[index - scale] and x[index] > x[index + scale]
                rows.append(row)
            chosen = max(range(len(rows)), key=lambda k: np.count_nonzero(rows[k]))
            expected = [i for i in range(n) if all(rows[k][i] for k in range(chosen + 1))]
            np.testing.assert_array_equal(_multiscale_maxima(signal, fs), expected)

    def test_raw_maxima_onsets_at_native_400_and_64_hz(self):
        for fs in (64., 400.):
            t = np.arange(int(30 * fs)) / fs
            # The raw-intensity diastolic onset is the maximum, not a trough.
            raw = 1000. + 10. * np.cos(2 * np.pi * (t - .5))
            saved = raw.copy()
            detected = detect_onsets(raw, fs)
            interior = detected[(detected > 2.) & (detected < 28.)]
            expected = np.arange(2.5, 28., 1.)
            self.assertEqual(len(interior), len(expected))
            np.testing.assert_allclose(interior, expected, atol=2/fs)
            np.testing.assert_array_equal(raw, saved)
            self.assertTrue((np.diff(detected) > 0).all())

    def test_acquisition_polarity_is_explicit_not_label_selected(self):
        fs = 64.
        t = np.arange(int(20 * fs)) / fs
        raw = np.cos(2 * np.pi * (t - .5))
        a = detect_onsets(raw, fs)
        b = detect_onsets(-raw, fs, polarity="blood_volume")
        np.testing.assert_array_equal(a, b)
        opposite = detect_onsets(-raw, fs)
        self.assertGreater(abs(np.median((opposite % 1.)) - np.median(a % 1.)), .3)

    def test_tail_window_coverage_no_duplicate_or_out_of_range_events(self):
        np.testing.assert_array_equal(_window_starts(100, 20, 16), [0, 16, 32, 48, 64, 80])
        np.testing.assert_array_equal(_window_starts(103, 20, 16), [0, 16, 32, 48, 64, 80, 83])
        np.testing.assert_array_equal(_window_starts(13, 20, 16), [0])
        fs = 400.
        t = np.arange(8123) / fs
        events = detect_onsets(np.cos(2*np.pi*t), fs)
        self.assertTrue(((events >= 0.) & (events < len(t)/fs)).all())
        self.assertEqual(len(events), len(np.unique(events)))

    def test_invalid_input_and_flatline_preserve_failure_semantics(self):
        self.assertEqual(len(detect_onsets(np.ones(1000), 64)), 0)
        self.assertEqual(len(detect_onsets(np.arange(10.), 64)), 0)
        for signal, fs, polarity in (([1., np.nan], 64, "raw_intensity"),
                                     (np.ones((2, 10)), 64, "raw_intensity"),
                                     (np.ones(100), 16, "raw_intensity"),
                                     (np.ones(100), 64, "automatic")):
            with self.assertRaises(ValueError):
                detect_onsets(signal, fs, polarity)


if __name__ == "__main__":
    unittest.main()
