"""Signal-level checks for the fixed, label-free HR comparators."""

import unittest

import numpy as np

from scripts.ppg_hr_baselines import predict_hr_windows


def pulse_train(fs, bpm, duration=8.0, phase=0.2):
    time = np.arange(int(fs * duration)) / fs
    cycle = (time * bpm / 60.0 + phase) % 1.0
    # Smooth, asymmetric pulses plus a small diastolic shoulder. Construct the
    # waveform from its known period, independently of either estimator.
    main = np.exp(-0.5 * ((cycle - 0.25) / 0.075) ** 2)
    shoulder = 0.15 * np.exp(-0.5 * ((cycle - 0.55) / 0.10) ** 2)
    return main + shoulder


class TestPPGHRBaselines(unittest.TestCase):
    def test_known_pulse_rates_across_sampling_rates(self):
        target = np.array([54.0, 84.0, 126.0, 174.0])
        for fs in (50, 64, 125):
            with self.subTest(fs=fs):
                signals = np.array([pulse_train(fs, rate) for rate in target])
                predictions = predict_hr_windows(signals, fs)
                np.testing.assert_allclose(predictions["welch"], target, atol=2.0, rtol=0)
                # Peak timing is quantized at the native sampling rate.
                np.testing.assert_allclose(predictions["elgendi"], target, atol=5.0, rtol=0)

    def test_drift_and_small_noise(self):
        fs = 64
        time = np.arange(8 * fs) / fs
        signal = pulse_train(fs, 84) + 0.5 * np.sin(2 * np.pi * 0.15 * time)
        signal += np.random.default_rng(7).normal(0, 0.015, len(time))
        for name, prediction in predict_hr_windows(signal[None], fs).items():
            with self.subTest(method=name):
                self.assertAlmostEqual(float(prediction[0]), 84.0, delta=3.0)

    def test_flatline_and_nonfinite_windows_fail_explicitly(self):
        signals = np.stack([np.zeros(512), np.full(512, 5.0), pulse_train(64, 84), pulse_train(64, 84)])
        signals[2, 100] = np.nan
        signals[3, 200] = np.inf
        for prediction in predict_hr_windows(signals, 64).values():
            self.assertTrue(np.isnan(prediction).all())

    def test_gain_offset_and_batch_composition_do_not_change_predictions(self):
        signals = np.stack([pulse_train(64, rate) for rate in (63, 96, 145)])
        original = signals.copy()
        expected = predict_hr_windows(signals, 64)
        transformed = predict_hr_windows(signals * 1e-7 + 4e-7, 64)
        for name in expected:
            np.testing.assert_allclose(transformed[name], expected[name], atol=1e-9, rtol=0)
            alone = predict_hr_windows(signals[1:2], 64)[name]
            np.testing.assert_allclose(alone, expected[name][1:2], atol=1e-9, rtol=0)
        np.testing.assert_array_equal(signals, original)

    def test_batch_boundary_preserves_order_and_values(self):
        signals = np.stack([pulse_train(64, 72), pulse_train(64, 120)])
        expected = predict_hr_windows(signals, 64)
        repeated = predict_hr_windows(np.tile(signals, (257, 1)), 64)
        for name in expected:
            np.testing.assert_allclose(repeated[name], np.tile(expected[name], 257), atol=1e-9, rtol=0)

    def test_empty_and_too_short_inputs(self):
        for prediction in predict_hr_windows(np.empty((0, 512)), 64).values():
            self.assertEqual(prediction.shape, (0,))
        for prediction in predict_hr_windows(np.ones((2, 32)), 64).values():
            self.assertTrue(np.isnan(prediction).all())

    def test_invalid_shapes_sampling_rates_and_types(self):
        for fs in (0, 16, np.nan, np.inf):
            with self.subTest(fs=fs), self.assertRaises(ValueError):
                predict_hr_windows(np.ones((1, 512)), fs)
        for signals in (np.ones(512), np.ones((2, 1, 512)), np.array([["x"]]), np.ones((1, 512), complex)):
            with self.assertRaises(ValueError):
                predict_hr_windows(signals, 64)

if __name__ == "__main__":
    unittest.main()
