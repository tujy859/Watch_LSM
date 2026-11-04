"""Leakage, reproducibility and basic learning checks for the small HR CNN."""

import unittest

import numpy as np
import torch

from scripts.ppg_hr_cnn import (
    SmallHRCNN, cpu_state_dict, fit_cnn, normalize_windows,
    predict_cnn, subject_weights,
)


def pulse_windows(n=12):
    time = np.arange(512, dtype=np.float32) / 64
    rates = np.linspace(60, 140, n, dtype=np.float32)
    phase = np.arange(n, dtype=np.float32)[:, None] / 3
    fundamental = 2 * np.pi * rates[:, None] / 60 * time[None, :] + phase
    x = np.sin(fundamental) + 0.2 * np.sin(2 * fundamental)
    return x.astype(np.float32), rates


class SmallCNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_subject_weights_give_equal_total_mass(self):
        ids = np.array(["a"] * 2 + ["b"] * 8)
        weights = subject_weights(ids)
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertAlmostEqual(float(weights[ids == "a"].sum()),
                               float(weights[ids == "b"].sum()))

    def test_window_normalization_has_no_other_window_dependency(self):
        x, _ = pulse_windows(3)
        one = normalize_windows(torch.from_numpy(x[:1, None]))
        combined = normalize_windows(torch.from_numpy(np.concatenate([
            x[:1], x[1:] * 1000 + 90000
        ]).astype(np.float32)[:, None]))
        torch.testing.assert_close(one, combined[:1])
        torch.testing.assert_close(normalize_windows(torch.ones(2, 1, 512)),
                                   torch.zeros(2, 1, 512))

    def test_training_validation_subject_overlap_fails(self):
        x, y = pulse_windows(4)
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_cnn(x, y, ["s1"] * 4, val_x=x, val_y=y,
                    val_subjects=["s1"] * 4, max_epochs=1)

    def test_selection_refit_and_state_reload(self):
        x, y = pulse_windows(12)
        untouched = SmallHRCNN()
        original = cpu_state_dict(untouched)
        model, metadata = fit_cnn(
            x[:8], y[:8], ["a"] * 4 + ["b"] * 4,
            val_x=x[8:10], val_y=y[8:10], val_subjects=["c"] * 2,
            seed=7, max_epochs=3, batch_size=4,
        )
        self.assertLess(metadata["parameter_count"], 100000)
        self.assertIn(metadata["best_epoch"], [1, 2, 3])
        best = min(metadata["history"], key=lambda r: r["validation_subject_mae_bpm"])
        self.assertEqual(metadata["best_epoch"], best["epoch"])
        pred = predict_cnn(model, x[10:])
        self.assertEqual(pred.shape, (2,))
        self.assertTrue(np.isfinite(pred).all())
        with torch.random.fork_rng():
            torch.manual_seed(7)
            initialized = cpu_state_dict(SmallHRCNN())
        self.assertTrue(any(not torch.equal(initialized[k], v)
                            for k, v in cpu_state_dict(model).items()))
        loaded = SmallHRCNN()
        loaded.load_state_dict(cpu_state_dict(model))
        np.testing.assert_allclose(pred, predict_cnn(loaded, x[10:]), atol=1e-5)
        for key, value in original.items():
            torch.testing.assert_close(untouched.state_dict()[key], value)
        _, refit = fit_cnn(x, y, ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
                          seed=7, fixed_epochs=metadata["best_epoch"], batch_size=4)
        self.assertEqual(refit["epochs_run"], metadata["best_epoch"])
        self.assertIsNone(refit["best_validation_subject_mae_bpm"])
        self.assertEqual(refit["validation_windows"], 0)

    def test_repeated_seed_is_deterministic_and_restores_rng(self):
        x, y = pulse_windows(8)
        before = torch.get_rng_state().clone()
        first, _ = fit_cnn(x, y, ["a"] * 4 + ["b"] * 4,
                           fixed_epochs=2, seed=19, batch_size=4)
        torch.testing.assert_close(torch.get_rng_state(), before)
        second, _ = fit_cnn(x, y, ["a"] * 4 + ["b"] * 4,
                            fixed_epochs=2, seed=19, batch_size=4)
        np.testing.assert_array_equal(predict_cnn(first, x), predict_cnn(second, x))

    def test_invalid_training_and_prediction_coverage(self):
        x, y = pulse_windows(4)
        with self.assertRaisesRegex(ValueError, "no windows"):
            fit_cnn(x[:0], y[:0], [], fixed_epochs=1)
        invalid = x.copy()
        invalid[1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            fit_cnn(invalid, y, ["a"] * 4, fixed_epochs=1)
        with self.assertRaisesRegex(ValueError, "must not receive validation"):
            fit_cnn(x, y, ["a"] * 4, fixed_epochs=1, val_x=x)
        model, _ = fit_cnn(x, y, ["a"] * 4, fixed_epochs=1)
        pred = predict_cnn(model, invalid)
        self.assertTrue(np.isnan(pred[1]))
        self.assertTrue(np.isfinite(pred[[0, 2, 3]]).all())
        self.assertEqual(predict_cnn(model, x[:0]).shape, (0,))


if __name__ == "__main__":
    unittest.main()
