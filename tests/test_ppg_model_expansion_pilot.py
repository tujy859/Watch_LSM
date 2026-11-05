"""Guards for limited sample selection, leakage and controlled task heads."""
import unittest

import numpy as np

from scripts.ppg_model_expansion_pilot import (
    extract_features, fit_projection, fit_small_head, project,
    select_pilot_rows, validate_roles,
)


class ModelExpansionPilotTests(unittest.TestCase):
    def test_sample_selection_ignores_labels_and_is_subject_disjoint(self):
        rows = {"subject": np.repeat([f"S{i}" for i in range(1, 8)], 8),
                "start_s": np.tile(np.arange(8) * 2, 7), "hr": np.arange(56)}
        first = select_pilot_rows(rows, 4)
        rows["hr"] = rows["hr"][::-1] * 100
        second = select_pilot_rows(rows, 4)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual({key: len(value) for key, value in first[2].items()},
                         {"train": 3, "validation": 1, "test": 2})
        self.assertEqual(len(first[0]), 24)
        self.assertEqual(len(set(first[0])), 24)

    def test_duplicate_sample_and_large_request_fail(self):
        rows = {"subject": np.repeat([f"S{i}" for i in range(6)], 2), "start_s": np.zeros(12)}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_pilot_rows(rows, 2)
        with self.assertRaisesRegex(ValueError, "1-48"):
            select_pilot_rows(rows, 49)

    def test_subject_reuse_in_a_different_split_fails(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_roles(["A", "B", "A"], ["train", "validation", "test"])

    def test_projection_does_not_refit_on_test_distribution(self):
        rng = np.random.default_rng(5)
        train = rng.normal(size=(40, 24))
        transforms = fit_projection(train)
        before = [(obj.mean_.copy(), obj.components_.copy() if hasattr(obj, "components_") else obj.scale_.copy())
                  for obj in transforms]
        ztrain = project(train, transforms)
        ztest = project(rng.normal(1000, 10, size=(7, 24)), transforms)
        self.assertEqual(ztest.shape, (7, 16))
        np.testing.assert_allclose(ztrain.mean(axis=0), 0., atol=1e-12)
        np.testing.assert_allclose(ztrain.std(axis=0), 1., atol=1e-12)
        for obj, (mean, state) in zip(transforms, before):
            np.testing.assert_array_equal(obj.mean_, mean)
            np.testing.assert_array_equal(obj.components_ if hasattr(obj, "components_") else obj.scale_, state)

    def test_small_head_is_fitted_and_has_fixed_parameter_count(self):
        rng = np.random.default_rng(2)
        train = rng.normal(size=(36, 16))
        validation = rng.normal(size=(12, 16))
        model, info = fit_small_head(train, 80 + train[:, 0] * 10, np.repeat(["A", "B", "C"], 12),
                                     validation, 80 + validation[:, 0] * 10, np.repeat("V", 12), epochs=3)
        self.assertEqual(info["parameters"], 289)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 289)
        self.assertNotEqual(info["initial_state_sha256"], info["selected_state_sha256"])
        self.assertIn(info["selected_epoch"], [1, 2, 3])

    def test_head_validation_rejects_leakage_and_nonfinite(self):
        x = np.ones((24, 16))
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_small_head(x, np.ones(24), ["A"] * 24, x, np.ones(24), ["A"] * 24)
        x[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_projection(x)

    def test_random_init_cannot_silently_target_another_architecture(self):
        with self.assertRaisesRegex(ValueError, "random architecture"):
            extract_features("SigmaPPG", np.ones((2, 400)), 50, random_init=True)


if __name__ == "__main__":
    unittest.main()
