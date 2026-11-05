"""Guard patient leakage, respiratory information, and formal input boundaries."""
import unittest

import numpy as np

from scripts.prepare_ppg_formal_clinical import patient_folds, respiration_modulation, spectral_respiration
from scripts.ppg_formal_clinical import observation_views, indices


class FormalClinicalTests(unittest.TestCase):
    def test_observation_views_cover_full_reference_interval(self):
        for seconds, starts in ((10, (0, 2)), (32, (0, 8, 16, 24))):
            x = np.arange(seconds * 125)[None, :]
            views = observation_views(x)
            self.assertEqual(views.shape, (1, len(starts), 1000))
            np.testing.assert_array_equal(np.unique(views), x[0])
            self.assertEqual(views[0, -1, -1], x[0, -1])
        with self.assertRaisesRegex(ValueError, "10 or 32"):
            observation_views(np.zeros((3, 1500)))

    def test_patient_split_rejects_same_person_different_record(self):
        subjects = np.array(["a", "a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "leakage"):
            indices(subjects, {"train": ["a"], "validation": ["b"], "test": ["a", "c"]})

    def test_duplicate_recordings_do_not_create_new_patient_folds(self):
        people = [f"person{i}" for i in range(46)] + ["person3"] * 10
        splits = patient_folds(people)
        seen = []
        for split in splits:
            train, validation, test = [set(split[name]) for name in ("train", "validation", "test")]
            self.assertFalse(train & test or train & validation or validation & test)
            self.assertEqual(len(train | validation | test), 46)
            seen.extend(test)
        self.assertEqual(len(seen), 46)
        self.assertEqual(len(set(seen)), 46)

    def test_fold_seed_changes_membership_reproducibly(self):
        people = [f"p{i}" for i in range(20)]
        self.assertEqual(patient_folds(people, 42), patient_folds(people[::-1], 42))
        self.assertNotEqual(patient_folds(people, 42), patient_folds(people, 43))

    def test_pure_respiration_frequency_is_recovered(self):
        t = np.arange(4000) / 125
        self.assertAlmostEqual(spectral_respiration(np.sin(2 * np.pi * .25 * t), 125), 15, delta=.8)

    def test_baseline_and_amplitude_modulation_are_detected_without_reference(self):
        t = np.arange(4000) / 125
        x = .3 * np.sin(2 * np.pi * .25 * t) + (1 + .4 * np.sin(2 * np.pi * .25 * t)) * np.sin(2 * np.pi * 1.3 * t)
        estimates = respiration_modulation(x)
        self.assertAlmostEqual(estimates[0], 15, delta=.8)
        self.assertAlmostEqual(estimates[1], 15, delta=1)

    def test_flat_signal_abstains_and_nonfinite_input_is_rejected(self):
        self.assertTrue(np.isnan(respiration_modulation(np.ones(4000))).all())
        x = np.ones(4000); x[100] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            respiration_modulation(x)


if __name__ == "__main__":
    unittest.main()
