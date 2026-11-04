"""Guard feature cache identity and the cross-preprocessing row association."""
import unittest

import numpy as np

from scripts.ppg_hr_poc_features import identity_digest, row_mapping, verify_identity


class FrozenFeatureIntegrityTests(unittest.TestCase):
    def test_reordered_subject_start_mapping(self):
        old = [{"subject": "S1", "start_sec": "0"}, {"subject": "S2", "start_sec": "2"}]
        actual = row_mapping(old, np.array(["S2", "S1", "S2"]), np.array([0, 0, 2]))
        np.testing.assert_array_equal(actual, [1, 2])

    def test_duplicate_new_keys_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate PoC"):
            row_mapping([], np.array(["S1", "S1"]), np.array([0, 0]))

    def test_missing_legacy_window_fails(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            row_mapping([{"subject": "S2", "start_sec": 0}], np.array(["S1"]), np.array([0]))

    def test_changed_weight_or_data_identity_is_rejected(self):
        identity = {"weights": "original", "data": "original"}
        saved = {"identity": identity, "identity_sha256": identity_digest(identity)}
        verify_identity(saved, identity)
        with self.assertRaisesRegex(ValueError, "Cache identity differs"):
            verify_identity(saved, {**identity, "data": "different"})


if __name__ == "__main__":
    unittest.main()
