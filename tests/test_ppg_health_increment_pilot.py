"""Verify person isolation, raw continuity, and training-only health transforms."""
from io import BytesIO
import unittest
from zipfile import ZipFile

import numpy as np

from scripts.ppg_health_increment_pilot import (
    apply_preprocessor, check_roles, continuous_core, deterministic_split,
    feature_groups, fit_preprocessor, pulse_features, ridge_check, workbook_rows,
)


class HealthIncrementPilotTests(unittest.TestCase):
    def test_subject_split_is_disjoint_and_order_independent(self):
        ids = [str(i) for i in range(30)]
        split = deterministic_split(ids, (18, 6, 6))
        self.assertEqual(split, deterministic_split(ids[::-1], (18, 6, 6)))
        merged = sum(split.values(), [])
        self.assertEqual(len(merged), len(set(merged)))
        with self.assertRaisesRegex(ValueError, "distinct"):
            deterministic_split(ids + ["1"], (18, 6, 6))

    def test_duplicate_person_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            check_roles(["p1", "p1", "p2"], ["train", "validation", "test"])

    def test_raw_continuity_and_padding_are_explicit(self):
        signal = np.arange(2000, dtype=float)
        core, padded, identity = continuous_core(signal, 10, 10, duration_s=120, pad_s=2)
        np.testing.assert_array_equal(core, signal[100:1300])
        np.testing.assert_array_equal(padded, signal[80:1320])
        self.assertEqual(identity["stop_sample"] - identity["start_sample"], 1200)
        signal[400] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            continuous_core(signal, 10, 10)
        with self.assertRaisesRegex(ValueError, "padding"):
            continuous_core(np.zeros(1300), 10, 0)

    def test_constant_rate_pulses_have_small_prv(self):
        fs = 64
        x = np.sin(2 * np.pi * np.arange(124 * fs) / fs)
        features = pulse_features(x, fs)
        self.assertAlmostEqual(features[0], 60., delta=.1)
        self.assertLess(features[3], 2.)

    def test_missing_feature_imputation_uses_training_only(self):
        train = np.asarray([[1., np.nan], [3., np.nan], [np.nan, np.nan]])
        transform = fit_preprocessor(train)
        np.testing.assert_array_equal(transform[0], [2., 0.])
        external = apply_preprocessor([[999., 100.], [np.nan, np.nan]], transform)
        self.assertTrue(np.isfinite(external).all())
        np.testing.assert_array_equal(transform[0], [2., 0.])

    def test_test_targets_do_not_select_alpha_or_modify_predictions(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(24, 4))
        y = x[:, 0] * 3 + rng.normal(size=24)
        roles = np.asarray(["train"] * 14 + ["validation"] * 5 + ["test"] * 5)
        a = ridge_check(x, y, roles)
        altered = y.copy(); altered[roles == "test"] += 1000
        b = ridge_check(x, altered, roles)
        self.assertEqual(a["alpha"], b["alpha"])
        np.testing.assert_array_equal(a["test_prediction"], b["test_prediction"])
        self.assertGreater(b["test_mae"], a["test_mae"] + 900)

    def test_increment_groups_are_nested(self):
        cov = np.zeros((5, 3)); traditional = np.ones((5, 11)); embedding = np.full((5, 8), 2.)
        groups = feature_groups(cov, traditional, embedding)
        self.assertEqual(groups["demographics_traditional_embedding"].shape, (5, 22))
        np.testing.assert_array_equal(groups["demographics_traditional_embedding"][:, :14], groups["demographics_traditional"])
        np.testing.assert_array_equal(groups["demographics_embedding"][:, 3:], embedding)

    def test_xlsx_preserves_shared_strings_and_sparse_columns(self):
        stream = BytesIO()
        ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
        with ZipFile(stream, "w") as z:
            z.writestr("xl/sharedStrings.xml", f'<sst {ns}><si><t>Female</t></si></sst>')
            z.writestr("xl/worksheets/sheet1.xml", f'<worksheet {ns}><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="AA1"><v>42.5</v></c></row></sheetData></worksheet>')
        stream.seek(0)
        self.assertEqual(workbook_rows(stream), [{0: "Female", 26: "42.5"}])


if __name__ == "__main__":
    unittest.main()
