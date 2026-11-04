"""Protect scientific split and weighted linear-readout semantics."""
import sys
from pathlib import Path
import unittest

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
if not (Path(__file__).resolve().parents[1] / "repos/ppg-bench/ppg_bench/preprocess.py").is_file():
    raise unittest.SkipTest("Protocol integration requires the optional pinned PPG-Bench checkout")
from run_ppg_hr_poc import make_splits, validate_split, fit_ridge, weights, macro_mae


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.subjects = {"Dalia": [f"s{i}" for i in range(15)], "WildPPG": [f"w{i}" for i in range(16)]}

    def test_fold_coverage_nested_budgets_and_disjoint_people(self):
        jobs = make_splits(self.subjects)
        self.assertEqual(len(jobs), 93)
        for dataset in self.subjects:
            for seed in (42, 43, 44):
                full = [j for j in jobs if j["source"] == dataset and j["scenario"] == "within" and j["seed"] == seed and j["budget"] == "full"]
                seen = [s for j in full for s in j["test"]]
                self.assertEqual(sorted(seen), sorted(self.subjects[dataset]))
                for job in full:
                    validate_split(job)
                    group = [j for j in jobs if j["source"] == dataset and j["scenario"] == "within" and j["seed"] == seed and j["fold"] == job["fold"]]
                    for small, large in zip(group, group[1:]):
                        self.assertTrue(set(small["train"]) <= set(large["train"]))
                        self.assertTrue(set(small["validation"]) <= set(large["validation"]))
                        self.assertEqual(small["test"], large["test"])
                    self.assertEqual([len(g["validation"]) for g in group], [1, 2, 3])
                    self.assertEqual([len(g["train"]) + len(g["validation"]) for g in group[:2]], [3, 6])
        self.assertEqual(make_splits(self.subjects), jobs)

    def test_transfer_has_no_target_fit_people(self):
        jobs = [j for j in make_splits(self.subjects) if j["scenario"] == "transfer"]
        self.assertEqual(len(jobs), 3)
        for j in jobs:
            self.assertEqual(set(j["train"] + j["validation"]), set(self.subjects["Dalia"]))
            self.assertEqual(j["test"], sorted(self.subjects["WildPPG"]))
            self.assertEqual((len(j["train"]), len(j["validation"])), (12, 3))

    def test_rejects_leakage(self):
        j = make_splits(self.subjects)[0]
        j["train"].append(j["test"][0])
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_split(j)

    def test_ridge_matches_weighted_sklearn_including_constant_feature(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(27, 8)); x[:, 0] = 0
        y = 80 + 4 * x[:, 2] + rng.normal(size=27)
        ids = np.array(["a"] * 5 + ["b"] * 22)
        vx = rng.normal(size=(9, 8)); vx[:, 0] = 0
        vy = 80 + 4 * vx[:, 2]
        vi = np.array(["c"] * 4 + ["d"] * 5)
        coef, intercept, metadata = fit_ridge(x, y, ids, vx, vy, vi, alphas=(0.1, 2.))
        sw = weights(ids)
        scale = StandardScaler().fit(x, sample_weight=sw)
        expected = Ridge(alpha=metadata["alpha"]).fit(scale.transform(x), y, sample_weight=sw)
        np.testing.assert_allclose(vx @ coef + intercept, expected.predict(scale.transform(vx)), atol=1e-10)
        self.assertAlmostEqual(sw[ids == "a"].sum(), sw[ids == "b"].sum())
        self.assertAlmostEqual(metadata["validation_subject_mae_bpm"], macro_mae(vy, vx @ coef + intercept, vi))

    def test_macro_weights_people_not_window_count(self):
        self.assertEqual(macro_mae(np.array([0, 0, 0, 0]), np.array([10, 0, 0, 0]), np.array(["a", "b", "b", "b"])), 5.)


if __name__ == "__main__":
    unittest.main()
