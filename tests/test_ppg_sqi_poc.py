"""Evidence for physical decoding, participant isolation and SQI scoring."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.ppg_sqi_poc import (
    FEATURE_NAMES, choose_threshold, decode_ppg, fit_logistic, handcrafted,
    make_jobs, metrics, weights,
)


class TestSQIData(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[1] / ".local"
        self.scratch.mkdir(exist_ok=True)

    def test_legacy_channels_encode_time_in_gain_not_zero_adc(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            directory = root / "100001"
            directory.mkdir()
            np.zeros(300, dtype="<i2").tofile(directory / "100001_PPG.dat")
            expected = np.linspace(-254, -253, 300)
            lines = ["100001_PPG 300 30 1"]
            lines += [f"100001_PPG.dat 16 {32767 / v}(-32767)/a.u. 0 0 0 0 0" for v in expected]
            (directory / "100001_PPG.hea").write_text("\n".join(lines))
            actual, meta = decode_ppg(root, "100001")
            np.testing.assert_allclose(actual, expected)
            self.assertTrue(meta["digital_constant"])
            self.assertFalse(meta["physical_constant"])

    def test_rgb_uses_red_physical_units(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            directory = root / "112001"
            directory.mkdir()
            digital = np.stack([np.arange(300), np.arange(300)*2, np.arange(300)*3], axis=1).astype("<i2")
            digital.tofile(directory / "112001_PPG.dat")
            lines = ["112001_PPG 3 30 300"]
            lines += [f"112001_PPG.dat 16 2(-100)/a.u. 0 0 0 0 0 PPG_{c}" for c in "RGB"]
            (directory / "112001_PPG.hea").write_text("\n".join(lines))
            actual, meta = decode_ppg(root, "112001")
            np.testing.assert_array_equal(actual, (np.arange(300)+100)/2)
            self.assertEqual(meta["channel"], "PPG_R")

    def test_features_retain_flat_data_and_reject_nan(self):
        t = np.arange(300) / 30
        x = np.stack([np.zeros(300), np.sin(2*np.pi*1.2*t)])
        f = handcrafted(x)
        self.assertEqual(f.shape, (2, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(f).all())
        self.assertEqual(f[0, 0], 1)
        self.assertEqual(f[1, 0], 0)
        self.assertAlmostEqual(f[1, FEATURE_NAMES.index("dominant_frequency")], 1.2, delta=.1)
        with self.assertRaisesRegex(ValueError, "finite"):
            handcrafted(np.full((1, 300), np.nan))


class TestSQIProtocol(unittest.TestCase):
    def test_split_isolation_and_exact_held_out_coverage(self):
        people = [str(i) for i in range(50)]
        jobs = make_jobs(people)
        self.assertEqual(len(jobs), 15)
        for job in jobs:
            tr, va, te = [set(job[k]) for k in ("train", "validation", "test")]
            self.assertEqual((len(tr), len(va), len(te)), (32, 8, 10))
            self.assertFalse(tr & va or tr & te or va & te)
            self.assertEqual(tr | va | te, set(people))
        for seed in (42, 43, 44):
            self.assertEqual(sorted(p for j in jobs if j["seed"] == seed for p in j["test"]), sorted(people))

    def test_subject_weights_prevent_record_count_domination(self):
        subject = np.array(["a"]*9 + ["b"])
        w = weights(subject)
        self.assertAlmostEqual(w[:9].sum(), w[9])
        y = np.array([0]*9 + [1])
        score = metrics(y, np.array([.1]*9 + [.9]), sample_weight=w)
        self.assertEqual(score["auroc"], 1)
        self.assertEqual(score["balanced_accuracy"], 1)
        self.assertEqual(score["good_F1"], 1)
        self.assertEqual(choose_threshold(y, np.ones(10)*.2, w), .5)

    def test_holdout_labels_cannot_affect_fit_or_threshold(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(60, 4))
        y = (x[:, 0] > 0).astype(int)
        subjects = np.repeat(["a", "b", "c", "d", "e", "f"], 10)
        tr, va = np.arange(40), np.arange(40, 50)
        changed = y.copy()
        changed[50:] = 1 - y[50:]
        p, info = fit_logistic(x, y, tr, va, subjects)
        p_changed, info_changed = fit_logistic(x, changed, tr, va, subjects)
        np.testing.assert_array_equal(p, p_changed)
        self.assertEqual(info["threshold"], info_changed["threshold"])


if __name__ == "__main__":
    unittest.main()
