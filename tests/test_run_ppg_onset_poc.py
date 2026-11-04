"""Scientific isolation and original-clock contracts for the manual-onset PoC."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts import run_ppg_onset_poc as runner


def record(subject="S1", name="S1_rest1", seconds=24, intervals=None):
    x = np.arange(seconds * 64, dtype=np.float64) + 10
    intervals = np.asarray([[0., float(seconds)]] if intervals is None else intervals, dtype=float).reshape(-1, 2)
    events = np.arange(2., seconds, 6.)
    valid = np.zeros(len(events), bool)
    for left, right in intervals:
        valid |= (events >= left) & (events < right)
    return {"subject": subject, "record_id": name, "activity": "rest", "fs": 64.,
            "duration_s": float(seconds), "raw_ppg": x, "input_ppg": x.copy(),
            "onset_times_s": events[valid], "annotated_intervals_s": intervals}


class TestManualOnsetRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.scratch = Path(__file__).resolve().parents[1] / ".local"
        cls.scratch.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_loso_split_has_seven_distinct_tests_and_independent_validation(self):
        jobs = runner.make_jobs([f"S{i}" for i in range(1, 8)])
        self.assertEqual([j["test"] for j in jobs], [[f"S{i}"] for i in range(1, 8)])
        for j in jobs:
            train, val, test = map(set, (j["train"], j["validation"], j["test"]))
            self.assertEqual((len(train), len(val), len(test)), (5, 1, 1))
            self.assertFalse(train & val or train & test or val & test)
            remaining = sorted(train | val)
            self.assertEqual(j["validation"], [np.random.default_rng(42 + j["fold"]).choice(remaining)])
        self.assertEqual(jobs, runner.make_jobs(reversed([f"S{i}" for i in range(1, 8)])))
        with self.assertRaises(ValueError):
            runner.make_jobs(["S1"] * 7)

    def test_complete_context_not_just_core_and_no_gap_or_record_join(self):
        r = record(intervals=[[0, 8], [9, 24]])
        np.testing.assert_array_equal(runner.full_frame_starts(r), [0., 6., 12.])
        np.testing.assert_array_equal(runner.annotated_frame_starts(r), [0., 12.])
        # The center [7,13) is annotated, but its surrounding [6,14) is not.
        np.testing.assert_array_equal(runner.annotated_frame_starts(record(intervals=[[7, 13]])), [])
        np.testing.assert_array_equal(runner.annotated_frame_starts(record(intervals=[])), [])
        np.testing.assert_array_equal(runner.full_frame_starts(record(seconds=7)), [])
        # The next acquisition always starts again at zero, not at prior end.
        np.testing.assert_array_equal(runner.annotated_frame_starts(record(name="S1_rest2", seconds=8)), [0.])

    def test_missing_samples_never_enter_labeled_context_and_are_not_hidden(self):
        r = record(intervals=[[0, 10], [10 + 1/64, 24]])
        r["raw_ppg"][640] = np.nan
        runner.validate_records([r])
        np.testing.assert_array_equal(runner.annotated_frame_starts(r), [0., 12.])
        digest = runner.record_digest(r)
        r["input_ppg"][640] += 1
        self.assertNotEqual(runner.record_digest(r), digest)
        r["annotated_intervals_s"] = np.array([[0., 24.]])
        with self.assertRaisesRegex(ValueError, "missing native"):
            runner.validate_records([r])
        r = record()
        r["input_ppg"][20] += 1
        with self.assertRaisesRegex(ValueError, "unchanged"):
            runner.validate_records([r])
        r = record(intervals=[])
        r["onset_times_s"] = np.array([2.])
        with self.assertRaisesRegex(ValueError, "outside annotation"):
            runner.validate_records([r])

    def test_sampling_is_person_bounded_and_deterministic(self):
        rows = {"subject": np.array(["trainA"] * 600 + ["trainB"] * 20 + ["val"] * 300 + ["test"] * 400)}
        train = runner.sample_person_indices(rows, ["trainA", "trainB"], 256)
        validation = runner.sample_person_indices(rows, ["val"], 128)
        self.assertEqual(len(train), 276)
        self.assertEqual(len(validation), 128)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(rows["subject"][train]), {"trainA", "trainB"})
        self.assertEqual(set(rows["subject"][validation]), {"val"})
        self.assertEqual((train[0], train[255]), (0, 599))
        with self.assertRaises(ValueError):
            runner.sample_person_indices(rows, ["absent"], 256)

    def test_cache_retains_record_clock_polarity_and_integrity(self):
        records = [record(intervals=[[0, 8], [9, 24]]), record(name="S1_rest2", seconds=8),
                   record(name="S1_rest3", seconds=8, intervals=[])]
        def stages(encoder, value):
            return [torch.zeros(len(value), c, 2) for c in (64, 160, 160, 400, 400, 512)]
        with tempfile.TemporaryDirectory(dir=self.scratch) as temporary:
            root = Path(temporary)
            (root / "protocol.json").write_text("{}")
            with patch.object(runner, "frozen_stages", side_effect=stages), patch("builtins.print"):
                runner.prepare_frames(root, records, None, "cpu")
                with np.load(root / "frames/rows.npz") as rows:
                    np.testing.assert_array_equal(rows["start_s"], [0., 12., 0.])
                    np.testing.assert_array_equal(rows["record_id"], ["S1_rest1", "S1_rest1", "S1_rest2"])
                    np.testing.assert_array_equal(rows["target"][:, 128], np.ones(3))
                cached = np.load(root / "frames/x64.npy")
                np.testing.assert_array_equal(cached[0], -records[0]["input_ppg"][:512])
                np.testing.assert_array_equal(cached[1], -records[0]["input_ppg"][768:1280])
                runner.prepare_frames(root, records, None, "cpu")
                np.save(root / "frames/x64.npy", cached + 1)
                with self.assertRaisesRegex(ValueError, "cache changed"):
                    runner.prepare_frames(root, records, None, "cpu")

    def test_validation_uses_only_requested_records_and_primary_50ms(self):
        r = record()
        rows = {"subject": np.array(["S1", "test"]), "record_id": np.array(["S1_rest1", "forbidden"]),
                "start_s": np.array([0., 0.])}
        probabilities = np.zeros((1, 512))
        probabilities[0, 132] = 1.  # 62.5ms late: fails 50ms, would pass old100ms.
        self.assertEqual(runner.frame_f1(probabilities, rows, np.array([0]), {"S1_rest1": r}, .5), 0.)
        probabilities[0, 132] = 0
        probabilities[0, 130] = 1
        self.assertEqual(runner.frame_f1(probabilities, rows, np.array([0]), {"S1_rest1": r}, .5), 1.)

    def test_inference_stitches_only_real_center_cores_on_original_clock(self):
        class FakeCNN(nn.Module):
            def forward(self, x):
                result = torch.full((len(x), 512), -20.)
                result[:, 128] = 20.
                return result
        with tempfile.TemporaryDirectory(dir=self.scratch) as temporary:
            root = Path(temporary)
            job = {"id": "fold", "test": ["S1"]}
            directory = root / "runs/fold/SmallCNN"
            directory.mkdir(parents=True)
            (directory / "fit.json").write_text("{}")
            r = record(intervals=[[0, 8], [9, 24]])
            runner.infer_record(root, job, r, "SmallCNN", FakeCNN(), {"threshold": .5}, None, "cpu")
            with np.load(directory / "S1_rest1.npz") as data:
                np.testing.assert_array_equal(data["event_times_s"], [2., 8., 14.])
                self.assertEqual(float(data["start_s"]), 1.)
                self.assertEqual(float(data["end_s"]), 19.)
                self.assertEqual(len(data["probability"]), 3 * 384)
            # Event8 remains in predictions, but is outside annotation coverage:
            # independent scoring, not the inference engine, applies that mask.
            meta = json.loads((directory / "S1_rest1.json").read_text())
            self.assertTrue(meta["annotation_mask_applied_at_scoring_only"])
            self.assertEqual(meta["identity"]["record_sha256"], runner.record_digest(r))

    def test_fit_never_consumes_test_waveforms_or_labels(self):
        seen, selected = [], []
        class TinyCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(.1))
            def forward(self, x):
                seen.append((self.training, x[:, 0, 0].detach().cpu().tolist()))
                return x[:, 0] * self.weight
        def validation(probability, rows, indices, references, threshold):
            selected.append(indices.tolist())
            return .5
        with tempfile.TemporaryDirectory(dir=self.scratch) as temporary:
            root = Path(temporary)
            (root / "frames").mkdir()
            (root / "protocol.json").write_text("{}")
            np.save(root / "frames/x64.npy", np.repeat(np.arange(1, 7, dtype=np.float32)[:, None], 512, axis=1))
            np.savez(root / "frames/rows.npz", subject=np.array(["train"] * 2 + ["val"] * 2 + ["test"] * 2),
                     record_id=np.array(["r"] * 6), start_s=np.arange(6) * 6., target=np.zeros((6, 512), np.float32))
            job = {"id": "test", "seed": 42, "train": ["train"], "validation": ["val"], "test": ["test"]}
            with patch.object(runner, "SmallEventCNN", TinyCNN), patch.object(runner, "frame_f1", side_effect=validation), patch("builtins.print"):
                _, fit = runner.fit_model(root, {}, job, "SmallCNN", "cpu")
        self.assertEqual(fit["train_frame_indices"], [0, 1])
        self.assertEqual(fit["validation_frame_indices"], [2, 3])
        self.assertTrue(all(s == [2, 3] for s in selected))
        self.assertTrue(all(set(values) <= ({1., 2.} if training else {3., 4.}) for training, values in seen))


if __name__ == "__main__":
    unittest.main()
