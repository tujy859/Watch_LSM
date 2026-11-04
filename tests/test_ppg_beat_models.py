"""Dense-event head contracts and subject/time isolation, using synthetic data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts import ppg_beat_models as models
from scripts import run_ppg_beat_poc as runner


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv = nn.Conv1d(1, 16, 3, padding=1)
        self.first_bn = nn.BatchNorm1d(16)
        self.first_activation = nn.ReLU()
        channels = (16, *models.STAGE_CHANNELS)
        self.stage_list = nn.ModuleList([
            nn.Sequential(nn.Conv1d(a, b, 1, stride=2), nn.BatchNorm1d(b), nn.ReLU())
            for a, b in zip(channels[:-1], channels[1:])])


class TestBeatModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.scratch = Path(__file__).resolve().parents[1] / ".local"
        cls.scratch.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_frozen_encoder_bn_and_gradient_are_unchanged(self):
        encoder = FakeEncoder().eval().requires_grad_(False)
        state = {k: v.clone() for k, v in encoder.state_dict().items()}
        x = torch.randn(2, 1, 1000, requires_grad=True)
        stages = models.frozen_stages(encoder, x)
        self.assertEqual([z.shape[1] for z in stages], list(models.STAGE_CHANNELS))
        self.assertTrue(all(not z.requires_grad for z in stages))
        head = models.AnyPPGEventHead()
        logits = head(stages)
        self.assertEqual(logits.shape, (2, 512))
        logits.sum().backward()
        self.assertTrue(any(p.grad is not None for p in head.parameters()))
        self.assertIsNone(x.grad)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        for key, value in encoder.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        encoder.train()
        with self.assertRaisesRegex(ValueError, "frozen"):
            models.frozen_stages(encoder, x)
        encoder.eval().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            models.frozen_stages(encoder, x)

    def test_small_cnn_shape_finite_and_declared_parameter_budget(self):
        network = models.SmallEventCNN()
        output = network(torch.randn(3, 1, 512))
        self.assertEqual(output.shape, (3, 512))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(models.parameter_counts(), {"AnyPPG_head": 26737, "SmallCNN": 17353})
        self.assertTrue(torch.isfinite(network(torch.ones(1, 1, 512))).all())

    def test_soft_target_keeps_global_clock_and_event_width(self):
        target = models.soft_events(np.array([10., 10.5]), np.array([8., 10., 20.]))
        self.assertEqual(target.shape, (3, 512))
        self.assertEqual(target.dtype, np.float32)
        self.assertEqual(target[0, 128], 1.)
        self.assertEqual(target[0, 160], 1.)
        self.assertEqual(target[1, 0], 1.)
        self.assertAlmostEqual(float(target[0, 129]), np.exp(-.5 * ((1/64) / .04) ** 2), places=6)
        self.assertEqual(float(target[2].max()), 0.)
        self.assertTrue(((target >= 0) & (target <= 1)).all())

    def test_frame_cache_time_alignment_and_integrity(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            (root / "protocol.json").write_text("{}")
            reference = Path(temporary) / "reference"
            (reference / "Dalia/S1").mkdir(parents=True)
            np.savez(reference / "Dalia/S1/beats.npz", duplicate_event_times_s=np.array([]))
            x = np.arange(24 * 64, dtype=np.float32)
            peaks = np.array([2., 8., 14.])

            def stages(encoder, value):
                return [torch.zeros(len(value), c, 2) for c in models.STAGE_CHANNELS]

            with patch.object(runner, "load_reference", return_value=(x, 64., peaks)), \
                 patch.object(runner, "frozen_stages", side_effect=stages), patch("builtins.print"):
                runner.prepare_frames(root, reference, None, "cpu")
                with np.load(root / "frames/rows.npz") as rows:
                    np.testing.assert_array_equal(rows["start_s"], [0., 6., 12.])
                    np.testing.assert_array_equal(rows["subject"], ["S1"] * 3)
                    np.testing.assert_array_equal(rows["target"][:, 128], [1., 1., 1.])
                saved = np.load(root / "frames/x64.npy")
                np.testing.assert_array_equal(saved[1], x[384:896])
                np.testing.assert_allclose(np.load(root / "frames/x125.npy")[0], runner.resample(x, 64., 125)[:1000])
                runner.prepare_frames(root, reference, None, "cpu")
                np.save(root / "frames/x64.npy", saved + 1)
                with self.assertRaisesRegex(ValueError, "cache changed"):
                    runner.prepare_frames(root, reference, None, "cpu")

    def test_validation_event_selection_never_reads_test_references(self):
        rows = {"subject": np.array(["test", "valA", "valB"]), "start_s": np.array([0., 0., 10.])}
        probability = np.zeros((2, 512))
        probability[:, 128] = 1.
        seen = []

        def reference(root, dataset, subject):
            seen.append(subject)
            return np.zeros(128), 64., np.array([2., 12.])

        with patch.object(runner, "load_reference", side_effect=reference):
            value = runner.frame_f1(probability, rows, np.array([1, 2]), Path("unused"), .5)
        self.assertEqual(value, 1.)
        self.assertEqual(seen, ["valA", "valB"])

    def test_fit_training_and_validation_access_exclude_test_rows(self):
        calls, selected = [], []

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(.1))

            def forward(self, x):
                calls.append((self.training, x[:, 0, 0].detach().cpu().tolist()))
                return x[:, 0] * self.weight

        def validation(probabilities, rows, indices, reference, threshold):
            selected.append(indices.tolist())
            return .5

        with tempfile.TemporaryDirectory(dir=self.scratch) as temporary:
            root = Path(temporary)
            (root / "frames").mkdir()
            (root / "protocol.json").write_text("{}")
            np.save(root / "frames/x64.npy", np.repeat(np.arange(1, 7, dtype=np.float32)[:, None], 512, axis=1))
            np.savez(root / "frames/rows.npz", subject=np.array(["train"] * 2 + ["val"] * 2 + ["test"] * 2),
                     start_s=np.arange(6) * 6., target=np.zeros((6, 512), np.float32))
            job = {"id": "synthetic", "seed": 42, "train": ["train"], "validation": ["val"], "test": ["test"]}
            with patch.object(runner, "SmallEventCNN", TinyModel), patch.object(runner, "frame_f1", side_effect=validation), patch("builtins.print"):
                _, fit = runner.fit_model(root, Path("unused"), job, "SmallCNN", "cpu")
        self.assertEqual(fit["train_frames"], 2)
        self.assertEqual(fit["validation_frames"], 1)
        self.assertTrue(all(s == [2] for s in selected))
        self.assertTrue(all(set(values) <= ({1., 2.} if training else {3.}) for training, values in calls))

    def test_registered_six_jobs_use_only_source_labels_and_heldout_people(self):
        source = [f"S{i}" for i in range(1, 16)]
        target = [f"w{i}" for i in range(16)]
        jobs = []
        for fold in range(5):
            test = source[3 * fold:3 * fold + 3]
            others = [s for s in source if s not in test]
            jobs.append({"id": f"fold{fold}", "scenario": "within", "source": "Dalia", "target": "Dalia", "budget": "full", "seed": 42,
                         "fold": fold, "train": others[3:], "validation": others[:3], "test": test})
        jobs.append({"id": "transfer", "scenario": "transfer", "source": "Dalia", "target": "WildPPG", "budget": "full", "seed": 42,
                     "fold": -1, "train": source[3:], "validation": source[:3], "test": target})
        with tempfile.TemporaryDirectory(dir=self.scratch) as temporary:
            root = Path(temporary)
            phase = root / ".local/ppg-hr-poc"
            phase.mkdir(parents=True)
            (phase / "protocol.json").write_text(json.dumps({"jobs": jobs}))
            reference = root / "reference"
            for dataset, subjects in (("Dalia", source), ("WildPPG", target)):
                for subject in subjects:
                    directory = reference / dataset / subject
                    directory.mkdir(parents=True)
                    (directory / "manifest.json").write_text(json.dumps({"outputs_sha256": {"beats.npz": "synthetic-digest", "ppg_native.npy": "synthetic-digest"}}))
            with patch.object(runner, "ROOT", root), patch.object(runner, "sha256", return_value="synthetic-digest"):
                protocol = runner.make_protocol(reference)
                altered = json.loads(json.dumps(jobs))
                altered[0]["train"][0] = altered[0]["test"][0]
                (phase / "protocol.json").write_text(json.dumps({"jobs": altered}))
                with self.assertRaises(ValueError):
                    runner.make_protocol(reference)
                (phase / "protocol.json").write_text(json.dumps({"jobs": jobs}))
                (reference / "Dalia/S1/manifest.json").write_text(json.dumps({"outputs_sha256": {"beats.npz": "wrong-digest"}}))
                with self.assertRaisesRegex(ValueError, "modified"):
                    runner.make_protocol(reference)
        self.assertEqual(len(protocol["jobs"]), 6)
        for job in protocol["jobs"]:
            self.assertFalse(set(job["train"]) & set(job["validation"]))
            self.assertTrue(set(job["train"]) | set(job["validation"]) <= set(source))
            if job["scenario"] == "within":
                self.assertFalse((set(job["train"]) | set(job["validation"])) & set(job["test"]))
                self.assertEqual(len(job["train"]), 9)
            else:
                self.assertEqual(len(job["train"]), 12)
                self.assertEqual(set(job["test"]), set(target))


if __name__ == "__main__":
    unittest.main()
