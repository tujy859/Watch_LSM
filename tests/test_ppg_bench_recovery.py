"""Never publish incomplete benchmark jobs during Windows output recovery."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_ppg_bench import recover_completed_jobs


class BenchmarkRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "jobs").mkdir()
        (self.root / "run.json").write_text(json.dumps({"jobs": ["known-job"]}))

    def tearDown(self):
        self.temp.cleanup()

    def saved_job(self, name, prediction=True):
        folder = self.root / "jobs" / f"{name}.working"
        folder.mkdir()
        record = {"results": [{"predictions": "predictions_repeat0.npz"}]}
        (folder / "results.json").write_text(json.dumps(record))
        if prediction:
            (folder / "predictions_repeat0.npz").write_bytes(b"preserved-prediction-bytes")
        return folder

    def test_completed_result_is_published_without_changing_artifacts(self):
        folder = self.saved_job("known-job")
        original = {p.name: p.read_bytes() for p in folder.iterdir()}
        self.assertEqual(recover_completed_jobs(self.root), ["known-job"])
        final = self.root / "jobs/known-job"
        self.assertEqual({p.name: p.read_bytes() for p in final.iterdir()}, original)
        self.assertFalse(folder.exists())

    def test_missing_predictions_are_never_counted_as_complete(self):
        folder = self.saved_job("known-job", prediction=False)
        self.assertEqual(recover_completed_jobs(self.root), [])
        self.assertTrue(folder.exists())
        self.assertFalse((self.root / "jobs/known-job").exists())

    def test_jobs_outside_the_run_manifest_are_not_published(self):
        folder = self.saved_job("other-job")
        self.assertEqual(recover_completed_jobs(self.root), [])
        self.assertTrue(folder.exists())


if __name__ == "__main__":
    unittest.main()
