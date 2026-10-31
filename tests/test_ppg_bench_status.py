"""Progress remains accurate after interruption or delayed upstream summaries."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.ppg_bench_status import published_status


class PublishedStatusTests(unittest.TestCase):
    def test_stale_summary_does_not_hide_published_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run.json").write_text(json.dumps({"jobs": ["done"]}))
            (root / "status.json").write_text(json.dumps({"completed": 0, "planned": 1, "complete": False}))
            (root / "jobs/done").mkdir(parents=True)
            (root / "jobs/done/results.json").write_text("{}")
            status = published_status(root)
            self.assertEqual(status["completed"], 1)
            self.assertEqual(status["upstream_reported_completed"], 0)
            self.assertTrue(status["complete"])
            self.assertTrue(status["summary_may_lag"])

    def test_working_and_unknown_jobs_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run.json").write_text(json.dumps({"jobs": ["pending"]}))
            for name in ("pending.working", "unknown"):
                folder = root / "jobs" / name
                folder.mkdir(parents=True)
                (folder / "results.json").write_text("{}")
            status = published_status(root)
            self.assertEqual(status["completed"], 0)
            self.assertEqual(status["planned"], 1)
            self.assertFalse(status["complete"])

    def test_manifest_without_first_summary_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run.json").write_text(json.dumps({"jobs": ["pending"]}))
            status = published_status(root)
            self.assertEqual(status["completed"], 0)
            self.assertEqual(status["planned"], 1)
            self.assertIsNone(status["upstream_reported_completed"])


if __name__ == "__main__":
    unittest.main()
