"""A queue must not report completion from an incomplete adopted process."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.continue_ppg_bench import MODELS, complete_linear_scope


class QueueCompletionTests(unittest.TestCase):
    def scope(self):
        return {"scope_complete": True, "scope": {"datasets": ["bidmc"], "models": sorted(MODELS), "methods": ["linear"]}, "jobs": {f"bidmc__{name}__linear": {"status": "complete"} for name in MODELS}}

    def check_scope(self, value):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "run-status.json").write_text(json.dumps(value))
            return complete_linear_scope(target, "bidmc")

    def test_all_seven_models_are_required(self):
        value = self.scope()
        self.assertTrue(self.check_scope(value))
        value["scope"]["models"].pop()
        self.assertFalse(self.check_scope(value))

    def test_failed_job_is_not_completed_by_successful_exit(self):
        value = self.scope()
        next(iter(value["jobs"].values()))["status"] = "failed"
        self.assertFalse(self.check_scope(value))

    def test_missing_final_scope_flag_is_incomplete(self):
        value = self.scope()
        value.pop("scope_complete")
        self.assertFalse(self.check_scope(value))
