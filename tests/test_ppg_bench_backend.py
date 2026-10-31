"""Backend adaptation must not silently mix existing benchmark outputs."""
from pathlib import Path
import tempfile
import unittest

from scripts.run_ppg_bench import validate_backend_scope


class BackendScopeTests(unittest.TestCase):
    def test_new_output_can_select_either_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validate_backend_scope(root, "upstream")
            validate_backend_scope(root, "native")

    def test_original_results_cannot_be_resumed_as_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run.json").write_text("{}")
            with self.assertRaises(ValueError):
                validate_backend_scope(root, "native")

    def test_native_results_cannot_be_resumed_as_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "backend-profile.json").write_text("{}")
            (root / "run.json").write_text("{}")
            validate_backend_scope(root, "native")
            with self.assertRaises(ValueError):
                validate_backend_scope(root, "upstream")
