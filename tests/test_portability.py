"""Portable paths and explicit device requests must survive a device migration."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from watch_lsm.paths import data_path, PROJECT_ROOT, project_path
from watch_lsm.runtime import select_device


class PortabilityTests(unittest.TestCase):
    def test_shared_locks_never_pin_local_wheels_or_hardware_builds(self):
        for path in (PROJECT_ROOT / "requirements/locks").glob("*.txt"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                self.assertNotIn(" @ ", line)
                self.assertNotIn("file:", line)
                name = line.split("==")[0].lower().replace("_", "-")
                self.assertNotEqual(name, "torch")
                self.assertFalse(name.startswith(("nvidia-", "triton", "pytorch-triton")))

    def test_default_paths_do_not_depend_on_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                with patch.dict(os.environ, {}, clear=True), patch("watch_lsm.paths.machine_config", return_value={}):
                    self.assertEqual(data_path("wrist"), PROJECT_ROOT / "datasets/wrist")
                    self.assertEqual(project_path("models"), PROJECT_ROOT / "models")
            finally:
                os.chdir(previous)

    def test_environment_override_takes_precedence(self):
        with patch.dict(os.environ, {"WATCH_LSM_DATASETS_DIR": "elsewhere"}):
            self.assertEqual(data_path(), PROJECT_ROOT / "elsewhere")

    def test_cpu_can_be_selected_on_gpu_machines(self):
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(select_device("cpu"), torch.device("cpu"))

    def test_unavailable_cuda_fails_instead_of_silently_falling_back(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA requested"):
                select_device("cuda")

    def test_auto_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(select_device("auto"), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
