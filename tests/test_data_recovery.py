"""Test interruption recovery and preservation of existing data on validation failure."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from scripts.sync_pulsedb import copy_one, safe_child, exclusive_lock


def write_waveform(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("Subj_Wins")
        for name in ["PPG_Raw", "ECG_Raw", "ABP_Raw", "SegSBP", "SegDBP"]:
            raw = handle.create_dataset(name + "_data", data=np.ones((1, 1250 if name.endswith("Raw") else 1)))
            refs = group.create_dataset(name, (1, 1), dtype=h5py.ref_dtype)
            refs[0, 0] = raw.ref


class RecoveryTests(unittest.TestCase):
    def test_fresh_machine_aami_profile_is_complete_not_only_recovery_delta(self):
        root = Path(__file__).resolve().parents[1]
        recovery = json.loads((root / "data_manifests/pulsedb_recovery.json").read_text(encoding="utf-8"))
        full = json.loads((root / "data_manifests/pulsedb_aami.json").read_text(encoding="utf-8"))
        delta_paths = {entry["path"] for entry in recovery["files"] if "aami" in entry["profiles"]}
        all_paths = {entry["path"] for entry in full["files"]}
        self.assertTrue(delta_paths < all_paths)
        self.assertEqual(len([p for p in all_paths if p.startswith("Segment_Files/")]), 242)
        self.assertIn("Info_Files/AAMI_Test_Info.mat", all_paths)
        for entry in full["files"]:
            safe_child(self.target, entry["path"])
            self.assertGreater(entry["bytes"], 0)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base / "source"
        self.target = self.base / "target"
        self.state = self.base / "state"
        self.state.mkdir()
        self.relative = "Segment_Files/PulseDB_MIMIC/example.mat"
        self.source_file = self.source / self.relative
        self.target_file = self.target / self.relative
        write_waveform(self.source_file)
        self.target_file.parent.mkdir(parents=True)
        self.target_file.write_bytes(b"old truncated content")
        self.entry = {"path": self.relative, "bytes": self.source_file.stat().st_size}

    def tearDown(self):
        self.temp.cleanup()

    def test_resumes_and_verifies_before_replacing(self):
        partial = self.target_file.with_name(self.target_file.name + ".part")
        partial.write_bytes(self.source_file.read_bytes()[:1024])
        result = copy_one(self.entry, self.source, self.target, self.state, retries=1)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(self.target_file.read_bytes(), self.source_file.read_bytes())
        self.assertFalse(partial.exists())
        again = copy_one(self.entry, self.source, self.target, self.state, retries=1)
        self.assertEqual(again["sha256"], result["sha256"])

    def test_invalid_source_never_replaces_existing_file(self):
        self.source_file.write_bytes(b"not hdf5")
        self.entry["bytes"] = self.source_file.stat().st_size
        result = copy_one(self.entry, self.source, self.target, self.state, retries=1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.target_file.read_bytes(), b"old truncated content")

    def test_corrupt_partial_is_preserved_for_inspection(self):
        partial = self.target_file.with_name(self.target_file.name + ".part")
        partial.write_bytes(b"wrong prefix")
        result = copy_one(self.entry, self.source, self.target, self.state, retries=1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.target_file.read_bytes(), b"old truncated content")
        self.assertEqual(partial.read_bytes(), b"wrong prefix")

    def test_source_change_from_pinned_digest_is_rejected(self):
        self.entry["sha256"] = "0" * 64
        result = copy_one(self.entry, self.source, self.target, self.state, retries=1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.target_file.read_bytes(), b"old truncated content")

    def test_manifest_path_cannot_escape_source_or_target(self):
        for path in ["../outside.mat", "/outside.mat", "C:/outside.mat", "..\\outside.mat"]:
            with self.assertRaises(ValueError):
                safe_child(self.target, path)

    def test_parallel_process_lock_rejects_a_second_writer(self):
        path = self.state / "writer.lock"
        with exclusive_lock(path):
            with self.assertRaises(OSError):
                with exclusive_lock(path):
                    pass


if __name__ == "__main__":
    unittest.main()
