"""Inspect only this machine's selected recovery profile and transfer journal."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from watch_lsm.paths import data_path
from scripts.sync_pulsedb import exclusive_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-hdf5", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/data-status.json")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data_manifests/pulsedb_recovery.json").read_text())
    entries = [e for e in manifest["files"] if {"repair", "aami"}.intersection(e["profiles"])]
    statuses = Counter()
    per_profile = {"repair": Counter(), "aami": Counter()}
    verified_bytes = 0
    failures = []
    for entry in entries:
        path = data_path("pulsedb", entry["path"])
        state_path = ROOT / ".local/pulsedb-sync" / (hashlib.sha256(entry["path"].encode()).hexdigest() + ".json")
        record = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if record.get("status") == "verified" and path.is_file() and path.stat().st_size == entry["bytes"]:
            status = "verified"
            verified_bytes += entry["bytes"]
        elif record.get("status") == "failed":
            status = "failed"
            failures.append({"path": entry["path"], "error": record["error"]})
        elif path.with_name(path.name + ".part").exists():
            status = "partial"
        else:
            status = "pending"
        statuses[status] += 1
        for profile in entry["profiles"]:
            per_profile[profile][status] += 1
    active = False
    lock_path = ROOT / ".local/pulsedb-sync/sync.lock"
    if lock_path.exists():
        try:
            with exclusive_lock(lock_path):
                pass
        except OSError:
            active = True
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "transfer_active": active,
              "files": len(entries), "status": dict(statuses), "profiles": per_profile,
              "verified_bytes": verified_bytes, "planned_bytes": sum(e["bytes"] for e in entries), "failures": failures}
    if args.verify_hdf5:
        import h5py
        errors = []
        files = list(data_path("pulsedb", "Segment_Files").rglob("*.mat"))
        for path in files:
            try:
                with h5py.File(path, "r") as handle:
                    list(handle.keys())
            except Exception as error:
                errors.append({"path": path.relative_to(data_path("pulsedb")).as_posix(), "error": str(error)})
        report["waveform_hdf5_open_check"] = {"files": len(files), "passed": len(files) - len(errors), "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
