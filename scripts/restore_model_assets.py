"""Verify model snapshots, or restore from an explicit offline source/Git history.

The Git-history source is a migration bridge for the already-tracked legacy
Watch-LSM checkpoint. New model binaries must remain outside Git.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from watch_lsm.paths import model_path
from scripts.sync_pulsedb import file_hash, safe_child


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="An offline project copy containing the manifest's relative paths")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data_manifests/model_assets.json", help="Core or PPG-Bench asset manifest")
    parser.add_argument("--only", action="append", help="Select a manifest path or basename; repeat for multiple assets")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = []
    selected = [entry for entry in manifest.get("files", manifest.get("new_files", [])) if not args.only
                or entry["path"] in args.only or Path(entry["path"]).name in args.only]
    if not selected:
        parser.error("No model manifest entries match --only")
    for entry in selected:
        relative = Path(entry["path"]).relative_to("models").as_posix()
        target = safe_child(model_path(), relative)
        if target.is_file() and target.stat().st_size == entry["bytes"] and file_hash(target) == entry["sha256"]:
            print("VERIFIED", entry["path"])
            continue
        if args.verify_only:
            errors.append(entry["path"])
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        digest = hashlib.sha256()
        process = None
        if args.source_root:
            stream = safe_child(args.source_root.resolve(), entry["path"]).open("rb")
        elif entry.get("legacy_git_source"):
            source = entry["legacy_git_source"]
            process = subprocess.Popen(["git", "show", f"{source['revision']}:{source['path']}"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stream = process.stdout
        else:
            errors.append(entry["path"] + " (provide --source-root; provenance: " + entry.get("source_record", entry.get("source", "manifest")) + ")")
            continue
        with stream, partial.open("wb") as dest:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
                dest.write(block)
            dest.flush()
            os.fsync(dest.fileno())
        if process is not None and process.wait() != 0:
            raise RuntimeError("Legacy Git object unavailable; use a full clone or explicit offline source")
        if partial.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Source does not match model manifest: {entry['path']}")
        if file_hash(partial) != entry["sha256"]:
            raise ValueError("Model destination checksum mismatch")
        os.replace(partial, target)
        print("RESTORED", entry["path"])
    if errors:
        print("Missing or mismatched assets:", *errors, sep="\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
