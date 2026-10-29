"""Copy only an explicit PulseDB profile, with resume and verified replacement.

Source stays read-only. State belongs to this machine under .local/, not Git.
Run with --dry-run first to inspect the plan. No full-library default exists.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from watch_lsm.paths import data_path, pulsedb_source
CHUNK = 4 * 1024 * 1024
PRINT_LOCK = threading.Lock()


def log(message):
    with PRINT_LOCK:
        print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def safe_child(root, relative):
    """Reject traversal and absolute paths before any read/write."""
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe manifest path: {relative}")
    child = (root / path).resolve()
    if not child.is_relative_to(root.resolve()) or child == root.resolve():
        raise ValueError(f"Manifest path escapes root: {relative}")
    return child


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_mat(path, waveform=True):
    """Read actual first and last waveform/label cells, not just HDF5 header."""
    import h5py
    import numpy as np
    with h5py.File(path, "r") as handle:
        if not waveform:
            roots = [key for key in handle if not key.startswith("#")]
            if not roots:
                raise ValueError("Index has no data root")
            return
        group = handle["Subj_Wins"]
        for field in ["PPG_Raw", "ECG_Raw", "ABP_Raw", "SegSBP", "SegDBP"]:
            cells = group[field]
            if not cells.size:
                raise ValueError(f"Empty {field}")
            for index in [0, cells.shape[-1] - 1]:
                value = handle[cells[0, index]][()]
                if not value.size or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {field} at {index}")


@contextmanager
def exclusive_lock(path):
    """OS-held lock is released on process exit, including interrupted jobs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def copy_one(entry, source_root, target_root, state_root, retries=6):
    relative = entry["path"]
    source = safe_child(source_root, relative)
    target = safe_child(target_root, relative)
    expected = entry["bytes"]
    record = state_root / (hashlib.sha256(relative.encode()).hexdigest() + ".json")
    try:
        if record.is_file() and target.is_file() and target.stat().st_size == expected:
            old = json.loads(record.read_text(encoding="utf-8"))
            if old.get("status") == "verified" and file_hash(target) == old.get("sha256"):
                validate_mat(target, waveform=relative.startswith("Segment_Files/"))
                log(f"SKIP verified {relative}")
                return old
    except (OSError, ValueError, KeyError) as error:
        log(f"Rechecking source for {relative}: local validation/state failed: {error}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    atomic_json(record, {"path": relative, "status": "copying"})
    for attempt in range(1, retries + 1):
        try:
            info = source.stat()
            if info.st_size != expected:
                raise ValueError(f"Source size {info.st_size} != expected {expected}")
            offset = partial.stat().st_size if partial.is_file() else 0
            if offset > expected:
                raise ValueError(f"Partial larger than expected: {partial.name}")
            digest = hashlib.sha256()
            started = time.monotonic()
            log(f"COPY {relative} bytes={expected} resume={offset} attempt={attempt}")
            with source.open("rb") as src:
                if offset:
                    prefix = hashlib.sha256()
                    remaining = offset
                    with partial.open("rb") as previous:
                        while remaining:
                            block = src.read(min(CHUNK, remaining))
                            if not block:
                                raise IOError("Source ended during resume verification")
                            digest.update(block)
                            prefix.update(previous.read(len(block)))
                            remaining -= len(block)
                    if prefix.hexdigest() != digest.hexdigest():
                        raise ValueError("Partial prefix differs from source; preserve and inspect it")
                last_report = time.monotonic()
                with partial.open("ab" if offset else "wb") as dest:
                    for block in iter(lambda: src.read(CHUNK), b""):
                        dest.write(block)
                        digest.update(block)
                        offset += len(block)
                        if time.monotonic() - last_report > 20:
                            dest.flush()
                            log(f"PROGRESS {relative} {offset}/{expected}")
                            last_report = time.monotonic()
                    dest.flush()
                    os.fsync(dest.fileno())
            current = source.stat()
            if (current.st_size, current.st_mtime_ns) != (info.st_size, info.st_mtime_ns):
                raise ValueError("Source changed during copy")
            if partial.stat().st_size != expected:
                raise ValueError("Copied size differs from manifest")
            sha = digest.hexdigest()
            if file_hash(partial) != sha:
                raise ValueError("Destination SHA-256 differs from source stream")
            if entry.get("sha256") and entry["sha256"] != sha:
                raise ValueError("Source differs from pinned manifest SHA-256")
            validate_mat(partial, waveform=relative.startswith("Segment_Files/"))
            # Old damaged file survives until every validation above succeeds.
            os.replace(partial, target)
            result = {"path": relative, "bytes": expected, "sha256": sha,
                      "status": "verified", "seconds": time.monotonic() - started,
                      "verified_at": datetime.now(timezone.utc).isoformat()}
            atomic_json(record, result)
            log(f"VERIFIED {relative} seconds={result['seconds']:.1f}")
            return result
        except Exception as error:
            log(f"RETRY {relative} {type(error).__name__}: {error}")
            if attempt == retries:
                result = {"path": relative, "status": "failed", "error": str(error)}
                atomic_json(record, result)
                return result
            time.sleep(min(2 * attempt, 5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Override this machine's configured read-only cloud source")
    parser.add_argument("--destination", type=Path, default=data_path("pulsedb"))
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data_manifests/pulsedb_recovery.json")
    parser.add_argument("--profile", choices=["repair", "aami", "repair-aami", "indices"], default="repair")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--retries", type=int, choices=range(1, 21), default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state-dir", type=Path, default=PROJECT_ROOT / ".local/pulsedb-sync")
    args = parser.parse_args()
    source = (args.source or pulsedb_source()).resolve(strict=True)
    target = args.destination.resolve()
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Source and destination must be separate directories")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = {"repair", "aami"} if args.profile == "repair-aami" else {args.profile}
    entries = [entry for entry in manifest["files"] if selected.intersection(entry["profiles"])]
    # Repairs first, smallest first within a profile: early verification and feedback.
    entries.sort(key=lambda e: ("repair" not in e["profiles"], e["bytes"], e["path"]))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        entries = entries[:args.limit]
    if not entries:
        raise ValueError("Manifest contains no files for the requested profile")
    for entry in entries:
        safe_child(source, entry["path"])
        safe_child(target, entry["path"])
    log(f"PLAN files={len(entries)} bytes={sum(e['bytes'] for e in entries)} profile={args.profile}")
    if args.dry_run:
        return 0
    args.state_dir.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    additional = sum(max(0, e["bytes"] - (safe_child(target, e["path"]).stat().st_size
                     if safe_child(target, e["path"]).is_file() else 0)) for e in entries)
    temporary = sum(sorted((e["bytes"] for e in entries), reverse=True)[:args.workers])
    if shutil.disk_usage(target).free < additional + temporary:
        raise OSError("Insufficient disk space including temporary files")
    with exclusive_lock(args.state_dir / "sync.lock"):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(copy_one, e, source, target, args.state_dir, args.retries) for e in entries]
            results = [future.result() for future in as_completed(futures)]
        summary = {"finished_at": datetime.now(timezone.utc).isoformat(),
                   "profile": args.profile, "files": len(entries),
                   "verified": sum(r["status"] == "verified" for r in results),
                   "failed": sum(r["status"] != "verified" for r in results)}
        atomic_json(args.state_dir / "summary.json", summary)
        log(f"FINISHED {summary}")
        return int(summary["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
