"""Download public PPG-Bench datasets with durable per-dataset status.

DREAMT is deliberately excluded: the user must obtain project access first.
Existing unrelated datasets and invalid historical archives are not modified.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ("dalia", "bidmc", "butppg", "ecsmp", "wesad", "wildppg")


def resilient_download(dataset, directory):
    """Retain upstream sizes/checksums while retrying transient publisher errors."""
    import ppg_bench.download as upstream
    original_open = upstream.open_url
    original_fetch = upstream.fetch

    def open_checked(url, headers=None):
        for attempt in range(5):
            query = "&" if "?" in url else "?"
            request_url = url if attempt == 0 else f"{url}{query}watch_lsm_retry={time.time_ns()}"
            try:
                response = original_open(request_url, headers)
                if "data.mendeley.com/public-api/" not in url:
                    return response
                with response:
                    body = response.read()
                payload = json.loads(body)
                if not isinstance(payload, list):
                    raise ValueError(f"Unexpected publisher API response: {body[:160]!r}")
                return io.BytesIO(body)
            except urllib.error.HTTPError as error:
                if error.code in (401, 403, 404) or attempt == 4:
                    raise
            except (urllib.error.URLError, TimeoutError, ValueError):
                if attempt == 4:
                    raise
            time.sleep(attempt + 1)

    def fetch_retry(url, destination, size=None, checksum=None, headers=None):
        for attempt in range(5):
            query = "&" if "?" in url else "?"
            request_url = url if attempt == 0 else f"{url}{query}watch_lsm_retry={time.time_ns()}"
            try:
                return original_fetch(request_url, destination, size, checksum, headers)
            except urllib.error.HTTPError as error:
                if error.code in (401, 403, 404) or attempt == 4:
                    raise
            except (urllib.error.URLError, TimeoutError, ValueError, OSError):
                if attempt == 4:
                    raise
            time.sleep(attempt + 1)

    upstream.open_url = open_checked
    upstream.fetch = fetch_retry
    if dataset == "butppg":
        # Small record pairs benefit from bounded parallel I/O. Use exactly the
        # same quality filter, URLs and raw layout as the upstream downloader.
        directory.mkdir(parents=True, exist_ok=True)
        base = "https://physionet.org/files/butppg/2.0.0/"
        for name in ("subject-info.csv", "quality-hr-ann.csv"):
            fetch_retry(base + name, directory / name)
        with (directory / "quality-hr-ann.csv").open(encoding="utf-8-sig") as stream:
            records = [row["ID"] for row in csv.DictReader(stream) if int(row["Quality"]) == 1]

        def record_files(record):
            if not record.isdigit():
                raise ValueError(f"Unexpected record ID: {record}")
            for extension in ("dat", "hea"):
                name = f"{record}/{record}_PPG.{extension}"
                fetch_retry(base + name, directory / name)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(record_files, records))
        (directory / "source.json").write_text(json.dumps({"dataset": dataset, "source": upstream.SOURCES[dataset]}, indent=2))
    else:
        upstream.download_one(dataset, directory)


def process_alive(pid):
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 258
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=PUBLIC, default=list(PUBLIC))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/ppg_bench_raw")
    parser.add_argument("--state", type=Path, default=ROOT / ".local/ppg-bench-repro/downloads-v2.json")
    parser.add_argument("--adopt", action="append", default=[], help="Wait for an already running dataset=PID worker before retrying")
    parser.add_argument("--worker", choices=PUBLIC, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        resilient_download(args.worker, args.output / args.worker)
        return 0
    args.state.parent.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock
    from ppg_bench.download import SOURCES, fetch, unpack
    lock = FileLock(args.state.with_suffix(".lock"))
    lock.acquire(timeout=0)
    mutex = threading.Lock()
    adopted = {name: int(pid) for name, pid in (item.split("=", 1) for item in args.adopt)}
    state = {"datasets": {name: {"status": "pending"} for name in args.datasets}}

    def update(name, **fields):
        with mutex:
            state["datasets"][name].update(fields, updated_at=time.time())
            temporary = args.state.with_suffix(".tmp.json")
            temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temporary.replace(args.state)

    def one(name):
        directory = args.output / name
        source = directory / "source.json"
        if name in adopted:
            update(name, status="running", adopted_pid=adopted[name])
            while process_alive(adopted[name]):
                time.sleep(5)
            if source.is_file():
                update(name, status="downloaded", source=str(source.relative_to(ROOT)))
                return
        for attempt in range(1, 4):
            update(name, status="running", attempt=attempt)
            # A size-less upstream download can leave a truncated final ZIP.
            # Preserve it as a partial rather than allowing it to be skipped.
            for archive in directory.glob("*.zip"):
                if not zipfile.is_zipfile(archive):
                    partial = archive.with_name(archive.name + ".part")
                    if not partial.exists():
                        archive.replace(partial)
            log_path = args.state.parent / f"download-v2-{name}.log"
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", name, "--output", str(args.output)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                update(name, status="downloaded", source=str(source.relative_to(ROOT)))
                return
            if name == "wesad":
                # Same published dataset from its official UCI repository;
                # avoid disabling TLS verification for the author share.
                try:
                    url = "https://archive.ics.uci.edu/static/public/465/wesad+wearable+stress+and+affect+detection.zip"
                    archive = fetch(url, directory / "wesad-uci.zip")
                    unpack(archive, directory)
                    if not list(directory.rglob("S2.pkl")):
                        raise ValueError("The UCI archive does not contain WESAD subject data")
                    source.write_text(json.dumps({"dataset": name, "source": SOURCES[name], "download_source": url}, indent=2))
                    update(name, status="downloaded", download_source=url)
                    return
                except Exception as error:
                    update(name, last_error=str(error))
            update(name, status="retrying", exit_code=result.returncode)
        update(name, status="failed")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one, name): name for name in args.datasets}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as error:
                    update(name, status="failed", last_error=str(error))
                print(name, state["datasets"][name]["status"], flush=True)
    finally:
        lock.release()
    return int(any(v["status"] != "downloaded" for v in state["datasets"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
