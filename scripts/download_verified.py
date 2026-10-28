"""Resume independent HTTP ranges, then require a pinned SHA-256 before publishing."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import re
import subprocess
import time


def download(url, output, size, sha256, workers=8, chunk_mb=16):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = output.parent / (output.name + ".chunks")
    parts.mkdir(exist_ok=True)
    width = chunk_mb * 1024 * 1024
    ranges = [(i, start, min(start + width, size) - 1) for i, start in enumerate(range(0, size, width))]
    def get(item):
        index, start, end = item
        path = parts / str(index)
        expected = end - start + 1
        if path.is_file() and path.stat().st_size == expected:
            return path
        temp = path.with_suffix(".part")
        headers = path.with_suffix(".headers")
        for attempt in range(3):
            result = subprocess.run(["curl.exe" if os.name == "nt" else "curl", "-sS", "-L",
                                     "--connect-timeout", "15", "--max-time", "240", "--fail",
                                     "--range", f"{start}-{end}", "-D", str(headers), "-o", str(temp), url],
                                    capture_output=True, text=True)
            text = headers.read_text(errors="replace") if headers.exists() else ""
            matches = re.findall(r"content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)", text, re.I)
            if (result.returncode == 0 and temp.is_file() and temp.stat().st_size == expected
                    and matches and tuple(map(int, matches[-1])) == (start, end, size)):
                os.replace(temp, path)
                print(f"CHUNK {index + 1}/{len(ranges)} verified range {start}-{end}", flush=True)
                return path
            time.sleep(attempt + 1)
        raise IOError(f"Range {start}-{end} failed: {result.stderr[-300:]}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(get, item) for item in ranges]):
            future.result()
    digest = hashlib.sha256()
    partial = output.with_suffix(output.suffix + ".part")
    with partial.open("wb") as dest:
        for index, _, _ in ranges:
            with (parts / str(index)).open("rb") as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    digest.update(block)
                    dest.write(block)
        dest.flush()
        os.fsync(dest.fileno())
    if partial.stat().st_size != size or digest.hexdigest() != sha256:
        raise ValueError("Assembled download SHA-256 or size differs from trusted metadata")
    os.replace(partial, output)
    print(f"VERIFIED {output.name} sha256={sha256}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mb", type=int, default=16)
    args = parser.parse_args()
    download(args.url, args.output, args.size, args.sha256, args.workers, args.chunk_mb)
