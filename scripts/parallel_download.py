"""Legacy command name for resumable, SHA-256-pinned range downloads.

Unverified preallocation-based downloads have been retired.
"""
import argparse
from pathlib import Path
from download_verified import download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mb", type=int, default=16)
    args = parser.parse_args()
    download(args.url, args.output, args.size, args.sha256, args.workers, args.chunk_mb)
