"""Compatibility entry point for verified copies of official index files."""
import sys
from sync_pulsedb import main

if __name__ == "__main__":
    sys.argv.extend(["--profile", "indices"])
    raise SystemExit(main())
