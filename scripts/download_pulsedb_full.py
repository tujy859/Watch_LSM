"""Legacy name: use explicit verified profiles; full-library download is disabled.

This entry point now uses the mounted source and the curated recovery manifest.
"""
from sync_pulsedb import main

if __name__ == "__main__":
    print("Full-library default retired. Select --profile repair, aami or repair-aami.")
    raise SystemExit(main())
