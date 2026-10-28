"""Compatibility entry point: verified profiles replace unsafe full sync."""
from sync_pulsedb import main

if __name__ == "__main__":
    print("Using verified profiles (default: repair, never full-library).")
    raise SystemExit(main())
