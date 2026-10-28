import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def sync(dest_dir):
    dest = Path(dest_dir).resolve()
    if dest == PROJECT_ROOT or dest.is_relative_to(PROJECT_ROOT) or PROJECT_ROOT.is_relative_to(dest):
        raise ValueError("Destination must be separate from the source tree")
    if sys.platform != "win32":
        raise RuntimeError("This offline asset-copy helper requires Windows robocopy")
    print(f"=================================================")
    print(f" Watch_LSM Fast Transfer Tool")
    print(f" Source:      {PROJECT_ROOT}")
    print(f" Destination: {dest}")
    print(f"=================================================")

    # Exclusion rules:
    # 1. .venv: contains 33,000+ tiny files that choke USB random write IOPS, and is non-portable across machines
    # 2. __pycache__, .git, *.tmp: cache/junk files
    exclude_dirs = [".venv", "venv", "__pycache__", ".git", ".local"]
    exclude_files = ["*.tmp", "*.pyc", "*.part", "watch_lsm.local.toml"]

    cmd = [
        "robocopy",
        str(PROJECT_ROOT),
        str(dest),
        "/E",
        "/MT:4",
        "/Z",
        "/XD", *exclude_dirs,
        "/XF", *exclude_files,
        "/R:1",
        "/W:1",
    ]

    print("Copying offline assets only. Use Git for code and rebuild venv on each machine.")
    res = subprocess.run(cmd)
    # robocopy exit codes: 0 = no change, 1 = files copied successfully, 2/3 = extra files / success
    if res.returncode < 8:
        print(f"\n[SUCCESS] Copy completed to {dest}; validate asset checksums separately.")
    else:
        raise RuntimeError(f"Robocopy failed with code {res.returncode}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = input("Enter destination directory (e.g. E:\\Watch_LSM): ").strip()

    if not target:
        print("Error: Destination path cannot be empty.")
        sys.exit(1)

    sync(target)
