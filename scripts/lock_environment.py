"""Export verified package versions without leaking local wheel URLs into Git."""

import argparse
import importlib.metadata
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-shared", action="store_true", help="Update this OS/Python's shared common lock after validation")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error("Export from the isolated project venv")
    rows = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"].lower().replace("_", "-")
        if name in ["watch-lsm", "torch", "pip", "setuptools", "wheel"]:
            continue
        if name.startswith(("nvidia-", "triton", "pytorch-triton")):
            continue
        rows[name] = distribution.version
    text = "# Verified common versions; install torch/accelerator dependencies via their hardware profile.\n"
    text += "\n".join(f"{name}=={version}" for name, version in sorted(rows.items())) + "\n"
    suffix = f"{platform.system().lower()}-py{sys.version_info.major}{sys.version_info.minor}-common.txt"
    output = ROOT / ("requirements/locks" if args.write_shared else ".local") / suffix
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Exported {len(rows)} portable version pins: {output}")


if __name__ == "__main__":
    main()
