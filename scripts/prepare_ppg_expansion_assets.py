"""Restore small public supplements used by the evaluation planning pilots.

This downloads data only. It never starts a benchmark or training job.
Existing files must agree with publisher checksums; mismatches are not replaced.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data_manifests/ppg_expansion_assets.json"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    if algorithm == "sha1":
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_destination(relative):
    destination = (ROOT / relative).resolve()
    if not destination.is_relative_to((ROOT / "datasets").resolve()):
        raise ValueError("Downloaded assets must stay in datasets")
    return destination


def fetch(item):
    destination = checked_destination(item["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    algorithm, expected = next((a, item[k]) for a, k in
                               [("sha256", "sha256"), ("sha1", "git_blob_sha1"), ("md5", "md5")]
                               if k in item)

    def validate(path):
        if "size" in item and path.stat().st_size != item["size"]:
            raise ValueError(f"Size differs: {item['path']}")
        if digest(path, algorithm) != expected:
            raise ValueError(f"Publisher checksum differs: {item['path']}")

    if destination.exists():
        validate(destination)
    else:
        partial = destination.with_name(destination.name + ".partial")
        with urllib.request.urlopen(item["url"], timeout=60) as response, partial.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        validate(partial)
        partial.replace(destination)
    result = {"path": item["path"], "url": item["url"], "bytes": destination.stat().st_size,
              "sha256": digest(destination), "publisher_checksum_verified": True}
    print(json.dumps({"verified": item["path"], "bytes": result["bytes"]}), flush=True)
    return result


def bidmc_items(info):
    url = info["base"] + "SHA256SUMS.txt"
    with urllib.request.urlopen(url, timeout=45) as response:
        raw = response.read()
    checksum_path = checked_destination(info["output"] + "/SHA256SUMS-expansion.txt")
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    if checksum_path.exists() and checksum_path.read_bytes() != raw:
        raise ValueError("Publisher checksum manifest changed")
    checksum_path.write_bytes(raw)
    checksums = {}
    for line in raw.decode().splitlines():
        value, relative = line.split(maxsplit=1)
        checksums[relative.lstrip("*").removeprefix("./")] = value
    result = []
    for subject in range(1, 54):
        for suffix in ("Breaths.csv", "Fix.txt"):
            relative = f"bidmc_csv/bidmc_{subject:02d}_{suffix}"
            result.append({"path": info["output"] + "/" + relative,
                           "url": info["base"] + relative, "sha256": checksums[relative]})
    return result


def extract_epfl(info):
    archive = checked_destination(info["path"])
    destination = archive.parent / "extracted"
    names = subprocess.check_output(["tar", "-tf", str(archive)], text=True).splitlines()
    for name in names:
        parts = PurePosixPath(name.replace("\\", "/")).parts
        if not parts or ".." in parts or parts[0].endswith(":") or name.startswith(("/", "\\")):
            raise ValueError("Unsafe archive member")
    if not destination.exists():
        destination.mkdir()
        subprocess.run(["tar", "-xf", str(archive), "-C", str(destination)], check=True)
    missing = [name for name in names if not (destination / name).exists()]
    if missing:
        raise ValueError(f"Incomplete extracted archive: {missing[:3]}")
    return {"archive_entries": len(names), "extracted_to": destination.relative_to(ROOT).as_posix(),
            "archive_md5": digest(archive, "md5")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["bidmc", "segade", "epfl"], required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four download workers")
    manifest = json.loads(MANIFEST.read_text())
    items = []
    for name in args.datasets:
        if name == "bidmc":
            items.extend(bidmc_items(manifest[name]))
        elif name == "segade":
            items.extend(manifest[name]["files"])
        else:
            items.append(manifest[name])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outputs = list(pool.map(fetch, items))
    report = {"datasets": args.datasets, "files": outputs, "bytes": sum(x["bytes"] for x in outputs)}
    if "epfl" in args.datasets:
        report["epfl_extraction"] = extract_epfl(manifest["epfl"])
    output = ROOT / ".local/ppg-expansion-pilot"
    output.mkdir(parents=True, exist_ok=True)
    (output / ("assets-" + "-".join(args.datasets) + ".json")).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files_verified": len(outputs), "bytes": report["bytes"]}), flush=True)


if __name__ == "__main__":
    main()
