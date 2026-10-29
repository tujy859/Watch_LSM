"""Read-only PulseDB source inventory and local repair/coverage assessment.

Only file metadata is read from --source. Cloud MAT files are not opened or copied.
The source location is machine-specific and must be supplied explicitly.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import h5py


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def assess(source, output):
    source = source.resolve(strict=True)
    local_root = PROJECT_ROOT / "datasets" / "pulsedb"
    if source == local_root.resolve():
        raise ValueError("Source must be distinct from the local PulseDB directory")
    output.mkdir(parents=True, exist_ok=True)
    audit_path = PROJECT_ROOT / "docs" / "migration_audit_2026-10-28.json"
    previous = json.loads(audit_path.read_text(encoding="utf-8"))
    known_bad = {Path(row["path"]).relative_to("datasets/pulsedb").as_posix()
                 for row in previous["pulsedb_errors"]}
    inventory = []
    for folder in ["Info_Files", "Supplementary_Info_Files", "Segment_Files"]:
        print(f"Scanning source metadata: {folder}", flush=True)
        for path in sorted((source / folder).rglob("*.mat")):
            rel = path.relative_to(source).as_posix()
            source_bytes = path.stat().st_size
            target = local_root / rel
            local_bytes = target.stat().st_size if target.is_file() else None
            status = ("missing" if local_bytes is None else
                      "size_mismatch" if local_bytes != source_bytes else
                      "known_hdf5_error" if rel in known_bad else "same_size")
            inventory.append({"path": rel, "source_bytes": source_bytes,
                              "local_bytes": local_bytes, "status": status,
                              "previous_hdf5_error": rel in known_bad})
    by_path = {row["path"]: row for row in inventory}
    repair = [row for row in inventory if row["local_bytes"] is not None
              and row["status"] != "same_size"]
    names = []
    index_file = local_root / "Info_Files" / "AAMI_Test_Info.mat"
    with h5py.File(index_file, "r") as handle:
        group = handle["AAMI_Test_Subset"]
        def decode(ref):
            return "".join(chr(int(v)) for v in handle[ref][()].ravel(order="F"))
        subject_names = [decode(ref) for ref in group["Subj_Name"][()].ravel()]
        sources = [decode(ref) for ref in group["Source"][()].ravel()]
    missing_names = []
    for subject, origin in zip(subject_names, sources):
        folder = {"MIMIC": "PulseDB_MIMIC", "VitalDB": "PulseDB_Vital"}[origin]
        # File-level inference only; segment offsets are deliberately not assumed.
        exact = f"Segment_Files/{folder}/{subject}.mat"
        prefix = f"Segment_Files/{folder}/{subject.split('_')[0]}.mat"
        rel = exact if exact in by_path else prefix
        row = by_path.get(rel)
        if row is None:
            missing_names.append(subject)
        else:
            names.append(rel)
    aami_files = [by_path[rel] for rel in sorted(set(names))]
    aami_copy = [row for row in aami_files if row["status"] != "same_size"]
    def summarize(rows):
        return {"files": len(rows), "source_bytes": sum(r["source_bytes"] for r in rows),
                "local_bytes": sum(r["local_bytes"] or 0 for r in rows),
                "status_counts": dict(Counter(r["status"] for r in rows))}
    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source_kind": "user-specified mounted Google Drive folder",
        "scope": "Source metadata only; local HDF5 errors reused from migration audit. No source content validation or copy.",
        "all": summarize(inventory), "repair_existing": summarize(repair),
        "missing": summarize([r for r in inventory if r["status"] == "missing"]),
        "families": {folder: summarize([r for r in inventory if folder in Path(r["path"]).parts])
                     for folder in ["PulseDB_MIMIC", "PulseDB_Vital", "Info_Files", "Supplementary_Info_Files"]},
        "known_bad_not_in_source": sorted(known_bad - set(by_path)),
        "aami_inferred": {"index_entries": len(subject_names),
                          "source_counts": dict(Counter(sources)),
                          "mapped_entries": len(names), "unmapped_names": missing_names,
                          "needed_files": summarize(aami_files), "copy_candidates": summarize(aami_copy),
                          "note": "Exact filename, else first underscore prefix; segment offsets and index semantics require loader validation."},
        "local_disk_free_bytes": shutil.disk_usage(PROJECT_ROOT).free,
    }
    fields = ["path", "source_bytes", "local_bytes", "status", "previous_hdf5_error"]
    write_csv(output / "pulsedb_source_inventory.csv", inventory, fields)
    write_csv(output / "pulsedb_repair_candidates.csv", repair, fields)
    write_csv(output / "pulsedb_aami_candidates.csv", aami_copy, fields)
    (output / "pulsedb_source_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs" / "source_assessment_2026-10-28")
    args = parser.parse_args()
    assess(args.source, args.output)
