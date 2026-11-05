"""Check at most 24 SEGADE/DaLiA annotation rows before proposing subject splits.

The downloaded rows lack subject IDs. Labels never participate in waveform
matching. Only complete 1920-point equality (absolute tolerance 1e-10) is
accepted; correlation alone cannot transfer a label to a raw recording.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from scripts.ppg_hr_poc_features import sha256, write_json
from scripts.ppg_hr_baselines import predict_hr_windows

N_SAMPLES = 1920
FS = 64
ATOL = 1e-10
UPSTREAM = "https://github.com/chengstark/Segade/blob/f5b2ad6e180a1474bc47d6c5ec5d482edaed6797/"


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    low, span = values.min(axis=-1, keepdims=True), np.ptp(values, axis=-1, keepdims=True)
    return (values-low)/np.where(span > 0, span, 1.)


def raw_affine_matches(trace, query, atol=ATOL):
    """Search all sample offsets using an affine-invariant candidate triple.

    Candidate tolerance is intentionally looser than final tolerance. Full
    min-max equality rejects mere periodic/correlation similarity.
    """
    trace, query = np.asarray(trace), np.asarray(query)
    if query.ndim != 1 or len(query) < 3 or np.ptp(query) == 0:
        raise ValueError("Need a nonconstant query")
    anchor = int(np.argmax(np.abs(np.diff(query)[:-1])))
    denominator = query[anchor+1]-query[anchor]
    if abs(denominator) <= 1e-15:
        raise ValueError("Degenerate matching anchor")
    ratio = (query[anchor+2]-query[anchor])/denominator
    first, second = trace[1:-1]-trace[:-2], trace[2:]-trace[:-2]
    tolerance = 1e-8 * (1 + np.abs(first) + np.abs(second))
    candidates = np.flatnonzero((np.abs(second-ratio*first) <= tolerance) & (np.abs(first) > 1e-12))-anchor
    candidates = candidates[(candidates >= 0) & (candidates+len(query) <= len(trace))]
    found = []
    for start in candidates:
        raw = trace[start:start+len(query)]
        error = float(np.max(np.abs(minmax(raw)-query)))
        if error <= atol:
            found.append({"start_sample": int(start), "method": "raw_window_minmax", "max_absolute_error": error})
    return found


def grid_matches(trace, queries, atol=ATOL):
    """Try the explicitly defined upstream filter on the natural 30-s grid.

    These are restricted hypotheses, not an exhaustive search over filters or
    time shifts. An unmatched result remains unmatched.
    """
    trace = np.asarray(trace, dtype=np.float64)
    n = len(trace)//N_SAMPLES
    raw = trace[:n*N_SAMPLES].reshape(n, N_SAMPLES)
    sos = butter(2, [0.9/(FS/2), 5/(FS/2)], btype="bandpass", output="sos")
    candidates = {
        "raw_window_minmax_grid": minmax(raw),
        "upstream_0.9_5hz_causal_filter_per_30s_then_minmax": minmax(sosfilt(sos, raw, axis=-1)),
        "upstream_0.9_5hz_causal_filter_continuous_then_window_minmax": minmax(sosfilt(sos, trace)[:n*N_SAMPLES].reshape(n, N_SAMPLES)),
    }
    matches = [[] for _ in queries]
    for method, blocks in candidates.items():
        for qi, query in enumerate(queries):
            # The initial check is only a speed-up; accepted matches must pass
            # every original point, without lag or fitted amplitude parameters.
            possible = np.flatnonzero(np.max(np.abs(blocks[:, :32]-query[:32]), axis=1) <= atol)
            for index in possible:
                error = float(np.max(np.abs(blocks[index]-query)))
                if error <= atol:
                    matches[qi].append({"start_sample": int(index*N_SAMPLES), "method": method, "max_absolute_error": error})
    return matches


def validate_annotation_pair(waveforms, labels):
    if waveforms.shape != labels.shape or waveforms.ndim != 2 or waveforms.shape[1] != N_SAMPLES:
        raise ValueError("Expected aligned N x 1920 waveform and point-label arrays")
    if not np.isfinite(waveforms).all() or not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Nonfinite waveform or nonbinary point labels")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-group", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.rows_per_group <= 12:
        raise ValueError("Pilot limit is twelve rows from each group")
    local = ROOT / ".local/ppg-expansion-pilot/artifact-reference"
    shared = ROOT / "docs/results/ppg_expansion_pilot"
    local.mkdir(parents=True, exist_ok=True)
    shared.mkdir(parents=True, exist_ok=True)
    queries, labels, identities, files, inventory = [], [], [], [], []
    for group in ("train", "test"):
        base = ROOT / "datasets/segade" / f"new_PPG_DaLiA_{group}/processed_dataset"
        waves = np.load(base / "scaled_ppgs.npy", mmap_mode="r")
        masks = np.load(base / "seg_labels.npy", mmap_mode="r")
        validate_annotation_pair(waves, masks)
        selected = np.linspace(0, len(waves)-1, min(args.rows_per_group, len(waves)), dtype=int)
        for index in selected:
            queries.append(np.asarray(waves[index]).copy())
            labels.append(np.asarray(masks[index]).copy())
            identities.append({"published_group": group, "published_row": int(index)})
        inventory.append({"published_group": group, "rows": len(waves), "samples_per_row": N_SAMPLES,
                          "artifact_points": int(masks.sum()), "all_rows_min0_max1": bool(np.all(waves.min(axis=1)==0) and np.all(waves.max(axis=1)==1))})
        files.extend({"path": (base/name).relative_to(ROOT).as_posix(), "sha256": sha256(base/name)} for name in ("scaled_ppgs.npy", "seg_labels.npy"))
    queries, labels = np.asarray(queries), np.asarray(labels)
    matches = [[] for _ in queries]
    source_files = []
    for path in sorted((ROOT / "datasets/ppg_bench_raw/dalia/PPG_FieldStudy").glob("S*/S*.pkl")):
        with path.open("rb") as stream:
            source = pickle.load(stream, encoding="latin1")
        trace = np.asarray(source["signal"]["wrist"]["BVP"]).ravel()
        del source
        grid = grid_matches(trace, queries)
        for i, query in enumerate(queries):
            found = raw_affine_matches(trace, query) + grid[i]
            for item in found:
                matches[i].append({"subject": path.parent.name, **item, "start_seconds": item["start_sample"]/FS})
        source_files.append({"subject": path.parent.name, "path": path.relative_to(ROOT).as_posix(),
                             "sha256": sha256(path), "wrist_samples": len(trace)})
        print(json.dumps({"subject_searched": path.parent.name, "matched_queries_so_far": sum(bool(m) for m in matches)}), flush=True)
    if len(source_files) != 15:
        raise ValueError("Need all fifteen raw DaLiA subjects to test uniqueness")
    mapped = []
    for identity, found, mask in zip(identities, matches, labels):
        unique = sorted(set((item["subject"], item["start_sample"]) for item in found))
        mapped.append({**identity, "unique_raw_locations": len(unique), "matches": found,
                       "artifact_fraction": float(np.mean(mask)), "status": "unique" if len(unique)==1 else "unresolved" if not unique else "ambiguous"})
    short = np.concatenate([q[:1536].reshape(3, 512) for q in queries])
    short_masks = np.concatenate([m[:1536].reshape(3, 512) for m in labels])
    outputs = predict_hr_windows(short, FS)
    short_rows = []
    for i, item in enumerate(mapped):
        for section in range(3):
            j = i*3+section
            short_rows.append({"published_group": item["published_group"], "published_row": item["published_row"],
                               "offset_seconds": section*8, "artifact_fraction": float(short_masks[j].mean()),
                               "welch_hr_bpm": float(outputs["welch"][j]), "elgendi_hr_bpm": float(outputs["elgendi"][j]),
                               "reference_identity_status": item["status"]})
    with (local / "short_window_outputs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(short_rows[0]))
        writer.writeheader()
        writer.writerows(short_rows)
    np.savez_compressed(local / "selected_waveforms_labels.npz", waveforms=queries, labels=labels)
    write_json(local / "mapping.json", mapped)
    unique_count = sum(item["status"] == "unique" for item in mapped)
    group_subjects = {group: sorted(set(m["subject"] for item in mapped if item["published_group"]==group and item["status"]=="unique" for m in item["matches"])) for group in ("train", "test")}
    report = {"status": "sample_mapping_passed" if unique_count==len(mapped) else "reference_mapping_incomplete",
              "purpose": "small annotation identity and traditional-output workflow check, not artifact detection efficacy",
              "upstream_commit": "f5b2ad6e180a1474bc47d6c5ec5d482edaed6797", "sources": [UPSTREAM+"README.md", UPSTREAM+"db2npy.py", UPSTREAM+"data_split.py"],
              "downloaded_arrays": inventory, "selected_rows": len(mapped), "uniquely_mapped_rows": unique_count,
              "ambiguous_rows": sum(item["status"]=="ambiguous" for item in mapped), "sampled_group_subjects": group_subjects,
              "sampled_group_subject_overlap": sorted(set(group_subjects["train"]) & set(group_subjects["test"])),
              "selected_mapping": mapped, "point_label_definition": "1=human-marked artifact, 0=not marked; db2npy floor(start), ceil(end) at 64Hz",
              "matching_rule": "all 1920 points within absolute 1e-10; raw minmax exhaustive sample offsets, two explicit upstream-filter variants on a 30-second grid only",
              "waveform_preprocessing_observation": "Every downloaded row has exactly min=0, max=1; db2npy writes database values unchanged, so it alone does not establish upstream raw-waveform preprocessing.",
              "eight_second_windows_checked": len(short), "traditional_output_coverage": {k: float(np.isfinite(v).mean()) for k,v in outputs.items()},
              "model_fits": 0, "new_subject_independent_evaluation_started": False,
              "source_file_hashes": source_files, "annotation_file_hashes": files,
              "script_sha256": sha256(Path(__file__)), "local_mapping_sha256": sha256(local / "mapping.json"),
              "limitations": ["Only sampled rows were investigated; full subject identity and temporal overlap audit remains required.",
                              "Official data_split.py performs KFold over array rows; do not reuse it as subject-independent validation.",
                              "Traditional HR output is a pipeline check, not an artifact classifier score or ECG-validated HR metric.",
                              "No approximate correlation match is accepted as annotation provenance."]}
    write_json(shared / "artifact_reference_pilot.json", report)
    print(json.dumps({"status": report["status"], "selected_rows": len(mapped), "uniquely_mapped": unique_count,
                      "short_windows": len(short)}), flush=True)


if __name__ == "__main__":
    main()
