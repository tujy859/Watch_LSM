"""Bounded PulseDB loading / regression pilot; never a full AAMI evaluation.

Select people through official index positions and local availability only.
Train_Info supplies disjoint fit/validation people; CalFree supplies test.
The small AAMI read audit is separate and is never used for model selection.
Outputs retain row identity so the extraction can be independently replayed.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import signal, stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def select_available_people(root, info_file, n_people, n_segments, seed, source="MIMIC"):
    """Bounded metadata sampling, independent of BP labels and waveform quality."""
    from watch_lsm.data.pulsedb import PulseDBIndex, segment_file, subject_identity
    rng = np.random.default_rng(seed)
    rows, selected = [], set()
    candidates = 0
    with PulseDBIndex(root / "Info_Files" / info_file) as index:
        # A bounded sample is intentional: this is not a coverage audit of all rows.
        positions = rng.choice(index.length, size=min(4000, index.length), replace=False)
        for position in positions:
            candidates += 1
            name = index.subject(int(position))
            origin, patient = subject_identity(name)
            if (origin != source or patient in selected
                    or not segment_file(root, name).is_file()):
                continue
            # Official subject rows are contiguous in the files inspected. Check,
            # do not assume, identity for every neighbouring row we select.
            neighbours = [int(position)]
            for step in range(1, n_segments * 2):
                for offset in (step, -step):
                    trial = int(position) + offset
                    if 0 <= trial < index.length and index.subject(trial) == name:
                        neighbours.append(trial)
                    if len(neighbours) == n_segments:
                        break
                if len(neighbours) == n_segments:
                    break
            if len(neighbours) != n_segments:
                continue
            chosen = [index.row(i) for i in sorted(neighbours)]
            if len({row.segment_position_one_based for row in chosen}) != n_segments:
                raise ValueError("Duplicate segment positions within selected subject")
            rows.extend(chosen)
            selected.add(patient)
            if len(selected) == n_people:
                break
        total_rows = index.length
    if len(selected) != n_people:
        raise RuntimeError(f"Only {len(selected)}/{n_people} available {source} people in bounded search")
    return rows, {"official_rows": total_rows, "candidate_rows_examined": candidates,
                  "selected_people": len(selected), "source": source}


def ppg_features(ppg, fs=125):
    """Small PPG-only spectral/shape baseline; no ABP, ECG or demographics."""
    x = np.asarray(ppg, dtype=float)
    x = (x - x.mean()) / max(x.std(), 1e-8)
    frequencies, power = signal.welch(x, fs=fs, nperseg=min(len(x), 512))
    cardiac = (frequencies >= 0.5) & (frequencies <= 4)
    peak = frequencies[cardiac][np.argmax(power[cardiac])]
    total = max(float(power.sum()), 1e-12)
    bands = [float(power[(frequencies >= lo) & (frequencies < hi)].sum()) / total
             for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8)]]
    dx = np.diff(x)
    result = [peak * 60, stats.skew(x), stats.kurtosis(x),
              np.mean(np.abs(dx)), np.std(dx), np.std(np.diff(dx)),
              *np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95]), *bands]
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite PPG-only feature")
    return result


def person_mae(y, predicted, people):
    return float(np.mean([np.mean(np.abs(y[people == p] - predicted[people == p]))
                          for p in np.unique(people)]))


def fit_heads(features, labels, split, people):
    """Select alpha on held-out validation people; test never enters fitting."""
    fit, val, test = (split == s for s in ("train", "validation", "test"))
    results, predictions = [], {}
    for col, target in enumerate(["sbp", "dbp"]):
        y = labels[:, col]
        candidates = []
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(features[fit], y[fit])
            loss = person_mae(y[val], model.predict(features[val]), people[val])
            candidates.append((loss, alpha, model))
        loss, alpha, model = min(candidates, key=lambda item: (item[0], item[1]))
        predicted = model.predict(features[test])
        constant = np.full(test.sum(), y[fit].mean())
        predictions[target] = predicted
        predictions[target + "_train_mean"] = constant
        results.extend([
            {"target": target, "method": "PPG_features_Ridge", "alpha": alpha,
             "validation_person_mae_mmhg": loss,
             "test_person_mae_mmhg": person_mae(y[test], predicted, people[test]),
             "test_window_mae_mmhg": float(np.mean(np.abs(y[test] - predicted)))},
            {"target": target, "method": "Train_mean", "alpha": None,
             "validation_person_mae_mmhg": None,
             "test_person_mae_mmhg": person_mae(y[test], constant, people[test]),
             "test_window_mae_mmhg": float(np.mean(np.abs(y[test] - constant)))},
        ])
    return results, predictions


def foundation_pilot(names, signals, labels, splits, people, output, device):
    """Optional three-encoder interface check on the exact same selected rows."""
    from scripts.ppg_model_expansion_pilot import extract_features
    results = []
    for name in names:
        features, evidence = extract_features(name, np.asarray(signals), 125,
                                              device=device, batch_size=16)
        fitted, predictions = fit_heads(features, labels, splits, people)
        fitted = [row for row in fitted if row["method"] == "PPG_features_Ridge"]
        for row in fitted:
            row["method"] = name + "_frozen_Ridge"
        np.savez_compressed(output / (name + "_features_predictions.npz"),
                            features=features, **predictions)
        results.append({"model": name, "evidence": evidence, "results": fitted})
    return results


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_prepared_arrays(output, report, arrays):
    """Verify cached row, label and waveform identity without reopening big MATs."""
    rows_path = output / "selected_rows.json"
    if hashlib.sha256(rows_path.read_bytes()).hexdigest() != report["selected_rows_sha256"]:
        raise ValueError("Selected official-row manifest changed")
    manifest = json.loads(rows_path.read_text(encoding="utf-8"))
    if len(manifest) != len(arrays["ppg"]):
        raise ValueError("Prepared waveform count differs from manifest")
    for i, row in enumerate(manifest):
        if (arrays["patient_id"][i] != row["patient_id"] or arrays["split"][i] != row["split"]
                or not np.array_equal(arrays["labels"][i], [row["sbp"], row["dbp"]])
                or hashlib.sha256(arrays["ppg"][i].tobytes()).hexdigest() != row["ppg_sha256"]):
            raise ValueError(f"Prepared sample {i} differs from official-row extraction")
    return len(manifest)


def add_foundation_check(args, report, arrays):
    """This step can run in the existing model venv, which lacks h5py."""
    report["foundation_input_identity_checked_rows"] = validate_prepared_arrays(args.output, report, arrays)
    report["foundation_interface_check"] = foundation_pilot(
        args.foundation_models, arrays["ppg"], arrays["labels"], arrays["split"],
        arrays["patient_id"], args.output.resolve(), args.device)
    report["pretrained_encoder_used"] = True
    exposure_note = "The reported test subset is now exposed by this pilot; later design choices must not treat it as a fresh blind holdout"
    if exposure_note not in report["limitations"]:
        report["limitations"].append(exposure_note)
    report["foundation_input_note"] = (
        "PPG_F was already filtered/normalized by PulseDB, then processed by each native model port. "
        "This checks execution, not the optimal preprocessing protocol. Actual native_input_samples/native_used_samples "
        "are reported; at 10 seconds SigmaPPG uses 500/500 samples at 50 Hz, with no trailing crop. "
        "The three encoders use identical fit/validation/test rows and Ridge alpha candidates; native "
        "embedding sizes differ, unlike the controlled task-head experiment in the separate model pilot.")
    write_json(args.output / "report.json", report)
    if args.shared_report:
        write_json(args.shared_report, report)


def run(args):
    if args.foundation_only:
        if not args.foundation_models:
            raise ValueError("--foundation-only requires --foundation-models")
        report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
        with np.load(args.output / "selected_segments.npz", allow_pickle=False) as arrays:
            add_foundation_check(args, report, arrays)
        print(json.dumps({"foundation_models": args.foundation_models,
                          "report": str(args.output / "report.json")}, indent=2))
        return
    from watch_lsm.data.pulsedb import (
        PulseDBIndex, UPSTREAM_COMMIT, assert_patient_disjoint, read_segment, subject_identity,
    )
    root, output = args.root.resolve(), args.output.resolve()
    if not (1 <= args.segments_per_person <= 16 and 2 <= args.train_people <= 16
            and 1 <= args.validation_people <= 8 and 1 <= args.test_people <= 8):
        raise ValueError("This pilot is deliberately bounded; full experiments need a separate approved protocol")
    output.mkdir(parents=True, exist_ok=True)
    train_rows, train_search = select_available_people(
        root, "Train_Info.mat", args.train_people + args.validation_people,
        args.segments_per_person, args.seed)
    names = list(dict.fromkeys(row.patient_id for row in train_rows))
    train_names = set(names[:args.train_people])
    test_rows, test_search = select_available_people(
        root, "CalFree_Test_Info.mat", args.test_people,
        args.segments_per_person, args.seed + 1)
    partitions = {"train": [r for r in train_rows if r.patient_id in train_names],
                  "validation": [r for r in train_rows if r.patient_id not in train_names],
                  "test": test_rows}
    identities = assert_patient_disjoint(partitions)
    manifest, features, labels, signals, splits, people = [], [], [], [], [], []
    mapping_differences = 0
    for name, rows in partitions.items():
        for row in rows:
            segment = read_segment(root, row, include_reference_signals=True)
            manifest.append({"split": name, **row.to_dict(),
                             "segment_file": segment["file"].relative_to(root).as_posix(),
                             "original_segment_id": segment["segment_id"],
                             "time_first": float(segment["T"][0]),
                             "ppg_sha256": hashlib.sha256(segment["PPG_F"].tobytes()).hexdigest()})
            mapping_differences += segment["segment_id"] != row.segment_position_one_based
            signals.append(segment["PPG_F"])
            features.append(ppg_features(segment["PPG_F"]))
            labels.append([row.sbp, row.dbp]); splits.append(name); people.append(row.patient_id)
    features, labels, splits, people = map(np.asarray, [features, labels, splits, people])
    results, predictions = fit_heads(features, labels, splits, people)
    np.savez_compressed(output / "selected_segments.npz", ppg=np.asarray(signals),
                        features=features, labels=labels, split=splits, patient_id=people,
                        fs=np.asarray(125), **predictions)
    write_json(output / "selected_rows.json", manifest)
    # AAMI is inspected only for mapping consistency, never as development/test input.
    aami_audit = []
    with PulseDBIndex(root / "Info_Files" / "AAMI_Test_Info.mat") as index:
        seen_sources = set()
        for position in range(index.length):
            name = index.subject(position); source, _ = subject_identity(name)
            if source in seen_sources:
                continue
            row = index.row(position)
            segment = read_segment(root, row, include_reference_signals=True)
            aami_audit.append({"source": source, "info_row_zero_based": position,
                               "patient_id": row.patient_id,
                               "segment_position_one_based": row.segment_position_one_based,
                               "labels_and_waveform_checked": True})
            seen_sources.add(source)
            if len(seen_sources) == 2:
                break
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Small-sample execution check only; not a performance ranking or clinical validation",
        "data": "PulseDB v2, MIMIC source only for regression",
        "upstream_commit": UPSTREAM_COMMIT,
        "mapping_source": f"https://github.com/pulselabteam/PulseDB/blob/{UPSTREAM_COMMIT}/Generate_Subsets.m",
        "seed": args.seed, "sampling_hz": 125, "window_seconds": 10,
        "waveform_field": "PPG_F", "reference_fields": ["SegSBP", "SegDBP"],
        "pretrained_encoder_used": bool(args.foundation_models), "new_downloads": [],
        "selection": "Fixed seeded official-row sampling, source and local-file availability; no BP-based selection",
        "search": {"Train_Info": train_search, "CalFree_Test_Info": test_search},
        "patient_identity": "source plus first seven characters of Subj_Name, checked against waveform SubjectID",
        "patient_disjoint": True, "split_patients": identities,
        "split_sizes": {name: {"people": len(identities[name]), "segments": len(rows)}
                        for name, rows in partitions.items()},
        "mapping_checks": {"rows_verified": len(manifest), "label_tolerance_mmhg": 1e-6,
                           "segment_id_differs_from_position": int(mapping_differences)},
        "aami_mapping_only": aami_audit,
        "features": {"n_features": features.shape[1], "input": "PPG only, no ECG/ABP/demographics"},
        "results": results,
        "selected_rows_sha256": hashlib.sha256((output / "selected_rows.json").read_bytes()).hexdigest(),
        "limitations": [
            "Only the selected train/validation/test patient sets were audited for intersection; not a whole-index audit",
            "File availability and source restrict the pilot; it is not representative of PulseDB or AAMI",
            "Validation people come exclusively from official Train_Info; no official standalone validation split exists here",
            "Test comes from official CalFree_Test_Info; AAMI labels did not enter fitting, selection or reported errors",
            "Future foundation-model results require pretraining-overlap disclosure: AnyPPG includes PulseDB; PaPaGei uses source cohorts",
            "PPG_F is already filtered/normalized by PulseDB and is not clean/noisy denoising ground truth",
            "The dataset is clinical finger PPG, not wrist blood-pressure validation",
            "The reported test subset is now exposed by this pilot; later design choices must not treat it as a fresh blind holdout",
        ],
    }
    if args.foundation_models:
        add_foundation_check(args, report, {"ppg": np.asarray(signals), "labels": labels,
                                          "split": splits, "patient_id": people})
    write_json(output / "report.json", report)
    if args.shared_report:
        write_json(args.shared_report, report)
    print(json.dumps({"split_sizes": report["split_sizes"], "mapping_checks": report["mapping_checks"],
                      "results": results, "report": str(output / "report.json")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "datasets/pulsedb")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/ppg-expansion-pilot/pulsedb")
    parser.add_argument("--shared-report", type=Path)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--train-people", type=int, default=8)
    parser.add_argument("--validation-people", type=int, default=4)
    parser.add_argument("--test-people", type=int, default=4)
    parser.add_argument("--segments-per-person", type=int, default=8)
    parser.add_argument("--foundation-models", nargs="*", choices=["AnyPPG", "PaPaGei-P", "SigmaPPG"], default=[])
    parser.add_argument("--foundation-only", action="store_true",
                        help="Read the prepared small NPZ in the existing model venv, without h5py")
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())
