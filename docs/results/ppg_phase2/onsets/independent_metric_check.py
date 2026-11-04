"""Independent numeric spot/full check; imports no scorer or event metrics."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
sys.path.insert(0, str(ROOT / "scripts"))
from ppg_embc21_data import load_records

BASE = ROOT / ".local/ppg-phase2/onsets"
OUT = BASE / "summary"


def read(name):
    with (OUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crop(events, start, end):
    return events[(events >= start) & (events < end)]


def greedy_pairs(reference, prediction, tolerance):
    # For sorted points and an interval tolerance, chronological greedy obtains
    # maximum match cardinality. When neighborhoods are unique, pairs are also
    # uniquely minimum-distance and independently check the scorer's RR support.
    # Native 400 Hz and output 64 Hz share exact 1/1600-second ticks. Integer
    # comparisons independently implement inclusive tolerance boundaries.
    reference, prediction = np.rint(reference * 1600).astype(np.int64), np.rint(prediction * 1600).astype(np.int64)
    tolerance = round(tolerance * 1600)
    i, j, pairs = 0, 0, []
    while i < len(reference) and j < len(prediction):
        if abs(reference[i] - prediction[j]) <= tolerance:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif prediction[j] < reference[i] - tolerance:
            j += 1
        else:
            i += 1
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


records = {r["record_id"]: r for r in load_records(ROOT / "datasets/embc21")}
predictions = {}
for path in sorted((BASE / "runs").glob("*/*/*.npz")):
    method = "MSPTD" if path.parent.name == "MSPTDfast" else path.parent.name
    with np.load(path) as z:
        predictions[(path.stem, method)] = z["event_times_s"]

comparisons, maximum_difference = 0, 0.


def close(actual, expected):
    global comparisons, maximum_difference
    actual, expected = float(actual), float(expected)
    comparisons += 1
    if np.isnan(actual) and np.isnan(expected):
        return
    assert np.isfinite([actual, expected]).all(), (actual, expected)
    maximum_difference = max(maximum_difference, abs(actual - expected))
    assert np.isclose(actual, expected, rtol=1e-10, atol=1e-8), (actual, expected)


intervals, windows = read("intervals.csv"), read("prv_windows.csv")
rr_checked, rr_ambiguous = 0, 0
for row in intervals:
    start, end = float(row["start_s"]), float(row["end_s"])
    ref = crop(records[row["record_id"]]["onset_times_s"], start, end)
    pred = crop(predictions[(row["record_id"], row["method"])], start, end)
    for tolerance in (.025, .05, .1):
        pairs = greedy_pairs(ref, pred, tolerance)
        tp, fp, fn = len(pairs), len(pred) - len(pairs), len(ref) - len(pairs)
        suffix = str(round(tolerance * 1000)) + "ms"
        for key, value in (("tp", tp), ("fp", fp), ("fn", fn), ("f1", 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan)):
            close(row[key + "_" + suffix], value)
        if tolerance != .05:
            continue
        # Skip the conditional interval-error check only for genuinely ambiguous
        # matching neighborhoods; event counts above are still checked in full.
        neighborhoods = abs(np.rint(ref[:, None] * 1600).astype(np.int64) - np.rint(pred[None, :] * 1600).astype(np.int64)) <= round(tolerance * 1600)
        unique = (neighborhoods.sum(0) <= 1).all() and (neighborhoods.sum(1) <= 1).all()
        if not unique:
            rr_ambiguous += 1
            continue
        consecutive = np.all(np.diff(pairs, axis=0) == 1, axis=1)
        a, b = pairs[:-1][consecutive], pairs[1:][consecutive]
        errors = ((pred[b[:, 1]] - pred[a[:, 1]]) - (ref[b[:, 0]] - ref[a[:, 0]])) * 1000
        close(row["matched_pp_count"], len(errors))
        close(row["matched_pp_mae_ms"], np.abs(errors).mean() if len(errors) else np.nan)
        close(row["matched_pp_coverage"], len(errors) / (len(ref) - 1) if len(ref) > 1 else np.nan)
        rr_checked += 1

for row in windows:
    start, end = float(row["start_s"]), float(row["start_s"]) + float(row["duration_s"])
    ref = crop(records[row["record_id"]]["onset_times_s"], start, end)
    pred = crop(predictions[(row["record_id"], row["method"])], start, end)
    own = []
    for prefix, sequence in (("reference", ref), ("prediction", pred)):
        # Work in seconds and convert only final values, independently of the
        # published scorer's helper. All detected beats remain in this sequence.
        pp = sequence[1:] - sequence[:-1]
        values = {
            "mean_pp_ms": np.sum(pp) / len(pp) * 1000 if len(pp) else np.nan,
            "sdpp_ms": np.sqrt(np.sum((pp - pp.mean()) ** 2) / (len(pp) - 1)) * 1000 if len(pp) > 1 else np.nan,
            "rmssd_ms": np.sqrt(np.sum((pp[1:] - pp[:-1]) ** 2) / (len(pp) - 1)) * 1000 if len(pp) > 1 else np.nan,
        }
        close(row[prefix + "_intervals"], len(pp))
        assert (row[prefix + "_eligible"] == "True") == (len(pp) >= 20)
        for name, value in values.items():
            close(row[prefix + "_" + name], value)
        own.append(values)
    for name in own[0]:
        expected = abs(own[1][name] - own[0][name]) if min(len(ref), len(pred)) >= 21 else np.nan
        close(row[name + "_absolute_error"], expected)

# Subject-level aggregation checked independently from published interval/block
# rows after all their primitive values above have been independently verified.
subject_rows = read("per_subject.csv")
for row in subject_rows:
    if row["metric"] != "f1_50ms":
        continue
    selected = [r for r in intervals if r["subject"] == row["subject"] and r["method"] == row["method"] and (row["group"] == "all" or r["activity"] == row["group"])]
    tp, fp, fn = [sum(int(r[key + "_50ms"]) for r in selected) for key in ("tp", "fp", "fn")]
    close(row["value"], 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan)
for row in read("summary.csv"):
    selected = [float(r["value"]) for r in subject_rows if all(r[key] == row[key] for key in ("method", "group", "metric", "duration_s"))]
    finite = np.asarray(selected)[np.isfinite(selected)]
    close(row["mean"], finite.mean() if len(finite) else np.nan)
    close(row["n_finite_subjects"], len(finite))

report = {"status": "passed", "metric_imports": "No production scoring or event-metric module imported",
          "method": "Independent sorted greedy maximum match count; unique-neighborhood consecutive PP intervals; direct numpy PP formulas and participant means",
          "record_method_predictions": len(predictions), "intervals_all_three_tolerances": len(intervals),
          "consecutive_pp_intervals_checked": rr_checked, "consecutive_pp_intervals_ambiguous_not_recomputed": rr_ambiguous,
          "prv_windows_all_methods": len(windows), "scalar_comparisons": comparisons,
          "maximum_absolute_float_difference": maximum_difference,
          "script_sha256": digest(__file__), "production_audit_sha256": digest(OUT / "audit.json"),
          "protocol_sha256": digest(BASE / "protocol.json"),
          "tables_sha256": {n: digest(OUT / n) for n in ("intervals.csv", "prv_windows.csv", "per_subject.csv", "summary.csv")}}
(OUT / "independent_metric_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
