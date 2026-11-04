"""Independent manual-onset support, pairing and artifact-audit invariants."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import summarize_ppg_onset_poc as summary


ROOT = Path(__file__).resolve().parents[1]


def record(duration=130., intervals=None, events=None, subject="S1", record_id="S1_rest1", activity="rest"):
    return {"subject": subject, "record_id": record_id, "activity": activity, "duration_s": duration,
            "fs": 64., "raw_ppg": np.zeros(int(duration * 64), np.float64),
            "input_ppg": np.zeros(int(duration * 64), np.float64),
            "annotated_intervals_s": np.asarray([[0., duration]] if intervals is None else intervals, float).reshape(-1, 2),
            "onset_times_s": np.arange(1.5, duration - 1, 1.) if events is None else np.asarray(events, float)}


def metric(rows, name, duration=0):
    return next(r for r in rows if r["metric"] == name and r["duration_s"] == duration)


class TestOnsetSummary(unittest.TestCase):
    def test_half_open_boundaries_and_gap_pairs_are_never_joined(self):
        r = record(140., [[0., 61.], [62., 130.]], [1., 2., 60., 62., 63., 128., 129.])
        intervals, windows, coverage = summary.score_record(r, "AnyPPG", [1., 2., 60., 62., 63., 128.], 1., 129.)
        self.assertEqual([x["reference_pp_count"] for x in intervals], [2, 2])
        self.assertEqual([x["matched_pp_count"] for x in intervals], [2, 2])
        self.assertEqual([w["start_s"] for w in windows], [1., 62.])
        self.assertTrue(all(w["duration_s"] == 60 for w in windows))
        self.assertEqual(coverage["evaluated_seconds"], 127.)
        self.assertEqual(coverage["evaluated_reference_events"], 6)

    def test_empty_predictions_fail_f1_but_prv_error_is_missing_with_zero_coverage(self):
        r = record()
        events, windows, _ = summary.score_record(r, "AnyPPG", [], 1., 127.)
        rows = summary.participant_metrics(events, windows, "S1", "AnyPPG", "all")
        self.assertEqual(metric(rows, "f1_50ms")["value"], 0.)
        self.assertEqual(metric(rows, "matched_rr_coverage")["value"], 0.)
        self.assertTrue(np.isnan(metric(rows, "matched_rr_mae_ms")["value"]))
        self.assertEqual(metric(rows, "rmssd_ms_coverage", 60)["value"], 0.)
        self.assertTrue(np.isnan(metric(rows, "rmssd_ms_mae", 60)["value"]))
        self.assertEqual(metric(rows, "rmssd_ms_mae", 60)["n_eligible"], 2)
        self.assertTrue(np.isnan(metric(rows, "rmssd_ms_coverage", 300)["value"]))
        aggregate = summary.aggregate_people(rows)
        self.assertEqual(metric(aggregate, "rmssd_ms_mae", 300)["n_finite_subjects"], 0)

    def test_false_positives_in_reference_empty_interval_count_as_failures(self):
        r = record(events=[])
        events, windows, _ = summary.score_record(r, "AnyPPG", [2., 4., 6.], 1., 127.)
        self.assertEqual(events[0]["fp_50ms"], 3)
        rows = summary.participant_metrics(events, windows, "S1", "AnyPPG", "all")
        self.assertEqual(metric(rows, "f1_50ms")["value"], 0.)
        self.assertEqual(metric(rows, "precision_50ms")["value"], 0.)

    def test_pp_pairs_require_both_sequences_consecutive_and_common_reference(self):
        r = record(10., events=[1., 2., 3., 4.])
        a, _, _ = summary.score_record(r, "AnyPPG", [1., 2.04, 3.01, 4.03], 0., 9.)
        b, _, _ = summary.score_record(r, "SmallCNN", [1., 3.01, 4.02], 0., 9.)
        pair = summary.paired_participant(a + b, [], "S1", "SmallCNN", "all", "matched_rr_mae_ms", 0)
        self.assertAlmostEqual(pair["difference"], 10.)
        self.assertEqual(pair["n_common_supported"], 1)
        self.assertEqual(pair["n_eligible"], 3)
        self.assertAlmostEqual(pair["common_support_fraction"], 1/3)
        self.assertEqual(b[0]["matched_pp_count"], 1)

    def test_prv_pair_uses_common_finite_windows_and_raw_formula(self):
        r = record()
        predicted = r["onset_times_s"][r["onset_times_s"] < 127.].copy()
        predicted[10] += .1
        a, wa, _ = summary.score_record(r, "AnyPPG", predicted, 1., 127.)
        b, wb, _ = summary.score_record(r, "SmallCNN", r["onset_times_s"][(r["onset_times_s"] >= 61.) & (r["onset_times_s"] < 127.)], 1., 127.)
        intervals = np.diff(predicted[predicted < 61.]) * 1000
        self.assertAlmostEqual(wa[0]["prediction_sdpp_ms"], np.std(intervals, ddof=1))
        self.assertAlmostEqual(wa[0]["prediction_rmssd_ms"], np.sqrt(np.mean(np.diff(intervals)**2)))
        # A strong difference in an unsupported comparator window must not enter a pair.
        wa[0]["rmssd_ms_absolute_error"], wa[1]["rmssd_ms_absolute_error"] = 100., 5.
        wb[1]["rmssd_ms_absolute_error"] = 1.
        pair = summary.paired_participant(a + b, wa + wb, "S1", "SmallCNN", "all", "rmssd_ms_mae", 60)
        self.assertEqual(pair["difference"], 4.)
        self.assertEqual(pair["n_common_supported"], 1)
        self.assertEqual(pair["n_eligible"], 2)

    def test_aggregate_is_equal_person_weight_not_event_weight_and_reports_support(self):
        base = {"dataset": "EMBC21", "group": "all", "method": "AnyPPG", "metric": "f1_50ms", "duration_s": 0,
                "support": "test", "coverage": 1., "n_eligible": 100, "n_supported": 100}
        rows = [{**base, "subject": "S1", "value": 0., "n_eligible": 10000},
                {**base, "subject": "S2", "value": 1.}, {**base, "subject": "S3", "value": np.nan}]
        result = summary.aggregate_people(rows)[0]
        self.assertEqual(result["mean"], .5)
        self.assertEqual(result["n_subjects"], 3)
        self.assertEqual(result["n_finite_subjects"], 2)
        self.assertEqual((result["ci95_low"], result["ci95_high"]), (0., 1.))
        self.assertTrue(all(np.isnan(v) for v in summary.bootstrap([1., np.nan])))

    def test_record_identity_preserves_missing_native_samples_and_input_copy(self):
        r = record(10.)
        r["raw_ppg"][0] = np.nan
        first = summary.independent_record_identity(r)
        self.assertEqual(first, summary.independent_record_identity(r))
        r["input_ppg"][0] = 1.
        self.assertNotEqual(first["record_sha256"], summary.independent_record_identity(r)["record_sha256"])
        with self.assertRaises(ValueError):
            summary.validate_intervals([0., 10.], 10.)

    def test_complete_independent_audit_and_prediction_tamper_detection(self):
        (ROOT / ".local").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".local") as temporary:
            root = Path(temporary)
            subjects = [f"S{i}" for i in range(1, 8)]
            records = [record(70., subject=s, record_id=f"{s}_{activity}{i}", activity=activity,
                              events=np.arange(1.5, 67., 1.)) for s in subjects for activity in ("rest", "squat", "step") for i in range(1, 6)]
            jobs = []
            for fold, s in enumerate(subjects):
                remaining = [p for p in subjects if p != s]
                val = str(np.random.default_rng(42 + fold).choice(remaining))
                jobs.append({"id": f"EMBC21-fold{fold}-seed42", "fold": fold, "seed": 42,
                             "scenario": "within", "source": "EMBC21", "target": "EMBC21", "test": [s],
                             "validation": [val], "train": [p for p in remaining if p != val]})
            protocol = {"version": "manual-wrist-onset-poc-v1", "models": list(summary.METHODS), "jobs": jobs,
                        "records": {r["record_id"]: summary.independent_record_identity(r) for r in records},
                        "traditional": {"fixed": True}, "fit": {"thresholds": [.2, .5]}}
            summary.write_json(root / "protocol.json", protocol)
            phash = summary.sha256(root / "protocol.json")
            frame_rows = [(r["subject"], r["record_id"], start) for r in sorted(records, key=lambda r: r["record_id"]) for start in np.arange(11) * 6.]
            frame_dir = root / "frames"
            frame_dir.mkdir()
            rows = {"subject": np.array([r[0] for r in frame_rows]), "record_id": np.array([r[1] for r in frame_rows]),
                    "start_s": np.array([r[2] for r in frame_rows])}
            np.savez(frame_dir / "rows.npz", **rows)
            summary.write_json(frame_dir / "complete.json", {"protocol_sha256": phash, "frames": len(frame_rows),
                                                             "sha256": {"rows.npz": summary.sha256(frame_dir / "rows.npz")}})
            for job in jobs:
                for method in summary.METHODS:
                    directory = root / "runs" / job["id"] / method
                    directory.mkdir(parents=True)
                    fit = {"identity": {"protocol_sha256": phash, "job": job, "method": method}}
                    if method == "MSPTDfast":
                        fit["traditional"] = protocol["traditional"]
                    else:
                        (directory / "head.pt").write_bytes(b"synthetic test head")
                        train, val = summary.selected_indices(rows, job["train"], 256), summary.selected_indices(rows, job["validation"], 128)
                        fit.update(head_sha256=summary.sha256(directory / "head.pt"), best_epoch=1, threshold=.5, validation_f1_50ms=1.,
                                   history=[{"epoch": 1, "threshold": .5, "validation_f1_50ms": 1., "threshold_scores": {"0.2": .8, "0.5": 1.}}],
                                   train_frame_indices=train, validation_frame_indices=val, train_frames=len(train), validation_frames=len(val),
                                   train_frames_by_person={s: int(np.sum(rows["subject"][train] == s)) for s in job["train"]})
                    summary.write_json(directory / "fit.json", fit)
                    for r in records:
                        if r["subject"] not in job["test"]:
                            continue
                        path = directory / (r["record_id"] + ".npz")
                        probability = np.zeros(11 * 384, np.float32)
                        probability[((r["onset_times_s"] - 1) * 64).astype(int)] = 1.
                        np.savez(path, event_times_s=r["onset_times_s"], start_s=1., end_s=67., probability=probability, probability_fs=64)
                        summary.write_json(path.with_suffix(".json"), {"sha256": summary.sha256(path), "events": len(r["onset_times_s"]),
                            "identity": {"fit_sha256": summary.sha256(directory / "fit.json"), "subject": r["subject"], "record_id": r["record_id"],
                                         "target": "EMBC21", "record_sha256": protocol["records"][r["record_id"]]["record_sha256"]}})
            with patch.object(summary, "load_records", return_value=records), patch.object(summary, "verify_data_and_code", return_value=["synthetic"]):
                audit = summary.summarize(root, root, root / "summary")
                self.assertEqual(audit["record_method_predictions"], 315)
                with (root / "summary/summary.csv").open() as stream:
                    results = list(csv.DictReader(stream))
                main = next(r for r in results if r["metric"] == "f1_50ms" and r["method"] == "AnyPPG" and r["group"] == "all")
                self.assertEqual(float(main["mean"]), 1.)
                self.assertEqual(int(main["n_finite_subjects"]), 7)
                target = root / "runs/EMBC21-fold0-seed42/AnyPPG/S1_rest1.npz"
                target.write_bytes(target.read_bytes() + b"tamper")
                with self.assertRaisesRegex(ValueError, "Prediction identity or bytes changed"):
                    summary.summarize(root, root, root / "summary")


if __name__ == "__main__":
    unittest.main()
