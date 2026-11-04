"""Common-support PRV comparison and participant-macro scoring invariants."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.ppg_beat_metrics import evaluate_event_frame
from scripts.summarize_ppg_beat_poc import (
    paired_participant, paired_summary, participant_result, reference_eligible,
    sha256, summarize,
)


def window(method, start=0., reference=None, prediction=None, valid=True, inside=True, subject="s1"):
    reference = np.arange(start + .5, start + 60., 1.) if reference is None else np.asarray(reference)
    prediction = reference.copy() if prediction is None else np.asarray(prediction)
    result = {"dataset": "Dalia", "scenario": "within", "subject": subject, "method": method,
              "start_s": start, "duration_s": 60, "reference_valid": valid,
              "inside_inference": inside, "reference_quality": "synthetic", "reference_exclusion_reason": ""}
    if inside:
        result.update(evaluate_event_frame(reference, prediction, start, start + 60, return_rr_detail=True))
    return result


class TestBeatSummaries(unittest.TestCase):
    def test_strict99_filters_wild_agreement_only_and_keeps_primary(self):
        a, b = window("AnyPPG"), window("AnyPPG", 60.)
        for row, agreement in ((a, .97), (b, .995)):
            row.update(dataset="WildPPG", scenario="transfer", xqrs_agreement_f1=agreement)
        primary = participant_result([a, b], "WildPPG", "transfer", "s1", "AnyPPG", 60, "reference_valid")
        strict = participant_result([a, b], "WildPPG", "transfer", "s1", "AnyPPG", 60, "reference_strict99")
        self.assertEqual(primary["evaluated_windows"], 2)
        self.assertEqual(strict["evaluated_windows"], 1)
        self.assertEqual(strict["scope_reference_valid_fraction"], .5)
        self.assertTrue(reference_eligible(window("AnyPPG"), "reference_strict99"))
        self.assertFalse(reference_eligible({**a, "reference_valid": False}, "reference_strict99"))

    def test_prv_pairs_use_common_finite_windows_not_separate_means(self):
        rows = [window("AnyPPG", 0.), window("AnyPPG", 60.),
                window("SmallCNN", 0., prediction=[]), window("SmallCNN", 60.)]
        for row, value in zip(rows, (1., 5., np.nan, 1.)):
            row["sdnn_rr_ms_absolute_error"] = value
        paired = paired_participant(rows, "Dalia", "within", "s1", "SmallCNN", 60, "reference_valid", "sdnn_rr_ms_mae")
        self.assertEqual(paired["difference"], 4.)
        self.assertEqual(paired["common_finite_windows"], 1)
        self.assertEqual(paired["eligible_common_windows"], 2)
        self.assertEqual(paired["common_support_fraction"], .5)

    def test_matched_rr_pairs_share_the_same_reference_intervals(self):
        reference = np.array([.5, 1.5, 2.5, 3.5])
        rows = [window("AnyPPG", reference=reference, prediction=[.5, 1.59, 2.51, 3.53]),
                window("SmallCNN", reference=reference, prediction=[.5, 2.51, 3.52])]
        paired = paired_participant(rows, "Dalia", "within", "s1", "SmallCNN", 60, "reference_valid", "matched_rr_mae_ms")
        self.assertAlmostEqual(paired["difference"], 10.)
        self.assertEqual(paired["common_reference_pairs"], 1)
        self.assertEqual(paired["pair_denominator"], 3)
        self.assertEqual(paired["common_support_fraction"], 1 / 3)

    def test_empty_predictions_are_f1_failures_and_remain_in_prv_coverage(self):
        rows = [window("AnyPPG", prediction=[])]
        result = participant_result(rows, "Dalia", "within", "s1", "AnyPPG", 60, "reference_valid")
        self.assertEqual(result["f1_100ms"], 0.)
        self.assertEqual(result["recall_100ms"], 0.)
        self.assertEqual(result["matched_rr_coverage"], 0.)
        self.assertEqual(result["rmssd_rr_ms_coverage"], 0.)
        self.assertTrue(np.isnan(result["rmssd_rr_ms_mae"]))
        rows.append(window("SmallCNN"))
        paired = paired_participant(rows, "Dalia", "within", "s1", "SmallCNN", 60, "reference_valid", "f1_100ms")
        self.assertEqual(paired["difference"], -1.)
        self.assertEqual(paired["common_support_fraction"], 1.)

    def test_reference_and_complete_inference_coverage_are_separate(self):
        rows = [window("AnyPPG", 0., inside=False), window("AnyPPG", 60., valid=False), window("AnyPPG", 120.)]
        result = participant_result(rows, "Dalia", "within", "s1", "AnyPPG", 60, "reference_valid")
        self.assertEqual(result["candidate_windows"], 3)
        self.assertEqual(result["reference_valid_fraction"], 2/3)
        self.assertEqual(result["inference_window_fraction"], 2/3)
        self.assertEqual(result["evaluated_window_fraction"], 1/3)
        sensitivity = participant_result(rows, "Dalia", "within", "s1", "AnyPPG", 60, "all_reference")
        self.assertEqual(sensitivity["evaluated_windows"], 2)

    def test_participant_bootstrap_does_not_weight_people_by_window_counts(self):
        records = [{"difference": d, "common_finite_windows": n, "eligible_common_windows": n,
                    "common_reference_pairs": 0, "common_support_fraction": 1., "support": "common_finite_PRV_windows"}
                   for d, n in ((1., 100), (9., 1))]
        summary = paired_summary(records)
        self.assertEqual(summary["difference"], 5.)
        self.assertEqual(summary["n_finite_subjects"], 2)
        self.assertEqual(summary["ci95_low"], 1.)
        self.assertEqual(summary["ci95_high"], 9.)
        self.assertEqual(summary["common_finite_windows"], 101)

    def test_300_second_raw_rr_formulas_and_half_open_endpoint(self):
        intervals = np.resize(np.array([1., 1.1, .9]), 299)
        events = np.r_[0., np.cumsum(intervals)]
        events = np.r_[events[events < 300.], 300.]
        result = evaluate_event_frame(events, events, 0., 300., return_rr_detail=True)
        expected_rr = np.diff(events[events < 300.]) * 1000.
        self.assertEqual(result["n_reference_events"], len(events) - 1)
        self.assertAlmostEqual(result["reference_sdnn_rr_ms"], np.std(expected_rr, ddof=1))
        self.assertAlmostEqual(result["reference_rmssd_rr_ms"], np.sqrt(np.mean(np.diff(expected_rr) ** 2)))
        self.assertAlmostEqual(result["sdnn_rr_ms_absolute_error"], 0.)

    def test_duplicate_or_different_candidate_rows_are_rejected(self):
        rows = [window("AnyPPG"), window("SmallCNN")]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_participant(rows + [rows[0]], "Dalia", "within", "s1", "SmallCNN", 60, "reference_valid", "f1_100ms")
        rows[1]["start_s"] = 60.
        with self.assertRaisesRegex(ValueError, "differ"):
            paired_participant(rows, "Dalia", "within", "s1", "SmallCNN", 60, "reference_valid", "f1_100ms")


def full_fixture(root):
    """31 participants, 93 outputs, with missing predictions and invalid labels."""
    reference = root / "reference"
    run = root / "run"
    run.mkdir()
    source, target = [f"S{i}" for i in range(1, 16)], [f"w{i}" for i in range(16)]
    events = np.r_[np.arange(.5, 60., 1.), np.arange(120.5, 601., 1.)]
    manifests = {}
    for dataset, people in (("Dalia", source), ("WildPPG", target)):
        for person in people:
            directory = reference / dataset / person
            directory.mkdir(parents=True)
            np.savez(directory / "beats.npz", r_peak_times_s=events, duration_s=601.)
            windows = []
            for start, duration in ((0, 60), (60, 60), (120, 60), (0, 300), (300, 300)):
                selected = events[(events >= start) & (events < start + duration)]
                rr = np.diff(selected) * 1000.
                windows.append({"start_s": start, "duration_s": duration,
                                "reference_valid": start not in (0, 60), "reference_quality": "synthetic",
                                "xqrs_agreement_f1": 1., "exclusion_reason": "synthetic-gap" if start in (0, 60) else "",
                                "mean_rr_ms": rr.mean() if len(rr) else "",
                                "sdnn_ms": rr.std(ddof=1) if len(rr) > 1 else "",
                                "rmssd_ms": np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else ""})
            with (directory / "windows.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(windows[0]))
                writer.writeheader()
                writer.writerows(windows)
            manifest = {"outputs_sha256": {name: sha256(directory / name) for name in ("beats.npz", "windows.csv")}}
            (directory / "manifest.json").write_text(json.dumps(manifest))
            manifests[(directory / "manifest.json").relative_to(reference).as_posix()] = sha256(directory / "manifest.json")
    jobs = []
    for fold in range(5):
        test = source[3*fold:3*fold + 3]
        available = [s for s in source if s not in test]
        jobs.append({"id": f"fold{fold}", "source": "Dalia", "target": "Dalia", "scenario": "within",
                     "train": available[3:], "validation": available[:3], "test": test})
    jobs.append({"id": "transfer", "source": "Dalia", "target": "WildPPG", "scenario": "transfer",
                 "train": source[3:], "validation": source[:3], "test": target})
    protocol = {"jobs": jobs, "reference_manifests": manifests}
    (run / "protocol.json").write_text(json.dumps(protocol))
    protocol_hash = sha256(run / "protocol.json")
    for job in jobs:
        for method in ("AnyPPG", "SmallCNN", "Elgendi"):
            directory = run / "runs" / job["id"] / method
            directory.mkdir(parents=True)
            fit = {"identity": {"protocol_sha256": protocol_hash, "job": job, "method": method}}
            if method == "Elgendi":
                fit["pat"] = {"subjects": job["train"], "delay_s": .25}
            else:
                (directory / "head.pt").write_bytes(b"synthetic-state-not-loaded")
                fit["head_sha256"] = sha256(directory / "head.pt")
            (directory / "fit.json").write_text(json.dumps(fit))
            for subject in job["test"]:
                predicted = events[(events >= 1.) & (events < 601.)]
                if subject == "S1" and method == "SmallCNN":
                    predicted = np.array([])
                np.savez(directory / f"{subject}.npz", event_times_s=predicted, start_s=1., end_s=601.)
                meta = {"identity": {"fit_sha256": sha256(directory / "fit.json"), "subject": subject, "target": job["target"]},
                        "sha256": sha256(directory / f"{subject}.npz")}
                (directory / f"{subject}.json").write_text(json.dumps(meta))
    return run, reference


class TestCompleteBeatSummary(unittest.TestCase):
    def test_complete_scoring_preserves_missing_output_and_reference_coverage(self):
        scratch = Path(__file__).resolve().parents[1] / ".local"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary)
            run, reference = full_fixture(root)
            with patch("builtins.print"):
                audit = summarize(run, reference, root / "summary")
            self.assertEqual(audit["status"], "complete")
            self.assertEqual(audit["subject_method_predictions"], 93)
            with (root / "summary/summary.csv").open() as stream:
                summaries = list(csv.DictReader(stream))
            cnn_prv = next(r for r in summaries if r["dataset"] == "Dalia" and r["duration_s"] == "300" and r["scope"] == "reference_valid" and r["method"] == "SmallCNN" and r["metric"] == "rmssd_rr_ms_mae")
            self.assertEqual(int(cnn_prv["n_subjects"]), 15)
            self.assertEqual(int(cnn_prv["n_finite_subjects"]), 14)
            with (root / "summary/paired.csv").open() as stream:
                pairs = list(csv.DictReader(stream))
            f1 = next(r for r in pairs if r["dataset"] == "Dalia" and r["duration_s"] == "300" and r["scope"] == "reference_valid" and r["comparator"] == "SmallCNN" and r["metric"] == "f1_100ms")
            self.assertAlmostEqual(float(f1["difference"]), 1/15)
            prv = next(r for r in pairs if r["dataset"] == "Dalia" and r["duration_s"] == "300" and r["scope"] == "reference_valid" and r["comparator"] == "SmallCNN" and r["metric"] == "rmssd_rr_ms_mae")
            self.assertEqual(int(prv["n_finite_subjects"]), 14)
            self.assertAlmostEqual(float(prv["participant_mean_common_support"]), 14/15)
            with (root / "summary/windows.csv").open() as stream:
                scored = list(csv.DictReader(stream))
            self.assertTrue(all(not any(k.startswith("_") for k in row) for row in scored))
            self.assertTrue(any(row["inside_inference"] == "False" for row in scored))
            # Do not trust an unchanged manifest if its actual reference changes.
            (reference / "Dalia/S1/beats.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Reference arrays changed"):
                summarize(run, reference, root / "summary")


if __name__ == "__main__":
    unittest.main()
