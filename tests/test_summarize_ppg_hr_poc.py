"""Independent synthetic evidence for PoC aggregation and corruption checks."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.summarize_ppg_hr_poc import (
    Auditor, DATASETS, METHODS, MODELS, SEEDS, paired_aggregate, paired_records,
    participant_aggregate, sha256, subject_records, summarize, summary_aggregate,
    validate_plan,
)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_jobs(subjects):
    jobs = []
    for dataset, people in subjects.items():
        for fold in range(5):
            test = people[fold::5]
            outer = [person for person in people if person not in test]
            for seed in SEEDS:
                order = np.random.default_rng(seed + fold).permutation(outer).tolist()
                for budget, size, positions in (("3", 3, (0,)), ("6", 6, (0, 3)), ("full", len(outer), (0, 3, 6))):
                    val = [order[i] for i in positions]
                    jobs.append({"id": f"{dataset}-{fold}-{seed}-{budget}", "scenario": "within", "source": dataset, "target": dataset,
                                 "fold": fold, "seed": seed, "budget": budget, "train": [s for s in order[:size] if s not in val], "validation": val, "test": test})
    for seed in SEEDS:
        jobs.append({"id": f"transfer-{seed}", "scenario": "transfer", "source": "Dalia", "target": "WildPPG", "fold": -1,
                     "seed": seed, "budget": "full", "train": subjects["Dalia"][3:], "validation": subjects["Dalia"][:3], "test": subjects["WildPPG"]})
    return jobs


def fixture(root):
    repository = root / "repository"
    for filename in ("scripts/ppg_hr_baselines.py", "scripts/ppg_hr_poc_features.py", "watch_lsm/configs/ppg_bench_models.yaml", "repos/ppg-bench/ppg_bench/example.py", "weights/model.bin"):
        path = repository / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture", encoding="utf-8")
    subjects = {d: [f"S{i:02d}" for i in range(15 if d == "Dalia" else 16)] for d in DATASETS}
    all_rows = {}
    manifest_hashes, row_hashes = {}, {}
    for dataset in DATASETS:
        directory = root / "data" / dataset
        directory.mkdir(parents=True)
        people = np.repeat(subjects[dataset], 3)
        n = len(people)
        rows = {"subject": people, "start_s": np.tile([0., 2., 4.], len(subjects[dataset])), "hr": 60. + np.arange(n) % 20,
                "motion_bin": np.tile([0, 1, 2], len(subjects[dataset])), "activity": np.tile([0, 1, -2], len(subjects[dataset])),
                "official_include": np.tile([True, True, False], len(subjects[dataset]))}
        all_rows[dataset] = rows
        np.savez(directory / "rows.npz", **rows)
        np.save(directory / "x64.npy", np.zeros((n, 512), np.float32))
        np.save(directory / "stats64.npy", np.ones((n, 2), np.float32))
        hashes = {name: sha256(directory / name) for name in ("rows.npz", "x64.npy")}
        save_json(directory / "manifest.json", {"dataset": dataset, "subjects": subjects[dataset], "windows": n, "hashes": hashes, "sources": []})
        manifest_hashes[dataset], row_hashes[dataset] = sha256(directory / "manifest.json"), hashes["rows.npz"]
    plan = {"models": list(MODELS), "seeds": list(SEEDS), "outer_folds": 5, "jobs": make_jobs(subjects),
            "data_manifest_sha256": manifest_hashes, "data_rows_sha256": row_hashes,
            "code_sha256": {"ppg_hr_baselines.py": sha256(repository / "scripts/ppg_hr_baselines.py")}}
    save_json(root / "protocol.json", plan)
    protocol_hash = sha256(root / "protocol.json")
    feature_hashes = {}
    for dataset in DATASETS:
        directory = root / "features" / dataset
        directory.mkdir(parents=True)
        n = len(all_rows[dataset]["hr"])
        for model in MODELS:
            np.save(directory / f"{model}.npy", np.ones((n, 2), np.float32))
            identity = {"dataset": dataset, "model": {"name": model, "embedding_dim": 2},
                        "data": {"manifest_sha256": manifest_hashes[dataset], "raw_sources": [],
                                 "files_sha256": {f: sha256(root / "data" / dataset / f) for f in ("rows.npz", "x64.npy", "stats64.npy")}},
                        "weight_files_sha256": {"weights/model.bin": sha256(repository / "weights/model.bin")},
                        "source": {"extraction_script_sha256": sha256(repository / "scripts/ppg_hr_poc_features.py"),
                                   "config_sha256": sha256(repository / "watch_lsm/configs/ppg_bench_models.yaml"),
                                   "upstream_python_files_sha256": {"ppg_bench/example.py": sha256(repository / "repos/ppg-bench/ppg_bench/example.py")}}}
            metadata = {"status": "complete", "identity": identity, "identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                        "completed_windows": n, "shape": [n, 2], "dtype": "float32", "features_sha256": sha256(directory / f"{model}.npy")}
            save_json(directory / f"{model}.json", metadata)
            feature_hashes[(dataset, model)] = sha256(directory / f"{model}.json")
        traditional = root / "traditional" / dataset
        traditional.mkdir(parents=True)
        np.savez(traditional / "predictions.npz", welch=all_rows[dataset]["hr"] + 4, elgendi=all_rows[dataset]["hr"] + 5)
        save_json(traditional / "complete.json", {"identity": {"data_manifest_sha256": manifest_hashes[dataset], "algorithm_sha256": plan["code_sha256"]["ppg_hr_baselines.py"]},
                                                 "sha256": sha256(traditional / "predictions.npz"), "coverage": {"welch": 1., "elgendi": 1.}})
    for job in plan["jobs"]:
        source, target = all_rows[job["source"]], all_rows[job["target"]]
        selected = np.flatnonzero(np.isin(target["subject"], job["test"]))
        for index, method in enumerate(METHODS):
            directory = root / "runs" / job["id"] / method
            directory.mkdir(parents=True)
            np.savez(directory / "predictions.npz", row=selected, subject=target["subject"][selected], start_s=target["start_s"][selected],
                     reference=target["hr"][selected], prediction=target["hr"][selected] + index + 1)
            identity = {"protocol_sha256": protocol_hash, "job": job, "method": method}
            if method in MODELS:
                identity["feature_metadata_sha256"] = {d: feature_hashes[(d, method)] for d in set((job["source"], job["target"]))}
            save_json(directory / "complete.json", {"identity": identity, "predictions_sha256": sha256(directory / "predictions.npz"), "subject_mae_bpm": index + 1.,
                                                     "train_windows": int(np.isin(source["subject"], job["train"]).sum()),
                                                     "validation_windows": int(np.isin(source["subject"], job["validation"]).sum()), "test_windows": len(selected)})
    return repository, plan


class TestAggregation(unittest.TestCase):
    def test_participant_then_seed_average_avoids_window_pseudoreplication(self):
        job = {"scenario": "within", "source": "Dalia", "target": "Dalia", "budget": "full", "seed": 42}
        rows = {"subject": np.array(["a"] * 9 + ["b"]), "hr": np.zeros(10), "motion_bin": np.zeros(10, int),
                "official_include": np.ones(10, bool), "activity": np.zeros(10, int)}
        first = subject_records(job, rows, {"SmallCNN": np.array([1.] * 9 + [9.])})
        second = subject_records({**job, "seed": 43}, rows, {"SmallCNN": np.array([3.] * 9 + [7.])})
        participants = participant_aggregate(first + second)
        summary = next(r for r in summary_aggregate(participants, first + second) if r["stratum"] == "all")
        self.assertEqual(summary["conditional_participant_mae_bpm"], 5.)
        self.assertEqual(summary["n_subjects"], 2)
        self.assertEqual(summary["unique_windows"], 10)

    def test_common_window_pairing_and_seed_unit(self):
        job = {"scenario": "within", "source": "Dalia", "target": "Dalia", "budget": "full", "seed": 42}
        rows = {"subject": np.array(["a", "a", "b", "b"]), "hr": np.zeros(4)}
        predictions = {"AnyPPG": np.array([1., 100., 3., 100.]), "welch": np.array([4., np.nan, 5., np.nan])}
        records = paired_records(job, rows, predictions)
        records += paired_records({**job, "seed": 43}, rows, predictions)
        pairs, participants = paired_aggregate(records, samples=10000)
        self.assertEqual(len(participants), 2)
        self.assertEqual(pairs[0]["n_subjects"], 2)
        self.assertEqual(pairs[0]["unique_windows"], 4)
        self.assertEqual(pairs[0]["paired_window_coverage"], .5)
        self.assertEqual(pairs[0]["difference_bpm"], -2.5)
        self.assertEqual(pairs[0]["ci95_low_bpm"], -3.)
        self.assertEqual(pairs[0]["ci95_high_bpm"], -2.)

    def test_missing_estimates_remain_in_coverage(self):
        job = {"scenario": "within", "source": "Dalia", "target": "Dalia", "budget": "full", "seed": 42}
        rows = {"subject": np.array(["a", "b"]), "hr": np.ones(2), "motion_bin": np.zeros(2, int), "activity": np.zeros(2, int), "official_include": np.ones(2, bool)}
        records = subject_records(job, rows, {"elgendi": np.array([np.nan, 3.])})
        summary = next(r for r in summary_aggregate(participant_aggregate(records), records) if r["stratum"] == "all")
        self.assertEqual(summary["n_subjects"], 2)
        self.assertEqual(summary["n_subjects_with_estimate"], 1)
        self.assertEqual(summary["window_coverage"], .5)
        self.assertEqual(summary["conditional_participant_mae_bpm"], 2.)
        self.assertIsNone(summary["full_coverage_participant_mae_bpm"])
        pairs, _ = paired_aggregate(paired_records(job, rows, {"AnyPPG": np.array([2., 2.]), "elgendi": np.array([np.nan, 3.])}))
        self.assertEqual(pairs[0]["n_subjects"], 1)
        self.assertEqual(pairs[0]["n_subjects_total"], 2)
        self.assertEqual(pairs[0]["paired_window_coverage"], .5)
        self.assertEqual(pairs[0]["min_seeds_with_pair"], 0)


class TestFullAudit(unittest.TestCase):
    def test_full_study_missing_completion_and_forged_row(self):
        scratch = Path(__file__).resolve().parents[1] / ".local"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary)
            repository, plan = fixture(root)
            audit = summarize(root, repository=repository, bootstrap_samples=100)
            self.assertEqual(audit["status"], "complete")
            self.assertEqual(audit["completed_learned_results"], 558)
            self.assertEqual(audit["verified_feature_caches"], 8)
            with (root / "summary/summary.csv").open(newline="", encoding="utf-8") as stream:
                results = list(csv.DictReader(stream))
            self.assertEqual(len(results), 7 * 8)
            self.assertTrue(all(int(r["n_subjects"]) in (15, 16) for r in results))
            self.assertTrue(all(int(r["min_seeds_per_subject"]) == 3 for r in results))
            first = plan["jobs"][0]
            directory = root / "runs" / first["id"] / "TrainMean"
            complete_path = directory / "complete.json"
            original = complete_path.read_text()
            complete_path.unlink()
            with self.assertRaisesRegex(ValueError, "557/558"):
                summarize(root, repository=repository, bootstrap_samples=100)
            partial = summarize(root, repository=repository, allow_partial=True, bootstrap_samples=100)
            self.assertEqual(partial["status"], "partial")
            complete_path.write_text(original)
            with np.load(directory / "predictions.npz") as loaded:
                prediction = {k: loaded[k] for k in loaded.files}
            prediction["start_s"][0] += .5
            np.savez(directory / "predictions.npz", **prediction)
            metadata = json.loads(original)
            metadata["predictions_sha256"] = sha256(directory / "predictions.npz")
            save_json(complete_path, metadata)
            with self.assertRaisesRegex(ValueError, "row identity mismatch"):
                summarize(root, repository=repository, allow_partial=True, bootstrap_samples=100)

    def test_split_leakage_is_detected_without_results(self):
        subjects = {d: {f"S{i:02d}" for i in range(15 if d == "Dalia" else 16)} for d in DATASETS}
        jobs = make_jobs({d: sorted(s) for d, s in subjects.items()})
        plan = {"models": MODELS, "seeds": SEEDS, "outer_folds": 5, "jobs": jobs}
        validate_plan(plan, subjects)
        corrupted = copy.deepcopy(plan)
        corrupted["jobs"][0]["train"][0] = corrupted["jobs"][0]["test"][0]
        with self.assertRaisesRegex(ValueError, "Held-out subject leakage"):
            validate_plan(corrupted, subjects)


if __name__ == "__main__":
    unittest.main()
