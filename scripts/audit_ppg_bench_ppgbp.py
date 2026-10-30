"""Audit the completed PPG-BP linear scope and export small, shareable results.

Raw predictions remain local. This checks saved artifacts rather than rerunning
training, and fails instead of publishing a partial seven-model result table.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"AnyPPG", "PaPaGei-S", "PaPaGei-P", "PulsePPG", "SigmaPPG", "PRVCtrl", "MOMENT-1-small"}
SPLITS = {(seed, fold) for seed in range(42, 47) for fold in range(5)}
TASKS = {"ppgbp-" + name for name in ("age", "bmi", "cerebral-infarction", "cerebrovascular-disease", "diabetes", "diasbp", "hr", "hypertension", "hypertension-binary", "sysbp")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, default=[ROOT / ".local/ppg-bench-reproduction-py312", ROOT / ".local/ppg-bench-reproduction-extra"])
    parser.add_argument("--output", type=Path, default=ROOT / ".local/ppg-bench-ppgbp-audit")
    args = parser.parse_args()
    groups = {}
    artifacts = hashlib.sha256()
    evidence = []
    seen_jobs = set()
    for root in args.roots:
        for manifest in sorted((root / "ppgbp").glob("*/linear/run.json")):
            run = json.loads(manifest.read_text())
            model = manifest.parent.parent.name
            require(model in MODELS, f"Unexpected model: {model}")
            require(set(run["seeds"]) == set(range(42, 47)) and set(run["folds"]) == set(range(5)), "Incomplete run scope")
            require(run["protocol"]["linear"]["inner_folds"] == 4, "Changed inner CV protocol")
            evidence.append({"model": model, "run_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "upstream_code_sha256": run["identity"]["code_sha256"], "versions": run["identity"]["versions"]})
            for job in run["jobs"]:
                require((model, job) not in seen_jobs, f"Duplicate job: {job}")
                seen_jobs.add((model, job))
                directory = manifest.parent / "jobs" / job
                record_path = directory / "results.json"
                raw = record_path.read_bytes()
                record = json.loads(raw)
                require(record["model"] == model and record["method"] == "linear", f"Wrong result identity: {job}")
                require(record["task"].startswith("ppgbp-"), f"Wrong dataset: {job}")
                split = (record["seed"], record["fold"])
                require(split in SPLITS, f"Unexpected split: {job}")
                train, test = set(record["train_groups"]), set(record["test_groups"])
                require(not train.intersection(test), f"Train/test subject overlap: {job}")
                require(len(train | test) == 218, f"Unexpected subject coverage: {job}")
                require(len(record["results"]) == 1 and record["results"][0]["repeat"] == 0, f"Wrong linear repeat count: {job}")
                item = record["results"][0]
                prediction = (directory / item["predictions"]).resolve()
                require(prediction.is_relative_to(directory.resolve()), f"Prediction escaped job: {job}")
                with np.load(prediction, allow_pickle=False) as arrays:
                    subjects = arrays["subjects"].astype(str)
                    require(set(subjects) == test, f"Prediction subjects differ from test split: {job}")
                    require(len(arrays["y_true"]) == len(arrays["y_pred"]) == len(subjects), f"Prediction length mismatch: {job}")
                    for name in arrays.files:
                        values = arrays[name]
                        if np.issubdtype(values.dtype, np.number):
                            require(np.isfinite(values).all(), f"Non-finite {name}: {job}")
                    if record["task"] in ("ppgbp-sysbp", "ppgbp-diasbp"):
                        # The benchmark scores the mean prediction per subject.
                        errors = [abs(float(arrays["y_pred"][subjects == name].mean()) - float(arrays["y_true"][subjects == name][0])) for name in test]
                        require(np.isclose(np.mean(errors), item["test"]["mae"], rtol=1e-6, atol=1e-5), f"Saved MAE disagrees with predictions: {job}")
                for path, content in [(record_path, raw), (prediction, prediction.read_bytes())]:
                    artifacts.update(f"{model}/{job}/{path.name}\n".encode())
                    artifacts.update(hashlib.sha256(content).digest())
                key = (model, record["task"])
                group = groups.setdefault(key, {})
                require(split not in group, f"Duplicate split: {key} {split}")
                metric = "mae" if "mae" in item["test"] else "argmax_balanced_accuracy" if "argmax_balanced_accuracy" in item["test"] else "balanced_accuracy"
                value = item["test"][metric]
                require(value is not None and np.isfinite(value), f"Missing primary metric: {job}")
                group[split] = (metric, value, record["baseline"][metric], test)
    require(set(groups) == {(model, task) for model in MODELS for task in TASKS}, "Expected seven models and the exact ten official tasks each")
    rows = []
    for (model, task), splits in sorted(groups.items()):
        require(set(splits) == SPLITS, f"Missing folds or seeds: {model} {task}")
        for seed in range(42, 47):
            tests = [splits[seed, fold][3] for fold in range(5)]
            require(sum(map(len, tests)) == len(set.union(*tests)) == 218, f"Outer test folds overlap: {model} {task} {seed}")
        means = [np.mean([splits[seed, fold][1] for fold in range(5)]) for seed in range(42, 47)]
        baseline = float(np.mean([value[2] for value in splits.values()]))
        rows.append({"model": model, "task": task, "method": "linear", "metric": next(iter(splits.values()))[0], "mean": float(np.mean(means)), "ci95": float(t.ppf(0.975, 4) * np.std(means, ddof=1) / np.sqrt(5)), "baseline_mean": baseline, "seeds": 5, "outer_evaluations": 25})
    require(len(seen_jobs) == 1750, "Incomplete evaluation count")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "scope": "PPG-BP, seven benchmark objects, linear only, all ten tasks", "completed_evaluations": len(seen_jobs), "full_reproduction_complete": False, "checks": ["five seeds and five folds per model/task", "218 subjects per outer partition", "no train/test subject overlap", "disjoint test folds covering all subjects per seed", "prediction subjects agree with test split", "finite numeric predictions and primary metrics", "SBP/DBP subject-mean MAE recomputed from predictions"], "artifact_set_sha256": artifacts.hexdigest(), "artifact_hash_order": "CLI root order, sorted model run manifests, upstream job order, result then prediction SHA-256", "environment_evidence": evidence, "note": "Checkpoints and source revisions are separately recorded in data_manifests/ppg_bench_assets.json and model_assets.json. Raw data, split lists and predictions stay local; this summary is not the full-paper result."}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "completed_evaluations": len(seen_jobs), "summary": str(args.output / "summary.csv")}))


if __name__ == "__main__":
    main()
