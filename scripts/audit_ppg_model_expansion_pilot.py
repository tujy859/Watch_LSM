"""Recompute small-pilot predictions and refit its 14 Ridge heads independently."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ppg_model_expansion_pilot import METHODS, validate_roles
from scripts.ppg_hr_poc_features import sha256, write_json


def main():
    local = ROOT / ".local/ppg-expansion-pilot/models"
    shared = ROOT / "docs/results/ppg_expansion_pilot"
    report = json.loads((shared / "models_pilot.json").read_text())
    with (shared / "models_pilot_scores.csv").open(newline="", encoding="utf-8") as stream:
        summaries = list(csv.DictReader(stream))
    errors, refits, mlp_reloads, checked = [], [], [], 0
    for dataset in ("Dalia", "WildPPG"):
        path = local / dataset
        rows = np.load(path / "rows.npz")
        validate_roles(rows["subject"], rows["role"])
        train, val, test = (rows["role"] == role for role in ("train", "validation", "test"))
        for item in [r for r in summaries if r["dataset"] == dataset]:
            name, head = item["method"], item["head"]
            suffix = f"{name}-{head}" if head != "task_control" else name
            result_path = path / f"{suffix}-prediction.npz"
            assert sha256(result_path) == item["prediction_sha256"]
            result = np.load(result_path)
            np.testing.assert_array_equal(result["row"], rows["row"][test])
            np.testing.assert_array_equal(result["reference"], rows["reference"][test])
            np.testing.assert_array_equal(result["subject"], rows["subject"][test])
            valid = np.isfinite(result["prediction"])
            error = np.abs(result["prediction"][valid] - result["reference"][valid])
            ids = result["subject"][valid]
            mean = np.mean([np.mean(error[ids == subject]) for subject in np.unique(ids)])
            errors.append(abs(mean - float(item["test_mae_bpm"])))
            assert np.isclose(valid.mean(), float(item["coverage"]))
            checked += 1
        for name in METHODS:
            features = np.load(path / f"{name}-features.npy")
            projection = np.load(path / f"{name}-projection.npz")
            # Applying the saved transform independently must yield a centered
            # training matrix, not a transform fitted on all three partitions.
            np.testing.assert_allclose(projection["mean"], features[train].astype(float).mean(axis=0), atol=1e-12)
            z = ((features.astype(float)-projection["mean"])/projection["scale"]-projection["pca_mean"]) @ projection["components"].T
            z = (z-projection["output_mean"])/projection["output_scale"]
            np.testing.assert_allclose(z[train].mean(axis=0), 0, atol=1e-10)
            ridge_info = json.loads((path / f"{name}-PCA16-Ridge.json").read_text())
            scaler = StandardScaler().fit(z[train])
            independent = Ridge(alpha=ridge_info["alpha"], solver="svd").fit(scaler.transform(z[train]), rows["reference"][train])
            independent_prediction = independent.predict(scaler.transform(z[test]))
            published = np.load(path / f"{name}-PCA16-Ridge-prediction.npz")["prediction"]
            refits.append(float(np.max(np.abs(independent_prediction-published))))
            mlp_info = json.loads((path / f"{name}-PCA16-MLP.json").read_text())
            model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
            model.load_state_dict(torch.load(path / f"{name}-mlp.pt", map_location="cpu", weights_only=True))
            model.eval()
            with torch.inference_mode():
                independent_prediction = model(torch.tensor(z[test], dtype=torch.float32)).squeeze(-1).numpy()*mlp_info["target_scale"]+mlp_info["target_center"]
            published = np.load(path / f"{name}-PCA16-MLP-prediction.npz")["prediction"]
            mlp_reloads.append(float(np.max(np.abs(independent_prediction-published))))
    for run in report["feature_runs"]:
        assert run["state_before_sha256"] == run["state_after_sha256"]
        assert run["encoder_trainable_parameters"] == 0
        path = local / run["dataset"] / f"{run['model']}{'-random-frozen' if run['random_init'] else ''}-features.npy"
        assert sha256(path) == run["features_sha256"]
    for dataset in ("Dalia", "WildPPG"):
        pair = [r for r in report["feature_runs"] if r["dataset"] == dataset and r["model"] == "AnyPPG"]
        assert len(pair) == 2 and pair[0]["encoder_parameters"] == pair[1]["encoder_parameters"]
        assert pair[0]["state_before_sha256"] != pair[1]["state_before_sha256"]
        assert pair[1]["random_init"] and not pair[1]["pretrained_weights_loaded"] and not pair[1]["checkpoint_sha256"]
    assert checked == 36 and max(errors) < 1e-10 and max(refits) < 1e-8 and max(mlp_reloads) < 1e-5
    output = {"status": "passed", "prediction_files_checked": checked,
              "ridge_independent_sklearn_refits": len(refits), "mlp_weight_reloads": len(mlp_reloads),
              "maximum_summary_recompute_difference_bpm": max(errors),
              "maximum_independent_ridge_prediction_difference_bpm": max(refits),
              "maximum_reloaded_mlp_prediction_difference_bpm": max(mlp_reloads),
              "subject_splits_disjoint": True, "random_anyppg_same_parameter_count_different_weights": True,
              "frozen_state_pairs_checked": len(report["feature_runs"]),
              "pilot_report_sha256": sha256(shared / "models_pilot.json"),
              "scores_sha256": sha256(shared / "models_pilot_scores.csv"),
              "audit_script_sha256": sha256(Path(__file__)),
              "scope": "prediction integrity and small workflow only, not full efficacy evaluation"}
    write_json(shared / "models_pilot_audit.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
