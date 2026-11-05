"""Small end-to-end checks for respiration, SpO2, affect and quality tasks.

Only one fixed split and at most eight people are used. Outputs are plumbing
evidence, never a performance ranking. Existing published results are untouched.
Run in the existing ppg-bench venv after prepare_ppg_expansion_assets.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ppg_model_expansion_pilot import extract_features, fit_projection, project, validate_roles
from scripts.ppg_sqi_poc import handcrafted, weights
from scripts.ppg_hr_poc_features import sha256, write_json

MODELS = ("AnyPPG", "PaPaGei-P", "SigmaPPG")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def split_people(people):
    people = sorted(set(people))[:8]
    if len(people) != 8:
        raise ValueError("Pilot needs eight distinct people")
    return {"train": people[:4], "validation": people[4:6], "test": people[6:]}


def roles_for(subjects, split):
    roles = np.array([next(role for role, ids in split.items() if subject in ids) for subject in subjects])
    validate_roles(np.asarray(subjects), roles)
    return roles


def manual_rate(samples, start, stop, fs=125):
    """Rate from complete manual inter-breath intervals within one window.

    Both annotators use the same boundary rule; no PPG prediction enters the
    reference. Official CSV preserves MATLAB one-based signal sample numbers,
    so subtract one before applying the half-open time-window boundaries.
    """
    times = (np.asarray(samples, dtype=float)-1) / fs
    times = times[np.isfinite(times) & (times >= start) & (times < stop)]
    if len(times) < 3 or np.any(np.diff(times) <= 0):
        return np.nan
    return float(60 / np.diff(times).mean())


def load_breaths(path):
    rows = read_csv(path)
    names = list(rows[0])
    if len(names) != 2:
        raise ValueError("Expected two breath annotators")
    return [np.array([float(r[name]) for r in rows if r[name].strip()]) for name in names]


def bidmc_data():
    raw = ROOT / "datasets/ppg_bench_raw/bidmc/bidmc_csv"
    cohort = []
    for fix in sorted(raw.glob("*_Fix.txt")):
        text = fix.read_text()
        match = re.search(r"MIMIC II matched wdb ID:\s*(\S+)", text)
        if not match:
            raise ValueError("Missing parent patient ID")
        cohort.append((match[1], fix.name.split("_")[1]))
    split = split_people([x[0] for x in cohort])
    selected = set(sum(split.values(), []))
    waveforms, rows, source_hashes = [], [], {}
    disagreements = 0
    used_patients = set()
    for patient, rid in cohort:
        if patient not in selected or patient in used_patients:
            continue
        used_patients.add(patient)
        paths = {kind: raw / f"bidmc_{rid}_{kind}" for kind in ("Signals.csv", "Numerics.csv", "Breaths.csv", "Fix.txt")}
        for path in paths.values():
            source_hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
        signal = np.genfromtxt(paths["Signals.csv"], delimiter=",", names=True)
        numeric = np.genfromtxt(paths["Numerics.csv"], delimiter=",", names=True)
        breaths = load_breaths(paths["Breaths.csv"])
        starts = np.arange(0, len(signal) // 125 - 31, 32, dtype=int)
        starts = starts[np.linspace(0, len(starts)-1, min(8, len(starts)), dtype=int)]
        for start in starts:
            rates = [manual_rate(x, start, start+32) for x in breaths]
            x = signal["PLETH"][start*125:(start+32)*125]
            target_values = numeric["SpO2"][(numeric["Time_s"] >= start) & (numeric["Time_s"] < start+32)]
            if not np.isfinite(rates).all() or abs(rates[0]-rates[1]) > 2:
                disagreements += 1
                continue
            if not np.isfinite(x).all() or len(x) != 4000 or len(target_values) != 32 or not np.isfinite(target_values).all():
                continue
            waveforms.append(x)
            rows.append({"subject": patient, "record": rid, "start_s": int(start),
                         "resp_manual": float(np.mean(rates)), "annotator_difference_bpm": abs(rates[0]-rates[1]),
                         "spo2_monitor": float(target_values.mean())})
    x = np.asarray(waveforms, dtype=np.float32)
    return x, rows, split, {"source_files_sha256": source_hashes, "fs": 125, "seconds": 32,
        "reference": "Manual two-annotator mean of within-window inter-breath rates; agreement <=2 breaths/min; monitor SpO2 average",
        "manual_reference_rejected_windows": disagreements, "subject_identity": "MIMIC-II parent ID from Fix.txt, not file number",
        "pilot_record_selection": "First recording per selected patient; at most 8 evenly spaced 32-second windows; no labels used to choose recording"}


def processed_data(name, target, classes):
    root = ROOT / "datasets/ppg_bench_processed/fs_125" / name
    split = split_people([p.name for p in root.iterdir() if p.is_dir()])
    xs, rows, hashes = [], [], {}
    for subject in sum(split.values(), []):
        path = root / subject
        x = np.load(path / "ppg.npy", mmap_mode="r")
        y = np.load(path / f"{target}.npy")
        metadata = read_csv(path / "metadata.csv")
        if len(x) != len(y) or len(x) != len(metadata):
            raise ValueError("Processed label/waveform/identity count mismatch")
        valid = np.flatnonzero(np.isin(y, classes))
        selected = valid[np.linspace(0, len(valid)-1, min(18, len(valid)), dtype=int)]
        for i in selected:
            xs.append(x[i])
            rows.append({"subject": subject, "record": subject, "processed_row": int(i),
                         "start_s": float(metadata[i]["start_sec"]), target: int(y[i])})
        for name in ("ppg.npy", f"{target}.npy", "metadata.csv"):
            hashes[(path/name).relative_to(ROOT).as_posix()] = sha256(path/name)
    return np.asarray(xs, np.float32), rows, split, {"source_files_sha256": hashes, "fs": 125, "seconds": 10,
        "reference": "Existing pinned preprocessing: WESAD laboratory conditions / ECSMP video stimulus identity; not psychiatric diagnosis"}


def sqi_data():
    source = ROOT / ".local/ppg-phase2/sqi"
    rows_all = read_csv(source / "rows.csv")
    split = split_people([r["subject"] for r in rows_all])
    x = np.load(source / "x125.npy", mmap_mode="r")
    chosen = []
    for subject in sum(split.values(), []):
        indices = [i for i,r in enumerate(rows_all) if r["subject"] == subject]
        chosen.extend(np.asarray(indices)[np.linspace(0,len(indices)-1,min(18,len(indices)),dtype=int)].tolist())
    rows = [{"subject": rows_all[i]["subject"], "record": rows_all[i]["record_id"], "start_s": 0,
             "quality": int(rows_all[i]["quality"])} for i in chosen]
    return np.array(x[chosen]), rows, split, {"fs":125,"seconds":10,
        "source_manifest_sha256":sha256(source / "data_manifest.json"),
        "source_x_sha256":sha256(source / "x125.npy"), "source_rows_sha256":sha256(source / "rows.csv"),
        "reference":"Original 10-second expert HR-readability quality; no shortening to 8 seconds"}


def metric(y, p, subjects, kind):
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite predictions")
    if kind == "regression":
        return float(np.mean([mean_absolute_error(y[subjects==s],p[subjects==s]) for s in np.unique(subjects)]))
    return float(balanced_accuracy_score(y, p, sample_weight=weights(subjects)))


def fit_one(z, y, subjects, roles, kind):
    train, val, test = [roles == role for role in ("train","validation","test")]
    scale = StandardScaler().fit(z[train])
    z = scale.transform(z)
    if kind == "classification" and not set(y).issubset(set(y[train])):
        raise ValueError("Training people do not cover all pilot classes")
    best = None
    for regularization in (0.1,1.,10.):
        model = (Ridge(alpha=regularization) if kind == "regression" else
                 LogisticRegression(C=1/regularization, max_iter=2000, random_state=42))
        model.fit(z[train], y[train], sample_weight=weights(subjects[train]))
        score = metric(y[val], model.predict(z[val]), subjects[val], kind)
        selection = score if kind == "regression" else -score
        if best is None or selection < best[0]:
            best = (selection, regularization, model)
    prediction = best[2].predict(z[test])
    return prediction, {"selected_regularization":best[1],"train_rows":int(train.sum()),"validation_rows":int(val.sum()),
                        "test_rows":int(test.sum()),"input_features":z.shape[1]}


def run_dataset(name, data, tasks, output, device):
    x, rows, split, evidence = data
    subjects = np.array([r["subject"] for r in rows])
    roles = roles_for(subjects, split)
    if len(x) > 160 or len(set(subjects)) > 8:
        raise ValueError("Small-sample pilot size cap exceeded")
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "input.npz", x=x, subject=subjects, role=roles)
    (folder / "rows.json").write_text(json.dumps(rows, indent=2)+"\n")
    representations = {"HandFeatures":handcrafted(x,fs=evidence["fs"])}
    encoder_evidence = {}
    for model in MODELS:
        z, info = extract_features(model,x,evidence["fs"],device=device,batch_size=8)
        dimensions = min(16, int((roles=="train").sum())-1)
        transforms = fit_projection(z[roles=="train"], n_components=dimensions)
        representations[model] = project(z,transforms)
        encoder_evidence[model] = info
        np.save(folder / f"{model}-features.npy", z)
    results, predictions = [], []
    test = roles == "test"
    for target, kind, unit in tasks:
        y = np.array([r[target] for r in rows])
        for method,z in representations.items():
            p, details = fit_one(z,y,subjects,roles,kind)
            results.append({"dataset":name,"target":target,"method":method,"metric":"MAE" if kind=="regression" else "balanced_accuracy",
                            "value":metric(y[test],p,subjects[test],kind),"unit":unit,**details})
            for i, pred in zip(np.flatnonzero(test),p):
                predictions.append({"target":target,"method":method,**rows[i],"prediction":float(pred)})
        if kind == "regression":
            baseline = np.average(y[roles=="train"],weights=weights(subjects[roles=="train"]))
        else:
            values = np.unique(y[roles=="train"])
            counts = [weights(subjects[roles=="train"])[y[roles=="train"]==v].sum() for v in values]
            baseline = values[np.argmax(counts)]
        p=np.full(test.sum(),baseline)
        results.append({"dataset":name,"target":target,"method":"TrainConstant","metric":"MAE" if kind=="regression" else "balanced_accuracy",
                        "value":metric(y[test],p,subjects[test],kind),"unit":unit,"train_rows":int((roles=="train").sum()),
                        "validation_rows":int((roles=="validation").sum()),"test_rows":int(test.sum())})
        for i, pred in zip(np.flatnonzero(test),p):
            predictions.append({"target":target,"method":"TrainConstant",**rows[i],"prediction":float(pred)})
    write_json(folder / "predictions.json", predictions)
    report = {"dataset":name,"status":"small_sample_pipeline_verified_not_performance_evidence","split":split,
              "windows":len(rows),"subjects":len(set(subjects)),"tasks":[t[0] for t in tasks],"data":evidence,
              "rows_sha256":sha256(folder/"rows.json"),"input_sha256":sha256(folder/"input.npz"),
              "encoders":encoder_evidence,"results":results,"predictions_sha256":sha256(folder/"predictions.json"),
              "selection":"First eight sorted people; at most 18 evenly spaced eligible windows per person (BIDMC at most 8); single split/seed",
              "code_sha256":sha256(Path(__file__)),
              "head":"train-only standardization/PCA min(16, training rows minus 1) for foundation features; Ridge or logistic; 3 validation-only regularization choices",
              "pretraining":False,"encoder_finetuning":False}
    write_json(folder / "report.json",report)
    print(json.dumps({"dataset":name,"windows":len(rows),"outputs":len(results),"status":"passed"}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--datasets",nargs="+",choices=["BIDMC","WESAD","ECSMP","BUT-PPG"],default=["BIDMC","WESAD","ECSMP","BUT-PPG"])
    parser.add_argument("--output",type=Path,default=ROOT/".local/ppg-expansion-pilot/additional-tasks")
    args=parser.parse_args()
    reports=[]
    for name in args.datasets:
        if name=="BIDMC":
            data=bidmc_data(); tasks=[("resp_manual","regression","breaths/min"),("spo2_monitor","regression","percentage points")]
        elif name=="WESAD":
            data=processed_data(name,"condition_3class",[0,1,2]);tasks=[("condition_3class","classification","fraction")]
        elif name=="ECSMP":
            data=processed_data(name,"stimulus",list(range(6)));tasks=[("stimulus","classification","fraction")]
        else:
            data=sqi_data();tasks=[("quality","classification","fraction")]
        with threadpool_limits(limits=4):
            reports.append(run_dataset(name,data,tasks,args.output,args.device))
    shared=ROOT/"docs/results/ppg_expansion_pilot"
    shared.mkdir(parents=True,exist_ok=True)
    # Include completed sibling pilot reports when rerunning only one dataset.
    reports = [json.loads(path.read_text()) for path in sorted(args.output.glob("*/report.json"))]
    write_json(shared/"additional_tasks_pilot.json",{"summary_code_sha256":sha256(Path(__file__)),"reports":reports,
        "interpretation":"Integration checks only; small chosen sample cannot support model rankings or clinical claims"})


if __name__=="__main__":
    main()
