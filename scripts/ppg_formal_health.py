"""Person-level health increment evaluation with frozen PPG representations.

ECSMP: 60 people, continuous 120 s neutral wrist BVP, 15 nonoverlapping 8 s
encoder inputs. PPG-BP: 218 people, first 2.0 s of each of three distinct
clips, not a continuous PRV trace. Labels never select people or time spans.
Five fixed person folds use the following fold for validation; three seeds
vary head/random-encoder initialization, not the held-out cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import resample_poly
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.ppg_health_increment_pilot import (
    workbook_rows, continuous_core, pulse_features, signal_features,
    feature_groups, fit_preprocessor, apply_preprocessor,
)
from scripts.ppg_model_expansion_pilot import MODELS, _upstream, _weights, _state_hash
from scripts.ppg_hr_poc_features import sha256

SEEDS = (42, 43, 44)
ALPHAS = (.1, 1., 10., 100., 1000.)
GROUPS = ("demographics", "demographics_traditional", "embedding",
          "demographics_embedding", "demographics_traditional_embedding")
RANDOM = "AnyPPG-random-frozen"
PROTOCOL = {
    "version": "formal-health-v1", "split_seed": 20261105,
    "five_folds": "test fold k, validation fold (k+1)%5, other three train",
    "initialization_seeds": list(SEEDS), "pca_components": 8,
    "ridge_alphas": list(ALPHAS), "mlp_hidden": 16, "mlp_epochs": 100,
    "mlp_lr": .01, "mlp_weight_decay": 1e-4,
    "mlp_selection": "lowest validation MAE; earliest epoch breaks ties",
    "equal_head_width": "zero-pad all input groups to twice the full fused dimension after training-only missing flags/scaling",
    "primary_comparison": "demographics+traditional+embedding vs demographics+traditional, linear head",
    "secondary_comparison": "same input groups with fixed small MLP; standalone and demographic-only increment",
    "testing_unit": "one person, seed-averaged absolute error; no windows as independent observations",
    "multiple_comparisons": "Holm across six pretrained-model fusion increments separately for each of SDS, SBP, DBP; other comparisons exploratory",
    "bootstrap": "10000 paired person resamples, percentile 95% intervals; fixed split uncertainty not fully represented",
    "permutation": "100000 paired person sign flips, two-sided; overlapping training folds can induce dependent errors",
    "encoders_frozen": True, "pretraining": False,
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def array_hash(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(x.dtype.str.encode()+str(x.shape).encode()+x.tobytes()).hexdigest()


def person_folds(subjects):
    """Stable identity-only outer folds, with disjoint inner validation people."""
    people = np.asarray(sorted(map(str, subjects)))
    if len(people) != len(set(people)) or len(people) < 20:
        raise ValueError("Need at least 20 distinct people")
    folds = np.array_split(np.random.default_rng(PROTOCOL["split_seed"]).permutation(people), 5)
    result = []
    for k in range(5):
        test, validation = folds[k].tolist(), folds[(k+1)%5].tolist()
        train = [p for j, f in enumerate(folds) if j not in (k, (k+1)%5) for p in f.tolist()]
        if set(train)&set(validation) or set(train)&set(test) or set(validation)&set(test):
            raise ValueError("Person leakage")
        result.append({"fold": k, "train": train, "validation": validation, "test": test})
    return result


def build_data(dataset):
    """Read raw signals and labels without looking at model outcomes."""
    cov, traditional, targets, windows, owners, stats, intervals = [], [], [], [], [], [], []
    hashes = {}
    if dataset == "PPG-BP":
        raw_root = ROOT / "datasets/ppg_bp/Data File"
        workbook = raw_root / "PPG-BP dataset.xlsx"
        raw_rows = workbook_rows(workbook)
        header = {v: k for k, v in raw_rows[1].items()}
        records = {str(int(float(r[header["subject_ID"]]))): r for r in raw_rows[2:]}
        people = sorted(set(records)-{"231"})
        if len(people) != 218:
            raise ValueError("PPG-BP participant inventory changed")
        hashes[workbook.relative_to(ROOT).as_posix()] = sha256(workbook)
        for subject in people:
            r = records[subject]
            if r[header["Sex(M/F)"]] not in ("Male", "Female"):
                raise ValueError("Missing sex")
            cov.append([float(r[header["Age(year)"]]), float(r[header["Sex(M/F)"]]=="Male"), float(r[header["BMI(kg/m^2)"]])])
            targets.append([float(r[header["Systolic Blood Pressure(mmHg)"]]), float(r[header["Diastolic Blood Pressure(mmHg)"]])])
            clips = []
            for j in (1, 2, 3):
                path = raw_root / "0_subject" / f"{subject}_{j}.txt"
                raw = np.fromstring(path.read_text(), sep="\t")
                if raw.shape != (2100,) or not np.isfinite(raw).all():
                    raise ValueError("Expected complete 2100-point source clip")
                core = raw[:2000]
                clips.append(core)
                windows.append(resample_poly(core, 1, 8).astype(np.float32))
                owners.append(subject)
                hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
                intervals.append({"subject": subject, "clip": j, "source_fs": 1000, "start_sample": 0, "stop_sample": 2000})
            all_observed = np.concatenate(clips)
            stats.extend([[float(all_observed.mean()), float(all_observed.std())]]*3)
            traditional.append([float(r[header["Heart Rate(b/m)"]]), *np.mean([signal_features(x, 1000) for x in clips], axis=0)])
        inventory = {"eligible_people": 218, "raw_people": 219, "excluded_ids": ["231"],
                     "disease_labels_used": False, "observation": "First 2.0 s of each of three separate 2.1 s fingertip clips; total 6.0 s, never joined for PRV"}
        target_names = ["cuff_sbp_mmhg", "cuff_dbp_mmhg"]
        covariate_names = ["age_years", "male_indicator", "bmi_kg_m2"]
        limitations = ["Finger auxiliary domain, not wrist blood pressure validation", "One cuff label per person, no continuous ABP", "Source clips shortened from 2.1 to 2.0 s for complete common model patches; not numerically interchangeable with old protocol", "Traditional baseline includes independently recorded table HR", "Pretraining overlap with source patients not certified"]
    elif dataset == "ECSMP":
        raw_root = ROOT / "datasets/ppg_bench_raw/ecsmp"
        processed = ROOT / "datasets/ppg_bench_processed/fs_125/ECSMP"
        workbook = raw_root / "scale.xlsx"
        source = workbook_rows(workbook, "SDS")
        labels = {str(int(float(row[0]))).zfill(3): row for row in source[1:] if row.get(0)}
        people = sorted(p.name for p in processed.iterdir() if p.is_dir())
        if len(people) != 60:
            raise ValueError("ECSMP participant inventory changed")
        hashes[workbook.relative_to(ROOT).as_posix()] = sha256(workbook)
        for subject in people:
            r = labels[subject]
            if r[1] not in ("Female", "Male"):
                raise ValueError("Missing sex")
            cov.append([float(r[2]), float(r[1]=="Male")]); targets.append([float(r[25])])
            info = json.loads((processed/subject/"stats.json").read_text())
            if float(r[25]) != float(info["sds_score"]):
                raise ValueError("Raw and processed SDS disagree")
            metadata_path = processed/subject/"metadata.csv"
            with metadata_path.open(newline="") as stream:
                neutral = [x for x in csv.DictReader(stream) if x["stimulus"]=="0" and x["has_video_label"]=="1"]
            start = min(float(x["start_sec"]) for x in neutral)
            if start+120 > min(float(x["video_end_sec"]) for x in neutral):
                raise ValueError("Insufficient continuous neutral observation")
            path = raw_root/"E4"/subject/"BVP.csv"
            with path.open() as stream:
                utc, fs = float(stream.readline()), int(float(stream.readline()))
            if fs != 64 or utc != float(info["bvp_start_utc"]):
                raise ValueError("Raw BVP timing differs from stimulus metadata")
            raw = np.loadtxt(path, delimiter=",", skiprows=2)
            core, extended, interval = continuous_core(raw, fs, start)
            interval.update(subject=subject, start_seconds=start, bvp_start_utc=utc)
            intervals.append(interval)
            resampled = resample_poly(core, 125, 64).astype(np.float32)
            pieces = resampled.reshape(15, 1000)
            windows.extend(pieces); owners.extend([subject]*15)
            stats.extend([[float(core.mean()), float(core.std())]]*15)
            traditional.append([*np.mean([signal_features(x, 125) for x in pieces], axis=0), *pulse_features(extended, fs)])
            hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
            hashes[metadata_path.relative_to(ROOT).as_posix()] = sha256(metadata_path)
        inventory = {"eligible_people": 60, "age_range": [min(x[0] for x in cov), max(x[0] for x in cov)],
                     "sex_counts": dict(Counter("Male" if x[1] else "Female" for x in cov)),
                     "sds_range": [min(x[0] for x in targets), max(x[0] for x in targets)],
                     "observation": "One continuous 120 s neutral raw BVP interval, 15 nonoverlapping 8 s clips; 2 s extra context only for conventional peak filtering"}
        target_names = ["sds_self_report_score"]; covariate_names = ["age_years", "male_indicator"]
        limitations = ["Self-reported SDS symptoms, not clinical depression diagnosis", "Only 60 young people aged 18-28", "Traditional PRV estimates have no ECG validation", "Stimulus timing uses source metadata, not new annotations", "Pretraining overlap not certified"]
    else:
        raise ValueError("Unknown dataset")
    data = {"dataset": dataset, "subjects": np.asarray(people), "covariates": np.asarray(cov, dtype=float),
            "traditional": np.asarray(traditional, dtype=float), "targets": np.asarray(targets, dtype=float),
            "windows": np.asarray(windows, dtype=np.float32), "waveform_subjects": np.asarray(owners),
            "stats": np.asarray(stats, dtype=np.float32), "fs": 125, "target_names": target_names,
            "covariate_names": covariate_names, "source_hashes": hashes, "intervals": intervals,
            "inventory": inventory, "limitations": limitations}
    if not np.isfinite(data["targets"]).all() or not np.isfinite(data["covariates"]).all():
        raise ValueError("Nonfinite labels or demographic data")
    return data


def extract_native_features(model_name, windows, fs, stats, *, device="cpu", seed=42,
                            random_init=False, batch_size=8):
    """Frozen official port with explicit whole-patch variable length support.

    Unlike the small-sample wrapper, MOMENT may receive a sequence shorter than
    512 if it contains complete 8-point patches. No padding is introduced.
    Both MOMENT and Sigma crop nothing: incomplete patches are rejected here.
    """
    import torch
    configs, revision = _upstream()
    from ppg_bench.models import external_module, get_port
    config = configs[model_name]
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all() or not len(x):
        raise ValueError("Invalid input")
    if fs != config["fs"]:
        x = resample_poly(x, config["fs"], fs, axis=-1).astype(np.float32)
    patch = 100 if model_name=="SigmaPPG" else 8 if model_name=="MOMENT-1-small" else 1
    if x.shape[-1] % patch:
        raise ValueError("Whole native patches required; implicit cropping forbidden")
    stats = np.asarray(stats, dtype=np.float32)
    if stats.shape != (len(x),2) or not np.isfinite(stats).all():
        raise ValueError("Invalid available-observation statistics")
    torch.set_num_threads(2)
    port = get_port(config)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.startswith("cuda") else []):
        torch.manual_seed(seed)
        if random_init:
            if model_name != "AnyPPG":
                raise ValueError("Only AnyPPG random architecture registered")
            from ppg_bench.encoders.anyppg import ANYPPG_CFG
            encoder = external_module(config["init"]["source_root"], "load_anyppg.resnet1d").Net1D(**ANYPPG_CFG)
            checkpoint = {}
        else:
            checkpoint = _weights(config); encoder = port.load_encoder()
    encoder.eval().requires_grad_(False).to(device)
    before = _state_hash(encoder)
    output, prepared_lengths = [], []
    started = time.monotonic()
    for a in range(0, len(x), batch_size):
        batch = {"x": torch.tensor(x[a:a+batch_size]), "ppg_mean": torch.tensor(stats[a:a+batch_size,0]), "ppg_std": torch.tensor(stats[a:a+batch_size,1])}
        with torch.inference_mode():
            prepared = port.prepare_batch(batch, config, device)
            prepared_lengths.append(int(prepared.shape[-1]))
            output.append(encoder(prepared).cpu().float().numpy())
    z = np.concatenate(output)
    after = _state_hash(encoder)
    if before != after or z.shape != (len(x), config["embedding_dim"]) or not np.isfinite(z).all():
        raise ValueError("Frozen encoder or representation audit failed")
    if set(prepared_lengths) != {x.shape[-1]}:
        raise ValueError("Native preparation changed input observation length")
    evidence = {"model": model_name, "random_init": random_init, "seed": seed, "device": device,
                "precision": "float32; no autocast", "input_sha256": array_hash(windows),
                "native_input_sha256": array_hash(x), "statistics_sha256": array_hash(stats),
                "normalization_scope": "unlabeled available observation only; 120 s per ECSMP person or all three observed 2 s PPG-BP clips",
                "native_fs": config["fs"], "native_input_samples": x.shape[-1], "prepared_samples": sorted(set(prepared_lengths)),
                "padding_samples": 0, "cropped_samples": 0, "native_patch_size": patch,
                "encoder_state_before": before, "encoder_state_after": after,
                "encoder_parameters": sum(p.numel() for p in encoder.parameters()), "trainable_encoder_parameters": 0,
                "checkpoint_sha256": checkpoint, "upstream_commit": revision, "feature_shape": list(z.shape),
                "seconds": time.monotonic()-started}
    del encoder, port
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return z, evidence


def fit_projection(raw, train):
    with threadpool_limits(limits=2):
        scale = StandardScaler().fit(raw[train])
        pca = PCA(n_components=8, svd_solver="full").fit(scale.transform(raw[train]))
        reduced = pca.transform(scale.transform(raw))
    return reduced, {"scale_mean": scale.mean_, "scale_scale": scale.scale_, "pca_mean": pca.mean_, "pca_components": pca.components_}


def prepared_group(x, train, width):
    fitted = fit_preprocessor(x[train])
    z = apply_preprocessor(x, fitted)
    if z.shape[1] > width:
        raise ValueError("Controlled head dimension exceeded")
    z = np.pad(z, ((0,0),(0,width-z.shape[1])))
    median, scale = fitted
    return z, {"imputation_median": median, "input_mean": scale.mean_, "input_scale": scale.scale_, "actual_columns": np.array(len(scale.mean_))}


def fit_head(z, y, train, validation, test, kind, seed):
    if any(set(a)&set(b) for a,b in ((train,validation),(train,test),(validation,test))):
        raise ValueError("Overlapping person indices")
    if kind == "linear":
        best = None; validation_candidates = []
        with threadpool_limits(limits=2):
            for alpha in ALPHAS:
                reg = Ridge(alpha=alpha).fit(z[train],y[train])
                score = float(np.mean(np.abs(reg.predict(z[validation])-y[validation])))
                validation_candidates.append({"alpha": alpha, "mae": score})
                if best is None or score < best[0]:
                    best = score, alpha, reg
        score, alpha, reg = best
        pred = reg.predict(z[test])
        centered = z[train]-z[train].mean(axis=0)
        beta = np.linalg.solve(centered.T@centered+alpha*np.eye(z.shape[1]),centered.T@(y[train]-y[train].mean()))
        independent = (z[test]-z[train].mean(axis=0))@beta+y[train].mean()
        state = {"coef": reg.coef_, "intercept": np.array(reg.intercept_)}
        meta = {"alpha": alpha, "validation_mae": score, "epochs": 0,
                "validation_candidates": validation_candidates,
                "trainable_head_parameters": z.shape[1]+1,
                "independent_prediction_max_difference": float(np.max(np.abs(pred-independent)))}
    elif kind == "mlp":
        import torch
        torch.set_num_threads(2)
        center, scale = float(y[train].mean()), max(float(y[train].std()),1.)
        tx = torch.tensor(z[train],dtype=torch.float32); vx = torch.tensor(z[validation],dtype=torch.float32)
        ty = torch.tensor((y[train]-center)/scale,dtype=torch.float32)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            head = torch.nn.Sequential(torch.nn.Linear(z.shape[1],16),torch.nn.ReLU(),torch.nn.Linear(16,1))
            optimizer = torch.optim.Adam(head.parameters(),lr=PROTOCOL["mlp_lr"],weight_decay=PROTOCOL["mlp_weight_decay"])
            best_score, best_epoch, best = float("inf"), None, None
            validation_curve = []
            for epoch in range(1,PROTOCOL["mlp_epochs"]+1):
                optimizer.zero_grad(set_to_none=True)
                loss = (head(tx).flatten()-ty).square().mean()
                loss.backward(); optimizer.step()
                with torch.no_grad():
                    val = head(vx).flatten().numpy()*scale+center
                score = float(np.mean(np.abs(val-y[validation])))
                validation_curve.append(score)
                if score < best_score:
                    best_score,best_epoch = score,epoch
                    best = {k:v.detach().clone() for k,v in head.state_dict().items()}
            head.load_state_dict(best)
            with torch.no_grad():
                pred = head(torch.tensor(z[test],dtype=torch.float32)).flatten().numpy().astype(float)*scale+center
        state = {"w1":best["0.weight"].numpy(),"b1":best["0.bias"].numpy(),
                 "w2":best["2.weight"].numpy(),"b2":best["2.bias"].numpy(),
                 "target_center":np.array(center),"target_scale":np.array(scale)}
        independent = (np.maximum(z[test].astype(np.float32)@state["w1"].T+state["b1"],0)@state["w2"].T+state["b2"]).flatten().astype(float)*scale+center
        meta = {"alpha": None,"validation_mae":best_score,"epochs":best_epoch,
                "validation_mae_by_epoch": validation_curve,
                "trainable_head_parameters":sum(p.numel() for p in head.parameters()),
                "independent_prediction_max_difference":float(np.max(np.abs(pred-independent)))}
    else:
        raise ValueError("Unknown head")
    if not np.isfinite(pred).all() or meta["independent_prediction_max_difference"]>1e-4:
        raise ValueError("Prediction independent reconstruction failed")
    return pred,state,meta


def prepare_dataset(dataset, output):
    folder = output/dataset
    folder.mkdir(parents=True,exist_ok=True)
    data = build_data(dataset)
    arrays = {k:data[k] for k in ("subjects","covariates","traditional","targets","windows","waveform_subjects","stats")}
    np.savez_compressed(folder/"data.npz",**arrays)
    meta = {k:v for k,v in data.items() if k not in arrays}
    meta.update(protocol=PROTOCOL,split=person_folds(data["subjects"]),input_sha256=array_hash(data["windows"]),
                data_npz_sha256=sha256(folder/"data.npz"),code_sha256=sha256(Path(__file__)))
    write_json(folder/"manifest.json",meta)
    return data,meta


def load_prepared(dataset, output):
    folder = output/dataset
    meta = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if meta["protocol"] != PROTOCOL or sha256(folder/"data.npz") != meta["data_npz_sha256"]:
        raise ValueError("Saved input/protocol changed")
    with np.load(folder/"data.npz",allow_pickle=False) as blob:
        data = {k:blob[k] for k in blob.files}
    return {**meta,**data},meta


def extract_dataset(dataset, output, models, device):
    data,meta = load_prepared(dataset,output)
    folder = output/dataset/"features"; folder.mkdir(exist_ok=True)
    for model in models:
        for seed in SEEDS if model==RANDOM else (42,):
            key = f"{model}__seed{seed}"
            path = folder/f"{key}.npz"; evidence_path = folder/f"{key}.json"
            if path.exists() and evidence_path.exists():
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                if evidence["input_sha256"] != array_hash(data["windows"]) or evidence["output_sha256"] != sha256(path):
                    raise ValueError("Feature cache identity mismatch")
                continue
            z,evidence = extract_native_features("AnyPPG" if model==RANDOM else model,data["windows"],data["fs"],data["stats"],
                                                 device=device,seed=seed,random_init=model==RANDOM)
            aggregate = np.asarray([z[data["waveform_subjects"]==person].mean(axis=0) for person in data["subjects"]])
            np.savez_compressed(path,subjects=data["subjects"],window_features=z,person_features=aggregate)
            evidence.update(output_sha256=sha256(path),aggregation="mean over fixed clips, one vector per person",data_sha256=meta["data_npz_sha256"])
            write_json(evidence_path,evidence)
            print(json.dumps({"dataset":dataset,"model":model,"seed":seed,"phase":"features_complete","seconds":evidence["seconds"]}),flush=True)


def fit_dataset(dataset,output,models):
    data,manifest = load_prepared(dataset,output)
    folder = output/dataset; predictions = folder/"predictions"; predictions.mkdir(exist_ok=True)
    subjects = data["subjects"]
    width = 2*(data["covariates"].shape[1]+data["traditional"].shape[1]+8)
    all_rows = []
    for split in manifest["split"]:
        train,validation,test = [np.flatnonzero(np.isin(subjects,split[role])) for role in ("train","validation","test")]
        for seed in SEEDS:
            # Shared controls and train mean are emitted once per fold/seed.
            for model in ["shared_control",*models]:
                if model=="shared_control":
                    reduced = np.zeros((len(subjects),8)); projection={}
                else:
                    source_seed = seed if model==RANDOM else 42
                    feature_path = folder/"features"/f"{model}__seed{source_seed}.npz"
                    with np.load(feature_path,allow_pickle=False) as blob:
                        if not np.array_equal(blob["subjects"],subjects):
                            raise ValueError("Feature person order differs")
                        raw = blob["person_features"].astype(float)
                    reduced,projection = fit_projection(raw,train)
                groups=feature_groups(data["covariates"],data["traditional"],reduced)
                for group,x in groups.items():
                    if (model=="shared_control") != (group in GROUPS[:2]):
                        continue
                    z,preprocess=prepared_group(x,train,width)
                    for target_index,target in enumerate(data["target_names"]):
                        y=data["targets"][:,target_index]
                        for head in ("linear","mlp"):
                            identity=f"{model}__{group}__{target}__{head}__s{seed}__f{split['fold']}"
                            path=predictions/f"{identity}.npz"; record=predictions/f"{identity}.json"
                            if path.exists() and record.exists():
                                row=json.loads(record.read_text())
                                if sha256(path)!=row["prediction_sha256"] or row["data_sha256"]!=manifest["data_npz_sha256"]:
                                    raise ValueError("Result cache identity mismatch")
                            else:
                                pred,state,evidence=fit_head(z,y,train,validation,test,head,seed)
                                np.savez_compressed(path,subjects=subjects[test],y_true=y[test],y_pred=pred,
                                                    train_subjects=subjects[train],validation_subjects=subjects[validation],
                                                    projected_features=reduced,prepared_features=z,
                                                    **state,**preprocess,**projection)
                                row={"dataset":dataset,"target":target,"model":model,"inputs":group,"head":head,
                                     "seed":seed,"fold":split["fold"],"train_people":len(train),"validation_people":len(validation),"test_people":len(test),
                                     "mae":float(np.mean(np.abs(pred-y[test]))),"rmse":float(np.sqrt(np.mean((pred-y[test])**2))),
                                     "bias":float(np.mean(pred-y[test])),"prediction_file":path.relative_to(output).as_posix(),
                                     "prediction_sha256":sha256(path),"data_sha256":manifest["data_npz_sha256"],**evidence}
                                write_json(record,row)
                            all_rows.append(row)
                if model=="shared_control":
                    for ti,target in enumerate(data["target_names"]):
                        y=data["targets"][:,ti]; pred=np.full(len(test),y[train].mean())
                        identity=f"train_mean__{target}__s{seed}__f{split['fold']}"
                        path=predictions/f"{identity}.npz"
                        np.savez_compressed(path,subjects=subjects[test],y_true=y[test],y_pred=pred,train_subjects=subjects[train],validation_subjects=subjects[validation])
                        all_rows.append({"dataset":dataset,"target":target,"model":"train_mean","inputs":"none","head":"constant","seed":seed,"fold":split["fold"],
                                         "train_people":len(train),"validation_people":len(validation),"test_people":len(test),"mae":float(np.mean(np.abs(pred-y[test]))),
                                         "rmse":float(np.sqrt(np.mean((pred-y[test])**2))),"bias":float(np.mean(pred-y[test])),
                                         "prediction_file":path.relative_to(output).as_posix(),"prediction_sha256":sha256(path),"data_sha256":manifest["data_npz_sha256"]})
            print(json.dumps({"dataset":dataset,"fold":split["fold"],"seed":seed,"phase":"fit_complete","results":len(all_rows)}),flush=True)
    write_json(folder/"results.json",all_rows)
    return all_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("prepare","features","fit","all"),default="prepare")
    parser.add_argument("--datasets",nargs="+",choices=("ECSMP","PPG-BP"),default=["ECSMP","PPG-BP"])
    parser.add_argument("--models",nargs="+",choices=(*MODELS,RANDOM),default=[*MODELS,RANDOM])
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--output",type=Path,default=ROOT/".local/ppg-formal-expansion/health")
    args=parser.parse_args()
    for dataset in args.datasets:
        if args.stage in ("prepare","all"):
            data,_=prepare_dataset(dataset,args.output)
            print(json.dumps({"dataset":dataset,"phase":"prepared","people":len(data["subjects"]),"input_shape":list(data["windows"].shape)}),flush=True)
        if args.stage in ("features","all"):
            extract_dataset(dataset,args.output,args.models,args.device)
        if args.stage in ("fit","all"):
            with threadpool_limits(limits=2):
                fit_dataset(dataset,args.output,args.models)


if __name__=="__main__":
    main()
