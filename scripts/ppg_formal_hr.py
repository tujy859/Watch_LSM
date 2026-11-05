"""Formal frozen-model HR extension using the original 93 subject splits.

Six pretrained encoders, three seed-matched random AnyPPG encoders, native
Ridge and identical PCA16 linear/MLP heads. No encoder optimization occurs.
Existing phase-one results are referenced and hashed, never overwritten.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from scripts.ppg_hr_poc_features import sha256, write_json, validate_data, forward_rows
from scripts.ppg_model_expansion_pilot import MODELS, _upstream, _weights, _state_hash, fit_small_head
from scripts.ppg_model_expansion_pilot import macro_mae

OLD = ROOT / ".local/ppg-hr-poc"
DEFAULT = ROOT / ".local/ppg-formal-expansion/hr"
DATASETS = ("Dalia", "WildPPG")
SEEDS = (42, 43, 44)
OLD_MODELS = ("AnyPPG", "PaPaGei-S", "PulsePPG", "MOMENT-1-small")
METHODS = (*MODELS, "AnyPPG-random-frozen")
HEADS = ("Native-Ridge", "PCA16-Ridge", "PCA16-MLP")
ALPHAS = (0.01, 0.1, 1., 10., 100., 1000.)


def validate_split(job):
    train, val, test = (set(job[k]) for k in ("train", "validation", "test"))
    if not all((train, val, test)) or train & val:
        raise ValueError("Missing subjects or training/validation overlap")
    if job["source"] == job["target"] and (train | val) & test:
        raise ValueError("Held-out subject leakage")
    if any(len(job[k]) != len(set(job[k])) for k in ("train", "validation", "test")):
        raise ValueError("Duplicate subject identity")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def feature_name(method, seed):
    return f"{method}-seed{seed}" if method == "AnyPPG-random-frozen" else method


def register(root):
    old = read_json(OLD / "protocol.json")
    assert len(old["jobs"]) == 93
    for job in old["jobs"]:
        validate_split(job)
    plan = {
        "version": "formal-hr-extension-v1", "models": list(MODELS),
        "methods": list(METHODS), "heads": list(HEADS), "seeds": list(SEEDS),
        "jobs": old["jobs"], "old_protocol_sha256": sha256(OLD / "protocol.json"),
        "data_rows_sha256": {d: sha256(OLD / "data" / d / "rows.npz") for d in DATASETS},
        "data_manifest_sha256": {d: sha256(OLD / "data" / d / "manifest.json") for d in DATASETS},
        "random_encoder": "Fresh official AnyPPG constructor for each seed 42/43/44; paired to each split seed; no checkpoint or encoder optimization",
        "projection": "Training windows only: unweighted feature StandardScaler, exact covariance-eigendecomposition PCA16, output StandardScaler. Person-level sufficient statistics are algebraically equivalent to stacking all training windows. PCA sign is fixed by largest loading.",
        "ridge": {"alphas": list(ALPHAS), "fit": "person-balanced feature standardization and squared loss; select validation person MAE; no train+validation refit"},
        "mlp": {"width": 16, "parameters": 289, "epochs": 40, "lr": 0.01, "weight_decay": 1e-4,
                "fit": "full-batch person-balanced squared error; training target mean/std scaling; validation person MAE selects epoch; seed equals split seed"},
        "input": "All original finite eight-second windows; native fixed preprocessing. PulsePPG whole-record unlabeled normalization remains offline.",
        "encoder_optimization": False,
        "primary_comparisons": "Within each of 7 scenarios: six pretrained versus SmallCNN using PCA16-MLP; AnyPPG pretrained versus seed-matched random using PCA16-MLP. Paired person-level differences, Holm across these 7 planned comparisons within scenario.",
        "secondary": "Native Ridge, PCA16 Ridge, head changes and traditional comparisons are descriptive/exploratory; bootstrap resamples people after averaging the 3 seed errors.",
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "protocol.json"
    if path.exists() and read_json(path) != plan:
        raise ValueError("Registered formal HR protocol changed")
    if not path.exists():
        write_json(path, plan)
    return plan


def paths(root, dataset, method, seed=42):
    name = feature_name(method, seed)
    dest = root / "features" / dataset
    return dest / f"{name}.npy", dest / f"{name}.json"


def features(root, dataset, method, seed, device="cuda", batch_size=64):
    final, meta = paths(root, dataset, method, seed)
    meta.parent.mkdir(parents=True, exist_ok=True)
    rows_hash = sha256(OLD / "data" / dataset / "rows.npz")
    if meta.exists():
        info = read_json(meta)
        actual_path = ROOT / info["path"]
        if info["rows_sha256"] != rows_hash or sha256(actual_path) != info["features_sha256"]:
            raise ValueError("Feature cache content or row identity changed")
        return info
    if method in OLD_MODELS:
        source = OLD / "features" / dataset / f"{method}.npy"
        source_meta = OLD / "features" / dataset / f"{method}.json"
        previous = read_json(source_meta)
        if previous["status"] != "complete" or previous["identity"]["data"]["files_sha256"]["rows.npz"] != rows_hash:
            raise ValueError("Old feature rows/status mismatch")
        if sha256(source) != previous["features_sha256"]:
            raise ValueError("Old feature hash mismatch")
        info = {"status": "complete", "path": source.relative_to(ROOT).as_posix(), "model": method,
                "dataset": dataset, "rows_sha256": rows_hash, "features_sha256": previous["features_sha256"],
                "source_metadata_sha256": sha256(source_meta), "reused": True, "shape": previous["shape"]}
        write_json(meta, info)
        return info
    import torch
    configs, revision = _upstream()
    from ppg_bench.models import get_port, external_module
    random = method == "AnyPPG-random-frozen"
    model_name = "AnyPPG" if random else method
    cfg = configs[model_name]
    x, stats, _, data = validate_data(OLD / "data" / dataset, cfg["fs"])
    identity = {"dataset": dataset, "model": method, "seed": seed if random else None,
                "data": data, "upstream_commit": revision, "weights": {} if random else _weights(cfg),
                "config": cfg, "device": device, "precision": "float32", "deterministic_cudnn": True,
                "extractor_sha256": sha256(Path(__file__))}
    partial, progress_path = final.with_suffix(".partial.npy"), final.with_suffix(".progress.json")
    if progress_path.exists():
        progress = read_json(progress_path)
        if progress["identity"] != identity:
            raise ValueError("Partial feature identity changed")
        done = progress["completed"]
        output = np.load(partial, mmap_mode="r+")
        if output.shape != (len(x), cfg["embedding_dim"]) or not np.isfinite(output[:done]).all():
            raise ValueError("Corrupt partial feature cache")
    else:
        if partial.exists() or final.exists():
            raise ValueError("Unregistered feature output")
        done = 0
        output = np.lib.format.open_memmap(partial, mode="w+", dtype="float32", shape=(len(x), cfg["embedding_dim"]))
        progress = {"identity": identity, "completed": 0}
        write_json(progress_path, progress)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    port = get_port(cfg)
    if random:
        from ppg_bench.encoders.anyppg import ANYPPG_CFG
        source = external_module(cfg["init"]["source_root"], "load_anyppg.resnet1d")
        encoder = source.Net1D(**ANYPPG_CFG)
    else:
        encoder = port.load_encoder()
    encoder.float().eval().requires_grad_(False).to(device)
    state_before = _state_hash(encoder)
    nparams = sum(p.numel() for p in encoder.parameters())
    started = time.monotonic()
    last = started
    while done < len(x):
        ix = np.arange(done, min(len(x), done + batch_size))
        try:
            values = forward_rows(torch, port, encoder, cfg, x, stats, ix, device)
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 16:
                raise
            batch_size //= 2
            torch.cuda.empty_cache()
            continue
        output[ix] = values
        done += len(ix)
        if time.monotonic() - last > 30 or done == len(x):
            output.flush()
            progress.update(completed=done, elapsed_current_seconds=time.monotonic()-started)
            write_json(progress_path, progress)
            print(json.dumps({"dataset":dataset,"method":feature_name(method,seed),"completed":done,"total":len(x)}), flush=True)
            last = time.monotonic()
    if _state_hash(encoder) != state_before:
        raise ValueError("Frozen encoder state changed")
    shape = list(output.shape)
    del output, encoder, port
    gc.collect()
    torch.cuda.empty_cache()
    partial.replace(final)
    info = {"status": "complete", "path": final.relative_to(ROOT).as_posix(), "model":method,
            "dataset":dataset,"rows_sha256":rows_hash,"features_sha256":sha256(final),"shape":shape,
            "reused":False,"identity":identity,"encoder_parameters":nparams,"encoder_state_before":state_before,
            "encoder_state_after":state_before,"encoder_trainable_parameters":0,"elapsed_current_seconds":time.monotonic()-started}
    write_json(meta, info)
    return info


def load_feature(root, dataset, method, seed):
    _, p = paths(root, dataset, method, seed)
    info = read_json(p)
    return np.load(ROOT / info["path"], mmap_mode="r"), info


def sufficient_statistics(x, y, subjects):
    """Exact unweighted per-person sums; no label selection or test fitting."""
    result = {}
    for person in np.unique(subjects):
        selected = subjects == person
        a, b = np.asarray(x[selected], dtype=np.float64), np.asarray(y[selected], dtype=np.float64)
        result[str(person)] = {"n":len(a), "sum":a.sum(0), "cross":a.T@a, "ysum":b.sum(), "xy":a.T@b}
    return result


def sum_statistics(stats, people, balanced=False):
    n = sum(stats[s]["n"] for s in people)
    values = {}
    for key in ("sum", "cross", "ysum", "xy"):
        values[key] = sum(stats[s][key] * (n / (len(people)*stats[s]["n"]) if balanced else 1.) for s in people)
    values["n"] = n
    return values


def projection_from_stats(stats, people, dimension=16):
    """Equivalent to StandardScaler + full PCA + StandardScaler on train rows."""
    a = sum_statistics(stats, people)
    n, mean = a["n"], a["sum"]/a["n"]
    cov = a["cross"]/n - np.outer(mean,mean)
    scale = np.sqrt(np.maximum(np.diag(cov),0))
    scale[scale < 1e-12] = 1.
    standardized_cov = cov / np.outer(scale,scale)
    eigenvalues, vectors = np.linalg.eigh((standardized_cov+standardized_cov.T)/2)
    order = np.argsort(eigenvalues)[::-1][:dimension]
    components = vectors[:,order]
    signs = np.sign(components[np.argmax(np.abs(components),axis=0),np.arange(dimension)])
    components *= np.where(signs==0,1,signs)
    output_scale = np.sqrt(np.maximum(eigenvalues[order],0))
    output_scale[output_scale < 1e-12] = 1.
    return {"mean":mean,"scale":scale,"components":components,"output_scale":output_scale}


def project(x, p):
    return ((np.asarray(x,dtype=np.float64)-p["mean"])/p["scale"]) @ p["components"] / p["output_scale"]


def ridge_from_stats(stats, people, val_x, val_y, val_subjects, alphas=ALPHAS):
    a = sum_statistics(stats, people, balanced=True)
    n, mean, center = a["n"], a["sum"]/a["n"], a["ysum"]/a["n"]
    cov = a["cross"] - n*np.outer(mean,mean)
    scale = np.sqrt(np.maximum(np.diag(cov)/n,0))
    scale[scale < 1e-12] = 1.
    gram = cov/np.outer(scale,scale)
    rhs = (a["xy"] - mean*a["ysum"])/scale
    eigval,eigvec = np.linalg.eigh((gram+gram.T)/2)
    coefs = eigvec @ ((eigvec.T@rhs)[:,None]/(np.maximum(eigval,0)[:,None]+np.asarray(alphas)[None,:]))
    pred = ((val_x-mean)/scale)@coefs+center
    scores = [macro_mae(val_y,pred[:,j],val_subjects) for j in range(len(alphas))]
    chosen = int(np.argmin(scores))
    coef = coefs[:,chosen]/scale
    return coef,float(center-mean@coef),{"alpha":float(alphas[chosen]),"alpha_scores":dict(zip(map(str,alphas),scores)),"validation_subject_mae_bpm":scores[chosen]}


def fit_model(root, method, jobs, rows, plan):
    import torch
    torch.set_num_threads(4)
    cache = {}
    for seed in (SEEDS if method=="AnyPPG-random-frozen" else (42,)):
        for dataset in DATASETS:
            x, info = load_feature(root,dataset,method,seed)
            cache[(dataset,seed)] = (x, info, sufficient_statistics(x, rows[dataset]["hr"], rows[dataset]["subject"]))
    for job in jobs:
        seed = job["seed"] if method=="AnyPPG-random-frozen" else 42
        x, info, stats = cache[(job["source"],seed)]
        tx, tinfo, _ = cache[(job["target"],seed)]
        source,target = rows[job["source"]],rows[job["target"]]
        train,val = (np.flatnonzero(np.isin(source["subject"],job[k])) for k in ("train","validation"))
        test = np.flatnonzero(np.isin(target["subject"],job["test"]))
        destination = root/"runs"/job["id"]/method
        destination.mkdir(parents=True,exist_ok=True)
        identity = {"protocol_sha256":sha256(root/"protocol.json"),"job":job,"method":method,
                    "feature_hashes":{job["source"]:info["features_sha256"],job["target"]:tinfo["features_sha256"]}}
        remaining = []
        for head in HEADS:
            complete = destination/f"{head}.json"
            if complete.exists():
                previous=read_json(complete)
                prediction=ROOT/previous["prediction_path"]
                if previous["identity"] != identity or sha256(prediction) != previous["prediction_sha256"]:
                    raise ValueError("Completed prediction identity/hash mismatch")
            else:
                remaining.append(head)
        if not remaining:
            continue
        projection = projection_from_stats(stats,job["train"])
        np.savez(destination/"projection.npz",**projection)
        ztrain,zval,ztest = (project(a,projection) for a in (x[train],x[val],tx[test]))
        for head in remaining:
            start=time.monotonic()
            reused=False
            if head=="Native-Ridge" and method in OLD_MODELS:
                olddir=OLD/"runs"/job["id"]/method
                oldinfo=read_json(olddir/"complete.json")
                if oldinfo["identity"]["job"] != job or sha256(olddir/"predictions.npz") != oldinfo["predictions_sha256"]:
                    raise ValueError("Old prediction identity changed")
                prediction_path=olddir/"predictions.npz"
                prediction=npz(prediction_path)["prediction"]
                fit=oldinfo["fit"]
                reused=True
            elif head=="PCA16-MLP":
                model,fit=fit_small_head(ztrain,source["hr"][train],source["subject"][train],zval,source["hr"][val],source["subject"][val],seed=job["seed"],epochs=40)
                with torch.inference_mode():
                    prediction=model(torch.tensor(ztest,dtype=torch.float32)).squeeze(-1).numpy()*fit["target_scale"]+fit["target_center"]
                torch.save(model.state_dict(),destination/f"{head}.pt")
            else:
                if head=="Native-Ridge":
                    coef,intercept,fit=ridge_from_stats(stats,job["train"],np.asarray(x[val],dtype=float),source["hr"][val],source["subject"][val])
                    prediction=np.asarray(tx[test],dtype=float)@coef+intercept
                else:
                    projected_stats=sufficient_statistics(ztrain,source["hr"][train],source["subject"][train])
                    coef,intercept,fit=ridge_from_stats(projected_stats,job["train"],zval,source["hr"][val],source["subject"][val])
                    prediction=ztest@coef+intercept
                np.savez(destination/f"{head}.npz",coef=coef,intercept=intercept)
            if not np.isfinite(prediction).all() or prediction.shape!=(len(test),):
                raise ValueError("Nonfinite or wrong shape predictions")
            if not reused:
                prediction_path=destination/f"{head}-prediction.npz"
                np.savez_compressed(prediction_path,row=test,subject=target["subject"][test],start_s=target["start_s"][test],reference=target["hr"][test],prediction=prediction)
            report={"identity":identity,"head":head,"reused":reused,"fit":fit,"train_windows":len(train),"validation_windows":len(val),"test_windows":len(test),
                    "subject_mae_bpm":macro_mae(target["hr"][test],prediction,target["subject"][test]),"prediction_path":prediction_path.relative_to(ROOT).as_posix(),
                    "prediction_sha256":sha256(prediction_path),"elapsed_seconds":time.monotonic()-start,"runner_sha256":sha256(Path(__file__))}
            write_json(destination/f"{head}.json",report)
        print(json.dumps({"method":method,"job":job["id"],"heads":"complete"}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=DEFAULT)
    parser.add_argument("--stage",choices=("register","features","heads"),default="register")
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    plan=register(args.root)
    if args.stage=="register":
        print(json.dumps({"splits":len(plan["jobs"]),"methods":len(METHODS),"heads":len(HEADS),"outputs":93*7*3}))
        return
    with threadpool_limits(limits=4):
        if args.stage=="features":
            for method in args.methods:
                for seed in (SEEDS if method=="AnyPPG-random-frozen" else (42,)):
                    for dataset in DATASETS:
                        features(args.root,dataset,method,seed,args.device)
        else:
            rows={d:npz(OLD/"data"/d/"rows.npz") for d in DATASETS}
            for method in args.methods:
                fit_model(args.root,method,plan["jobs"],rows,plan)


if __name__=="__main__":
    main()
