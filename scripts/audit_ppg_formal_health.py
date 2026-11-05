"""Independently audit person-level health predictions and paired increments.

Resampling units are people, with absolute errors averaged across the three
initializations before inference. Repeated seeds are not extra participants.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.ppg_formal_health import MODELS,RANDOM,SEEDS,PROTOCOL,load_prepared,write_json
from scripts.ppg_hr_poc_features import sha256


def audit_age_inputs(data,manifest,output):
    """Trace age labels to the raw table and each predictor to its BP parent."""
    source,parent=load_prepared("PPG-BP",output)
    if manifest["source_parent_manifest_sha256"]!=sha256(output/"PPG-BP/manifest.json"):
        raise ValueError("Age parent manifest changed")
    if data["covariate_names"]!=["male_indicator","bmi_kg_m2"] or data["target_names"]!=["chronological_age_years"]:
        raise ValueError("Age target appears in predictor metadata")
    for key in ("subjects","traditional","windows","waveform_subjects","stats"):
        np.testing.assert_array_equal(data[key],source[key],err_msg=f"Age predictor differs from BP parent: {key}")
    np.testing.assert_array_equal(data["targets"],source["covariates"][:,:1])
    np.testing.assert_array_equal(data["covariates"],source["covariates"][:,1:])
    if manifest["split"]!=parent["split"]:
        raise ValueError("Age folds differ from parent identity-only split")
    from scripts.ppg_health_increment_pilot import workbook_rows
    workbook=ROOT/"datasets/ppg_bp/Data File/PPG-BP dataset.xlsx"
    rows=workbook_rows(workbook); columns={v:k for k,v in rows[1].items()}
    ages={str(int(float(r[columns["subject_ID"]]))):float(r[columns["Age(year)"]]) for r in rows[2:]}
    np.testing.assert_array_equal(data["targets"][:,0],[ages[p] for p in data["subjects"]])
    return {"raw_table_age_labels_verified":len(data["subjects"]),"age_removed_from_all_covariates":True,
            "all_nonlabel_arrays_identical_to_BP_parent":True,"predictor_covariates":data["covariate_names"],
            "parent_split_preserved":True}


def independent_input_rebuild(data,row,saved,train):
    """Rebuild every predictor column and its train-only imputation/scaling."""
    cov,traditional,z=data["covariates"],data["traditional"],saved["projected_features"]
    if row["inputs"]=="demographics": raw=cov
    elif row["inputs"]=="demographics_traditional": raw=np.column_stack([cov,traditional])
    elif row["inputs"]=="embedding": raw=z
    elif row["inputs"]=="demographics_embedding": raw=np.column_stack([cov,z])
    elif row["inputs"]=="demographics_traditional_embedding": raw=np.column_stack([cov,traditional,z])
    else: raise ValueError("Unknown predictor input group")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        median=np.nanmedian(raw[train],axis=0)
    median=np.where(np.isfinite(median),median,0.)
    augmented=np.column_stack([np.where(np.isnan(raw),median,raw),np.isnan(raw).astype(float)])
    mean=augmented[train].mean(axis=0); scale=augmented[train].std(axis=0)
    scale=np.where(scale<1e-12,1.,scale)
    np.testing.assert_allclose(saved["imputation_median"],median,rtol=0,atol=1e-12)
    np.testing.assert_allclose(saved["input_mean"],mean,rtol=0,atol=1e-10)
    np.testing.assert_allclose(saved["input_scale"],scale,rtol=1e-10,atol=1e-10)
    transformed=(augmented-mean)/scale
    rebuilt=np.pad(transformed,((0,0),(0,saved["prepared_features"].shape[1]-transformed.shape[1])))
    if int(saved["actual_columns"])!=transformed.shape[1]:
        raise ValueError("Predictor column count changed")
    return float(np.max(np.abs(rebuilt-saved["prepared_features"])))


def holm(p_values):
    p=np.asarray(p_values,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Invalid p values")
    order=np.argsort(p,kind="stable")
    out=np.empty(len(p)); running=0.
    for rank,index in enumerate(order):
        running=max(running,min(1.,float((len(p)-rank)*p[index])))
        out[index]=running
    return out


def paired_inference(differences,seed=20261105):
    x=np.asarray(differences,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError("Need finite person differences")
    rng=np.random.default_rng(seed)
    samples=rng.integers(0,len(x),size=(10000,len(x)))
    ci=np.percentile(x[samples].mean(axis=1),[2.5,97.5])
    observed=abs(float(x.mean())); extreme=0; total=100000
    for _ in range(total//1000):
        signs=rng.integers(0,2,size=(1000,len(x)))*2-1
        extreme+=int(np.sum(np.abs((signs*x).mean(axis=1))>=observed-1e-12))
    return {"difference_mae":float(x.mean()),"ci_low":float(ci[0]),"ci_high":float(ci[1]),
            "p_two_sided":float((extreme+1)/(total+1)),"people":len(x)}


def _mean_ci(x):
    rng=np.random.default_rng(20261105)
    samples=rng.integers(0,len(x),size=(10000,len(x)))
    low,high=np.percentile(x[samples].mean(axis=1),[2.5,97.5])
    return float(low),float(high)


def audit_dataset(dataset,output,shared):
    data,manifest=load_prepared(dataset,output)
    age_evidence=audit_age_inputs(data,manifest,output) if dataset=="PPG-BP-age" else None
    folder=output/dataset
    rows=json.loads((folder/"results.json").read_text())
    include_cnn=dataset!="PPG-BP-age"
    if include_cnn:
        rows+=json.loads((folder/"cnn_results.json").read_text())
    expected=5*3*(7*3*2+2*2+1+(2 if include_cnn else 0))*len(data["target_names"])
    if len(rows)!=expected:
        raise ValueError(f"Incomplete health evaluation: {len(rows)} vs {expected}")
    groups=defaultdict(dict)
    max_metric,max_direct,max_refit,max_transform,max_inputs=0.,0.,0.,0.,0.
    tested=set(); cnn_x=None
    for relative,digest in manifest["source_hashes"].items():
        if sha256(ROOT/relative)!=digest:
            raise ValueError("Raw source changed after cohort preparation")
    for feature_path in sorted((folder/"features").glob("*.npz")):
        evidence=json.loads(feature_path.with_suffix(".json").read_text())
        if sha256(feature_path)!=evidence["output_sha256"] or evidence["data_sha256"]!=manifest["data_npz_sha256"]:
            raise ValueError("Frozen feature cache hash/identity changed")
        if evidence["encoder_state_before"]!=evidence["encoder_state_after"] or evidence["trainable_encoder_parameters"]!=0:
            raise ValueError("Encoder was not kept frozen")
        if evidence["padding_samples"] or evidence["cropped_samples"]:
            raise ValueError("Native adapter changed declared observation")
        if dataset=="PPG-BP-age":
            from scripts.ppg_formal_health import array_hash
            if evidence["input_sha256"]!=array_hash(data["windows"]) or evidence["statistics_sha256"]!=array_hash(data["stats"]):
                raise ValueError("Age features used unexpected waveform inputs")
            parent_path=output/"PPG-BP/features"/feature_path.name
            if evidence["source_feature_path"]!=parent_path.relative_to(output).as_posix() or sha256(parent_path)!=sha256(feature_path):
                raise ValueError("Age frozen features differ from label-independent parent features")
        with np.load(feature_path,allow_pickle=False) as feat:
            if not np.array_equal(feat["subjects"],data["subjects"]):
                raise ValueError("Feature participant order changed")
            averaged=np.asarray([feat["window_features"][data["waveform_subjects"]==p].mean(axis=0) for p in data["subjects"]])
            np.testing.assert_array_equal(averaged,feat["person_features"])
    for row in rows:
        path=output/row["prediction_file"]
        if sha256(path)!=row["prediction_sha256"]:
            raise ValueError("Prediction hash changed")
        key=(row["target"],row["model"],row["inputs"],row["head"])
        identity=(key,row["fold"],row["seed"])
        if identity in tested:
            raise ValueError("Duplicate evaluation identity")
        tested.add(identity)
        split=manifest["split"][row["fold"]]
        with np.load(path,allow_pickle=False) as saved:
            people=saved["subjects"].astype(str)
            if set(people)!=set(split["test"]) or set(saved["train_subjects"])!=set(split["train"]) or set(saved["validation_subjects"])!=set(split["validation"]):
                raise ValueError("Split identities differ from predeclared person folds")
            true,pred=saved["y_true"],saved["y_pred"]
            test_idx=np.array([int(np.flatnonzero(data["subjects"]==p)[0]) for p in people])
            train=np.flatnonzero(np.isin(data["subjects"],split["train"]))
            ti=data["target_names"].index(row["target"])
            y=data["targets"][:,ti]
            if not np.array_equal(true,y[test_idx]) or not np.isfinite(pred).all():
                raise ValueError("Target identity or finite prediction failed")
            max_metric=max(max_metric,abs(float(np.mean(np.abs(pred-true)))-row["mae"]))
            if row["head"]=="constant":
                direct=np.full(len(test_idx),y[train].mean())
            elif row["head"]=="cnn":
                import torch
                from scripts.ppg_formal_health_cnn import PersonCNN,person_inputs,predict
                torch.set_num_threads(2)
                if cnn_x is None:
                    cnn_x=person_inputs(data)
                checkpoint=output/row["checkpoint_file"]
                if sha256(checkpoint)!=row["checkpoint_sha256"]:
                    raise ValueError("CNN checkpoint changed")
                model=PersonCNN(saved["clinical"].shape[1])
                model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True)
                model.eval()
                direct=predict(model,cnn_x,torch.tensor(saved["clinical"],dtype=torch.float32),test_idx,row["target_center"],row["target_scale"],batch=4)
                if int(np.argmin(row["validation_mae_by_epoch"]))+1!=row["epochs"]:
                    raise ValueError("CNN selected epoch differs from validation scores")
            else:
                z=saved["prepared_features"]
                max_inputs=max(max_inputs,independent_input_rebuild(data,row,saved,train))
                if row["model"]!="shared_control":
                    source_seed=row["seed"] if row["model"]==RANDOM else 42
                    with np.load(folder/"features"/f"{row['model']}__seed{source_seed}.npz",allow_pickle=False) as feat:
                        raw=feat["person_features"].astype(float)
                    max_transform=max(max_transform,float(np.max(np.abs(saved["scale_mean"]-raw[train].mean(axis=0)))))
                    reduced=(((raw-saved["scale_mean"])/saved["scale_scale"])-saved["pca_mean"])@saved["pca_components"].T
                    max_transform=max(max_transform,float(np.max(np.abs(reduced-saved["projected_features"]))))
                if row["head"]=="linear":
                    direct=z[test_idx]@saved["coef"]+saved["intercept"]
                    centered=z[train]-z[train].mean(axis=0)
                    beta=np.linalg.solve(centered.T@centered+row["alpha"]*np.eye(z.shape[1]),centered.T@(y[train]-y[train].mean()))
                    refit=(z[test_idx]-z[train].mean(axis=0))@beta+y[train].mean()
                    max_refit=max(max_refit,float(np.max(np.abs(refit-pred))))
                    candidates=row["validation_candidates"]
                    chosen=min(candidates,key=lambda c:(c["mae"],c["alpha"]))
                    if chosen["alpha"]!=row["alpha"]:
                        raise ValueError("Selected Ridge regularization differs from validation scores")
                else:
                    direct=(np.maximum(z[test_idx].astype(np.float32)@saved["w1"].T+saved["b1"],0)@saved["w2"].T+saved["b2"]).flatten().astype(float)*saved["target_scale"]+saved["target_center"]
                    if int(np.argmin(row["validation_mae_by_epoch"]))+1!=row["epochs"]:
                        raise ValueError("Selected epoch differs from validation-only rule")
            max_direct=max(max_direct,float(np.max(np.abs(direct-pred))))
            for p,t,v in zip(people,true,pred):
                if (p,row["seed"]) in groups[key]:
                    raise ValueError("Person has more than one test prediction per seed")
                groups[key][(p,row["seed"])]=(float(t),float(v))
    if max_metric>1e-8 or max_direct>.002 or max_refit>1e-7 or max_transform>1e-5 or max_inputs>1e-7:
        raise ValueError("Independent numeric audit tolerance exceeded")
    people=data["subjects"].astype(str).tolist()
    summary=[]; errors={}
    for key,values in groups.items():
        if set(values)!={(p,s) for p in people for s in SEEDS}:
            raise ValueError("Test predictions do not cover every person and seed exactly once")
        error=np.array([[values[(p,s)][1]-values[(p,s)][0] for s in SEEDS] for p in people])
        ae=np.abs(error).mean(axis=1); errors[key]=ae
        low,high=_mean_ci(ae)
        target,model,inputs,head=key
        summary.append({"dataset":dataset,"target":target,"model":model,"inputs":inputs,"head":head,
                        "people":len(people),"seeds":3,"mae":float(ae.mean()),"mae_ci_low":low,"mae_ci_high":high,
                        "rmse":float(np.sqrt((error**2).mean())),"bias":float(error.mean())})
    comparisons=[]
    for target in data["target_names"]:
        for head in ("linear","mlp"):
            base=errors[(target,"shared_control","demographics_traditional",head)]
            primary=[]
            for model in (*MODELS,RANDOM):
                fusion=errors[(target,model,"demographics_traditional_embedding",head)]
                comp={"dataset":dataset,"target":target,"model":model,"head":head,
                      "comparison":"fusion minus demographics+traditional",**paired_inference(fusion-base),
                      "primary":head=="linear" and model in MODELS and dataset!="PPG-BP-age"}
                comparisons.append(comp)
                if comp["primary"]:
                    primary.append(comp)
            if primary:
                adjusted=holm([x["p_two_sided"] for x in primary])
                for c,p in zip(primary,adjusted):
                    c["p_holm_six_models"]=float(p)
                    c["improvement_supported_exploratory_inference"]=bool(c["difference_mae"]<0 and c["ci_high"]<0 and p<.05)
    evidence={"dataset":dataset,"status":"complete_audited","people":len(people),"results":len(rows),"prediction_files":len(tested),
              "patient_folds":5,"initialization_seeds":list(SEEDS),"protocol":PROTOCOL,
              "maximum_mae_recalculation_difference":max_metric,"maximum_saved_model_reconstruction_difference":max_direct,
              "maximum_independent_ridge_refit_difference":max_refit,"maximum_pca_reconstruction_difference":max_transform,
              "maximum_predictor_input_reconstruction_difference":max_inputs,"predictor_inputs_rebuilt_from_declared_arrays":True,
              "every_person_tested_once_per_seed_per_method":True,"source_manifest_sha256":sha256(folder/"manifest.json"),
              "raw_source_files_rehashed":len(manifest["source_hashes"]),"person_feature_aggregation_independently_rebuilt":True,
              "raw_results_sha256":sha256(folder/"results.json"),"limitations":data["limitations"],
              "interval_interpretation":"Exploratory paired person bootstrap; seeds averaged inside people, split is fixed, shared training folds induce dependence",
              "feature_caches":{p.name:sha256(p) for p in sorted((folder/"features").glob("*.npz"))}}
    if age_evidence is not None:
        evidence["age_label_leakage_audit"]=age_evidence
    shared.mkdir(parents=True,exist_ok=True)
    write_json(shared/f"{dataset}_audit.json",evidence)
    write_json(shared/f"{dataset}_comparisons.json",comparisons)
    with (shared/f"{dataset}_summary.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",nargs="+",choices=("ECSMP","PPG-BP","PPG-BP-age"),default=["ECSMP","PPG-BP"])
    parser.add_argument("--output",type=Path,default=ROOT/".local/ppg-formal-expansion/health")
    parser.add_argument("--shared",type=Path,default=ROOT/"docs/results/ppg_formal_expansion/health")
    args=parser.parse_args()
    for dataset in args.datasets:
        with threadpool_limits(limits=2):
            evidence=audit_dataset(dataset,args.output,args.shared)
        print(json.dumps(evidence),flush=True)


if __name__=="__main__":
    main()
