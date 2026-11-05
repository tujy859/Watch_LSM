"""Audit synthetic denoising by source people; EPFL copies are not people."""
from __future__ import annotations
import argparse
import csv
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import butter,resample_poly,sosfiltfilt
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.ppg_formal_denoising import METHODS,SEEDS,LEVELS,PROTOCOL,load_data,_network,predict
from scripts.ppg_formal_health import write_json,array_hash
from scripts.ppg_hr_poc_features import sha256
from scripts.ppg_hr_baselines import predict_hr_windows
from scripts.audit_ppg_formal_health import holm


def metrics(target,pred,reference_hr):
    """Separate metric implementation; Welch is the declared downstream algorithm."""
    actual=np.asarray(target,np.float64)[:,64:448]; predicted=np.asarray(pred,np.float64)[:,64:448]
    mse=np.sum(np.square(actual-predicted),axis=1)/actual.shape[1]
    energy=np.sum(np.square(actual),axis=1)/actual.shape[1]
    relative=np.sqrt(mse)/np.maximum(np.sqrt(energy),1e-12)
    a=actual-np.mean(actual,axis=1)[:,None]; b=predicted-np.mean(predicted,axis=1)[:,None]
    r=np.einsum("ij,ij->i",a,b)/np.maximum(np.sqrt(np.einsum("ij,ij->i",a,a)*np.einsum("ij,ij->i",b,b)),1e-12)
    hr=predict_hr_windows(predicted,64)["welch"]
    return {"nrmse":relative,"snr_db":-20*np.log10(np.maximum(relative,1e-12)),"correlation":r,
            "hr_absolute_error":np.abs(hr-np.asarray(reference_hr)),"hr_prediction":hr}


def audit_prepared_pairs(data,manifest,dataset):
    """Rebuild references from raw sources; no target controls input normalization."""
    from scripts.ppg_embc21_data import load_records
    from scripts.ppg_denoising_data_pilot import load_vector
    if dataset=="EMBC21":
        records={(r["subject"],r["record_id"]):r for r in load_records(ROOT/"datasets/embc21")}
    else:
        source=ROOT/"datasets/epfl_synthetic/extracted/DataPPGwNoise/Datasets"
        raw=load_vector(source/"GoodSignals/PPG/ppg.mat","ppg")
    max_target,max_input,max_hr,max_snr=0.,0.,0.,0.
    references={}; external_noisy={}
    for i,row in enumerate(manifest["rows"]):
        identity=(row["subject"],row["record_id"],row["start_s"])
        if row["subject"]!=data["subjects"][i]:
            raise ValueError("Prepared pair person identity mismatch")
        if dataset=="EMBC21":
            r=records[identity[:2]]; start=row["start_s"];a,b=int(start*400),int((start+8)*400)
            if r["activity"]!="rest" or not r["valid_mask"][a:b].all():
                raise ValueError("Reference is not a fully valid resting clip")
            reference=resample_poly(-r["raw_ppg"][a:b],5,16)
            reference-=reference.mean()
            measured=10*np.log10(np.mean(reference**2)/np.mean((data["x125"][i]-reference)**2))
            max_snr=max(max_snr,abs(measured-row["noise_snr_db"]))
            onsets=r["onset_times_s"]; onsets=onsets[(onsets>=start+1)&(onsets<start+7)]
            hr=60/np.mean(np.diff(onsets)) if len(onsets)>=3 else np.nan
            max_hr=max(max_hr,abs(hr-data["reference_hr"][i]))
            # Production stores the synthetic reference at float32 before resampling.
            reference=reference.astype(np.float32)
        else:
            a,b=int(row["start_s"]*250),int((row["start_s"]+8)*250)
            reference=resample_poly(raw[a:b],1,2)
            if row["record_id"] not in external_noisy:
                external_noisy[row["record_id"]]=load_vector(source/"NoisySignals/PPG"/(row["record_id"]+".mat"),"noise_signal")
            rebuilt=resample_poly(external_noisy[row["record_id"]][a:b],1,2).astype(np.float32)
            np.testing.assert_array_equal(rebuilt,data["x125"][i])
        clean64=resample_poly(reference.astype(np.float64),64,125)
        clean64-=clean64.mean()
        noisy64=resample_poly(data["x125"][i],64,125).astype(np.float64)
        mean,scale=noisy64.mean(),noisy64.std()
        normalized=((noisy64-mean)/scale).astype(np.float32)
        baseline=sosfiltfilt(butter(2,[.5,8],fs=64,btype="bandpass",output="sos"),normalized).astype(np.float32)
        for expected,actual in ((normalized,data["x64"][i]),(baseline,data["baseline"][i])):
            max_input=max(max_input,float(np.max(np.abs(expected-actual))))
        max_target=max(max_target,float(np.max(np.abs((clean64/scale).astype(np.float32)-data["target"][i]))))
        np.testing.assert_allclose([mean,scale],[data["noisy_mean"][i],data["noisy_std"][i]],rtol=0,atol=1e-12)
        references.setdefault(identity,set()).add(row.get("noise_snr_db"))
    if max_target>1e-6 or max_input>1e-6 or max_hr>1e-9 or max_snr>1e-5:
        raise ValueError("Raw-reference/normalization/pair audit exceeded tolerance")
    if dataset=="EMBC21" and (len(references)!=168 or any(v!=set(LEVELS) for v in references.values())):
        raise ValueError("Synthetic reference clip or noise-level coverage changed")
    return {"pairs":len(manifest["rows"]),"raw_reference_max_difference":max_target,"inference_input_max_difference":max_input,
            "manual_reference_hr_max_difference":max_hr,"realized_snr_max_difference_db":max_snr}


def bootstrap(x):
    x=np.asarray(x,dtype=float)
    rng=np.random.default_rng(20261105)
    sampled=x[rng.integers(0,len(x),(10000,len(x)))].mean(axis=1)
    return [float(v) for v in np.percentile(sampled,[2.5,97.5])]


def exact_paired(x):
    x=np.asarray(x,dtype=float)
    signs=np.asarray(list(product((-1,1),repeat=len(x))))
    return float(np.mean(np.abs((signs*x).mean(axis=1))>=abs(x.mean())-1e-12))


def save_csv(path,rows):
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def audit(output,shared):
    torch.set_num_threads(2)
    shared.mkdir(parents=True,exist_ok=True)
    records=json.loads((output/"results.json").read_text())
    if len(records)!=84 or len({(r["method"],r["seed"],r["fold"]) for r in records})!=84:
        raise ValueError("Need4methods x3seeds x7personfolds")
    data,manifest=load_data(output,"EMBC21"); epfl,external_manifest=load_data(output,"EPFL")
    if json.loads((output/"protocol.json").read_text())!=PROTOCOL:
        raise ValueError("Saved denoising protocol changed")
    pair_audit={"EMBC21":audit_prepared_pairs(data,manifest,"EMBC21"),"EPFL":audit_prepared_pairs(epfl,external_manifest,"EPFL")}
    for meta in (manifest,external_manifest):
        for relative,digest in meta["source_hashes"].items():
            if sha256(ROOT/relative)!=digest:
                raise ValueError("Source waveform/annotation changed")
    base={name:metrics(data["target"],data[key],data["reference_hr"]) for name,key in (("NoisyInput","x64"),("FixedBandpass","baseline"))}
    external_base={name:metrics(epfl["target"],epfl[key],epfl["reference_hr"]) for name,key in (("NoisyInput","x64"),("FixedBandpass","baseline"))}
    predictions={name:{seed:{metric:np.full(len(data["target"]),np.nan) for metric in base["NoisyInput"]} for seed in SEEDS} for name in METHODS}
    external_metrics={name:[] for name in METHODS}
    max_metric,max_cpu=0.,0.; rebuilt=0; cache_maps={}
    for row in records:
        folder=output/row["run_folder"]
        split=manifest["folds"][row["fold"]]
        if any(row[k]!=split[k] for k in ("train","validation","test")):
            raise ValueError("Person roles differ from declared fold")
        if set(row["train"])&set(row["test"]) or set(row["validation"])&set(row["test"]):
            raise ValueError("Held-out person appeared in development")
        if int(np.argmin(row["validation_curve"]))+1!=row["best_epoch"]:
            raise ValueError("Selected epoch violates validation-only rule")
        if sha256(folder/"predictions.npz")!=row["prediction_sha256"] or sha256(folder/"weights.pt")!=row["weights_sha256"]:
            raise ValueError("Saved weights/predictions changed")
        with np.load(folder/"predictions.npz",allow_pickle=False) as saved:
            idx=saved["indices"]
            if not np.array_equal(idx,np.flatnonzero(np.isin(data["subjects"],split["test"]))):
                raise ValueError("Wrong test pair identities")
            m=metrics(data["target"][idx],saved["prediction"],data["reference_hr"][idx])
            for key,value in m.items():
                max_metric=max(max_metric,float(np.nanmax(np.abs(value-saved[key]))))
                destination=predictions[row["method"]][row["seed"]][key]
                if np.isfinite(destination[idx]).any():
                    raise ValueError("Duplicate test pair predictions")
                destination[idx]=value
            model=_network(row["method"])
            model.load_state_dict(torch.load(folder/"weights.pt",map_location="cpu",weights_only=True),strict=True);model.eval()
            feature_seed=row["seed"] if row["method"]=="AnyPPG-random-frozen" else 42
            cache_key=(row["method"],feature_seed)
            maps=None
            if row["method"]!="SmallCNN":
                if cache_key not in cache_maps:
                    feature_dir=output/"features"/f"{row['method']}__seed{feature_seed}"
                    feature_info=json.loads((feature_dir/"complete.json").read_text())
                    if feature_info["state_before_sha256"]!=feature_info["state_after_sha256"] or feature_info["encoder_trainable_parameters"]!=0:
                        raise ValueError("Temporal encoder changed during inference")
                    if feature_info["input_sha256"]!=array_hash(data["x125"]) or any(sha256(feature_dir/name)!=digest for name,digest in feature_info["files"].items()):
                        raise ValueError("Temporal feature input or saved map changed")
                    cache_maps.clear()
                    cache_maps[cache_key]=[np.load(feature_dir/f"stage{i}.npy",mmap_mode="r") for i in range(6)]
                maps=[torch.tensor(np.array(a[idx[:3]])) for a in cache_maps[cache_key]]
            x=torch.tensor(data["x64"][idx[:3]]); baseline=torch.tensor(data["baseline"][idx[:3]])
            cpu=predict(model,row["method"],maps,x,baseline,np.arange(len(x)),batch=3)
            max_cpu=max(max_cpu,float(np.max(np.abs(cpu-saved["prediction"][:3]))));rebuilt+=len(x)
        external_info=json.loads((folder/"epfl.json").read_text())
        if row["method"]!="SmallCNN":
            encoder=external_info["encoder"]
            if encoder["state_before_sha256"]!=encoder["state_after_sha256"] or encoder["encoder_trainable_parameters"]!=0:
                raise ValueError("External encoder was not frozen")
        if sha256(folder/"epfl_predictions.npz")!=external_info["prediction_sha256"] or external_info["data_sha256"]!=external_manifest["data_sha256"]:
            raise ValueError("External test identity/hash changed")
        with np.load(folder/"epfl_predictions.npz",allow_pickle=False) as saved:
            m=metrics(epfl["target"],saved["prediction"],epfl["reference_hr"])
            for key,value in m.items():
                max_metric=max(max_metric,float(np.nanmax(np.abs(value-saved[key]))))
            external_metrics[row["method"]].append(m)
    if max_metric>1e-8 or max_cpu>.002:
        raise ValueError("Independent waveform/metric audit tolerance exceeded")
    people=sorted(set(data["subjects"])); summary=[]; person_values={}; strata=[]
    all_metrics={**base,**{method:{key:np.mean([predictions[method][seed][key] for seed in SEEDS],axis=0) for key in base["NoisyInput"]} for method in METHODS}}
    for method,m in all_metrics.items():
        if not np.isfinite(m["nrmse"]).all():
            raise ValueError("Missing denoising pair outputs")
        person_values[method]=np.array([m["nrmse"][data["subjects"]==p].mean() for p in people])
        for metric in ("nrmse","snr_db","snr_gain_db","correlation","hr_absolute_error"):
            values=m["snr_db"]-base["NoisyInput"]["snr_db"] if metric=="snr_gain_db" else m[metric]
            subject=np.array([np.nanmean(values[data["subjects"]==p]) for p in people])
            low,high=bootstrap(subject)
            summary.append({"dataset":"EMBC21","method":method,"metric":metric,"value":float(subject.mean()),"ci_low":low,"ci_high":high,"people":7,"pairs":504})
        for level in LEVELS:
            mask=np.array([r["noise_snr_db"]==level for r in manifest["rows"]])
            strata.append({"dataset":"EMBC21","method":method,"stratum":f"Generated SNR {level:g}dB","people":7,"pairs":int(mask.sum()),
                           "nrmse":float(m["nrmse"][mask].mean()),"snr_gain_db":float((m["snr_db"]-base["NoisyInput"]["snr_db"])[mask].mean()),
                           "correlation":float(m["correlation"][mask].mean()),"hr_absolute_error":float(np.nanmean(m["hr_absolute_error"][mask]))})
    comparisons=[]
    for model in ("AnyPPG","PaPaGei-S"):
        for control in ("FixedBandpass","SmallCNN"):
            difference=person_values[model]-person_values[control]
            low,high=bootstrap(difference)
            comparisons.append({"model":model,"control":control,"metric":"nrmse","difference":float(difference.mean()),"ci_low":low,"ci_high":high,"p_exact_signflip":exact_paired(difference),"people":7})
    for row,p in zip(comparisons,holm([r["p_exact_signflip"] for r in comparisons])):
        row["p_holm_four_contrasts"]=float(p)
    # Every external score first averages all21 trained fold/seed heads for a
    # given waveform window. It still represents ONE source participant.
    all_external={**external_base,**{method:{key:np.mean([v[key] for v in values],axis=0) for key in external_base["NoisyInput"]} for method,values in external_metrics.items()}}
    copy_ids=np.array([r["record_id"] for r in external_manifest["rows"]])
    ids=sorted(set(copy_ids))
    input_snr={copy:float(external_base["NoisyInput"]["snr_db"][copy_ids==copy].mean()) for copy in ids}
    ranges=[("input SNR <0dB",-np.inf,0),("input SNR 0-10dB",0,10),("input SNR 10-20dB",10,20),("input SNR >=20dB",20,np.inf)]
    for method,m in all_external.items():
        for metric in ("nrmse","snr_db","snr_gain_db","correlation","hr_absolute_error"):
            values=m["snr_db"]-external_base["NoisyInput"]["snr_db"] if metric=="snr_gain_db" else m[metric]
            summary.append({"dataset":"EPFL","method":method,"metric":metric,"value":float(np.nanmean(values)),"ci_low":"","ci_high":"","people":1,"pairs":3060})
        for label,lo,hi in ranges:
            copies=[copy for copy in ids if lo<=input_snr[copy]<hi]
            mask=np.isin(copy_ids,copies)
            if mask.any():
                strata.append({"dataset":"EPFL","method":method,"stratum":label,"people":1,"pairs":int(mask.sum()),
                               "nrmse":float(m["nrmse"][mask].mean()),"snr_gain_db":float((m["snr_db"]-external_base["NoisyInput"]["snr_db"])[mask].mean()),
                               "correlation":float(m["correlation"][mask].mean()),"hr_absolute_error":float(np.nanmean(m["hr_absolute_error"][mask]))})
    save_csv(shared/"summary.csv",summary);save_csv(shared/"noise_strata.csv",strata)
    write_json(shared/"paired_comparisons.json",comparisons)
    evidence={"status":"complete_audited","fitted_heads":84,"EMBC21_test_windows_per_method_seed":504,"EPFL_test_windows_per_head":3060,
              "metric_arrays_recomputed":84*2*5,"CPU_waveforms_recomputed":rebuilt,"maximum_metric_difference":max_metric,
              "maximum_CPU_waveform_difference":max_cpu,"subject_separation_verified":True,"every_pair_tested_once_per_method_seed":True,
              "raw_pair_reconstruction":pair_audit,"metrics_independent_numpy_implementation":True,
              "source_files_rehashed":len(manifest["source_hashes"])+len(external_manifest["source_hashes"]),
              "EPFL_population_CI":False,"EPFL_underlying_people":1,"limitations":["Resting reference signals are not guaranteed clean","Noise is synthetic","EMBC only7people, exploratory paired inference with shared training folds","EPFL340 copies share one clean waveform and cannot establish population generalization","Downstream HR is a fixed algorithm, not a learned clinical measurement"]}
    write_json(shared/"audit.json",evidence)
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/".local/ppg-formal-expansion/denoising")
    parser.add_argument("--shared",type=Path,default=ROOT/"docs/results/ppg_formal_expansion/denoising")
    args=parser.parse_args();print(json.dumps(audit(args.output,args.shared)),flush=True)


if __name__=="__main__":
    main()
