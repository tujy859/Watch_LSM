"""Small paired synthetic-noise study, with no claim of real clean wrist PPG.

EMBC21 manually valid resting clips are synthetic reference waveforms. People
are separated before downstream fitting. EPFL's 340 copies of one waveform
are an external stress test only, never training or validation examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.signal import butter,resample_poly,sosfiltfilt
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.ppg_embc21_data import load_records,eligible_windows
from scripts.ppg_denoising_data_pilot import load_vector
from scripts.ppg_temporal_adapters import TemporalEncoder,make_head
from scripts.ppg_beat_models import SmallEventCNN
from scripts.ppg_hr_baselines import predict_hr_windows
from scripts.ppg_hr_poc_features import sha256
from scripts.ppg_formal_health import write_json,array_hash

METHODS=("AnyPPG","PaPaGei-S","AnyPPG-random-frozen","SmallCNN")
SEEDS=(42,43,44)
LEVELS=(-5.,5.,15.)
PROTOCOL={
    "version":"synthetic-reference-denoising-v1","methods":list(METHODS),"seeds":list(SEEDS),
    "reference":"EMBC21 manually valid resting raw PPG, blood-volume polarity, per-clip DC removed; not guaranteed truly noise-free",
    "windows":"At most 24 uniformly selected nonoverlapping 8s clips/person, on record-zero 8s grid; all samples manually valid and finite",
    "split":"7 person folds, test S(k), next person validation, other five train; no noisy copies of a person cross roles",
    "noise_snr_db":list(LEVELS),"noise_recipe":"0.5 Gaussian white + sinusoid frequency U(0.1,0.4) Hz + 3 signed Gaussian transients width U(0.03,0.12)s; centered and scaled to prescribed realized energy SNR at125Hz",
    "noise_randomness":"stable SHA256 seed of subject/record/start/SNR; same pairs for all models and all optimization seeds",
    "normalization":"Only noisy 64Hz clip mean/std enter inference; reference AC waveform divided by same noisy std as target; no clean-driven alignment or gain fitting",
    "sampling":{"input_hz":125,"output_hz":64,"input_seconds":8,"scoring_seconds":[1,7]},
    "learned_output":"fixed 0.5-8Hz second-order zero-phase Butterworth output plus learned residual; last output convolution starts at zero",
    "fit":{"epochs":20,"patience":5,"batch":64,"lr":.001,"loss":"person-balanced MSE in noisy-input-normalized units, central6s",
           "selection":"validation-person mean NRMSE, earliest epoch on ties; no refit"},
    "primary_metrics":["NRMSE relative to reference RMS","SNR gain relative to noisy input","correlation"],
    "secondary_hr":"Fixed existing Welch algorithm on central6s; EMBC reference from manual onset mean intervals, EPFL reference same Welch on supplied clean waveform (algorithm consistency only)",
    "epfl":"All 340 copies, each9 nonoverlapping8s clips, evaluated by every trained fold/seed; one underlying person, descriptive noise strata only; never tune on EPFL",
    "inference":"EMBC seven person paired exploratory bootstrap; average optimization seeds inside person; EPFL no population CI",
    "pretraining":False,"encoder_finetuning":False,
}


def noisy_normalization(noisy64):
    x=np.asarray(noisy64,dtype=float)
    mean=x.mean(axis=-1,keepdims=True)
    std=x.std(axis=-1,keepdims=True)
    if not np.isfinite(x).all() or np.any(std<=1e-10):
        raise ValueError("Noisy input must be finite and nonconstant")
    return ((x-mean)/std).astype(np.float32),mean[:,0],std[:,0]


def synthetic_pair(reference,snr_db,key):
    clean=np.asarray(reference,dtype=float)
    clean=clean-clean.mean()
    if clean.shape!=(1000,) or not np.isfinite(clean).all() or clean.std()<=1e-10:
        raise ValueError("Expected nonconstant 8s125Hz reference")
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little"))
    t=np.arange(1000)/125
    noise=.5*rng.normal(size=1000)+np.sin(2*np.pi*rng.uniform(.1,.4)*t+rng.uniform(0,2*np.pi))
    for _ in range(3):
        noise+=rng.choice([-1.,1.])*rng.uniform(1,3)*np.exp(-.5*((t-rng.uniform(0,8))/rng.uniform(.03,.12))**2)
    noise-=noise.mean()
    noise*=np.sqrt(np.mean(clean**2)/np.mean(noise**2))*10**(-snr_db/20)
    return (clean+noise).astype(np.float32),clean.astype(np.float32)


def finalized_arrays(noisy125,clean125,subjects,rows,reference_hr):
    x125=np.asarray(noisy125,dtype=np.float32)
    clean64=resample_poly(np.asarray(clean125,dtype=np.float64),64,125,axis=-1)
    clean64-=clean64.mean(axis=-1,keepdims=True)
    noisy64=resample_poly(x125,64,125,axis=-1)
    x,means,scales=noisy_normalization(noisy64)
    target=(clean64/scales[:,None]).astype(np.float32)
    filtered=sosfiltfilt(butter(2,[.5,8],fs=64,btype="bandpass",output="sos"),x,axis=-1).astype(np.float32)
    return {"x125":x125,"x64":x,"target":target,"baseline":filtered,
            "noisy_mean":means,"noisy_std":scales,"subjects":np.asarray(subjects),"reference_hr":np.asarray(reference_hr,dtype=float)},rows


def prepare(output,shared):
    output.mkdir(parents=True,exist_ok=True); shared.mkdir(parents=True,exist_ok=True)
    existing=output/"protocol.json"
    if existing.exists() and json.loads(existing.read_text())!=PROTOCOL:
        raise ValueError("Protocol changed; use a distinct run directory")
    write_json(existing,PROTOCOL); write_json(shared/"protocol.json",PROTOCOL)
    records=load_records(ROOT/"datasets/embc21")
    people=sorted({r["subject"] for r in records})
    source_hashes={}; noisy,clean,subjects,rows,hrs=[],[],[],[],[]
    for person in people:
        candidates=[]
        for r in records:
            if r["subject"]==person and r["activity"]=="rest":
                for start,end in eligible_windows(r,8,8):
                    candidates.append((r,float(start),float(end)))
        if len(candidates)<24:
            raise ValueError("Fewer than 24 fully annotated resting clips for a person")
        choose=np.linspace(0,len(candidates)-1,24,dtype=int)
        for i in choose:
            r,start,end=candidates[i]
            a,b=int(start*400),int(end*400)
            reference=resample_poly(-r["raw_ppg"][a:b],5,16)
            onsets=r["onset_times_s"]
            onsets=onsets[(onsets>=start+1)&(onsets<start+7)]
            hr=60/np.diff(onsets).mean() if len(onsets)>=3 else np.nan
            for snr in LEVELS:
                identity=f"{person}/{r['record_id']}/{start:.3f}/{snr}"
                x,y=synthetic_pair(reference,snr,identity)
                noisy.append(x); clean.append(y); subjects.append(person); hrs.append(hr)
                rows.append({"subject":person,"record_id":r["record_id"],"start_s":start,"duration_s":8,"noise_snr_db":snr,"pair_key":identity})
            for path in r["source_paths"]:
                source_hashes[path.relative_to(ROOT).as_posix()]=sha256(path)
    data,rows=finalized_arrays(noisy,clean,subjects,rows,hrs)
    np.savez_compressed(output/"EMBC21.npz",**data)
    folds=[{"fold":k,"test":[person],"validation":[people[(k+1)%7]],
            "train":[p for p in people if p not in (person,people[(k+1)%7])]} for k,person in enumerate(people)]
    write_json(output/"EMBC21.json",{"rows":rows,"source_hashes":source_hashes,"folds":folds,"data_sha256":sha256(output/"EMBC21.npz")})
    source=ROOT/"datasets/epfl_synthetic/extracted/DataPPGwNoise/Datasets"
    clean_path=source/"GoodSignals/PPG/ppg.mat"
    reference=load_vector(clean_path,"ppg")
    files=sorted((source/"NoisySignals/PPG").glob("*.mat"),key=lambda p:tuple(map(int,re.findall(r"\d+",p.stem))))
    if len(files)!=340:
        raise ValueError("Expected all340 EPFL noise copies")
    noisy,clean,subjects,rows,hrs=[],[],[],[],[]
    source_hashes={clean_path.relative_to(ROOT).as_posix():sha256(clean_path)}
    for path in files:
        x=load_vector(path,"noise_signal")
        code=int(re.findall(r"\d+",path.stem)[0])
        for start in range(0,72,8):
            a,b=start*250,(start+8)*250
            noisy.append(resample_poly(x[a:b],1,2))
            clean.append(resample_poly(reference[a:b],1,2))
            subjects.append("EPFL-person1"); hrs.append(np.nan)
            rows.append({"subject":"EPFL-person1","record_id":path.stem,"start_s":start,"duration_s":8,"filename_noise_code":code})
        source_hashes[path.relative_to(ROOT).as_posix()]=sha256(path)
    data,rows=finalized_arrays(noisy,clean,subjects,rows,hrs)
    data["reference_hr"]=predict_hr_windows(data["target"][:,64:448],64)["welch"]
    np.savez_compressed(output/"EPFL.npz",**data)
    write_json(output/"EPFL.json",{"rows":rows,"source_hashes":source_hashes,"data_sha256":sha256(output/"EPFL.npz")})
    write_json(shared/"data_audit.json",{"reference_people_EMBC21":7,"rest_clips_per_person":24,"noise_realizations_per_clip":3,
               "EMBC21_pairs":504,"EPFL_underlying_people":1,"EPFL_noisy_copies":340,"EPFL_window_pairs":3060,
               "full_annotation_and_finite_window_policy":True,"clean_used_for_input_normalization":False,
               "data_sha256":{d:sha256(output/f"{d}.npz") for d in ("EMBC21","EPFL")},
               "source_note":"EMBC raw reference may retain residual noise; expert validity is an onset-annotation judgement, not proof of noise-free morphology"})
    print(json.dumps({"stage":"prepared","EMBC21_pairs":504,"EPFL_pairs":3060}),flush=True)


def load_data(output,dataset):
    meta=json.loads((output/f"{dataset}.json").read_text())
    if sha256(output/f"{dataset}.npz")!=meta["data_sha256"]:
        raise ValueError("Prepared pairs changed")
    with np.load(output/f"{dataset}.npz",allow_pickle=False) as blob:
        data={k:blob[k] for k in blob.files}
    return data,meta


def cache_features(output,methods,device):
    data,_=load_data(output,"EMBC21")
    torch.set_num_threads(2)
    for method in methods:
        if method=="SmallCNN":
            continue
        for seed in SEEDS if method=="AnyPPG-random-frozen" else (42,):
            folder=output/"features"/f"{method}__seed{seed}"
            folder.mkdir(parents=True,exist_ok=True)
            done=folder/"complete.json"
            if done.exists():
                info=json.loads(done.read_text())
                if info["input_sha256"]!=array_hash(data["x125"]) or any(sha256(folder/name)!=digest for name,digest in info["files"].items()):
                    raise ValueError("Temporal cache changed")
                continue
            encoder=TemporalEncoder(method,seed,device)
            parts=[[] for _ in range(6)]
            for start in range(0,len(data["x125"]),16):
                maps=encoder(data["x125"][start:start+16])
                for destination,x in zip(parts,maps):
                    destination.append(x.cpu().numpy())
            for i,items in enumerate(parts):
                np.save(folder/f"stage{i}.npy",np.concatenate(items))
            info=encoder.audit()
            info.update(input_sha256=array_hash(data["x125"]),files={f"stage{i}.npy":sha256(folder/f"stage{i}.npy") for i in range(6)})
            write_json(done,info)
            del encoder,maps,parts
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            print(json.dumps({"stage":"features_complete","method":method,"seed":seed}),flush=True)


def metrics(target,pred,reference_hr):
    truth=np.asarray(target,dtype=float)[:,64:448]; estimate=np.asarray(pred,dtype=float)[:,64:448]
    error=estimate-truth
    rms=np.sqrt(np.mean(truth**2,axis=1)); rmse=np.sqrt(np.mean(error**2,axis=1))
    nrmse=rmse/np.maximum(rms,1e-12)
    centered_y=truth-truth.mean(axis=1,keepdims=True); centered_p=estimate-estimate.mean(axis=1,keepdims=True)
    denominator=np.sqrt(np.sum(centered_y**2,axis=1)*np.sum(centered_p**2,axis=1))
    corr=np.sum(centered_y*centered_p,axis=1)/np.maximum(denominator,1e-12)
    hr=predict_hr_windows(estimate,64)["welch"]
    return {"nrmse":nrmse,"snr_db":-20*np.log10(np.maximum(nrmse,1e-12)),"correlation":corr,
            "hr_absolute_error":np.abs(hr-reference_hr),"hr_prediction":hr}


def _network(method):
    head=SmallEventCNN() if method=="SmallCNN" else make_head(method)
    last=head.net[-1] if method=="SmallCNN" else head.decoder[-1]
    nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
    return head


def _correction(head,method,maps,x,idx):
    return head(x[idx].unsqueeze(1)) if method=="SmallCNN" else head([z[idx] for z in maps])


@torch.no_grad()
def predict(head,method,maps,x,baseline,indices,batch=64):
    head.eval(); result=[]
    for start in range(0,len(indices),batch):
        idx=indices[start:start+batch]
        result.append((baseline[idx]+_correction(head,method,maps,x,idx)).cpu().numpy())
    return np.concatenate(result)


def run_fits(output,methods,device):
    torch.set_num_threads(2)
    data,manifest=load_data(output,"EMBC21")
    x=torch.tensor(data["x64"],device=device); baseline=torch.tensor(data["baseline"],device=device)
    target=torch.tensor(data["target"],device=device)
    all_rows=[]
    for method in methods:
        for seed in SEEDS:
            source_seed=seed if method=="AnyPPG-random-frozen" else 42
            maps=None
            if method!="SmallCNN":
                folder=output/"features"/f"{method}__seed{source_seed}"
                maps=[torch.tensor(np.load(folder/f"stage{i}.npy"),device=device) for i in range(6)]
            for split in manifest["folds"]:
                train,validation,test=[np.flatnonzero(np.isin(data["subjects"],split[k])) for k in ("train","validation","test")]
                folder=output/"runs"/f"{method}__s{seed}__f{split['fold']}"
                folder.mkdir(parents=True,exist_ok=True)
                record=folder/"complete.json"
                if record.exists():
                    row=json.loads(record.read_text())
                    if row["data_sha256"]!=manifest["data_sha256"] or sha256(folder/"predictions.npz")!=row["prediction_sha256"] or sha256(folder/"weights.pt")!=row["weights_sha256"]:
                        raise ValueError("Existing dense result changed")
                    all_rows.append(row); continue
                torch.manual_seed(seed)
                head=_network(method).to(device)
                optimizer=torch.optim.Adam(head.parameters(),lr=.001)
                rng=np.random.default_rng(seed); best_score=float("inf"); best_epoch=None; best=None; history=[]
                counts={p:int(np.sum(data["subjects"][train]==p)) for p in split["train"]}
                weight=torch.tensor([len(train)/(len(counts)*counts[p]) if p in counts else 0 for p in data["subjects"]],dtype=torch.float32,device=device)
                for epoch in range(1,21):
                    head.train(); order=rng.permutation(train)
                    for start in range(0,len(order),64):
                        idx=order[start:start+64]
                        optimizer.zero_grad(set_to_none=True)
                        pred=baseline[idx]+_correction(head,method,maps,x,idx)
                        loss=((pred[:,64:448]-target[idx,64:448]).square().mean(dim=1)*weight[idx]).mean()
                        loss.backward(); optimizer.step()
                    val=predict(head,method,maps,x,baseline,validation)
                    m=metrics(data["target"][validation],val,data["reference_hr"][validation])
                    score=float(np.mean(m["nrmse"])); history.append(score)
                    if score<best_score:
                        best_score,best_epoch=score,epoch
                        best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                    if epoch-best_epoch>=5:
                        break
                head.load_state_dict(best); head.eval()
                pred=predict(head,method,maps,x,baseline,test)
                values=metrics(data["target"][test],pred,data["reference_hr"][test])
                torch.save(best,folder/"weights.pt")
                np.savez_compressed(folder/"predictions.npz",indices=test,prediction=pred,**values)
                row={"method":method,"seed":seed,"fold":split["fold"],"train":split["train"],"validation":split["validation"],"test":split["test"],
                     "best_epoch":best_epoch,"validation_nrmse":best_score,"validation_curve":history,
                     "trainable_parameters":sum(p.numel() for p in head.parameters()),"nrmse":float(values["nrmse"].mean()),
                     "snr_db":float(values["snr_db"].mean()),"correlation":float(values["correlation"].mean()),
                     "data_sha256":manifest["data_sha256"],"prediction_sha256":sha256(folder/"predictions.npz"),"weights_sha256":sha256(folder/"weights.pt"),
                     "run_folder":folder.relative_to(output).as_posix(),"device":device}
                write_json(record,row); all_rows.append(row)
                print(json.dumps({"stage":"fit_complete","method":method,"seed":seed,"fold":split["fold"],"nrmse":row["nrmse"]}),flush=True)
                del head,optimizer
            del maps
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    write_json(output/"results.json",all_rows)


def external(output,methods,device):
    """Run all trained heads with shared frozen EPFL maps, batchwise."""
    torch.set_num_threads(2)
    data,manifest=load_data(output,"EPFL")
    for method in methods:
        seeds=SEEDS if method=="AnyPPG-random-frozen" else (None,)
        for feature_seed in seeds:
            selected=[r for r in json.loads((output/"results.json").read_text()) if r["method"]==method and (feature_seed is None or r["seed"]==feature_seed)]
            if len(selected)!=(7 if feature_seed is not None else 21):
                raise ValueError("External test requires every fitted fold and seed")
            heads=[]; predictions=[]; destinations=[]
            for row in selected:
                folder=output/row["run_folder"]
                head=_network(method).to(device)
                head.load_state_dict(torch.load(folder/"weights.pt",map_location=device,weights_only=True),strict=True); head.eval()
                heads.append(head); predictions.append([]); destinations.append(folder)
            encoder=None if method=="SmallCNN" else TemporalEncoder(method,feature_seed or 42,device)
            for start in range(0,len(data["x125"]),32):
                stop=start+32
                maps=None if encoder is None else encoder(data["x125"][start:stop])
                x=torch.tensor(data["x64"][start:stop],device=device)
                baseline=torch.tensor(data["baseline"][start:stop],device=device)
                with torch.no_grad():
                    for head,collector in zip(heads,predictions):
                        correction=head(x.unsqueeze(1)) if method=="SmallCNN" else head(maps)
                        collector.append((baseline+correction).cpu().numpy())
            evidence=encoder.audit() if encoder is not None else {"method":"SmallCNN","encoder_frozen":False,"task_supervised_control":True}
            for folder,items in zip(destinations,predictions):
                pred=np.concatenate(items); values=metrics(data["target"],pred,data["reference_hr"])
                np.savez_compressed(folder/"epfl_predictions.npz",prediction=pred,**values)
                write_json(folder/"epfl.json",{"data_sha256":manifest["data_sha256"],"prediction_sha256":sha256(folder/"epfl_predictions.npz"),"encoder":evidence,"people":1,"noisy_copies":340,"windows":3060})
            del encoder,heads,predictions,maps
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            print(json.dumps({"stage":"external_complete","method":method,"feature_seed":feature_seed,"heads":len(selected)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("prepare","features","fit","external"),default="prepare")
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--output",type=Path,default=ROOT/".local/ppg-formal-expansion/denoising")
    parser.add_argument("--shared",type=Path,default=ROOT/"docs/results/ppg_formal_expansion/denoising")
    args=parser.parse_args()
    if args.stage=="prepare": prepare(args.output,args.shared)
    elif args.stage=="features": cache_features(args.output,args.methods,args.device)
    elif args.stage=="fit": run_fits(args.output,args.methods,args.device)
    else: external(args.output,args.methods,args.device)


if __name__=="__main__":
    main()
