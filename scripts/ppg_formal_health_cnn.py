"""Supervised small CNN control: pool a person's clips before one person loss."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import resample_poly
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.ppg_formal_health import SEEDS,load_prepared,prepared_group,write_json
from scripts.ppg_hr_cnn import SmallHRCNN,normalize_windows
from scripts.ppg_hr_poc_features import sha256

CNN_PROTOCOL={"epochs":40,"patience":10,"learning_rate":.001,"weight_decay":1e-4,
              "person_batch":12,"embedding_dimension":16,"head_hidden":16,
              "loss":"one standardized squared error per person after averaging that person's clip embeddings",
              "features":"same four-block SmallHRCNN convolutional backbone with fixed adaptive pooling, new 16D projection",
              "input":"all common observed clips, resampled to 64 Hz and independently z-scored; no synthetic padding",
              "selection":"validation person MAE; earliest epoch ties; no refit on validation people"}


class PersonCNN(nn.Module):
    def __init__(self,clinical_width):
        super().__init__()
        self.features=SmallHRCNN().features
        self.projection=nn.Sequential(nn.Flatten(),nn.Linear(48*8,16),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(16+clinical_width,16),nn.ReLU(),nn.Linear(16,1))

    def forward(self,clips,clinical):
        if clips.ndim!=4 or clips.shape[2]!=1 or clips.shape[0]!=clinical.shape[0]:
            raise ValueError("Expected person,clip,channel,time input and one covariate row per person")
        n,k,_,length=clips.shape
        z=self.projection(self.features(clips.reshape(n*k,1,length))).reshape(n,k,16).mean(dim=1)
        return self.head(torch.cat([z,clinical],dim=1)).flatten()


def person_inputs(data):
    x=resample_poly(data["windows"],64,data["fs"],axis=-1).astype(np.float32)
    owners=data["waveform_subjects"]
    per_person=[x[owners==p] for p in data["subjects"]]
    if len({a.shape for a in per_person})!=1:
        raise ValueError("Every person must have the same predeclared number and length of clips")
    result=torch.tensor(np.asarray(per_person)).unsqueeze(2)
    return normalize_windows(result)


@torch.no_grad()
def predict(model,x,clinical,indices,center,scale,batch=12):
    output=[]
    for a in range(0,len(indices),batch):
        idx=indices[a:a+batch]
        output.append(model(x[idx],clinical[idx]).cpu().numpy().astype(float)*scale+center)
    return np.concatenate(output)


def fit_one(x,clinical,y,train,validation,test,seed,device):
    torch.set_num_threads(2)
    center=float(y[train].mean()); scale=max(float(y[train].std()),1.)
    tx=x.to(device); cov=torch.tensor(clinical,dtype=torch.float32,device=device)
    ty=torch.tensor((y-center)/scale,dtype=torch.float32,device=device)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.startswith("cuda") else []):
        torch.manual_seed(seed)
        model=PersonCNN(clinical.shape[1]).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=CNN_PROTOCOL["learning_rate"],weight_decay=CNN_PROTOCOL["weight_decay"])
        rng=np.random.default_rng(seed); best_score=float("inf"); best_epoch=None; best=None; history=[]
        for epoch in range(1,CNN_PROTOCOL["epochs"]+1):
            model.train()
            order=rng.permutation(train)
            for start in range(0,len(order),CNN_PROTOCOL["person_batch"]):
                idx=order[start:start+CNN_PROTOCOL["person_batch"]]
                optimizer.zero_grad(set_to_none=True)
                loss=(model(tx[idx],cov[idx])-ty[idx]).square().mean()
                loss.backward(); optimizer.step()
            model.eval()
            val=predict(model,tx,cov,validation,center,scale)
            score=float(np.mean(np.abs(val-y[validation]))); history.append(score)
            if score<best_score:
                best_score,best_epoch=score,epoch
                best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch-best_epoch>=CNN_PROTOCOL["patience"]:
                break
        model.load_state_dict(best); model.eval()
        pred=predict(model,tx,cov,test,center,scale)
    model.cpu()
    restored=PersonCNN(clinical.shape[1]); restored.load_state_dict(best,strict=True); restored.eval()
    cpu=predict(restored,x,torch.tensor(clinical,dtype=torch.float32),test,center,scale,batch=4)
    difference=float(np.max(np.abs(cpu-pred)))
    if not np.isfinite(pred).all() or difference>.002:
        raise ValueError("CPU reconstruction of CNN predictions failed")
    meta={"epochs":best_epoch,"validation_mae":best_score,"validation_mae_by_epoch":history,
          "trainable_parameters":sum(p.numel() for p in model.parameters()),"cpu_reload_max_difference":difference,
          "target_center":center,"target_scale":scale,"device":device,"protocol":CNN_PROTOCOL}
    del model,restored,tx,cov
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return pred,best,meta


def run(dataset,output,device):
    data,manifest=load_prepared(dataset,output)
    folder=output/dataset/"cnn"; folder.mkdir(exist_ok=True)
    x=person_inputs(data)
    people=data["subjects"]
    clinical_raw=np.column_stack([data["covariates"],data["traditional"]])
    rows=[]
    for split in manifest["split"]:
        train,validation,test=[np.flatnonzero(np.isin(people,split[k])) for k in ("train","validation","test")]
        clinical,preprocess=prepared_group(clinical_raw,train,2*clinical_raw.shape[1])
        for seed in SEEDS:
            for ti,target in enumerate(data["target_names"]):
                for group in ("embedding","demographics_traditional_embedding"):
                    identity=f"{target}__{group}__s{seed}__f{split['fold']}"
                    path=folder/f"{identity}.npz"; checkpoint=folder/f"{identity}.pt"; record=folder/f"{identity}.json"
                    if path.exists() and record.exists() and checkpoint.exists():
                        row=json.loads(record.read_text())
                        if row["data_sha256"]!=manifest["data_npz_sha256"] or row["prediction_sha256"]!=sha256(path) or row["checkpoint_sha256"]!=sha256(checkpoint):
                            raise ValueError("CNN result identity mismatch")
                    else:
                        inputs=clinical if group!="embedding" else np.zeros_like(clinical)
                        y=data["targets"][:,ti]
                        pred,state,meta=fit_one(x,inputs,y,train,validation,test,seed,device)
                        torch.save(state,checkpoint)
                        np.savez_compressed(path,subjects=people[test],y_true=y[test],y_pred=pred,
                                            train_subjects=people[train],validation_subjects=people[validation],clinical=inputs,**preprocess)
                        row={"dataset":dataset,"target":target,"model":"SmallCNN","inputs":group,"head":"cnn",
                             "seed":seed,"fold":split["fold"],"train_people":len(train),"validation_people":len(validation),"test_people":len(test),
                             "mae":float(np.mean(np.abs(pred-y[test]))),"rmse":float(np.sqrt(np.mean((pred-y[test])**2))),"bias":float(np.mean(pred-y[test])),
                             "prediction_file":path.relative_to(output).as_posix(),"prediction_sha256":sha256(path),
                             "checkpoint_file":checkpoint.relative_to(output).as_posix(),"checkpoint_sha256":sha256(checkpoint),
                             "data_sha256":manifest["data_npz_sha256"],**meta}
                        write_json(record,row)
                    rows.append(row)
            print(json.dumps({"dataset":dataset,"fold":split["fold"],"seed":seed,"phase":"person_cnn_complete","results":len(rows)}),flush=True)
    write_json(output/dataset/"cnn_results.json",rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",nargs="+",choices=("ECSMP","PPG-BP"),default=["ECSMP","PPG-BP"])
    parser.add_argument("--output",type=Path,default=ROOT/".local/ppg-formal-expansion/health")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=parser.parse_args()
    for dataset in args.datasets:
        run(dataset,args.output,args.device)


if __name__=="__main__":
    main()
