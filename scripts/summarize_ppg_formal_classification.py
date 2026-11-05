"""Audit held-out identities, replay fitted heads and aggregate by participant."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.ppg_formal_classification import (
    DEFAULT,ENCODERS,HEADS,SEEDS,feature_key,indices,linear_replay,paired_views,
    project,read_csv,read_json,score,validate,weights,write_csv,
)
from scripts.ppg_hr_poc_features import sha256,write_json


def weighted_auc(y,p,w):
    order=np.argsort(p,kind='stable'); y=y[order];p=p[order];w=w[order]
    ends=np.r_[np.flatnonzero(np.diff(p)!=0),len(p)-1]
    starts=np.r_[0,ends[:-1]+1]
    positive=np.add.reduceat(w*y,starts);negative=np.add.reduceat(w*(1-y),starts)
    denominator=positive.sum()*negative.sum()
    if denominator<=0: raise ValueError('Bootstrap contains only one class')
    return float(np.dot(positive,np.cumsum(negative)-.5*negative)/denominator)


def weighted_metric(y,p,decision,w,name):
    if name=='auroc':return weighted_auc(y,p[:,1],w)
    k=p.shape[1];cm=np.bincount(y*k+decision,weights=w,minlength=k*k).reshape(k,k)
    if name=='balanced_accuracy':return float(np.mean(np.diag(cm)/cm.sum(1)))
    if name=='macro_f1':return float(np.mean(2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1e-20)))
    raise ValueError(name)


def holm(values):
    order=np.argsort(values);out=np.empty(len(values));last=0.
    for rank,i in enumerate(order):
        last=max(last,min(1.,float(values[i])*(len(values)-rank)));out[i]=last
    return out


def replay_neural(folder,model,head,state_path,x,classes):
    import torch
    from torch import nn
    from scripts.ppg_hr_cnn import SmallHRCNN,normalize_windows
    torch.set_num_threads(2)
    cnn=model=='SmallCNN'
    if cnn:
        network=SmallHRCNN();network.head[-1]=nn.Linear(32,classes)
        tx=normalize_windows(torch.tensor(paired_views(x,64),dtype=torch.float32).reshape(-1,1,512)).reshape(len(x),2,1,512)
    else:
        network=nn.Sequential(nn.Linear(x.shape[1],16),nn.ReLU(),nn.Linear(16,classes))
        tx=torch.tensor(x,dtype=torch.float32)
    network.load_state_dict(torch.load(state_path,map_location='cpu',weights_only=True));network.eval()
    out=[]
    with torch.inference_mode():
        for start in range(0,len(tx),256):
            v=tx[start:start+256]
            logits=network(v.reshape(-1,1,512)).reshape(len(v),2,classes).mean(1) if cnn else network(v)
            out.append(logits.softmax(1).numpy())
    return np.concatenate(out)


def audit(dataset,root,shared,bootstrap=1000,permutations=2000):
    folder=root/dataset;protocol=validate(folder);rows=read_csv(folder/'rows.csv')
    y=np.asarray([int(r['target']) for r in rows]);subjects=np.asarray([r['subject'] for r in rows]);people,inv=np.unique(subjects,return_inverse=True)
    methods=[f'{m}/{h}' for m in ENCODERS for h in HEADS]+['HandFeatures/Linear','SmallCNN','TrainPrior']
    primary=protocol['primary_metric'];predictions={};decisions={};score_rows=[];subject_rows=[]
    maximum_replay=0.;checked=0;seen_hashes={};feature_cache={}
    for method in methods:
        predictions[method]=[];decisions[method]=[]
        for seed in SEEDS:
            p=np.full((len(y),int(y.max())+1),np.nan);d=np.full(len(y),-1);seen=np.zeros(len(y),int)
            for job in [j for j in protocol['jobs'] if j['seed']==seed]:
                tr,va,te=indices(subjects,job);dest=folder/'runs'/job['id']/method
                info=read_json(dest/'complete.json');fp=dest/'predictions.npz'
                if info['identity']['protocol_sha256']!=sha256(folder/'protocol.json') or info['prediction_sha256']!=sha256(fp):raise ValueError('Published result mismatch')
                if info['identity']['job']!=job or info['identity']['method']!=method:raise ValueError('Result split/method mismatch')
                with np.load(fp,allow_pickle=False) as f:
                    if not np.array_equal(f['row'],te) or not np.array_equal(f['subject'],subjects[te]) or not np.array_equal(f['reference'],y[te]):raise ValueError('Held-out identities differ')
                    p[te]=f['probability'];d[te]=f['decision'];seen[te]+=1
                threshold=info['selection']['threshold']
                expected=(p[te,1]>=threshold).astype(int) if p.shape[1]==2 else p[te].argmax(1)
                if not np.array_equal(d[te],expected):raise ValueError('Decision threshold mismatch')
                scores=score(y[te],p[te],subjects[te],d[te])
                if any(abs(scores[k]-info['metrics'][k])>1e-12 for k in scores):raise ValueError('Test metric mismatch')
                model,*head=method.split('/');head=head[0] if head else ''
                if model in ENCODERS:
                    key=feature_key(model,seed)
                    if key not in feature_cache:
                        mp=folder/'features'/f'{key}.json';metadata=read_json(mp);path=folder/'features'/f'{key}.npy'
                        if sha256(path)!=metadata['feature_sha256'] or metadata['evidence']['state_before_sha256']!=metadata['evidence']['state_after_sha256']:
                            raise ValueError('Feature hash or frozen encoder mismatch')
                        feature_cache[key]=np.load(path)
                    with np.load(dest/'projection.npz') as f: transform={k:f[k] for k in f.files}
                    x=project(feature_cache[key][te],transform)
                elif model=='HandFeatures':x=np.load(folder/'hand.npy')[te]
                elif model=='SmallCNN':x=np.load(folder/'x64.npy')[te]
                if head in ('PCA16-Linear','Linear'):
                    with np.load(dest/'model.npz') as f:state={k:f[k] for k in f.files}
                    repeated=linear_replay(x,state)
                elif model=='TrainPrior':
                    counts=np.bincount(y[tr],weights=weights(subjects[tr]),minlength=p.shape[1]);repeated=np.tile(counts/counts.sum(),(len(te),1))
                else:repeated=replay_neural(folder,model,head,dest/'model.pt',x,p.shape[1])
                difference=float(np.max(np.abs(repeated-p[te])));maximum_replay=max(maximum_replay,difference)
                if difference>2e-6:raise ValueError(f'Independent head replay differs: {method} {difference}')
                checked+=1;seen_hashes[f'{job["id"]}/{method}']=sha256(fp)
            if not np.all(seen==1):raise ValueError('Incomplete or duplicate held-out predictions')
            predictions[method].append(p);decisions[method].append(d)
            scores=score(y,p,subjects,d)
            score_rows.append(dict(dataset=dataset,method=method,seed=seed,**scores))
            for person in people:
                mask=subjects==person
                # Do not invent per-person AUROC for participants with one class.
                error=float(np.mean(d[mask]!=y[mask]));subject_rows.append(dict(dataset=dataset,method=method,seed=seed,subject=person,records=int(mask.sum()),error_rate=error))
    rng=np.random.default_rng(20261105);base=weights(subjects);replicates={m:[] for m in methods}
    estimate={m:float(np.mean([weighted_metric(y,p,d,base,primary) for p,d in zip(predictions[m],decisions[m])])) for m in methods}
    for _ in range(bootstrap):
        count=np.bincount(rng.integers(len(people),size=len(people)),minlength=len(people));w=base*count[inv]
        for method in methods:
            replicates[method].append(float(np.mean([weighted_metric(y,p,d,w,primary) for p,d in zip(predictions[method],decisions[method])])))
    summary=[]
    for method in methods:
        lo,hi=np.percentile(replicates[method],[2.5,97.5])
        summary.append(dict(dataset=dataset,method=method,metric=primary,estimate=estimate[method],ci95_low=float(lo),ci95_high=float(hi),participants=len(people),records=len(y),seeds=3))
    pairs=[(f'{m}/PCA16-Linear','HandFeatures/Linear') for m in ENCODERS if m!='AnyPPG-random-frozen']
    pairs.append(('AnyPPG/PCA16-Linear','AnyPPG-random-frozen/PCA16-Linear'))
    paired=[]
    for left,right in pairs:
        delta=estimate[left]-estimate[right];distribution=np.asarray(replicates[left])-replicates[right]
        lo,hi=np.percentile(distribution,[2.5,97.5]);extreme=0
        # Swap the complete prediction blocks of people, preserving every seed
        # and record from one person together; recompute the pooled estimand.
        for _ in range(permutations):
            swap=rng.integers(2,size=len(people)).astype(bool)[inv];values=[]
            for seed in range(3):
                pa,pb=predictions[left][seed],predictions[right][seed];da,db=decisions[left][seed],decisions[right][seed]
                a=np.where(swap[:,None],pb,pa);b=np.where(swap[:,None],pa,pb)
                ad=np.where(swap,db,da);bd=np.where(swap,da,db)
                values.append(weighted_metric(y,a,ad,base,primary)-weighted_metric(y,b,bd,base,primary))
            extreme+=abs(float(np.mean(values)))>=abs(delta)-1e-12
        paired.append(dict(dataset=dataset,model=left,comparator=right,metric=primary,difference=delta,
                           ci95_low=float(lo),ci95_high=float(hi),permutation_p=(extreme+1)/(permutations+1)))
    for row,pvalue in zip(paired,holm([r['permutation_p'] for r in paired])):row['holm_p']=float(pvalue)
    destination=shared/dataset;destination.mkdir(parents=True,exist_ok=True)
    for name,data in [('summary.csv',summary),('paired_primary.csv',paired),('seed_scores.csv',score_rows),('per_subject.csv',subject_rows)]:write_csv(destination/name,data)
    write_json(destination/'protocol.json',protocol)
    write_json(destination/'audit.json',dict(status='complete',fits_verified=checked,expected_fits=len(protocol['jobs'])*len(methods),
        participants=len(people),records=len(y),maximum_head_replay_probability_difference=maximum_replay,
        protocol_sha256=sha256(folder/'protocol.json'),predictions_sha256=seen_hashes,
        bootstrap_replicates=bootstrap,permutation_replicates=permutations,
        uncertainty='Participant bootstrap, average metric across seeds; CIs pointwise. Two-sided paired participant prediction-swap permutation p with +1 correction; Holm across seven fixed primary comparisons.',
        summary_sha256={n:sha256(destination/n) for n in ('summary.csv','paired_primary.csv','seed_scores.csv','per_subject.csv')},
        audit_code_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(dataset=dataset,verified=checked,maximum_replay=maximum_replay,summary=summary)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--datasets',nargs='+',default=['BUT-PPG']);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--shared',type=Path,default=ROOT/'docs/results/ppg_formal_expansion/classification');p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--permutations',type=int,default=2000)
    args=p.parse_args()
    with threadpool_limits(limits=2):
        for dataset in args.datasets:audit(dataset,args.output,args.shared,args.bootstrap,args.permutations)


if __name__=='__main__':main()
