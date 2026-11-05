"""Small frozen temporal heads for audited person-independent artifact labels."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.ppg_hr_poc_features import sha256,write_json
from scripts.ppg_formal_classification import indices,read_json,write_csv,weights
from scripts.ppg_sqi_poc import handcrafted
from scripts.ppg_temporal_adapters import TemporalEncoder,make_head
from scripts.ppg_beat_models import SmallEventCNN

DEFAULT=ROOT/'.local/ppg-formal-expansion/artifact'
NEURAL=('AnyPPG','PaPaGei-S','PaPaGei-P','AnyPPG-random-frozen','SmallCNN')
SEEDS=(42,43,44)
THRESHOLDS=np.arange(1,10)/10


def load(root):
    protocol=read_json(root/'protocol.json')
    if sha256(root/'data.npz')!=protocol['data_sha256'] or sha256(root/'rows.json')!=protocol['rows_sha256']:
        raise ValueError('Artifact reference cache changed')
    with np.load(root/'data.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    return protocol,data


def subset(subjects,people,limit):
    result=[]
    for person in people:
        ix=np.flatnonzero(subjects==person)
        if not len(ix):raise ValueError('Empty person')
        result.extend(ix[np.linspace(0,len(ix)-1,min(limit,len(ix)),dtype=int)].tolist())
    return np.asarray(result)


def local_features(x):
    """Label-free frame SQI plus local amplitude/variation at every sample."""
    x=np.asarray(x,dtype=float);z=(x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-6)
    derivative=np.diff(z,axis=1,prepend=z[:,:1])
    channels=[np.abs(z),np.abs(derivative),np.abs(np.diff(derivative,axis=1,prepend=derivative[:,:1]))]
    for width in (16,64,128):
        mean=uniform_filter1d(z,width,axis=1,mode='nearest')
        variance=np.maximum(uniform_filter1d(z*z,width,axis=1,mode='nearest')-mean*mean,0)
        channels.extend([mean,np.sqrt(variance)])
    global_features=handcrafted(x,64)
    return np.concatenate([np.stack(channels,axis=2),np.broadcast_to(global_features[:,None,:],(len(x),512,global_features.shape[1]))],axis=2).astype(np.float32)


def counts(y,p,t=.5):
    a=np.asarray(y,dtype=bool).ravel();b=np.asarray(p).ravel()>=t
    tp=int(np.sum(a&b));fp=int(np.sum(~a&b));fn=int(np.sum(a&~b));tn=int(np.sum(~a&~b))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,f1=2*tp/max(2*tp+fp+fn,1),
        sensitivity=tp/max(tp+fn,1),specificity=tn/max(tn+fp,1),
        balanced_accuracy=.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1)))


def person_f1(y,p,subjects,t):
    return float(np.mean([counts(y[subjects==s],p[subjects==s],t)['f1'] for s in np.unique(subjects)]))


def select_threshold(y,p,subjects):
    return max((person_f1(y,p,subjects,t),-abs(t-.5),-t,t) for t in THRESHOLDS)[-1]


def key(method,seed):
    return f'{method}-seed{seed}' if method=='AnyPPG-random-frozen' else method


def features(root,models,device):
    protocol,data=load(root);torch.set_num_threads(2)
    for method in models:
        if method=='SmallCNN':continue
        for seed in SEEDS if method=='AnyPPG-random-frozen' else [42]:
            folder=root/'features'/key(method,seed);folder.mkdir(parents=True,exist_ok=True)
            done=folder/'complete.json'
            if done.exists():
                info=read_json(done)
                if info['protocol_sha256']!=sha256(root/'protocol.json') or any(sha256(folder/n)!=h for n,h in info['files_sha256'].items()):raise ValueError('Temporal cache changed')
                continue
            encoder=TemporalEncoder(method,seed,device);arrays=None
            for begin in range(0,len(data['x125']),32):
                stages=encoder(data['x125'][begin:begin+32])
                if arrays is None:arrays=[np.lib.format.open_memmap(folder/f'stage{i}.npy',mode='w+',dtype=np.float32,shape=(len(data['x125']),*z.shape[1:])) for i,z in enumerate(stages)]
                for target,z in zip(arrays,stages):target[begin:begin+len(z)]=z.cpu().numpy()
            for a in arrays:a.flush()
            write_json(done,dict(protocol_sha256=sha256(root/'protocol.json'),encoder=encoder.audit(),
                files_sha256={f'stage{i}.npy':sha256(folder/f'stage{i}.npy') for i in range(6)}))
            print(json.dumps(dict(features=key(method,seed),frames=len(data['x125']))),flush=True)
            del encoder,arrays;gc.collect()


def inputs(root,data,method,seed):
    if method=='SmallCNN':return [data['x64']]
    folder=root/'features'/key(method,seed);info=read_json(folder/'complete.json')
    if info['protocol_sha256']!=sha256(root/'protocol.json') or any(sha256(folder/n)!=h for n,h in info['files_sha256'].items()):raise ValueError('Frozen feature identity mismatch')
    return [np.load(folder/f'stage{i}.npy',mmap_mode='r') for i in range(6)]


def neural_fit(root,data,job,method,device):
    dest=root/'runs'/job['id']/method;dest.mkdir(parents=True,exist_ok=True)
    complete=dest/'complete.json';identity=dict(protocol_sha256=sha256(root/'protocol.json'),job=job,method=method)
    if complete.exists():
        info=read_json(complete)
        if info['identity']!=identity or info['prediction_sha256']!=sha256(dest/'prediction.npz'):raise ValueError('Existing artifact result changed')
        return
    _,_,test=indices(data['subject'],job)
    train=subset(data['subject'],job['train'],128);val=subset(data['subject'],job['validation'],64)
    arrays=inputs(root,data,method,job['seed']);torch.manual_seed(job['seed'])
    model=(SmallEventCNN() if method=='SmallCNN' else make_head(method)).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);w=weights(data['subject'][train])
    def batch(ix):
        values=[torch.tensor(np.asarray(a[ix]),dtype=torch.float32,device=device) for a in arrays]
        return values[0].unsqueeze(1) if method=='SmallCNN' else values
    def predict(ix):
        model.eval()
        with torch.inference_mode():return np.concatenate([model(batch(ix[i:i+32]))[:,64:448].sigmoid().cpu().numpy() for i in range(0,len(ix),32)])
    best=(-np.inf,None,None,0);history=[];start=time.monotonic()
    for epoch in range(1,21):
        model.train();order=np.random.default_rng(job['seed']+epoch).permutation(len(train))
        for begin in range(0,len(order),32):
            local=order[begin:begin+32];ix=train[local]
            optimizer.zero_grad(set_to_none=True)
            target=torch.tensor(data['target'][ix,64:448],device=device)
            loss=F.binary_cross_entropy_with_logits(model(batch(ix))[:,64:448],target,reduction='none').mean(1)
            loss=(loss*torch.tensor(w[local],dtype=torch.float32,device=device)).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite artifact loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
        vp=predict(val);t=select_threshold(data['target'][val,64:448],vp,data['subject'][val]);f1=person_f1(data['target'][val,64:448],vp,data['subject'][val],t)
        history.append(dict(epoch=epoch,validation_f1=f1,threshold=t))
        if f1>best[0]:best=(f1,{n:v.detach().cpu().clone() for n,v in model.state_dict().items()},t,epoch)
        if epoch-best[3]>=5:break
    model.load_state_dict(best[1]);p=predict(test)
    torch.save(best[1],dest/'head.pt')
    np.savez_compressed(dest/'prediction.npz',row=test,subject=data['subject'][test],reference=data['target'][test,64:448],probability=p)
    write_json(complete,dict(identity=identity,threshold=best[2],best_epoch=best[3],validation_f1=best[0],history=history,
        train_rows=train.tolist(),validation_rows=val.tolist(),head_sha256=sha256(dest/'head.pt'),parameters=sum(v.numel() for v in model.parameters()),
        prediction_sha256=sha256(dest/'prediction.npz'),elapsed_seconds=time.monotonic()-start,code_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(job=job['id'],method=method,seconds=round(time.monotonic()-start,2))),flush=True)


def traditional(root,data,job,values):
    train=subset(data['subject'],job['train'],128);val=subset(data['subject'],job['validation'],64);_,_,test=indices(data['subject'],job)
    for method in ('HandFeatures','TrainPrior'):
        dest=root/'runs'/job['id']/method;dest.mkdir(parents=True,exist_ok=True)
        identity=dict(protocol_sha256=sha256(root/'protocol.json'),job=job,method=method)
        if (dest/'complete.json').exists():
            meta=read_json(dest/'complete.json')
            if meta['identity']!=identity or meta['prediction_sha256']!=sha256(dest/'prediction.npz'):raise ValueError('Artifact control changed')
            continue
        target=data['target'][train,64:448:4].ravel();w=np.repeat(weights(data['subject'][train]),96)
        if method=='HandFeatures':
            x=values[train,64:448:4].reshape(-1,values.shape[-1]).astype(float);mean=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1.;x=(x-mean)/scale
            vx=values[val,64:448].reshape(-1,values.shape[-1]).astype(float);vx=(vx-mean)/scale;best=None;candidates=[]
            for c in (.01,.1,1.,10.):
                model=LogisticRegression(C=c,max_iter=1000).fit(x,target,sample_weight=w)
                if model.n_iter_.max()>=1000:raise ValueError('Artifact logistic failed to converge')
                vp=model.predict_proba(vx)[:,1].reshape(len(val),384);t=select_threshold(data['target'][val,64:448],vp,data['subject'][val]);metric=person_f1(data['target'][val,64:448],vp,data['subject'][val],t)
                candidates.append(dict(C=c,validation_f1=metric))
                if best is None or metric>best[0]:best=(metric,model,t)
            p=best[1].predict_proba((values[test,64:448].reshape(-1,values.shape[-1])-mean)/scale)[:,1].reshape(len(test),384)
            t=best[2];np.savez(dest/'model.npz',mean=mean,scale=scale,coef=best[1].coef_,intercept=best[1].intercept_)
            details=dict(candidates=candidates,selected_C=best[1].C)
        else:
            prior=float(np.average(target,weights=w));vp=np.full((len(val),384),prior);p=np.full((len(test),384),prior)
            t=select_threshold(data['target'][val,64:448],vp,data['subject'][val]);details=dict(prior=prior)
        np.savez_compressed(dest/'prediction.npz',row=test,subject=data['subject'][test],reference=data['target'][test,64:448],probability=p)
        write_json(dest/'complete.json',dict(identity=identity,threshold=t,selection=details,train_rows=train.tolist(),validation_rows=val.tolist(),prediction_sha256=sha256(dest/'prediction.npz')))
        print(json.dumps(dict(job=job['id'],method=method)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['features','learned','traditional'],required=True)
    p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--device',default='cpu');p.add_argument('--models',nargs='+',choices=NEURAL,default=list(NEURAL));p.add_argument('--job')
    args=p.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.stage=='features':features(args.root,args.models,args.device);return
        protocol,data=load(args.root);values=local_features(data['x64']) if args.stage=='traditional' else None
        for job in protocol['jobs']:
            if args.job and job['id']!=args.job:continue
            if args.stage=='traditional':traditional(args.root,data,job,values)
            else:
                for method in args.models:neural_fit(args.root,data,job,method,args.device)


if __name__=='__main__':main()
