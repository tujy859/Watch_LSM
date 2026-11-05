"""Frozen six-model classification with identical observation and small heads.

BUT quality first; WESAD laboratory states and ECSMP stimulus identity follow.
Every ten-second record is represented by the same two eight-second views
(starts zero and two seconds), whose embeddings are averaged. Test people
never fit normalization, PCA, classifiers, epochs or decision thresholds.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import resample_poly
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ppg_hr_poc_features import sha256, write_json
from scripts.ppg_model_expansion_pilot import MODELS, extract_features
from scripts.ppg_sqi_poc import handcrafted, make_jobs as sqi_jobs, weights

DEFAULT = ROOT / '.local/ppg-formal-expansion/classification'
SEEDS = (42, 43, 44)
ENCODERS = (*MODELS, 'AnyPPG-random-frozen')
HEADS = ('PCA16-Linear', 'PCA16-MLP')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def split_jobs(subjects, dataset):
    people = np.asarray(sorted(set(subjects)))
    if dataset == 'BUT-PPG':
        return sqi_jobs(people)
    result = []
    for fold, (available, test) in enumerate(KFold(5, shuffle=True, random_state=20261105).split(people)):
        for seed in SEEDS:
            order = np.random.default_rng(seed + 1000 * fold).permutation(people[available])
            nval = len(order) // 4
            result.append(dict(id=f'fold{fold}-seed{seed}', fold=fold, seed=seed,
                               train=order[nval:].tolist(), validation=order[:nval].tolist(), test=people[test].tolist()))
    return result


def indices(subjects, job):
    groups = [set(job[key]) for key in ('train', 'validation', 'test')]
    if any(groups[i] & groups[j] for i, j in ((0,1),(0,2),(1,2))):
        raise ValueError('Participant leakage')
    values = [np.flatnonzero(np.isin(subjects, sorted(group))) for group in groups]
    if any(not len(v) for v in values):
        raise ValueError('Empty participant group')
    return values


def paired_views(x, fs):
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 10 * fs:
        raise ValueError('Expected complete ten-second records')
    return np.stack([x[:, :8*fs], x[:, 2*fs:10*fs]], axis=1)


def prepare(dataset, root):
    folder = root / dataset
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'protocol.json').exists():
        validate(folder); return
    hashes = {}; rows = []; arrays = []; statistics = []
    if dataset == 'BUT-PPG':
        source = ROOT / '.local/ppg-phase2/sqi'
        manifest = read_json(source/'data_manifest.json')
        for name in ('x125.npy','rows.csv'):
            if sha256(source/name) != manifest['files_sha256'][name]:
                raise ValueError('Audited quality data changed')
            hashes[(source/name).relative_to(ROOT).as_posix()] = sha256(source/name)
        x = np.load(source/'x125.npy')
        rows = [dict(subject=r['subject'], record=r['record_id'], start_s=0., target=int(r['quality'])) for r in read_csv(source/'rows.csv')]
        stats = np.column_stack([x.mean(1), x.std(1)])
    else:
        source = ROOT / 'datasets/ppg_bench_processed/fs_125' / dataset
        target = 'condition_3class' if dataset == 'WESAD' else 'stimulus'
        classes = range(3) if dataset == 'WESAD' else range(6)
        for person in sorted(p for p in source.iterdir() if p.is_dir()):
            xx = np.load(person/'ppg.npy'); yy = np.load(person/f'{target}.npy')
            meta = read_csv(person/'metadata.csv'); info = read_json(person/'stats.json')
            if len(xx) != len(yy) or len(xx) != len(meta):
                raise ValueError('Source waveform/label identity mismatch')
            selected = np.flatnonzero(np.isin(yy, list(classes)))
            for i in selected:
                if int(meta[i][target]) != int(yy[i]) or float(meta[i]['end_sec'])-float(meta[i]['start_sec']) != 10:
                    raise ValueError('Source reference or observation interval mismatch')
                rows.append(dict(subject=person.name, record=person.name, start_s=float(meta[i]['start_sec']), target=int(yy[i])))
            arrays.append(xx[selected])
            statistics.extend([[info['ppg_mean'],info['ppg_std']]]*len(selected))
            for name in ('ppg.npy', f'{target}.npy', 'metadata.csv', 'stats.json'):
                hashes[(person/name).relative_to(ROOT).as_posix()] = sha256(person/name)
        x = np.concatenate(arrays).astype(np.float32); stats = np.asarray(statistics, dtype=np.float32)
    if x.shape != (len(rows),1250) or not np.isfinite(x).all():
        raise ValueError('Invalid complete observation')
    identities = [(r['subject'],r['record'],r['start_s']) for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError('Duplicate sample identity')
    np.save(folder/'x125.npy', x); np.save(folder/'stats.npy', stats)
    np.save(folder/'hand.npy', handcrafted(x,125))
    np.save(folder/'x64.npy', resample_poly(x,64,125,axis=1).astype(np.float32))
    write_csv(folder/'rows.csv',rows)
    people = np.asarray([r['subject'] for r in rows])
    protocol = dict(dataset=dataset, records=len(rows), participants=len(set(people)),
        class_counts={str(c):sum(r['target']==c for r in rows) for c in sorted(set(r['target'] for r in rows))},
        jobs=split_jobs(people,dataset), encoders=list(ENCODERS), heads=list(HEADS),
        controls=['HandFeatures-Linear','SmallCNN','TrainPrior'],
        observation='Full ten seconds; all neural methods use two eight-second views starting 0 and 2 seconds; mean embeddings or CNN logits before prediction',
        normalization='Native per-model preprocessing; PulsePPG uses unlabeled whole-record statistics (BUT ten seconds, WESAD/ECSMP source acquisition)',
        feature_preprocessing='Training-only unweighted standardization, PCA16 covariance eigendecomposition, training component standardization; same features for both heads',
        linear={'C':[.01,.1,1.,10.,100.], 'max_iter':3000},
        mlp={'hidden':16,'epochs':40,'lr':.01,'weight_decay':.0001,'loss':'person-weighted cross entropy','select':'validation primary metric'},
        cnn={'epochs':20,'patience':5,'lr':.001,'batch_size':128,'loss':'person-weighted cross entropy','select':'validation primary metric'},
        primary_metric='auroc' if dataset=='BUT-PPG' else 'balanced_accuracy',
        binary_threshold='0.05 to 0.95 in steps .05, maximize validation person-weighted balanced accuracy; nearest .5 tie',
        test_protocol='Five person folds; 3 seed-matched random encoders and supervised heads; no train+validation refit',
        primary_comparisons='Six PCA16-Linear encoders vs HandFeatures-Linear, plus AnyPPG PCA16-Linear vs random same head; seven tests Holm corrected per dataset',
        other_comparisons='MLP head differences, CNN and remaining pairs exploratory; report paired intervals without choosing winner on test set',
        code_sha256=sha256(Path(__file__)), source_files_sha256=hashes,
        files_sha256={name:sha256(folder/name) for name in ('x125.npy','x64.npy','stats.npy','hand.npy','rows.csv')},
        pretraining=False, encoder_finetuning=False)
    write_json(folder/'protocol.json',protocol)
    print(json.dumps({'prepared':dataset,'records':len(rows),'participants':len(set(people))}),flush=True)


def validate(folder):
    protocol=read_json(folder/'protocol.json')
    for name,digest in protocol['files_sha256'].items():
        if sha256(folder/name)!=digest: raise ValueError(f'Prepared input changed: {name}')
    return protocol


def feature_key(model,seed):
    return f'{model}-seed{seed}' if model=='AnyPPG-random-frozen' else model


def features(dataset,root,device,models):
    folder=root/dataset; protocol=validate(folder)
    x=np.load(folder/'x125.npy'); stats=np.load(folder/'stats.npy')
    views=paired_views(x,125).reshape(-1,1000)
    stats=np.repeat(stats,2,axis=0)
    dest=folder/'features'; dest.mkdir(exist_ok=True)
    for method in models:
        for seed in SEEDS if method=='AnyPPG-random-frozen' else [42]:
            key=feature_key(method,seed); fp=dest/f'{key}.npy'; mp=dest/f'{key}.json'
            if mp.exists():
                info=read_json(mp)
                if info['protocol_sha256']!=sha256(folder/'protocol.json') or info['feature_sha256']!=sha256(fp):
                    raise ValueError('Feature cache identity mismatch')
                continue
            z,info=extract_features('AnyPPG' if method=='AnyPPG-random-frozen' else method,
                    views,125,stats=stats,random_init=method=='AnyPPG-random-frozen',device=device,batch_size=32,seed=seed)
            z=z.reshape(len(x),2,-1).mean(axis=1)
            np.save(fp,z)
            write_json(mp,dict(protocol_sha256=sha256(folder/'protocol.json'),feature_sha256=sha256(fp),
                               extractor_sha256=sha256(ROOT/'scripts/ppg_model_expansion_pilot.py'),evidence=info,aggregation='mean of two same-record embeddings'))
            print(json.dumps({'features':dataset,'model':key,'shape':list(z.shape)}),flush=True)


def projection(train,dim=16):
    train=np.asarray(train,dtype=np.float64)
    if len(train)<=dim or train.shape[1]<dim or not np.isfinite(train).all():
        raise ValueError('Insufficient finite training features')
    mean=train.mean(0); scale=train.std(0); scale[scale<1e-12]=1.
    z=(train-mean)/scale
    values,vectors=np.linalg.eigh(z.T@z/len(z))
    components=vectors[:,-dim:][:,::-1]
    component_scale=np.sqrt(np.maximum(values[-dim:][::-1],1e-12))
    return dict(mean=mean,scale=scale,components=components,component_scale=component_scale)


def project(x,state):
    return ((np.asarray(x,dtype=float)-state['mean'])/state['scale'])@state['components']/state['component_scale']


def score(y,p,subjects,decision=None):
    y=np.asarray(y); p=np.asarray(p); w=weights(subjects)
    if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('Invalid classification probability')
    d=p.argmax(1) if decision is None else decision
    result={'balanced_accuracy':float(balanced_accuracy_score(y,d,sample_weight=w)),
            'macro_f1':float(f1_score(y,d,average='macro',sample_weight=w,zero_division=0)),
            'coverage':1.}
    if p.shape[1]==2:
        result['auroc']=float(roc_auc_score(y,p[:,1],sample_weight=w))
        result['auprc']=float(average_precision_score(y,p[:,1],sample_weight=w))
    return result


def threshold(y,p,subjects):
    if p.shape[1]!=2: return .5
    return max((score(y,p,subjects,p[:,1]>=t)['balanced_accuracy'],-abs(t-.5),-t,t)
               for t in np.arange(1,20)/20)[-1]


def fit_linear(x,y,subjects,tr,va,primary):
    mean=x[tr].mean(0); scale=x[tr].std(0); scale[scale<1e-12]=1.
    z=(x-mean)/scale; best=None; candidates=[]
    for c in (.01,.1,1.,10.,100.):
        model=LogisticRegression(C=c,max_iter=3000,random_state=0).fit(z[tr],y[tr],sample_weight=weights(subjects[tr]))
        if np.max(model.n_iter_)>=3000: raise ValueError('Logistic classifier failed to converge')
        p=model.predict_proba(z[va]); metric=score(y[va],p,subjects[va])[primary]
        candidates.append(dict(C=c,validation=metric))
        if best is None or metric>best[0]: best=(metric,model)
    p=best[1].predict_proba(z)
    state=dict(mean=mean,scale=scale,coefficient=best[1].coef_,intercept=best[1].intercept_,classes=best[1].classes_)
    return p,dict(candidates=candidates,selected_C=best[1].C,threshold=threshold(y[va],p[va],subjects[va])),state


def linear_replay(x,state):
    logits=((x-state['mean'])/state['scale'])@state['coefficient'].T+state['intercept']
    if logits.shape[1]==1:
        from scipy.special import expit
        p=expit(logits[:,0]); return np.column_stack([1-p,p])
    return softmax(logits,axis=1)


def fit_neural(x,y,subjects,tr,va,primary,seed,cnn=False,device='cpu'):
    import torch
    from torch import nn
    from scripts.ppg_hr_cnn import SmallHRCNN,normalize_windows
    torch.set_num_threads(2); torch.manual_seed(seed)
    k=int(y.max())+1
    if cnn:
        model=SmallHRCNN(); model.head[-1]=nn.Linear(32,k)
        tx=normalize_windows(torch.tensor(paired_views(x,64),dtype=torch.float32).reshape(-1,1,512)).reshape(len(x),2,1,512).to(device)
    else:
        model=nn.Sequential(nn.Linear(x.shape[1],16),nn.ReLU(),nn.Linear(16,k))
        tx=torch.tensor(x,dtype=torch.float32,device=device)
    model=model.to(device); yy=torch.tensor(y,dtype=torch.long,device=device)
    wt=torch.tensor(weights(subjects[tr]),dtype=torch.float32,device=device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001 if cnn else .01,weight_decay=0 if cnn else .0001)
    rng=np.random.default_rng(seed); best=(-np.inf,None,0); history=[]
    def logits(ids):
        v=tx[ids]
        return model(v.reshape(-1,1,512)).reshape(len(ids),2,k).mean(1) if cnn else model(v)
    for epoch in range(1,(20 if cnn else 40)+1):
        model.train(); order=rng.permutation(len(tr)); batch=128 if cnn else len(tr)
        for start in range(0,len(tr),batch):
            local=order[start:start+batch]; idx=tr[local]
            optimizer.zero_grad(set_to_none=True)
            loss=(nn.functional.cross_entropy(logits(idx),yy[idx],reduction='none')*wt[local]).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite classifier loss')
            loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            vp=torch.cat([logits(va[i:i+256]).softmax(1) for i in range(0,len(va),256)]).cpu().numpy()
        metric=score(y[va],vp,subjects[va])[primary]; history.append(dict(epoch=epoch,validation=metric))
        if metric>best[0]: best=(metric,{n:v.detach().cpu().clone() for n,v in model.state_dict().items()},epoch)
        if cnn and epoch-best[2]>=5: break
    model.load_state_dict(best[1]); model.eval()
    with torch.inference_mode():
        p=torch.cat([logits(np.arange(i,min(i+256,len(x)))).softmax(1) for i in range(0,len(x),256)]).cpu().numpy()
    return p,dict(best_epoch=best[2],history=history,parameters=sum(v.numel() for v in model.parameters()),
                  threshold=threshold(y[va],p[va],subjects[va])),best[1]


def run(dataset,root,models,controls,device):
    folder=root/dataset; protocol=validate(folder); rows=read_csv(folder/'rows.csv')
    subjects=np.asarray([r['subject'] for r in rows]); y=np.asarray([int(r['target']) for r in rows]); primary=protocol['primary_metric']
    x64=np.load(folder/'x64.npy') if controls else None
    feature_cache={}; jobs=protocol['jobs']; methods=[(m,h) for m in models for h in HEADS]
    if controls: methods += [('HandFeatures','Linear'),('SmallCNN',''),('TrainPrior','')]
    for job in jobs:
        tr,va,te=indices(subjects,job)
        if set(y[tr])!=set(y) or set(y[va])!=set(y): raise ValueError('A class is absent from training or validation')
        transforms={}
        for model,head in methods:
            name=f'{model}/{head}' if head else model; dest=folder/'runs'/job['id']/name; dest.mkdir(parents=True,exist_ok=True)
            complete=dest/'complete.json'; predfile=dest/'predictions.npz'
            identity=dict(protocol_sha256=sha256(folder/'protocol.json'),job=job,method=name)
            if model in ENCODERS:
                key=feature_key(model,job['seed']); mp=folder/'features'/f'{key}.json'
                if not mp.exists(): raise ValueError(f'Missing feature cache {key}')
                info=read_json(mp); identity['features_sha256']=info['feature_sha256']
                if key not in feature_cache:
                    path=folder/'features'/f'{key}.npy'
                    if sha256(path)!=info['feature_sha256'] or info['protocol_sha256']!=identity['protocol_sha256']: raise ValueError('Feature mismatch')
                    feature_cache[key]=np.load(path)
            if complete.exists():
                prior=read_json(complete)
                if prior['identity']!=identity or prior['prediction_sha256']!=sha256(predfile): raise ValueError('Result identity mismatch')
                continue
            start=time.monotonic()
            if model in ENCODERS:
                if model not in transforms:
                    state=projection(feature_cache[key][tr]); transforms[model]=(project(feature_cache[key],state),state)
                x,proj=transforms[model]; np.savez(dest/'projection.npz',**proj)
            elif model=='HandFeatures': x=np.load(folder/'hand.npy')
            if head in ('PCA16-Linear','Linear'):
                p,details,state=fit_linear(x,y,subjects,tr,va,primary)
                if not np.allclose(p,linear_replay(x,state),atol=1e-12,rtol=1e-10): raise ValueError('Independent linear replay failed')
                np.savez(dest/'model.npz',**state)
            elif model=='TrainPrior':
                counts=np.bincount(y[tr],weights=weights(subjects[tr]),minlength=y.max()+1)
                p=np.tile(counts/counts.sum(),(len(y),1)); details=dict(threshold=threshold(y[va],p[va],subjects[va]))
            else:
                p,details,state=fit_neural(x64 if model=='SmallCNN' else x,y,subjects,tr,va,primary,job['seed'],cnn=model=='SmallCNN',device=device if model=='SmallCNN' else 'cpu')
                import torch
                torch.save(state,dest/'model.pt')
            decision=(p[:,1]>=details['threshold']).astype(int) if p.shape[1]==2 else p.argmax(1)
            np.savez(predfile,row=te,subject=subjects[te],reference=y[te],probability=p[te],decision=decision[te])
            write_json(complete,dict(identity=identity,selection=details,metrics=score(y[te],p[te],subjects[te],decision[te]),
                                    prediction_sha256=sha256(predfile),seconds=time.monotonic()-start,code_sha256=sha256(Path(__file__))))
            print(json.dumps(dict(dataset=dataset,job=job['id'],method=name,seconds=round(time.monotonic()-start,2))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','features','run'],required=True)
    parser.add_argument('--datasets',nargs='+',choices=['BUT-PPG','WESAD','ECSMP'],default=['BUT-PPG'])
    parser.add_argument('--models',nargs='*',choices=ENCODERS,default=list(ENCODERS))
    parser.add_argument('--controls',action='store_true'); parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        for dataset in args.datasets:
            if args.stage=='prepare': prepare(dataset,args.output)
            elif args.stage=='features': features(dataset,args.output,args.device,args.models)
            else: run(dataset,args.output,args.models,args.controls,args.device)


if __name__=='__main__': main()
