"""Rebuild formal HR identities, predictions and patient-level summaries.

Audits every registered result, independently reloads all heads and performs
three real-data direct sklearn refits to check the sufficient-statistic route.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.ppg_formal_hr import (DEFAULT,OLD,DATASETS,METHODS,HEADS,MODELS,SEEDS,paths,
                                  read_json,npz,sha256,write_json,load_feature,project)


def require(condition,message):
    if not condition:
        raise ValueError(message)


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def weighted_scaler(x,subjects):
    people,inv,counts=np.unique(subjects,return_inverse=True,return_counts=True)
    w=len(subjects)/(len(people)*counts[inv])
    return StandardScaler().fit(x,sample_weight=w),w


def macro(y,p,subjects):
    return float(np.mean([np.mean(np.abs(y[subjects==s]-p[subjects==s])) for s in np.unique(subjects)]))


def refit_checks(root,plan,rows):
    checks=[]
    choices=[('AnyPPG','Dalia-fold0-seed42-labelsfull'),('PaPaGei-P','WildPPG-fold1-seed43-labels6'),('AnyPPG-random-frozen','Dalia-to-WildPPG-seed44')]
    for method,jid in choices:
        job=next(j for j in plan['jobs'] if j['id']==jid)
        s,t=rows[job['source']],rows[job['target']]
        train,val=(np.flatnonzero(np.isin(s['subject'],job[k])) for k in ('train','validation'))
        test=np.flatnonzero(np.isin(t['subject'],job['test']))
        x,_=load_feature(root,job['source'],method,job['seed'])
        tx,_=load_feature(root,job['target'],method,job['seed'])
        train_x=np.asarray(x[train],dtype=float)
        sc=StandardScaler().fit(train_x)
        pc=PCA(16,svd_solver='full').fit(sc.transform(train_x))
        out=StandardScaler().fit(pc.transform(sc.transform(train_x)))
        z=[out.transform(pc.transform(sc.transform(a))) for a in (train_x,x[val],tx[test])]
        folder=root/'runs'/jid/method
        stored=npz(folder/'projection.npz')
        expected=project(train_x,stored)
        signs=np.sign(np.sum(expected*z[0],axis=0))
        projection_error=float(np.max(np.abs(expected-z[0]*signs)))
        require(projection_error<2e-5,'PCA sufficient-statistics disagreement with direct sklearn')
        results={}
        for head,arrays in [('Native-Ridge',(train_x,np.asarray(x[val],float),np.asarray(tx[test],float))),('PCA16-Ridge',z)]:
            a,b,c=arrays
            scaler,w=weighted_scaler(a,s['subject'][train])
            candidates=[]
            for alpha in plan['ridge']['alphas']:
                model=Ridge(alpha=alpha,solver='cholesky').fit(scaler.transform(a),s['hr'][train],sample_weight=w)
                score=macro(s['hr'][val],model.predict(scaler.transform(b)),s['subject'][val])
                candidates.append((score,alpha,model))
            score,alpha,model=min(candidates,key=lambda v:v[0])
            info=read_json(folder/f'{head}.json')
            pred=npz(ROOT/info['prediction_path'])['prediction']
            err=float(np.max(np.abs(model.predict(scaler.transform(c))-pred)))
            require(alpha==info['fit']['alpha'],'Independent alpha choice differs')
            require(err<2e-4,'Independent Ridge prediction differs')
            results[head]={'selected_alpha':alpha,'max_test_prediction_difference_bpm':err,'validation_mae_bpm':score}
        checks.append({'method':method,'job':jid,'training_windows':len(train),'pca_max_coordinate_difference_after_sign_alignment':projection_error,'independent_ridge_refits':results})
    return checks


def audit(root,shared):
    import torch
    torch.set_num_threads(2)
    shared.mkdir(parents=True,exist_ok=True)
    plan=read_json(root/'protocol.json')
    ph=sha256(root/'protocol.json')
    rows={d:npz(OLD/'data'/d/'rows.npz') for d in DATASETS}
    oldplan=read_json(OLD/'protocol.json')
    require(plan['jobs']==oldplan['jobs'] and len(plan['jobs'])==93,'Original 93 splits changed')
    files={}
    def check_hash(path,expected=None):
        key=path.relative_to(ROOT).as_posix()
        if key not in files:
            files[key]=sha256(path)
        require(expected is None or files[key]==expected,f'Content hash mismatch: {key}')
        return files[key]
    for d in DATASETS:
        check_hash(OLD/'data'/d/'rows.npz',plan['data_rows_sha256'][d])
        check_hash(OLD/'data'/d/'manifest.json',plan['data_manifest_sha256'][d])
    feature_infos=[]
    feature_arrays={}
    for method in METHODS:
        for seed in (SEEDS if method=='AnyPPG-random-frozen' else (42,)):
            for d in DATASETS:
                _,metadata=paths(root,d,method,seed)
                info=read_json(metadata)
                path=ROOT/info['path']
                check_hash(path,info['features_sha256'])
                require(info['rows_sha256']==plan['data_rows_sha256'][d],'Feature row identity changed')
                x=np.load(path,mmap_mode='r')
                require(list(x.shape)==info['shape'] and len(x)==len(rows[d]['hr']) and np.isfinite(x).all(),'Invalid features')
                if not info['reused']:
                    require(info['encoder_state_before']==info['encoder_state_after'] and info['encoder_trainable_parameters']==0,'Encoder state changed')
                    for filename,h in info['identity']['data']['files_sha256'].items():
                        check_hash(OLD/'data'/d/filename,h)
                    for filename,h in info['identity']['weights'].items():
                        check_hash(ROOT/filename,h)
                else:
                    oldmeta=OLD/'features'/d/f'{method}.json'
                    check_hash(oldmeta,info['source_metadata_sha256'])
                    inherited=read_json(oldmeta)['identity']
                    for filename,h in inherited['data']['files_sha256'].items():
                        check_hash(OLD/'data'/d/filename,h)
                    for filename,h in inherited['weight_files_sha256'].items():
                        check_hash(ROOT/filename,h)
                feature_infos.append(info)
                feature_arrays[(d,method,seed)]=x
    random_states={x['identity']['seed']:x['encoder_state_before'] for x in feature_infos if x['model']=='AnyPPG-random-frozen'}
    require(len(set(random_states.values()))==3,'Random encoders are not independent seeds')
    for seed in SEEDS:
        state=[x['encoder_state_before'] for x in feature_infos if x['model']=='AnyPPG-random-frozen' and x['identity']['seed']==seed]
        require(len(state)==2 and state[0]==state[1],'Random encoder differs between domains')
    from scripts.ppg_model_expansion_pilot import _upstream, _state_hash
    configs,_=_upstream()
    from ppg_bench.models import get_port
    pretrained_any=get_port(configs['AnyPPG']).load_encoder().eval().requires_grad_(False)
    pretrained_state=_state_hash(pretrained_any)
    pretrained_parameters=sum(p.numel() for p in pretrained_any.parameters())
    require(pretrained_state not in random_states.values(),'Random control reused pretrained parameters')
    require(all(x['encoder_parameters']==pretrained_parameters for x in feature_infos if x['model']=='AnyPPG-random-frozen'),'Random architecture parameter count differs')
    del pretrained_any
    participant=[]
    results=[]
    max_reload=0.
    def record(job,method,head,indices,pred,reference,subjects,coverage=None):
        for person in np.unique(subjects):
            m=subjects==person
            valid=np.isfinite(pred[m])
            require(valid.any(),'No outputs for a participant')
            participant.append({'scenario':job['scenario'],'source':job['source'],'target':job['target'],'budget':job['budget'],
                                'fold':job['fold'],'seed':job['seed'],'method':method,'head':head,'subject':person,
                                'mae_bpm':float(np.mean(np.abs(pred[m][valid]-reference[m][valid]))),
                                'bias_bpm':float(np.mean(pred[m][valid]-reference[m][valid])),
                                'coverage':float(valid.mean()),'windows':int(m.sum())})
    for job in plan['jobs']:
        source,target=rows[job['source']],rows[job['target']]
        train,val=(np.flatnonzero(np.isin(source['subject'],job[k])) for k in ('train','validation'))
        test=np.flatnonzero(np.isin(target['subject'],job['test']))
        require(not set(job['train']) & set(job['validation']),'Train/validation overlap')
        if job['source']==job['target']:
            require(not (set(job['train'])|set(job['validation'])) & set(job['test']),'Test leakage')
        for method in METHODS:
            seed=job['seed'] if method=='AnyPPG-random-frozen' else 42
            x=feature_arrays[(job['target'],method,seed)]
            folder=root/'runs'/job['id']/method
            projection=npz(folder/'projection.npz')
            z=project(x[test],projection)
            for head in HEADS:
                path=folder/f'{head}.json'
                info=read_json(path)
                require(info['identity']['job']==job and info['identity']['method']==method and info['identity']['protocol_sha256']==ph,'Result identity mismatch')
                ppath=ROOT/info['prediction_path']
                check_hash(ppath,info['prediction_sha256'])
                data=npz(ppath)
                for key,expected in [('row',test),('subject',target['subject'][test]),('start_s',target['start_s'][test]),('reference',target['hr'][test])]:
                    require(np.array_equal(data[key],expected),f'Prediction {key} mismatch')
                require(np.isfinite(data['prediction']).all(),'Invalid prediction')
                score=macro(data['reference'],data['prediction'],data['subject'])
                require(abs(score-info['subject_mae_bpm'])<1e-10,'Reported MAE mismatch')
                if head=='PCA16-MLP':
                    model=torch.nn.Sequential(torch.nn.Linear(16,16),torch.nn.ReLU(),torch.nn.Linear(16,1))
                    model.load_state_dict(torch.load(folder/f'{head}.pt',map_location='cpu',weights_only=True),strict=True)
                    model.eval()
                    f=info['fit']
                    require(abs(f['target_center']-float(np.mean(source['hr'][train])))<1e-10,'Target mean not fitted on training')
                    require(abs(f['target_scale']-max(float(np.std(source['hr'][train])),1.))<1e-10,'Target scale not fitted on training')
                    best=min(f['history'],key=lambda v:v['validation_mae_bpm'])
                    require(best['epoch']==f['selected_epoch'] and f['parameters']==289,'MLP selection or size differs')
                    with torch.inference_mode():
                        pred=model(torch.tensor(z,dtype=torch.float32)).squeeze(-1).numpy()*f['target_scale']+f['target_center']
                else:
                    coefpath=OLD/'runs'/job['id']/method/'ridge.npz' if info['reused'] else folder/f'{head}.npz'
                    coef=npz(coefpath)
                    pred=(np.asarray(x[test],float) if head=='Native-Ridge' else z)@coef['coef']+coef['intercept']
                err=float(np.max(np.abs(pred-data['prediction'])))
                require(err<5e-5,'Saved head reload prediction differs')
                max_reload=max(max_reload,err)
                record(job,method,head,test,data['prediction'],data['reference'],data['subject'])
                results.append({'job':job['id'],'method':method,'head':head,'mae_bpm':score,'reused':info['reused'],'maximum_reload_difference_bpm':err})
        for method in ('SmallCNN','TrainMean'):
            path=OLD/'runs'/job['id']/method
            info=read_json(path/'complete.json')
            require(info['identity']['job']==job,'Old control split mismatch')
            check_hash(path/'predictions.npz',info['predictions_sha256'])
            p=npz(path/'predictions.npz')
            require(np.array_equal(p['row'],test) and np.array_equal(p['reference'],target['hr'][test]) and np.array_equal(p['subject'],target['subject'][test]),'Old control row identity mismatch')
            record(job,method,'Task-control',test,p['prediction'],p['reference'],p['subject'])
        traditional=OLD/'traditional'/job['target']
        check_hash(traditional/'predictions.npz',read_json(traditional/'complete.json')['sha256'])
        p=npz(traditional/'predictions.npz')
        for method,pred in p.items():
            record(job,method,'Task-control',test,pred[test],target['hr'][test],target['subject'][test])
    require(len(results)==1953,'Incomplete formal result matrix')
    direct=refit_checks(root,plan,rows)
    csv_write(shared/'per_run.csv',results)
    csv_write(shared/'per_subject_seed.csv',participant)
    write_json(shared/'protocol.json',plan)
    write_json(shared/'feature_evidence.json',feature_infos)
    audit_report={'status':'passed','registered_outputs':1953,'audited_outputs':len(results),'reused_native_ridge_outputs':sum(r['reused'] for r in results),
                  'feature_caches':len(feature_infos),'all_input_and_prediction_identities_verified':True,'maximum_head_reload_difference_bpm':max_reload,
                  'random_encoder_state_hashes':random_states,'pretrained_any_state_sha256':pretrained_state,'matched_any_parameters':pretrained_parameters,
                  'direct_sklearn_checks':direct,'artifact_hashes':files,
                  'audit_script_sha256':sha256(Path(__file__)),'runner_sha256_at_audit':sha256(ROOT/'scripts/ppg_formal_hr.py')}
    write_json(shared/'audit.json',audit_report)
    summarize(participant,shared)
    return audit_report


def bootstrap(values):
    x=np.asarray(values,float)
    rng=np.random.default_rng(20261105)
    boot=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(1)
    return np.quantile(boot,[.025,.975]).tolist()


def exact_sign_p(values):
    x=np.asarray(values,float)
    require(len(x)<=20,'Exact sign test limited to 20 participants')
    signs=((np.arange(2**len(x),dtype=np.uint32)[:,None] >> np.arange(len(x))) & 1)*2.-1.
    return float(np.mean(np.abs(signs@x)>=abs(x.sum())-1e-12))


def summarize(participant,shared):
    pooled=defaultdict(list)
    for r in participant:
        key=tuple(r[k] for k in ('scenario','source','target','budget','method','head','subject'))
        pooled[key].append(r)
    subject=[]
    grouped=defaultdict(dict)
    for key,rs in pooled.items():
        require(len(rs)==3 and {r['seed'] for r in rs}==set(SEEDS),'Incomplete seed replicates')
        value=float(np.mean([r['mae_bpm'] for r in rs]))
        item=dict(zip(('scenario','source','target','budget','method','head','subject'),key))
        item.update(mae_bpm=value,coverage=float(np.mean([r['coverage'] for r in rs])))
        subject.append(item)
        grouped[key[:-1]][key[-1]]=value
    summary=[]
    for key,people in grouped.items():
        v=list(people.values());lo,hi=bootstrap(v)
        row=dict(zip(('scenario','source','target','budget','method','head'),key))
        coverage=[r['coverage'] for r in subject if tuple(r[k] for k in ('scenario','source','target','budget','method','head'))==key]
        row.update(mae_bpm=float(np.mean(v)),ci_low_bpm=lo,ci_high_bpm=hi,people=len(v),coverage=float(np.mean(coverage)))
        summary.append(row)
    paired=[]
    scenarios=sorted(set(key[:4] for key in grouped))
    for scenario in scenarios:
        family=[]
        for method,control,chead in [*((m,'SmallCNN','Task-control') for m in MODELS),('AnyPPG','AnyPPG-random-frozen','PCA16-MLP')]:
            a=grouped[(*scenario,method,'PCA16-MLP')]
            b=grouped[(*scenario,control,chead)]
            require(a.keys()==b.keys(),'Paired comparison participant mismatch')
            delta=np.array([a[s]-b[s] for s in sorted(a)])
            lo,hi=bootstrap(delta)
            row=dict(zip(('scenario','source','target','budget'),scenario))
            row.update(method=method,head='PCA16-MLP',control=control,control_head=chead,
                       delta_mae_bpm=float(delta.mean()),ci_low_bpm=lo,ci_high_bpm=hi,people=len(delta),p_exact_signflip=exact_sign_p(delta))
            family.append(row)
        order=np.argsort([r['p_exact_signflip'] for r in family]);running=0.
        for rank,index in enumerate(order):
            running=max(running,min(1.,(len(family)-rank)*family[index]['p_exact_signflip']))
            family[index]['p_holm_within_scenario']=running
        paired.extend(family)
    csv_write(shared/'per_subject.csv',subject)
    csv_write(shared/'summary.csv',summary)
    csv_write(shared/'paired_primary.csv',paired)
    write_json(shared/'inference_notes.json',{'unit':'person; average seed-specific MAEs first',
               'ci':'10000 participant bootstrap replicates, unadjusted 95% intervals',
               'p':'two-sided exact paired sign-flip; requires exchangeability/symmetry under null',
               'multiplicity':'Holm correction over 7 predeclared primary comparisons separately in each of 7 scenarios; not all-task/global correction',
               'limitation':'Only 15/16 participants; same fixed outer splits. Neither confidence interval nor p value establishes broad population or device superiority. Traditional errors are conditional on finite outputs; coverage reported.'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--shared',type=Path,default=ROOT/'docs/results/ppg_formal_expansion/hr')
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        report=audit(args.root,args.shared)
    print(json.dumps({k:v for k,v in report.items() if k not in ('artifact_hashes','direct_sklearn_checks')}))


if __name__=='__main__':
    main()
