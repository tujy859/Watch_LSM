"""Recalculate onset F1 and PRV directly, without the production scorers.

Sorted two-pointer matching independently verifies maximum event-match counts;
raw successive differences verify PRV without deleting any predicted intervals.
This does not independently repeat the minimum-cost timing-error assignment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.ppg_embc21_data import load_records


def matches(reference,prediction,tolerance):
    """Maximum cardinality for ordered events with a symmetric tolerance."""
    i=j=count=0
    while i<len(reference) and j<len(prediction):
        if prediction[j]<reference[i]-tolerance-1e-12:j+=1
        elif prediction[j]>reference[i]+tolerance+1e-12:i+=1
        else:i+=1;j+=1;count+=1
    return count


def variability(events):
    pp=np.diff(events)*1000.
    if len(pp)<2:return {'mean_pp_ms':float(np.mean(pp)) if len(pp) else np.nan,'sdpp_ms':np.nan,'rmssd_ms':np.nan}
    return {'mean_pp_ms':float(np.mean(pp)),
            'sdpp_ms':float(np.sqrt(np.sum((pp-pp.mean())**2)/(len(pp)-1))),
            'rmssd_ms':float(np.sqrt(np.sum((pp[1:]-pp[:-1])**2)/(len(pp)-1)))}


def table(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def audit(root,shared):
    records={r['record_id']:r for r in load_records(ROOT/'datasets/embc21')}
    plan=json.loads((root/'protocol.json').read_text())
    comparisons=0;max_error=0.
    def equal(actual,expected,label):
        nonlocal comparisons,max_error
        a,b=float(actual),float(expected);comparisons+=1
        if np.isnan(a) and np.isnan(b):return
        if not np.isfinite([a,b]).all() or not np.isclose(a,b,rtol=1e-10,atol=1e-9):
            raise ValueError(f'{label}: {a} != {b}')
        max_error=max(max_error,abs(a-b))
    def key(row):return (row['record_id'],row['method'],int(row['seed']))
    interval_rows=defaultdict(list);window_rows=defaultdict(list)
    for row in table(shared/'intervals.csv'):interval_rows[key(row)].append(row)
    for row in table(shared/'prv_windows.csv'):window_rows[key(row)].append(row)
    totals=defaultdict(lambda:np.zeros(3));errors=defaultdict(dict)
    prediction_count=interval_count=window_count=0
    for job in plan['jobs']:
        for method in plan['models']:
            for record in records.values():
                if record['subject'] not in job['test']:continue
                prediction_count+=1;identity=(record['record_id'],method,job['seed'])
                meta=json.loads((root/'runs'/job['id']/method/f"{record['record_id']}.json").read_text())
                with np.load(ROOT/meta['path'],allow_pickle=False) as z:
                    prediction=z['event_times_s'];begin,end=float(z['start_s']),float(z['end_s'])
                reference=record['onset_times_s']
                person=(record['subject'],method,job['seed'])
                saved_intervals={int(r['interval_index']):r for r in interval_rows[identity]}
                saved_windows={(int(r['interval_index']),float(r['start_s']),int(r['duration_s'])):r for r in window_rows[identity]}
                expected_intervals=expected_windows=0
                for index,(a,b) in enumerate(record['annotated_intervals_s']):
                    lo,hi=max(a,begin),min(b,end)
                    if hi<=lo:continue
                    interval_count+=1;expected_intervals+=1
                    row=saved_intervals[index]
                    equal(row['start_s'],lo,'interval start');equal(row['end_s'],hi,'interval end')
                    r=reference[(reference>=lo)&(reference<hi)]
                    p=prediction[(prediction>=lo)&(prediction<hi)]
                    for ms in (25,50,100):
                        tp=matches(r,p,ms/1000.)
                        for name,value in (('tp',tp),('fp',len(p)-tp),('fn',len(r)-tp),
                                           ('f1',2*tp/(len(r)+len(p)) if len(r)+len(p) else np.nan)):
                            equal(row[f'{name}_{ms}ms'],value,'event '+name)
                        if ms==50:totals[person]+=np.array([tp,len(p)-tp,len(r)-tp])
                    for duration in (60,300):
                        for n in range(int(np.floor((hi-lo+1e-9)/duration))):
                            window_count+=1;expected_windows+=1;left=lo+n*duration;right=left+duration
                            rb=r[(r>=left)&(r<right)];pb=p[(p>=left)&(p<right)]
                            saved=saved_windows[(index,left,duration)]
                            rf,pf=variability(rb),variability(pb)
                            for name in rf:
                                equal(saved['reference_'+name],rf[name],'reference '+name)
                                equal(saved['prediction_'+name],pf[name],'prediction '+name)
                                err=abs(rf[name]-pf[name]) if len(rb)>=21 and len(pb)>=21 else np.nan
                                equal(saved[name+'_absolute_error'],err,'PRV '+name)
                                if len(rb)>=21:
                                    errors[(*person,name+'_mae',duration)][(record['record_id'],index,left)]=err
                equal(len(saved_intervals),expected_intervals,'interval count')
                equal(len(saved_windows),expected_windows,'window count')
    metrics={}
    for person,(tp,fp,fn) in totals.items():metrics[(*person,'f1_50ms',0)]=2*tp/(2*tp+fp+fn)
    for k,values in errors.items():
        v=np.array(list(values.values()));v=v[np.isfinite(v)]
        metrics[k]=float(v.mean()) if len(v) else np.nan
    audited_metrics={'f1_50ms','mean_pp_ms_mae','sdpp_ms_mae','rmssd_ms_mae'}
    per_seed=table(shared/'per_subject_seed.csv')
    for row in per_seed:
        if row['group']=='all' and row['metric'] in audited_metrics:
            k=(row['subject'],row['method'],int(row['seed']),row['metric'],int(row['duration_s']))
            equal(row['value'],metrics.get(k,np.nan),'person/seed '+row['metric'])
    averaged=defaultdict(list)
    for (subject,method,seed,metric,duration),value in metrics.items():averaged[subject,method,metric,duration].append(value)
    means={k:float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for k,v in averaged.items()}
    for row in table(shared/'per_subject.csv'):
        if row['group']=='all' and row['metric'] in audited_metrics:
            equal(row['value'],means.get((row['subject'],row['method'],row['metric'],int(row['duration_s'])),np.nan),'three-seed mean')
    pair_values=defaultdict(list)
    subjects=sorted({r['subject'] for r in records.values()})
    paired_rows=table(shared/'paired.csv')
    for row in paired_rows:
        method,control,metric,duration=row['method'],row['comparator'],row['metric'],int(row['duration_s'])
        diffs=[]
        for subject in subjects:
            seed_values=[]
            for seed in plan['seeds']:
                if not duration:
                    delta=metrics[subject,method,seed,metric,duration]-metrics[subject,control,seed,metric,duration]
                else:
                    x=errors[subject,method,seed,metric,duration];y=errors[subject,control,seed,metric,duration]
                    if x.keys()!=y.keys():raise ValueError('Paired reference windows differ')
                    v=[x[k]-y[k] for k in x if np.isfinite(x[k]) and np.isfinite(y[k])]
                    delta=float(np.mean(v)) if v else np.nan
                seed_values.append(delta)
            diffs.append(float(np.nanmean(seed_values)))
        delta=np.asarray(diffs);pair_values[method,control,metric,duration]=delta
        equal(row['difference'],delta.mean(),'paired mean')
        signs=np.array(list(itertools.product((-1.,1.),repeat=len(delta))))
        p=float(np.mean(np.abs(signs@delta)>=abs(delta.sum())-1e-12))
        equal(row['p_exact_signflip'],p,'exact sign flip')
        rng=np.random.default_rng(20261105)
        ci=np.quantile(np.mean(delta[rng.integers(0,len(delta),(10000,len(delta)))],axis=1),[.025,.975])
        equal(row['ci95_low'],ci[0],'paired bootstrap low');equal(row['ci95_high'],ci[1],'paired bootstrap high')
    ordered=sorted(paired_rows,key=lambda r:float(r['p_exact_signflip']));running=0.
    for rank,row in enumerate(ordered):
        running=max(running,min(1.,(len(ordered)-rank)*float(row['p_exact_signflip'])))
        equal(row['p_holm_8_comparisons'],running,'Holm adjustment')
    if prediction_count!=1890:raise ValueError('Incomplete prediction matrix')
    result={'status':'passed','scored_record_outputs':prediction_count,'intervals':interval_count,'prv_windows':window_count,
            'scalar_comparisons':comparisons,'maximum_absolute_difference':max_error,
            'scope':'All 25/50/100ms TP/FP/FN/F1, raw mean-PP/SDPP/RMSSD per 60s/300s window, all-person per-seed and three-seed means, 8 paired primary estimates/CIs/exact p/Holm. Does not reimplement minimum-cost timing-error or matched-PP assignment.',
            'script_sha256':sha(Path(__file__)),'protocol_sha256':sha(root/'protocol.json'),
            'tables_sha256':{p.name:sha(p) for p in shared.glob('*.csv')}}
    (shared/'independent_metric_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'.local/ppg-formal-expansion/onsets')
    parser.add_argument('--shared',type=Path,default=ROOT/'docs/results/ppg_formal_expansion/onsets')
    args=parser.parse_args();audit(args.root,args.shared)
