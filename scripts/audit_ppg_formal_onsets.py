"""Identity, probability and paired-person audit for the formal onset extension."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.ppg_formal_onsets import DEFAULT,OLD,METHODS,SEEDS,cache_name
from scripts.ppg_formal_hr import read_json,npz,sha256,write_json
from scripts.ppg_temporal_adapters import TemporalEncoder,make_head
from scripts.ppg_beat_models import SmallEventCNN
from scripts.run_ppg_onset_poc import full_frame_starts,resample,probability_peaks
from scripts.ppg_embc21_data import load_records
from scripts.summarize_ppg_onset_poc import (score_record,participant_metrics,paired_participant,finite_mean,
                                           independent_record_identity,verify_frame_cache,verify_data_and_code,
                                           selected_indices,verify_fit as verify_original_fit)
from scripts.audit_ppg_formal_hr import exact_sign_p,bootstrap,require


def write_csv(path,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r if not k.startswith('_')))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        w.writerows({k:r.get(k,'') for k in columns} for r in rows)


def verify_selection(fit,job,rows,thresholds):
    """Rebuild subject sampling and validation-only selection independently."""
    for label,subjects,limit in (('train',job['train'],256),('validation',job['validation'],128)):
        indices=selected_indices(rows,subjects,limit)
        require(fit[label+'_frame_indices']==indices and fit[label+'_frames']==len(indices),
                'Train/validation frame identities or counts changed')
    history=fit['history']
    require(0<len(history)<=20,'Invalid training history length')
    for epoch,item in enumerate(history,1):
        require(set(item['threshold_scores'])=={str(t) for t in thresholds},'Validation threshold grid changed')
        scores=[item['threshold_scores'][str(t)] for t in thresholds]
        winner=int(np.argmax(scores))
        require(np.isfinite(scores).all() and item['epoch']==epoch and
                item['threshold']==thresholds[winner] and item['validation_f1_50ms']==scores[winner],
                'Threshold not selected by the first validation maximum')
    selected=history[int(np.argmax([h['validation_f1_50ms'] for h in history]))]
    require(all(fit[k]==selected[k] for k in ('threshold','validation_f1_50ms')) and
            fit['best_epoch']==selected['epoch'],'Epoch not selected by the first validation maximum')


def reload_check(root,job,method,record,fit,meta):
    if method=='MSPTDfast':return None
    if fit['reused']:
        oldfit=fit['fit'];head=(ROOT/fit['original_fit']).parent/'head.pt'
    else:
        oldfit=fit;head=root/'runs'/job['id']/method/'head.pt'
    model=SmallEventCNN() if method=='SmallCNN' else make_head(method)
    model.load_state_dict(torch.load(head,map_location='cpu',weights_only=True),strict=True);model.eval()
    adapter=None if method=='SmallCNN' else TemporalEncoder(method,job['seed'],'cpu')
    starts=full_frame_starts(record);fs=64 if method=='SmallCNN' else 125
    wave=resample(-np.asarray(record['input_ppg'],np.float32),record['fs'],fs)
    out=[]
    with torch.inference_mode():
        for offset in range(0,len(starts),16):
            x=np.asarray([wave[int(s*fs):int((s+8)*fs)] for s in starts[offset:offset+16]])
            inp=torch.tensor(x).unsqueeze(1) if method=='SmallCNN' else adapter(x)
            out.append(model(inp).sigmoid()[:,64:448].numpy().reshape(-1))
    actual=np.concatenate(out)
    previous=npz(ROOT/meta['path'])
    difference=float(np.max(np.abs(actual-previous['probability'])))
    require(difference<2e-4,'Independent CPU probability reload differs')
    events=probability_peaks(actual,oldfit['threshold'],1.)
    require(np.array_equal(events,previous['event_times_s']),'CPU reload changes discrete event detections')
    return {'method':method,'seed':job['seed'],'record_id':record['record_id'],'probability_max_abs_difference':difference,'events_identical':True,
            'encoder':adapter.audit() if adapter else None}


def audit(root,shared):
    shared.mkdir(parents=True,exist_ok=True)
    records=load_records(ROOT/'datasets/embc21');byid={r['record_id']:r for r in records}
    plan=read_json(root/'protocol.json');old=read_json(OLD/'protocol.json');ph=sha256(root/'protocol.json')
    require(plan['records']=={k:independent_record_identity(v) for k,v in byid.items()},'Acquisition/label identities changed')
    source_files=verify_data_and_code(old,ROOT/'datasets/embc21')
    frame_rows,frame_hashes=verify_frame_cache(OLD,sha256(OLD/'protocol.json'),records)
    expected=[{**j,'seed':s,'id':f"EMBC21-fold{j['fold']}-seed{s}"} for j in old['jobs'] for s in SEEDS]
    require(plan['jobs']==expected and tuple(plan['models'])==METHODS,'Split or methods changed')
    feature_evidence=[]
    for method in ('PaPaGei-S','PaPaGei-P','AnyPPG-random-frozen'):
        for seed in (SEEDS if method=='AnyPPG-random-frozen' else (42,)):
            d=root/'features'/cache_name(method,seed);info=read_json(d/'complete.json')
            require(info['identity']['protocol_sha256']==ph,'Feature protocol identity mismatch')
            require(info['identity']['adapter_sha256']==sha256(ROOT/'scripts/ppg_temporal_adapters.py'),'Temporal adapter changed after caching')
            require(info['identity']['rows_sha256']==frame_hashes['rows.npz'] and info['identity']['input_sha256']==frame_hashes['x125.npy'],'Feature inputs differ')
            for name,h in info['sha256'].items():require(sha256(d/name)==h,'Temporal feature bytes changed')
            enc=info['encoder'];require(enc['state_before_sha256']==enc['state_after_sha256'],'Encoder state changed')
            for p,h in enc['checkpoint_sha256'].items():require(sha256(ROOT/p)==h,'Encoder checkpoint changed')
            feature_evidence.append(info)
    inference_states=[]
    for method in ('AnyPPG','PaPaGei-S','PaPaGei-P','AnyPPG-random-frozen'):
        for seed in (SEEDS if method=='AnyPPG-random-frozen' else (42,)):
            info=read_json(root/f'{cache_name(method,seed)}-inference-state.json')
            require(info['state_before_sha256']==info['state_after_sha256'] and info['encoder_trainable_parameters']==0,'Inference encoder was not frozen')
            for p,h in info['checkpoint_sha256'].items():require(sha256(ROOT/p)==h,'Inference checkpoint changed')
            if method!='AnyPPG':
                cached=next(r['encoder'] for r in feature_evidence if r['identity']['method']==method and (method!='AnyPPG-random-frozen' or r['identity']['seed']==seed))
                require(cached['state_before_sha256']==info['state_before_sha256'],'Training-cache and inference encoder states differ')
            inference_states.append(info)
    any_states=[r['state_before_sha256'] for r in inference_states if r['method'].startswith('AnyPPG')]
    require(len(set(any_states))==4,'Random AnyPPG seeds are not distinct from one another and the pretrained state')
    all_intervals,all_windows,all_coverage,people,pair_people=[],[],[],[],[]
    prediction_hashes={};fit_hashes={};reload=[];new_fits=reused_fits=0
    seed_intervals=defaultdict(list);seed_windows=defaultdict(list)
    for job in plan['jobs']:
        for method in METHODS:
            folder=root/'runs'/job['id']/method;fit=read_json(folder/'fit.json')
            fit_hash=sha256(folder/'fit.json');fit_hashes[str((folder/'fit.json').relative_to(ROOT))]=fit_hash
            require(fit['identity']=={'protocol_sha256':ph,'job':job,'method':method},'Fit identity differs')
            if fit['reused']:
                require(sha256(ROOT/fit['original_fit'])==fit['original_fit_sha256'],'Original fit changed')
                require(read_json(ROOT/fit['original_fit'])==fit['fit'],'Embedded original fit differs')
                actual=fit['fit'];reused_fits+=1
                original_job={**job,'seed':42,'id':f"EMBC21-fold{job['fold']}-seed42"}
                verify_original_fit((ROOT/fit['original_fit']).parent,actual,original_job,method,
                                    plan['old_protocol_sha256'],frame_rows,old)
            else:
                require(sha256(folder/'head.pt')==fit['head_sha256'],'Saved dense head changed')
                require(fit['runner_sha256']==sha256(ROOT/'scripts/ppg_formal_onsets.py'),'Runner changed after fitting')
                actual=fit;new_fits+=1
            if method!='MSPTDfast':
                verify_selection(actual,job,frame_rows,old['fit']['thresholds'])
                if not fit['reused']:require(actual['parameters']==plan['parameters'][method],'Task-head parameter count changed')
            test_records=[r for r in records if r['subject'] in job['test']]
            for number,record in enumerate(test_records):
                meta=read_json(folder/f"{record['record_id']}.json")
                expected_identity={'fit_sha256':fit_hash,'record_id':record['record_id'],'subject':record['subject'],
                                   'record_sha256':plan['records'][record['record_id']]['record_sha256']}
                require(meta['identity']==expected_identity and meta['reused']==fit['reused'],'Prediction reference identity changed')
                path=ROOT/meta['path'];h=sha256(path);require(h==meta['sha256'],'Prediction bytes changed')
                prediction_hashes[path.relative_to(ROOT).as_posix()]=h
                p=npz(path);start,end=float(p['start_s']),float(p['end_s'])
                require(start==1. and end==1.+6*len(full_frame_starts(record)),'Inference clock changed')
                if 'probability' in p:
                    require(p['probability'].shape==(int((end-start)*64),) and np.isfinite(p['probability']).all() and
                            float(p['probability_fs'])==64 and np.all((p['probability']>=0)&(p['probability']<=1)),
                            'Invalid dense probability or clock')
                    require(np.array_equal(probability_peaks(p['probability'],actual['threshold'],1.),p['event_times_s']),'Events differ from saved threshold/probability')
                intervals,windows,coverage=score_record(record,method,p['event_times_s'],start,end)
                for table in (intervals,windows):
                    for item in table:item['seed']=job['seed']
                coverage['seed']=job['seed']
                all_intervals.extend(intervals);all_windows.extend(windows);all_coverage.append(coverage)
                seed_intervals[job['seed']].extend(intervals);seed_windows[job['seed']].extend(windows)
                if job['fold']==0 and number==0:
                    check=reload_check(root,job,method,record,fit,meta)
                    if check:reload.append(check)
        print(f"Audited {job['id']}: six methods and {len(test_records)} test records",flush=True)
    for seed in SEEDS:
        events,windows=seed_intervals[seed],seed_windows[seed]
        for subject in sorted({r['subject'] for r in records}):
            for group in ('all','rest','squat','step'):
                for method in METHODS:
                    for row in participant_metrics(events,windows,subject,method,group):people.append({**row,'seed':seed})
            for method,control in [('AnyPPG','SmallCNN'),('PaPaGei-S','SmallCNN'),('PaPaGei-P','SmallCNN'),('AnyPPG','AnyPPG-random-frozen')]:
                def remap(table):
                    return [{**r,'method':'AnyPPG' if r['method']==method else 'control'} for r in table if r['method'] in (method,control)]
                for metric,duration in [('f1_50ms',0),('rmssd_ms_mae',60)]:
                    row=paired_participant(remap(events),remap(windows),subject,'control','all',metric,duration)
                    row.update(method=method,comparator=control,seed=seed);pair_people.append(row)
    pooled=defaultdict(list)
    for row in people:pooled[tuple(row[k] for k in ('group','subject','method','metric','duration_s'))].append(row)
    averaged=[]
    for key,rs in pooled.items():
        require(len(rs)==3,'Missing seed replicate')
        require(len({r['n_eligible'] for r in rs})==1,'Reference support changed across seeds')
        averaged.append({**dict(zip(('group','subject','method','metric','duration_s'),key)),'value':finite_mean([r['value'] for r in rs]),
                         'n_supported_mean':float(np.mean([r['n_supported'] for r in rs])),'n_eligible':rs[0]['n_eligible'],
                         'coverage':finite_mean([r['coverage'] for r in rs]),'seed_count':3})
    group_summary=defaultdict(list)
    for r in averaged:group_summary[tuple(r[k] for k in ('group','method','metric','duration_s'))].append(r)
    summary=[]
    for key,rs in group_summary.items():
        values=np.array([r['value'] for r in rs]);valid=values[np.isfinite(values)]
        ci=bootstrap(valid) if len(valid)>=2 else [float('nan')]*2
        summary.append({**dict(zip(('group','method','metric','duration_s'),key)),'mean':finite_mean(values),'ci95_low':ci[0],'ci95_high':ci[1],
                        'n_subjects':len(rs),'n_finite_subjects':len(valid),'participant_mean_coverage':finite_mean([r['coverage'] for r in rs]),
                        'n_eligible':sum(r['n_eligible'] for r in rs)})
    grouped_pairs=defaultdict(list)
    for r in pair_people:grouped_pairs[tuple(r[k] for k in ('method','comparator','metric','duration_s','subject'))].append(r)
    pair_averages=[];families=defaultdict(list)
    for key,rs in grouped_pairs.items():
        require(len(rs)==3,'Missing paired seed')
        row={**dict(zip(('method','comparator','metric','duration_s','subject'),key)),'difference':finite_mean([r['difference'] for r in rs]),
             'common_support_fraction':finite_mean([r['common_support_fraction'] for r in rs])}
        pair_averages.append(row);families[key[:-1]].append(row)
    paired=[]
    for key,rs in families.items():
        v=np.array([r['difference'] for r in rs]);v=v[np.isfinite(v)];ci=bootstrap(v)
        paired.append({**dict(zip(('method','comparator','metric','duration_s'),key)),'difference':float(v.mean()),'ci95_low':ci[0],'ci95_high':ci[1],
                       'n_subjects':len(v),'p_exact_signflip':exact_sign_p(v),'participant_mean_common_support':finite_mean([r['common_support_fraction'] for r in rs])})
    order=np.argsort([r['p_exact_signflip'] for r in paired]);running=0.
    for rank,index in enumerate(order):
        running=max(running,min(1.,(len(paired)-rank)*paired[index]['p_exact_signflip']));paired[index]['p_holm_8_comparisons']=running
    for name,table in [('intervals',all_intervals),('prv_windows',all_windows),('coverage',all_coverage),('per_subject_seed',people),('per_subject',averaged),
                       ('summary',summary),('paired_per_subject_seed',pair_people),('paired_per_subject',pair_averages),('paired',paired)]:write_csv(shared/f'{name}.csv',table)
    require(len(all_coverage)==1890,'Incomplete six-method three-seed 105-record matrix')
    require(new_fits==91 and reused_fits==35 and len(fit_hashes)==126,'Incomplete fitted/reference matrix')
    require(len(prediction_hashes)==1680,'Unexpected unique prediction count after traditional reuse')
    write_json(shared/'protocol.json',plan)
    write_json(shared/'feature_evidence.json',feature_evidence)
    result={'status':'passed','scored_record_outputs':len(all_coverage),'unique_prediction_files':len(prediction_hashes),'new_neural_fits':new_fits,
            'reused_fit_references':reused_fits,'source_files_verified':len(source_files),'probability_reloads':reload,'feature_caches':len(feature_evidence),
            'prediction_sha256':prediction_hashes,'fit_sha256':fit_hashes,'audit_script_sha256':sha256(Path(__file__)),
            'inference_encoder_states':inference_states,
            'execution':'FP32 deterministic cuDNN kernels; independent artifact heads may run concurrently on the same GPU. Wall times are not model-efficiency evidence.',
            'inference':'Average three seeds within each person; 10000 paired-person bootstrap CI, exact sign-flip p, Holm over 8 primary tests. Only seven people; single expert and mostly resting PRV support. With seven people the smallest two-sided exact sign-flip p is 0.015625; the smallest first Holm-adjusted value for eight comparisons is 0.125, so this design cannot yield adjusted p<0.05.'}
    write_json(shared/'audit.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--shared',type=Path,default=ROOT/'docs/results/ppg_formal_expansion/onsets')
    args=p.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):r=audit(args.root,args.shared)
    print({k:v for k,v in r.items() if k not in ('prediction_sha256','fit_sha256','probability_reloads')})


if __name__=='__main__':main()
