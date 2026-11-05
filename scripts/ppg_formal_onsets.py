"""Three-seed, fixed-LOSO frozen-model extension of manual wrist onsets.

Original clocks, artifact/missing-sample exclusions, core intervals, sampling,
loss and validation selection are unchanged. Only models/seeds are extended.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'scripts'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from scripts.ppg_temporal_adapters import TemporalEncoder,make_head,PAPAGEI_TAPS,PAPAGEI_CHANNELS
from scripts.ppg_beat_models import SmallEventCNN,THRESHOLDS
from scripts.ppg_embc21_data import load_records
from scripts.run_ppg_onset_poc import (record_digest,validate_records,sample_person_indices,
                                      full_frame_starts,frame_f1,probability_peaks,resample)
from scripts.ppg_formal_hr import read_json,npz,sha256,write_json

OLD=ROOT/'.local/ppg-phase2/onsets'
DEFAULT=ROOT/'.local/ppg-formal-expansion/onsets'
METHODS=('AnyPPG','PaPaGei-S','PaPaGei-P','AnyPPG-random-frozen','SmallCNN','MSPTDfast')
SEEDS=(42,43,44)


def register(root,records):
    validate_records(records)
    old=read_json(OLD/'protocol.json')
    if any(record_digest(r)!=old['records'][r['record_id']]['record_sha256'] for r in records):
        raise ValueError('Original acquisition identities changed')
    jobs=[]
    for oldjob in old['jobs']:
        for seed in SEEDS:
            jobs.append({**oldjob,'seed':seed,'id':f"EMBC21-fold{oldjob['fold']}-seed{seed}"})
    plan={'version':'formal-manual-onset-extension-v1','models':list(METHODS),'seeds':list(SEEDS),'jobs':jobs,
          'old_protocol_sha256':sha256(OLD/'protocol.json'),'records':old['records'],
          'data_manifest_sha256':old['data_manifest_sha256'],'frames':old['frames'],'sampling':old['sampling'],
          'annotation_policy':old['annotation_policy'],'missing_samples':old['missing_samples'],'input_polarity':old['input_polarity'],
          'fit':{**old['fit'],'seed':'each of 42/43/44; fixed train/validation/test people independent of initialization seed'},
          'temporal':{'AnyPPG':'same six stage_list outputs and original NumPy preprocessing',
                      'PaPaGei':'native official preprocessing; residual blocks '+str(PAPAGEI_TAPS),
                      'PaPaGei_channels':list(PAPAGEI_CHANNELS),'output_clock_hz':64},
          'parameters':{m:sum(p.numel() for p in (SmallEventCNN() if m=='SmallCNN' else make_head(m)).parameters()) for m in METHODS if m!='MSPTDfast'},
          'reuse':'Original seed42 AnyPPG/SmallCNN fits and predictions, all fixed traditional predictions, original frame identities and AnyPPG stage caches; hash verified.',
          'primary_comparisons':'F1 at 50ms and 60-second RMSSD error: AnyPPG, PaPaGei-S/P versus SmallCNN and AnyPPG versus random AnyPPG; Holm over 8 planned tests, paired person-level exact sign-flip after averaging seeds.',
          'limitations':'7 people, one expert, predominantly resting PRV windows. PP/PRV algorithm agreement, not ECG HRV equivalence. Encoders/architectures/native preprocessing differ; random Any comparison isolates pretraining within one architecture.',
          'encoder_optimization':False}
    root.mkdir(parents=True,exist_ok=True)
    path=root/'protocol.json'
    if path.exists() and read_json(path)!=plan:raise ValueError('Registered onset protocol changed')
    if not path.exists():write_json(path,plan)
    oldcache=read_json(OLD/'frames/complete.json')
    if oldcache['protocol_sha256']!=plan['old_protocol_sha256']:raise ValueError('Original frame protocol differs')
    for name,h in oldcache['sha256'].items():
        if sha256(OLD/'frames'/name)!=h:raise ValueError('Original frame cache changed')
    return plan


def cache_name(method,seed):
    return f'{method}-seed{seed}' if method=='AnyPPG-random-frozen' else method


def prepare(root,method,seed,device):
    if method=='AnyPPG':return
    directory=root/'features'/cache_name(method,seed)
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'complete.json'
    identity={'protocol_sha256':sha256(root/'protocol.json'),'method':method,'seed':seed if method=='AnyPPG-random-frozen' else None,
              'rows_sha256':sha256(OLD/'frames/rows.npz'),'input_sha256':sha256(OLD/'frames/x125.npy'),
              'adapter_sha256':sha256(ROOT/'scripts/ppg_temporal_adapters.py')}
    if path.exists():
        info=read_json(path)
        if info['identity']!=identity or any(sha256(directory/name)!=h for name,h in info['sha256'].items()):raise ValueError('Temporal feature cache changed')
        return
    encoder=TemporalEncoder(method,seed,device)
    x=np.load(OLD/'frames/x125.npy',mmap_mode='r')
    outputs=None
    for start in range(0,len(x),64):
        maps=encoder(x[start:start+64])
        if outputs is None:
            outputs=[np.lib.format.open_memmap(directory/f'stage{i}.npy',mode='w+',dtype='float32',shape=(len(x),*v.shape[1:])) for i,v in enumerate(maps)]
        for out,v in zip(outputs,maps):out[start:start+len(v)]=v.cpu().numpy()
    shapes=[]
    for out in outputs:
        if not np.isfinite(out).all():raise ValueError('Nonfinite temporal cache')
        shapes.append(list(out.shape));out.flush()
    info=encoder.audit()
    del outputs,encoder
    gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    write_json(path,{'identity':identity,'frames':len(x),'shapes':shapes,'encoder':info,'sha256':{p.name:sha256(p) for p in directory.glob('stage*.npy')}})
    print(json.dumps({'prepared':cache_name(method,seed),'frames':len(x),'device':device}),flush=True)


def features(root,method,seed):
    if method=='SmallCNN':return [np.load(OLD/'frames/x64.npy',mmap_mode='r')]
    directory=OLD/'frames' if method=='AnyPPG' else root/'features'/cache_name(method,seed)
    return [np.load(directory/f'stage{i}.npy',mmap_mode='r') for i in range(6)]


def fit(root,job,method,records,device):
    directory=root/'runs'/job['id']/method
    directory.mkdir(parents=True,exist_ok=True)
    identity={'protocol_sha256':sha256(root/'protocol.json'),'job':job,'method':method}
    fitpath=directory/'fit.json'
    if fitpath.exists():
        info=read_json(fitpath)
        if info['identity']!=identity:raise ValueError('Head identity changed')
        if info.get('reused'):return None,info
        if sha256(directory/'head.pt')!=info['head_sha256']:raise ValueError('Head changed')
        model=(SmallEventCNN() if method=='SmallCNN' else make_head(method)).to(device)
        model.load_state_dict(torch.load(directory/'head.pt',map_location=device,weights_only=True))
        return model.eval(),info
    if method=='MSPTDfast' or (job['seed']==42 and method in ('AnyPPG','SmallCNN')):
        olddir=OLD/'runs'/f"EMBC21-fold{job['fold']}-seed42"/method
        previous=read_json(olddir/'fit.json')
        if method!='MSPTDfast' and sha256(olddir/'head.pt')!=previous['head_sha256']:raise ValueError('Old fitted head changed')
        oldjob={**job,'seed':42,'id':f"EMBC21-fold{job['fold']}-seed42"}
        if previous['identity']['job']!=oldjob:raise ValueError('Original fit subject split changed')
        info={'identity':identity,'reused':True,'original_fit':str((olddir/'fit.json').relative_to(ROOT)),'original_fit_sha256':sha256(olddir/'fit.json'),'fit':previous}
        write_json(fitpath,info)
        return None,info
    torch.manual_seed(job['seed'])
    model=(SmallEventCNN() if method=='SmallCNN' else make_head(method)).to(device)
    rows=npz(OLD/'frames/rows.npz')
    train=sample_person_indices(rows,job['train'],256)
    validation=sample_person_indices(rows,job['validation'],128)
    values=features(root,method,job['seed'])
    def batch(indices):
        tensors=[torch.tensor(np.asarray(v[indices]),dtype=torch.float32,device=device) for v in values]
        return tensors[0].unsqueeze(1) if method=='SmallCNN' else tensors
    counts={s:int(np.count_nonzero(rows['subject'][train]==s)) for s in job['train']}
    w=np.asarray([len(train)/(len(counts)*counts[rows['subject'][i]]) for i in train])
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    best,state,threshold,best_epoch,stale=-1.,None,None,None,0
    history=[];started=time.monotonic()
    for epoch in range(1,21):
        model.train();order=np.random.default_rng(job['seed']+epoch).permutation(len(train));total=0.
        for offset in range(0,len(train),64):
            pos=order[offset:offset+64];indices=train[pos]
            target=torch.tensor(rows['target'][indices,64:448],device=device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(batch(indices))[:,64:448]
            loss=F.binary_cross_entropy_with_logits(logits,target,pos_weight=torch.tensor(5.,device=device),reduction='none').mean(1)
            loss=(loss*torch.tensor(w[pos],dtype=torch.float32,device=device)).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite dense loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*len(indices)
        model.eval();prob=[]
        with torch.no_grad():
            for offset in range(0,len(validation),64):prob.extend(model(batch(validation[offset:offset+64])).sigmoid().cpu().numpy())
        scores=[frame_f1(prob,rows,validation,records,t) for t in THRESHOLDS]
        chosen=int(np.argmax(scores))
        history.append({'epoch':epoch,'loss':total/len(train),'validation_f1_50ms':scores[chosen],'threshold':THRESHOLDS[chosen],'threshold_scores':dict(zip(map(str,THRESHOLDS),scores))})
        if scores[chosen]>best:
            best,threshold,best_epoch,stale=scores[chosen],THRESHOLDS[chosen],epoch,0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=5:break
    torch.save(state,directory/'head.pt');model.load_state_dict(state)
    info={'identity':identity,'reused':False,'best_epoch':best_epoch,'threshold':threshold,'validation_f1_50ms':best,'history':history,
          'head_sha256':sha256(directory/'head.pt'),'parameters':sum(p.numel() for p in model.parameters()),
          'train_frames':len(train),'validation_frames':len(validation),'train_frame_indices':train.tolist(),'validation_frame_indices':validation.tolist(),
          'elapsed_seconds':time.monotonic()-started,'runner_sha256':sha256(Path(__file__))}
    write_json(fitpath,info)
    return model.eval(),info


@torch.no_grad()
def infer(root,job,method,record,model,fit,encoder,device):
    folder=root/'runs'/job['id']/method
    path,meta=folder/f"{record['record_id']}.npz",folder/f"{record['record_id']}.json"
    identity={'fit_sha256':sha256(folder/'fit.json'),'record_id':record['record_id'],'subject':record['subject'],'record_sha256':record_digest(record)}
    if meta.exists():
        old=read_json(meta)
        if old['identity']!=identity or sha256(ROOT/old['path'])!=old['sha256']:raise ValueError('Prediction changed')
        return
    if fit['reused']:
        olddir=OLD/'runs'/f"EMBC21-fold{job['fold']}-seed42"/method
        oldmeta=read_json(olddir/meta.name)
        oldpath=olddir/path.name
        if oldmeta['identity']['record_sha256']!=identity['record_sha256'] or sha256(oldpath)!=oldmeta['sha256']:raise ValueError('Old record prediction changed')
        write_json(meta,{'identity':identity,'reused':True,'path':oldpath.relative_to(ROOT).as_posix(),'sha256':oldmeta['sha256']})
        return
    starts=full_frame_starts(record);end=1.+6*len(starts)
    fs=64 if method=='SmallCNN' else 125
    volume=resample(-np.asarray(record['input_ppg'],np.float32),record['fs'],fs)
    output=[]
    for offset in range(0,len(starts),64):
        frames=np.asarray([volume[int(s*fs):int((s+8)*fs)] for s in starts[offset:offset+64]])
        inp=torch.from_numpy(frames).unsqueeze(1).to(device) if method=='SmallCNN' else encoder(frames)
        output.append(model(inp).sigmoid()[:,64:448].cpu().numpy().reshape(-1))
    probability=np.concatenate(output).astype(np.float32)
    if not np.isfinite(probability).all():raise ValueError('Nonfinite probability')
    events=probability_peaks(probability,fit['threshold'],1.)
    np.savez_compressed(path,event_times_s=events,start_s=1.,end_s=end,probability=probability,probability_fs=64)
    write_json(meta,{'identity':identity,'reused':False,'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path),'events':len(events)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=('register','prepare','run'),default='register')
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    records=load_records(ROOT/'datasets/embc21');plan=register(args.root,records)
    if args.stage=='register':print(json.dumps({'jobs':len(plan['jobs']),'records':len(records),'parameters':plan['parameters']}));return
    if args.stage=='prepare':
        for method in args.methods:
            if method in ('AnyPPG','SmallCNN','MSPTDfast'):continue
            for seed in (SEEDS if method=='AnyPPG-random-frozen' else (42,)):prepare(args.root,method,seed,args.device)
        return
    by_id={r['record_id']:r for r in records}
    for method in args.methods:
        for seed in SEEDS:
            encoder=TemporalEncoder(method,seed,args.device) if method not in ('SmallCNN','MSPTDfast') else None
            for job in plan['jobs']:
                if job['seed']!=seed:continue
                try:
                    model,fit_info=fit(args.root,job,method,by_id,args.device)
                    for record in records:
                        if record['subject'] in job['test']:infer(args.root,job,method,record,model,fit_info,encoder,args.device)
                    print(json.dumps({'method':method,'job':job['id'],'complete':True}),flush=True)
                    del model
                except Exception as error:
                    write_json(args.root/'failure.json',{'method':method,'job':job['id'],'error_type':type(error).__name__,'error':str(error)})
                    raise
            if encoder is not None:
                write_json(args.root/f'{cache_name(method,seed)}-inference-state.json',encoder.audit())
                del encoder
            gc.collect()
            if args.device.startswith('cuda'):torch.cuda.empty_cache()


if __name__=='__main__':main()
