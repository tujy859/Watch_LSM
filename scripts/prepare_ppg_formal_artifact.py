"""Map all public SEGADE rows exactly, then select label-blind center contexts."""
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.ppg_artifact_reference_pilot import minmax,validate_annotation_pair
from scripts.ppg_formal_classification import split_jobs
from scripts.ppg_hr_poc_features import sha256,write_json


def main():
    output=ROOT/'.local/ppg-formal-expansion/artifact';output.mkdir(parents=True,exist_ok=True)
    arrays=[];labels=[];identities=[];hashes={}
    for group in ('train','test'):
        base=ROOT/f'datasets/segade/new_PPG_DaLiA_{group}/processed_dataset'
        x=np.load(base/'scaled_ppgs.npy');y=np.load(base/'seg_labels.npy');validate_annotation_pair(x,y)
        arrays.append(x);labels.append(y)
        identities.extend(dict(published_group=group,published_row=i) for i in range(len(x)))
        for name in ('scaled_ppgs.npy','seg_labels.npy'):hashes[(base/name).relative_to(ROOT).as_posix()]=sha256(base/name)
    arrays=np.concatenate(arrays);labels=np.concatenate(labels);matches=[[] for _ in arrays]
    tree=cKDTree(arrays[:,:32]);sos=butter(2,[.9/32,5/32],btype='bandpass',output='sos')
    source=[]
    for path in sorted((ROOT/'datasets/ppg_bench_raw/dalia/PPG_FieldStudy').glob('S*/S*.pkl')):
        with path.open('rb') as f:trace=np.asarray(pickle.load(f,encoding='latin1')['signal']['wrist']['BVP']).ravel().astype(float)
        n=len(trace)//1920;blocks=minmax(sosfilt(sos,trace[:n*1920].reshape(n,1920),axis=1))
        for block,possible in enumerate(tree.query_ball_point(blocks[:,:32],np.sqrt(32)*1e-10)):
            for index in possible:
                error=float(np.max(np.abs(blocks[block]-arrays[index])))
                if error<=1e-10:matches[index].append(dict(subject=path.parent.name,start_sample=block*1920,max_error=error))
        source.append(dict(subject=path.parent.name,sha256=sha256(path),samples=len(trace)))
    if len(source)!=15 or any(len(m)!=1 for m in matches):
        write_json(output/'unresolved.json',dict(source=source,unresolved=[dict(row=i,matches=m) for i,m in enumerate(matches) if len(m)!=1]))
        raise ValueError('All annotation rows must map uniquely across all fifteen people')
    rows=[dict(**identity,**match[0]) for identity,match in zip(identities,matches)]
    keys=[(r['subject'],r['start_sample']) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate source acquisition intervals across annotation rows')
    subjects=np.asarray([r['subject'] for r in rows]);groups={g:sorted(set(r['subject'] for r in rows if r['published_group']==g)) for g in ('train','test')}
    # Center frame [11,19), center scored core [12,18), all at original 64Hz.
    x64=arrays[:,11*64:19*64].astype(np.float32);y64=labels[:,11*64:19*64].astype(np.float32)
    x125=resample_poly(x64,125,64,axis=1).astype(np.float32)
    np.savez(output/'data.npz',x64=x64,x125=x125,target=y64,subject=subjects,start_sample=np.asarray([r['start_sample'] for r in rows]))
    write_json(output/'rows.json',rows)
    protocol=dict(dataset='SEGADE-PPG-DaLiA',records=len(rows),participants=len(set(subjects)),
        reference='1=human artifact; every published row uniquely matches all 1920 original samples after fixed per-30s 0.9-5Hz causal filtering/minmax',
        observation='One fixed center 8-second context [11,19) per 30-second annotated row; score center [12,18), 64Hz. Selection never uses point labels.',
        score_seconds=len(rows)*6,available_annotation_seconds=len(rows)*30,coverage_fraction=.2,
        jobs=split_jobs(subjects,'DaLiA-artifact'),models=['AnyPPG','PaPaGei-S','PaPaGei-P','AnyPPG-random-frozen','SmallCNN','HandFeatures','TrainPrior'],
        training='At most 128 evenly spaced frame indices per training person; 64 per validation person; every test frame. Seed42/43/44,20 epochs,patience5,Adam.001; person-weighted BCE; validation person-mean point F1 selects epoch and threshold.',
        head='Frozen six-stage maps -> registered ~27k temporal head; randomAny exact same architecture; SmallEventCNN 17353 parameters',
        thresholds=[.1,.2,.3,.4,.5,.6,.7,.8,.9],primary='Mean participant point-level artifact F1; average seeds within person first',
        secondary=['sensitivity','specificity','balanced_accuracy','AUPRC'],
        primary_comparisons='Three pretrained models vs HandFeatures plus Any vs randomAny; four paired participant sign-flip tests with Holm correction',
        traditional='HandSQI frame features plus fixed local amplitude/derivative/rolling variation descriptors, logistic classifier; validation regularization and threshold only',
        original_published_subject_groups=groups,original_group_overlap=sorted(set(groups['train'])&set(groups['test'])),
        uniqueness='All 4305 rows and 15 source people searched on the fixed 30-second grid; max point error <=1e-10; duplicate mapped identities prohibited',
        annotation_files_sha256=hashes,raw_source_files=source,data_sha256=sha256(output/'data.npz'),rows_sha256=sha256(output/'rows.json'),
        preparation_code_sha256=sha256(Path(__file__)),pretraining=False,encoder_finetuning=False)
    write_json(output/'protocol.json',protocol)
    shared=ROOT/'docs/results/ppg_formal_expansion/artifact';shared.mkdir(parents=True,exist_ok=True)
    write_json(shared/'reference_audit.json',dict(status='complete',rows=len(rows),unique=len(set(keys)),participants=15,
        maximum_point_error=max(r['max_error'] for r in rows),original_groups=groups,scored_fraction=.2,
        protocol_sha256=sha256(output/'protocol.json'),annotation_files_sha256=hashes,source_files=source))
    print(json.dumps(dict(mapped=len(rows),participants=15,score_seconds=len(rows)*6)),flush=True)


if __name__=='__main__':main()
