"""Verify dense artifact predictions and report participant-level inference.

All held-out predictions and selected validation heads are replayed on CPU.
The independent unit is a participant, after averaging that person's three
seed-specific scores. Dense time samples never become independent replicates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt
from scipy.special import expit
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ppg_beat_models import SmallEventCNN
from scripts.ppg_formal_artifact import DEFAULT, NEURAL, SEEDS, key, local_features
from scripts.ppg_formal_classification import read_json, write_csv
from scripts.ppg_hr_poc_features import sha256, write_json
from scripts.ppg_temporal_adapters import make_head

THRESHOLDS = np.arange(1, 10) / 10
METRICS = ('f1', 'sensitivity', 'specificity', 'balanced_accuracy', 'auprc')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def point_metrics(y, probability, threshold):
    """Independent confusion counts; this deliberately does not use fit helpers."""
    y = np.asarray(y).ravel().astype(int)
    decision = (np.asarray(probability).ravel() >= threshold).astype(int)
    require(len(y) == len(decision) and np.isin(y, [0, 1]).all(), 'Invalid point references')
    tn, fp, fn, tp = np.bincount(y * 2 + decision, minlength=4)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                f1=float(2 * tp / max(2 * tp + fp + fn, 1)),
                sensitivity=float(sensitivity), specificity=float(specificity),
                balanced_accuracy=float((sensitivity + specificity) / 2),
                auprc=float(average_precision_score(y, np.asarray(probability).ravel())))


def validation_selection(y, p, subjects):
    values = []
    for threshold in THRESHOLDS:
        scores = []
        for person in np.unique(subjects):
            mask = subjects == person
            scores.append(f1_score(y[mask].ravel(), (p[mask] >= threshold).ravel(), zero_division=0))
        values.append((float(np.mean(scores)), -abs(threshold - .5), -threshold, float(threshold)))
    selected = max(values)
    return selected[3], selected[0]


def exact_signflip(difference):
    difference = np.asarray(difference, dtype=float)
    require(difference.ndim == 1 and 0 < len(difference) <= 20 and np.isfinite(difference).all(),
            'Exact sign-flip requires 1 to 20 finite participant differences')
    signs = 2 * ((np.arange(2 ** len(difference))[:, None] >> np.arange(len(difference))) & 1) - 1
    null = (signs * difference).mean(axis=1)
    return float(np.mean(np.abs(null) >= abs(difference.mean()) - 1e-12))


def holm(values):
    values = np.asarray(values, dtype=float)
    out = np.empty(len(values)); last = 0.
    for rank, index in enumerate(np.argsort(values)):
        last = max(last, min(1., values[index] * (len(values) - rank)))
        out[index] = last
    return out


def sample_indices(subjects, people, cap):
    result = []
    for person in people:
        rows = np.flatnonzero(subjects == person)
        require(len(rows) > 0, 'Unknown split participant')
        result.extend(rows[np.linspace(0, len(rows)-1, min(cap, len(rows)), dtype=int)])
    return np.asarray(result)


def source_audit(root, protocol, data):
    require(sha256(root/'data.npz') == protocol['data_sha256'] and
            sha256(root/'rows.json') == protocol['rows_sha256'], 'Reference cache changed')
    rows = read_json(root/'rows.json')
    require(len(rows) == len(data['subject']) == protocol['records'] == 4305, 'Incomplete annotation rows')
    require(len({(r['subject'], r['start_sample']) for r in rows}) == len(rows), 'Duplicate source interval')
    require(np.array_equal(data['subject'], [r['subject'] for r in rows]) and
            np.array_equal(data['start_sample'], [r['start_sample'] for r in rows]), 'Mapped row identity differs')
    published = {}
    for group in ('train', 'test'):
        folder = ROOT/f'datasets/segade/new_PPG_DaLiA_{group}/processed_dataset'
        published[group] = [np.load(folder/name) for name in ('scaled_ppgs.npy', 'seg_labels.npy')]
    for path, expected in protocol['annotation_files_sha256'].items():
        require(sha256(ROOT/path) == expected, 'Annotation source changed')
    maximum = 0.; sos = butter(2, [.9/32, 5/32], btype='bandpass', output='sos')
    for entry in protocol['raw_source_files']:
        person = entry['subject']; path = ROOT/f'datasets/ppg_bench_raw/dalia/PPG_FieldStudy/{person}/{person}.pkl'
        require(sha256(path) == entry['sha256'], 'Raw DaLiA source changed')
        with path.open('rb') as stream:
            original = pickle.load(stream, encoding='latin1')
        signal = np.asarray(original['signal']['wrist']['BVP']).ravel().astype(float)
        blocks = sosfilt(sos, signal[:len(signal)//1920*1920].reshape(-1, 1920), axis=1)
        blocks = (blocks-blocks.min(1, keepdims=True)) / np.maximum(np.ptp(blocks, axis=1, keepdims=True), 1e-12)
        for index in np.flatnonzero(data['subject'] == person):
            row = rows[index]; x, y = published[row['published_group']]
            j = row['published_row']; mapped = blocks[row['start_sample']//1920]
            error = float(np.max(np.abs(mapped-x[j]))); maximum = max(maximum, error)
            require(row['start_sample'] % 1920 == 0 and error <= 1e-10, 'Original source mapping mismatch')
            require(np.array_equal(data['x64'][index], x[j, 704:1216].astype(np.float32)) and
                    np.array_equal(data['target'][index], y[j, 704:1216].astype(np.float32)), 'Center data/labels differ')
    repeated = resample_poly(data['x64'], 125, 64, axis=1).astype(np.float32)
    require(np.array_equal(repeated, data['x125']), '125 Hz observation differs')
    require(np.isfinite(data['x64']).all() and np.isin(data['target'], [0, 1]).all(), 'Nonfinite or nonbinary source')
    require(len(set(data['subject'])) == 15, 'Incorrect participant coverage')
    return dict(all_4305_full_30s_mappings_rechecked=True, maximum_mapping_point_error=maximum,
                all_center_waveforms_and_point_labels_checked=True, raw_people=15,
                uniqueness='Published map has no duplicate person/start; preparation searched every candidate in all 15 sources')


def feature_arrays(root, method, seed, n, protocol_hash, cache, hashes):
    name = key(method, seed)
    if name not in cache:
        folder = root/'features'/name; info = read_json(folder/'complete.json')
        require(info['protocol_sha256'] == protocol_hash, 'Feature protocol changed')
        evidence = info['encoder']
        require(evidence['state_before_sha256'] == evidence['state_after_sha256'] and
                evidence['encoder_trainable_parameters'] == 0 and evidence['method'] == method,
                'Frozen encoder evidence differs')
        if method == 'AnyPPG-random-frozen':
            require(evidence['random_seed'] == seed and not evidence['checkpoint_sha256'], 'Random initialization differs')
        for path, expected in evidence['checkpoint_sha256'].items():
            require(sha256(ROOT/path) == expected, 'Pretrained checkpoint changed')
        arrays = []
        for i in range(6):
            path = folder/f'stage{i}.npy'; expected = info['files_sha256'][path.name]
            require(sha256(path) == expected, 'Feature cache hash mismatch')
            array = np.load(path, mmap_mode='r')
            require(array.shape[0] == n and np.isfinite(array).all(), 'Invalid feature array')
            arrays.append(array)
        cache[name] = arrays
        hashes[name] = dict(metadata_sha256=sha256(folder/'complete.json'), **info)
    return cache[name]


def replay_network(method, state_path, arrays, rows, device='cpu'):
    model = SmallEventCNN() if method == 'SmallCNN' else make_head(method)
    model.load_state_dict(torch.load(state_path, map_location='cpu', weights_only=True)); model.to(device).eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            ix = rows[start:start+32]
            batch = [torch.tensor(np.asarray(array[ix]), dtype=torch.float32, device=device) for array in arrays]
            x = batch[0].unsqueeze(1) if method == 'SmallCNN' else batch
            values.append(model(x)[:, 64:448].sigmoid().cpu().numpy())
    return np.concatenate(values), sum(p.numel() for p in model.parameters())


def history_check(info):
    history = info['history']
    require([r['epoch'] for r in history] == list(range(1, len(history)+1)) and 1 <= len(history) <= 20,
            'Invalid epoch history')
    require(all(np.isfinite(r['validation_f1']) and r['threshold'] in THRESHOLDS for r in history),
            'Nonfinite validation history or invalid threshold')
    selected = max(history, key=lambda r: r['validation_f1'])
    require(selected['epoch'] == info['best_epoch'] and selected['validation_f1'] == info['validation_f1'] and
            selected['threshold'] == info['threshold'], 'Selected epoch is not validation-best')
    require(len(history) == 20 or len(history)-info['best_epoch'] == 5, 'Early-stopping patience differs')


def audit(root, shared, bootstrap=10000, cuda_crosscheck=False):
    torch.set_num_threads(2)
    protocol = read_json(root/'protocol.json'); protocol_hash = sha256(root/'protocol.json')
    with np.load(root/'data.npz', allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    source = source_audit(root, protocol, data)
    subjects = data['subject']; y = data['target'][:, 64:448]; people = np.unique(subjects)
    methods = list(protocol['models']); feature_cache = {}; feature_hashes = {}; outputs = {}
    require(set(methods) == set(NEURAL) | {'HandFeatures', 'TrainPrior'}, 'Unexpected methods')
    require(len(protocol['jobs']) == 15 and len({j['id'] for j in protocol['jobs']}) == 15, 'Incomplete jobs')
    hand = local_features(data['x64']); person_rows = []; run_rows = []
    max_probability_error = 0.; max_validation_error = 0.; decisions_different = 0; cuda_checks = []
    source_hash = sha256(ROOT/'scripts/ppg_formal_artifact.py')
    for job in protocol['jobs']:
        groups = [set(job[k]) for k in ('train', 'validation', 'test')]
        require([len(g) for g in groups] == [9, 3, 3] and set.union(*groups) == set(people) and
                all(not groups[a] & groups[b] for a,b in ((0,1),(0,2),(1,2))), 'Participant split leaks or misses people')
        train = sample_indices(subjects, job['train'], 128); val = sample_indices(subjects, job['validation'], 64)
        test = np.flatnonzero(np.isin(subjects, job['test']))
        for method in methods:
            folder = root/'runs'/job['id']/method; info = read_json(folder/'complete.json')
            require(info['identity'] == dict(protocol_sha256=protocol_hash, job=job, method=method), 'Run identity differs')
            require(np.array_equal(train, info['train_rows']) and np.array_equal(val, info['validation_rows']),
                    'Train/validation row selection differs')
            require(sha256(folder/'prediction.npz') == info['prediction_sha256'], 'Prediction hash differs')
            with np.load(folder/'prediction.npz', allow_pickle=False) as f:
                require(np.array_equal(f['row'], test) and np.array_equal(f['subject'], subjects[test]) and
                        np.array_equal(f['reference'], y[test]), 'Held-out prediction identity differs')
                p = f['probability'].copy()
            require(p.shape == y[test].shape and np.isfinite(p).all() and np.all((0 <= p) & (p <= 1)),
                    'Invalid probabilities')
            threshold = info['threshold']; require(threshold in THRESHOLDS, 'Invalid selected threshold')
            artifacts = dict(prediction_sha256=sha256(folder/'prediction.npz'), metadata_sha256=sha256(folder/'complete.json'))
            if method in NEURAL:
                require(info['code_sha256'] == source_hash, 'Neural fit code changed')
                require(sha256(folder/'head.pt') == info['head_sha256'], 'Saved neural head changed')
                arrays = [data['x64']] if method == 'SmallCNN' else feature_arrays(root, method, job['seed'], len(y), protocol_hash, feature_cache, feature_hashes)
                repeated, parameters = replay_network(method, folder/'head.pt', arrays, np.r_[val, test])
                vp, replay = repeated[:len(val)], repeated[len(val):]
                require(parameters == info['parameters'], 'Head parameter count differs')
                history_check(info)
                artifacts['head_sha256'] = info['head_sha256']
            elif method == 'HandFeatures':
                with np.load(folder/'model.npz', allow_pickle=False) as f: state = {k: f[k] for k in f.files}
                tx = hand[train, 64:448:4].reshape(-1, hand.shape[-1]).astype(float)
                mean = tx.mean(0); scale = tx.std(0); scale[scale < 1e-12] = 1.
                require(np.array_equal(mean, state['mean']) and np.array_equal(scale, state['scale']),
                        'Hand-feature standardization was not fitted on training rows')
                def predict(ix):
                    x = hand[ix, 64:448].reshape(-1, hand.shape[-1]).astype(float)
                    return expit(((x-state['mean'])/state['scale']) @ state['coef'].T + state['intercept']).reshape(len(ix), 384)
                vp, replay = predict(val), predict(test)
                candidates = info['selection']['candidates']
                require([r['C'] for r in candidates] == [.01,.1,1.,10.] and
                        max(candidates, key=lambda r:r['validation_f1'])['C'] == info['selection']['selected_C'],
                        'Hand-feature C was not selected by validation score')
                artifacts['model_sha256'] = sha256(folder/'model.npz')
            else:
                train_subjects = subjects[train]
                prior = float(np.mean([y[train][train_subjects == person, ::4].mean() for person in np.unique(train_subjects)]))
                require(abs(prior-info['selection']['prior']) < 1e-7, 'Train prior differs')
                vp = np.full((len(val),384), info['selection']['prior']); replay = np.full(p.shape, info['selection']['prior'])
            selected_threshold, validation_f1 = validation_selection(y[val], vp, subjects[val])
            require(selected_threshold == threshold, 'Selected threshold does not reproduce on validation people')
            expected_val = info['validation_f1'] if method in NEURAL else (
                max(r['validation_f1'] for r in info['selection']['candidates']) if method == 'HandFeatures' else validation_f1)
            validation_error = abs(expected_val-validation_f1); max_validation_error = max(max_validation_error, validation_error)
            require(validation_error < 2e-5, 'Selected validation score differs')
            error = float(np.max(np.abs(replay-p))); max_probability_error = max(max_probability_error, error)
            require(error < 5e-5, f'CPU saved-head replay differs: {job["id"]} {method} {error}')
            nchanged = int(np.sum((replay >= threshold) != (p >= threshold))); decisions_different += nchanged
            if cuda_crosscheck and method == 'SmallCNN':
                gpu,_ = replay_network(method,folder/'head.pt',arrays,test,'cuda')
                gpu_error = float(np.max(np.abs(gpu-p)))
                require(gpu_error < 2e-6, 'CUDA SmallCNN replay differs')
                cuda_checks.append(dict(job=job['id'],maximum_cuda_probability_error=gpu_error,
                                        maximum_cpu_probability_error=error,cpu_decisions_changed=nchanged))
            outputs[f'{job["id"]}/{method}'] = artifacts
            run_rows.append(dict(job=job['id'], method=method, seed=job['seed'], fold=job['fold'], train_people=9,
                                 validation_people=3, test_people=3, train_frames=len(train), validation_frames=len(val),
                                 test_frames=len(test), threshold=threshold, validation_f1=validation_f1,
                                 maximum_cpu_probability_error=error, cpu_decisions_changed=nchanged))
            for person in job['test']:
                mask = subjects[test] == person
                metrics = point_metrics(y[test][mask], p[mask], threshold)
                tn,fp,fn,tp = confusion_matrix(y[test][mask].ravel(), (p[mask]>=threshold).ravel(), labels=[0,1]).ravel()
                require([metrics[k] for k in ('tn','fp','fn','tp')] == [tn,fp,fn,tp], 'Independent sklearn count mismatch')
                person_rows.append(dict(method=method, seed=job['seed'], subject=person, frames=int(mask.sum()),
                                        scored_points=int(mask.sum())*384, **metrics))
            print(json.dumps(dict(audited=job['id'],method=method,cpu_error=error)), flush=True)
    require(len(outputs) == 105, 'Incomplete runs')
    averaged = {}; participant_rows = []
    for method in methods:
        averaged[method] = []
        for person in people:
            rows = [r for r in person_rows if r['method'] == method and r['subject'] == person]
            require(len(rows) == 3 and {r['seed'] for r in rows} == set(SEEDS), 'Missing/duplicate held-out person seed')
            record = dict(method=method,subject=person,frames=rows[0]['frames'],scored_points=rows[0]['scored_points'],
                          **{metric:float(np.mean([r[metric] for r in rows])) for metric in METRICS})
            averaged[method].append([record[m] for m in METRICS]); participant_rows.append(record)
        averaged[method] = np.asarray(averaged[method])
    rng = np.random.default_rng(20261105); draws = rng.integers(0,len(people),size=(bootstrap,len(people)))
    summaries = []; distributions = {}
    for method in methods:
        values = averaged[method]; means = values.mean(0); distribution = values[draws].mean(1)
        distributions[method] = distribution[:, 0]; ci = np.percentile(distribution,[2.5,97.5],axis=0)
        record = dict(method=method,f1=float(means[0]),ci_low=float(ci[0,0]),ci_high=float(ci[1,0]),
                      participants=len(people),records=len(y),seeds=3,scored_points=int(y.size),scored_seconds=len(y)*6,
                      annotation_seconds=len(y)*30,coverage_fraction=.2,held_out_frame_coverage=1.,
                      artifact_point_prevalence=float(y.mean()))
        for index, metric in enumerate(METRICS[1:], 1):
            record.update({metric:float(means[index]),f'{metric}_ci_low':float(ci[0,index]),f'{metric}_ci_high':float(ci[1,index])})
        summaries.append(record)
    pairs = [('AnyPPG','HandFeatures'),('PaPaGei-S','HandFeatures'),('PaPaGei-P','HandFeatures'),('AnyPPG','AnyPPG-random-frozen')]
    paired = []
    for left,right in pairs:
        difference = averaged[left][:,0]-averaged[right][:,0]
        low,high = np.percentile(distributions[left]-distributions[right],[2.5,97.5])
        paired.append(dict(model=left,comparator=right,metric='participant_point_f1',difference=float(difference.mean()),
                           ci_low=float(low),ci_high=float(high),exact_signflip_p=exact_signflip(difference),participants=15))
    for row,pvalue in zip(paired,holm([r['exact_signflip_p'] for r in paired])):
        row['holm_p'] = float(pvalue)
    shared.mkdir(parents=True,exist_ok=True)
    for name,rows in [('summary.csv',summaries),('paired.csv',paired),('per_subject_seed.csv',person_rows),
                      ('per_subject.csv',participant_rows),('run_audit.csv',run_rows)]:
        write_csv(shared/name,rows)
    write_json(shared/'protocol.json',protocol)
    result = dict(status='passed',runs=105,neural_heads_cpu_replayed=75,traditional_outputs_replayed=30,
        source=source,source_protocol_sha256=protocol_hash,outputs_sha256=outputs,feature_caches=feature_hashes,
        maximum_cpu_probability_error=max_probability_error,maximum_validation_f1_error=max_validation_error,
        cpu_threshold_decisions_changed=decisions_different,participants=15,records=4305,point_labels_per_record=384,
        smallcnn_cuda_crosschecks=cuda_checks,
        cpu_tolerance_note='FP32 CPU/CUDA convolution differences allowed up to 5e-5 in probability, with discrete threshold changes counted explicitly. All 15 SmallCNN CUDA predictions can be crosschecked with --cuda-crosscheck.',
        statistics=dict(estimand='Average 3 seed scores within person, then equal mean over 15 people',
                        bootstrap_replicates=bootstrap,bootstrap_unit='participant with all seeds and points together',
                        exact_signflip_assignments=2**15,primary_tests=4,multiplicity='Holm across the 4 primary tests'),
        selection_check='All saved heads replayed on validation and test; validation threshold and best saved epoch history checked. Historical nonselected epoch states were not saved and cannot be independently replayed.',
        limitations=['Fixed center 6s score covers 20% of annotated time; no claim of full-duration coverage.',
                     'Public SEGADE waveforms already have per-30s causal filtering and min-max scaling; this is artifact detection on released signals, not raw-signal denoising.',
                     'No individual point is an independent statistical unit; 15 participant inference remains exploratory.',
                     'HandFeatures model hashes are recorded at audit time; original completion metadata did not bind model.npz hashes. Saved-model replay independently validates their predictions.'],
        audit_code_sha256=sha256(Path(__file__)))
    write_json(shared/'audit.json',result)
    print(json.dumps(dict(status='passed',runs=105,maximum_cpu_probability_error=max_probability_error,summary=summaries,paired=paired)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--shared',type=Path,default=ROOT/'docs/results/ppg_formal_expansion/artifact')
    parser.add_argument('--bootstrap',type=int,default=10000)
    parser.add_argument('--cuda-crosscheck',action='store_true',help='Also replay all SmallCNN test heads on original CUDA backend')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        audit(args.root,args.shared,args.bootstrap,args.cuda_crosscheck)
