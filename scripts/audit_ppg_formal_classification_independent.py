"""Independent raw-source, sklearn fit and statistical checks for classification.

Does not call the production data preparation, classifier, score or split code.
Published predictions remain untouched. Original source labels are all checked;
source waveforms are reread for all WESAD/ECSMP windows and 100 BUT records.
"""
from pathlib import Path
import csv
import json
import pickle
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ppg_hr_poc_features import sha256, write_json


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def person_weights(subjects):
    _, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    return len(subjects) / len(counts) / counts[inverse]


def metric(y, p, d, w, name):
    if name == 'auroc':
        return roc_auc_score(y, p[:, 1], sample_weight=w)
    if name == 'auprc':
        return average_precision_score(y, p[:, 1], sample_weight=w)
    if name == 'balanced_accuracy':
        return balanced_accuracy_score(y, d, sample_weight=w)
    return f1_score(y, d, average='macro', sample_weight=w, zero_division=0)


def cesaf_orders(path):
    """Read the original workbook without the benchmark's worksheet decoder."""
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    with ZipFile(path) as z:
        strings = [''.join(s.itertext()) for s in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        wb = ET.fromstring(z.read('xl/workbook.xml'))
        rels = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        sheet = next(s for s in wb.find(ns+'sheets') if s.attrib['name'] == 'CESAF')
        rel = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        target = rels[rel].split('/')[-1]
        result = {}
        for row in ET.fromstring(z.read('xl/worksheets/'+target)).find(ns+'sheetData'):
            cells = {}
            for cell in row:
                value = cell.find(ns+'v')
                if value is None:
                    continue
                key = ''.join(c for c in cell.attrib['r'] if c.isalpha())
                cells[key] = strings[int(value.text)] if cell.get('t') == 's' else value.text
            try:
                person = str(int(float(cells.get('A', 'nan')))).zfill(3)
            except (ValueError, OverflowError):
                continue
            if all(c in cells for c in ('AQ', 'AR', 'AS', 'AT', 'AU', 'AV')):
                result[person] = [cells[c] for c in ('AQ', 'AR', 'AS', 'AT', 'AU', 'AV')]
    return result


def source_check(dataset, folder, rows):
    x = np.load(folder/'x125.npy', mmap_mode='r')
    raw = ROOT/'datasets/ppg_bench_raw'
    max_error = 0.; waveform_rows = 0; hashes = {}
    if dataset == 'BUT-PPG':
        import wfdb
        source = raw/'butppg-full-v2'
        annotations = {r['ID']: int(r['Quality']) for r in read_csv(source/'quality-hr-ann.csv')}
        assert len(rows) == len(annotations) == 3888
        for row in rows:
            assert annotations[row['record']] == int(row['target']) and row['subject'] == row['record'][:3]
        for i in np.unique(np.linspace(0, len(rows)-1, 100, dtype=int)):
            rid = rows[i]['record']
            record = wfdb.rdrecord(str(source/rid/(rid+'_PPG')))
            if record.p_signal.shape == (1, 300):
                signal = record.p_signal[0]
            else:
                # WFDB 4.1 parses the a.u. unit token into sig_name in this
                # release; read the explicit channel label from the header.
                header = (source/rid/(rid+'_PPG.hea')).read_text().splitlines()
                names = [line.split()[-1] for line in header[1:4]]
                assert record.p_signal.shape == (300, 3) and names.count('PPG_R') == 1
                signal = record.p_signal[:, names.index('PPG_R')]
            repeated = resample_poly(signal, 25, 6).astype(np.float32)
            max_error = max(max_error, float(np.max(np.abs(x[i]-repeated)))); waveform_rows += 1
        hashes['quality-hr-ann.csv'] = sha256(source/'quality-hr-ann.csv')
    else:
        subjects = np.asarray([r['subject'] for r in rows])
        orders = cesaf_orders(raw/'ecsmp/scale.xlsx') if dataset == 'ECSMP' else None
        names = ['neutral','fear','sad','happy','anger','disgust']
        for person in np.unique(subjects):
            ids = np.flatnonzero(subjects == person)
            if dataset == 'WESAD':
                path = raw/f'wesad/WESAD/{person}/{person}.pkl'
                with path.open('rb') as f:
                    original = pickle.load(f, encoding='latin1')
                wave = np.asarray(original['signal']['wrist']['BVP']).ravel()
                labels = np.asarray(original['label']).ravel()
                for i in ids:
                    start = round(float(rows[i]['start_s'])*700)
                    u, c = np.unique(labels[start:start+7000], return_counts=True)
                    majority = int(u[np.argmax(c)])
                    assert c.max()/7000 >= .9 and majority in (1,2,3) and majority-1 == int(rows[i]['target'])
                hashes[str(path.relative_to(ROOT))] = sha256(path)
                del original, labels
            else:
                path = raw/f'ecsmp/E4/{person}/BVP.csv'
                with path.open() as f:
                    float(f.readline()); fs = float(f.readline())
                assert fs == 64
                wave = np.loadtxt(path, skiprows=2, delimiter=',').astype(np.float32)
                eeg_path = next((raw/'ecsmp/EEG_downsample').glob(f'*_{person}.mat'))
                eeg = loadmat(eeg_path, squeeze_me=True, struct_as_record=False)['EEG']
                events = [(int(e.type), float(e.latency)/float(eeg.srate)) for e in np.ravel(eeg.urevent)]
                starts = [(c,t) for c,t in events if 101 <= c <= 106]
                assert [names[c-101] for c,t in starts] == orders[person]
                bounds = []
                for j, (code, start) in enumerate(starts):
                    stop = starts[j+1][1] if j+1 < len(starts) else np.inf
                    end = [t for c,t in events if c == 20 and start < t < stop]
                    if not end: end = [t for c,t in events if c == 12 and start < t < stop]
                    bounds.append((start, end[0], code-101))
                for i in ids:
                    a = float(rows[i]['start_s'])
                    candidates = [c for lo,hi,c in bounds if lo <= a and a+10 <= hi]
                    assert candidates == [int(rows[i]['target'])]
                hashes[str(path.relative_to(ROOT))] = sha256(path)
                hashes[str(eeg_path.relative_to(ROOT))] = sha256(eeg_path)
            repeated = resample_poly(wave, 125, 64).astype(np.float32)
            for i in ids:
                a = round(float(rows[i]['start_s'])*125)
                max_error = max(max_error, float(np.max(np.abs(x[i]-repeated[a:a+1250])))); waveform_rows += 1
            print(json.dumps(dict(source=dataset, person=person, rows=len(ids))), flush=True)
    assert max_error < 1e-6, (dataset, max_error)
    return dict(all_labels_checked=len(rows), raw_waveforms_checked=waveform_rows, maximum_waveform_error=max_error,
                raw_source_sha256=hashes, eeg_timing_note='ECSMP reproduces released EEG latency/fs clock; not independent hardware synchronization validation')


def independent_refit(folder, protocol, subjects, y):
    job = protocol['jobs'][0]
    tr, va, te = [np.flatnonzero(np.isin(subjects, job[k])) for k in ('train','validation','test')]
    results = []
    for method in ('AnyPPG','PaPaGei-P'):
        dest = folder/'runs'/job['id']/method/'PCA16-Linear'
        features = np.load(folder/'features'/f'{method}.npy').astype(np.float64)
        scaler = StandardScaler().fit(features[tr]); scaled = scaler.transform(features)
        pca = PCA(n_components=16, svd_solver='full').fit(scaled[tr])
        z = pca.transform(scaled)
        z = StandardScaler().fit(z[tr]).transform(z)
        # The production logistic performs a second training-only standardization.
        z = StandardScaler().fit(z[tr]).transform(z)
        info = read_json(dest/'complete.json'); best = None; candidates = []
        for c in (.01,.1,1.,10.,100.):
            classifier = LogisticRegression(C=c,max_iter=3000,random_state=0).fit(z[tr],y[tr],sample_weight=person_weights(subjects[tr]))
            p = classifier.predict_proba(z[va]); value = metric(y[va],p,p.argmax(1),person_weights(subjects[va]),protocol['primary_metric'])
            candidates.append(value)
            if best is None or value > best[0]: best = (value, classifier)
        assert best[1].C == info['selection']['selected_C'], (method, best[1].C, info['selection'])
        stored = np.load(dest/'predictions.npz')['probability']
        error = float(np.max(np.abs(best[1].predict_proba(z[te])-stored)))
        assert error < 1e-5, (method, error)
        results.append(dict(method=method, job=job['id'], selected_C=best[1].C, maximum_probability_error=error,
                            pipeline='Independent sklearn StandardScaler/full-SVD PCA16/StandardScaler/logistic; training and validation only'))
    return results


def audit(dataset):
    folder = ROOT/'.local/ppg-formal-expansion/classification'/dataset
    shared = ROOT/'docs/results/ppg_formal_expansion/classification'/dataset
    rows = read_csv(folder/'rows.csv'); protocol = read_json(folder/'protocol.json')
    subjects = np.asarray([r['subject'] for r in rows]); y = np.asarray([int(r['target']) for r in rows])
    source = source_check(dataset, folder, rows)
    refits = independent_refit(folder, protocol, subjects, y)
    summaries = read_csv(shared/'summary.csv'); published = read_csv(shared/'seed_scores.csv')
    seen = 0; largest = 0.; examples = {}
    for summary in summaries:
        method = summary['method']; estimates = []
        for seed in (42,43,44):
            prob = np.full((len(rows), y.max()+1), np.nan); decision = np.full(len(rows), -1); hit = np.zeros(len(rows), int)
            for job in protocol['jobs']:
                if job['seed'] != seed: continue
                groups = [set(job[k]) for k in ('train','validation','test')]
                assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
                ids = np.flatnonzero(np.isin(subjects, job['test']))
                with np.load(folder/'runs'/job['id']/method/'predictions.npz') as p:
                    assert np.array_equal(p['row'], ids) and np.array_equal(p['reference'], y[ids]) and np.array_equal(p['subject'],subjects[ids])
                    prob[ids] = p['probability']; decision[ids] = p['decision']; hit[ids] += 1
                seen += 1
            assert np.all(hit == 1)
            reference = next(r for r in published if r['method'] == method and int(r['seed']) == seed)
            names = ['balanced_accuracy','macro_f1'] + (['auroc','auprc'] if y.max() == 1 else [])
            for name in names:
                value = metric(y,prob,decision,person_weights(subjects),name)
                largest = max(largest,abs(value-float(reference[name])))
            estimates.append(metric(y,prob,decision,person_weights(subjects),protocol['primary_metric']))
            if seed == 42 and method in ('AnyPPG/PCA16-Linear','HandFeatures/Linear'):
                examples[method] = (prob,decision)
        largest = max(largest,abs(np.mean(estimates)-float(summary['estimate'])))
    assert seen == 255 and largest < 1e-12
    # Independent numerical checks of the production custom metric under whole-person
    # resampling and block swaps; this is separate from source and fit checks above.
    from scripts.summarize_ppg_formal_classification import weighted_metric
    rng = np.random.default_rng(9182); people, inv = np.unique(subjects,return_inverse=True)
    a,ad = examples['AnyPPG/PCA16-Linear']; b,bd = examples['HandFeatures/Linear']; diff = 0.
    for _ in range(30):
        multiplicity = np.bincount(rng.integers(len(people),size=len(people)),minlength=len(people))
        w = person_weights(subjects)*multiplicity[inv]
        swap = rng.integers(2,size=len(people)).astype(bool)[inv]
        p = np.where(swap[:,None],b,a); d = np.where(swap,bd,ad)
        for name in ('balanced_accuracy','macro_f1', *(['auroc'] if y.max() == 1 else [])):
            diff = max(diff,abs(metric(y,p,d,w,name)-weighted_metric(y,p,d,w,name)))
    assert diff < 1e-12
    out = dict(status='passed',dataset=dataset,source=source,independent_refits=refits,
        held_out_outputs_checked=seen, maximum_sklearn_metric_difference=largest,
        independent_random_block_statistic_checks=30, maximum_resampled_metric_difference=diff,
        statistical_review='Bootstrap resamples people with all records and seeds together; paired permutation swaps complete person prediction blocks before recomputing pooled metric; seven primary tests use Holm per dataset.',
        observation_review='All methods observe the same ten seconds via starts 0 and 2 seconds, each eight seconds. Mean FM embeddings / mean CNN logits; neither test labels nor test cohort fits PCA.',
        limitations=['ECSMP is six stimulus identities, not six verified felt-emotion diagnoses.','WESAD is three laboratory states, not an independent clinical stress diagnosis.','BUT labels reflect HR readability, not morphological fidelity.', 'Intervals use 1000 participant bootstrap replicates; p values use 2000 random swaps, so exploratory precision is finite.'],
        protocol_sha256=sha256(folder/'protocol.json'),audit_code_sha256=sha256(Path(__file__)))
    write_json(shared/'independent_source_fit_statistics_audit.json',out)
    print(json.dumps(dict(dataset=dataset,status='passed',outputs=seen,metric_error=largest,refits=refits)),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        for dataset in sys.argv[1:] or ['BUT-PPG','WESAD','ECSMP']:
            audit(dataset)
