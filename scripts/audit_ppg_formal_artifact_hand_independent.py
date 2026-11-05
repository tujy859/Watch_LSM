"""Refit all artifact logistic controls from train-only rows without changing runs.

Independent sklearn fitting/threshold selection validates the saved coefficients,
whose initial completion metadata did not include a model-file hash. Descriptor
construction is shared, but sampling, standardization, weights, fitting and
validation selection below are reconstructed without the production fit helper.
"""
from pathlib import Path
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.ppg_formal_artifact import local_features
from scripts.ppg_hr_poc_features import sha256, write_json


def main():
    folder = ROOT/'.local/ppg-formal-expansion/artifact'
    protocol = json.loads((folder/'protocol.json').read_text())
    data = np.load(folder/'data.npz'); subjects = data['subject']; y = data['target'][:,64:448]
    values = local_features(data['x64']); report = []
    for job in protocol['jobs']:
        ix = []
        for group,cap in [('train',128),('validation',64)]:
            ids = []
            for person in job[group]:
                selected = np.where(subjects == person)[0]
                ids.extend(selected[np.linspace(0,len(selected)-1,min(cap,len(selected)),dtype=int)])
            ix.append(np.asarray(ids))
        train,val = ix; test = np.flatnonzero(np.isin(subjects,job['test']))
        dest = folder/'runs'/job['id']/'HandFeatures'
        info = json.loads((dest/'complete.json').read_text())
        raw = values[train,64:448:4].reshape(-1,values.shape[-1]).astype(float)
        scaler = StandardScaler().fit(raw); x = scaler.transform(raw)
        vx = scaler.transform(values[val,64:448].reshape(-1,values.shape[-1]).astype(float))
        _,inverse,counts = np.unique(subjects[train],return_inverse=True,return_counts=True)
        weight = np.repeat(len(train)/len(counts)/counts[inverse],96)
        candidates = []; best = None
        for c in [.01,.1,1.,10.]:
            model = LogisticRegression(C=c,max_iter=1000).fit(x,y[train,::4].ravel(),sample_weight=weight)
            probability = model.predict_proba(vx)[:,1].reshape(len(val),384)
            scores = []
            for threshold in np.arange(1,10)/10:
                score = np.mean([f1_score(y[val][subjects[val]==person].ravel(),
                                         (probability[subjects[val]==person]>=threshold).ravel(),zero_division=0)
                                 for person in np.unique(subjects[val])])
                scores.append((float(score),-abs(threshold-.5),-threshold,float(threshold)))
            selected = max(scores)
            candidates.append(dict(C=c,validation_f1=selected[0],threshold=selected[3]))
            if best is None or selected[0] > best[0]:
                best = (selected[0],model,selected[3])
        assert best[1].C == info['selection']['selected_C'] and best[2] == info['threshold']
        delta = max(abs(a['validation_f1']-b['validation_f1']) for a,b in zip(candidates,info['selection']['candidates']))
        assert delta < 1e-12
        test_x = scaler.transform(values[test,64:448].reshape(-1,values.shape[-1]).astype(float))
        repeated = best[1].predict_proba(test_x)[:,1].reshape(len(test),384)
        prediction = np.load(dest/'prediction.npz')['probability']
        error = float(np.max(np.abs(repeated-prediction)))
        assert error < 1e-8, (job['id'],error)
        record = dict(job=job['id'],selected_C=best[1].C,threshold=best[2],candidates=candidates,
                      maximum_probability_error=error,maximum_validation_metric_error=delta,
                      model_sha256=sha256(dest/'model.npz'),prediction_sha256=sha256(dest/'prediction.npz'))
        report.append(record)
        print(json.dumps({k:v for k,v in record.items() if k != 'candidates'}),flush=True)
    write_json(ROOT/'docs/results/ppg_formal_expansion/artifact/hand_independent_refit.json',
               dict(status='passed',jobs=len(report),candidate_refits=4*len(report),results=report,
                    audit_code_sha256=sha256(Path(__file__)),protocol_sha256=sha256(folder/'protocol.json'),
                    limitation='Label-free descriptor function reused; all sampling, scaling, sample weights, sklearn fitting and validation selection independently reconstructed.'))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
