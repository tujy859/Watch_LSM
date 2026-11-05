"""Person separation, complete observation and held-out classifier guards."""
import unittest

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from scripts.ppg_formal_classification import (
    fit_linear, indices, linear_replay, paired_views, project, projection,
    score, split_jobs,
)
from scripts.summarize_ppg_formal_classification import holm, weighted_auc


class FormalClassificationTests(unittest.TestCase):
    def test_weighted_auc_preserves_ties_and_cluster_weights(self):
        y=np.array([0,1,1,0,0,1]);p=np.array([.2,.2,.5,.5,.5,.8]);w=np.array([1,3,1,1,2,4])
        self.assertAlmostEqual(weighted_auc(y,p,w),roc_auc_score(y,p,sample_weight=w))
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])

    def test_two_views_cover_entire_record_without_crossing_record(self):
        x=np.arange(2500).reshape(2,1250)
        v=paired_views(x,125)
        np.testing.assert_array_equal(v[:,0],x[:,:1000])
        np.testing.assert_array_equal(v[:,1],x[:,250:])
        self.assertEqual(v.shape,(2,2,1000))
        with self.assertRaises(ValueError): paired_views(x[:,:1000],125)

    def test_each_person_tested_once_per_seed(self):
        subjects=np.repeat([f'p{i}' for i in range(15)],3)
        jobs=split_jobs(subjects,'WESAD')
        for seed in (42,43,44):
            seen=[]
            for job in jobs:
                indices(subjects,job)
                if job['seed']==seed: seen.extend(job['test'])
            self.assertEqual(sorted(seen),sorted(set(subjects)))
        bad=dict(jobs[0]);bad['test']=bad['train']
        with self.assertRaisesRegex(ValueError,'leakage'): indices(subjects,bad)

    def test_covariance_projection_matches_direct_svd_subspace(self):
        x=np.random.default_rng(4).normal(size=(100,24))
        state=projection(x); actual=project(x,state)
        z=StandardScaler().fit_transform(x)
        expected=StandardScaler().fit_transform(PCA(16,svd_solver='full').fit_transform(z))
        np.testing.assert_allclose(actual@actual.T,expected@expected.T,atol=1e-10)
        before=state['mean'].copy();project(x+10000,state)
        np.testing.assert_array_equal(before,state['mean'])

    def test_test_labels_do_not_change_fitted_linear_predictions(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(60,4));y=(x[:,0]>0).astype(int)
        subjects=np.repeat(['a','b','c','d','e','f'],10)
        tr=np.arange(30);va=np.arange(30,40)
        p,meta,state=fit_linear(x,y,subjects,tr,va,'auroc')
        y2=y.copy();y2[40:]=1-y2[40:]
        p2,meta2,_=fit_linear(x,y2,subjects,tr,va,'auroc')
        np.testing.assert_array_equal(p,p2);self.assertEqual(meta,meta2)
        np.testing.assert_allclose(p,linear_replay(x,state),atol=1e-14)

    def test_invalid_probability_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'probability'):
            score(np.array([0,1]),np.array([[.5,.7],[.2,.8]]),np.array(['a','b']))


if __name__=='__main__': unittest.main()
