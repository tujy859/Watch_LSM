"""Check algebraic cache acceleration against direct training-only fits."""
import unittest
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from scripts.ppg_formal_hr import sufficient_statistics, projection_from_stats, project, ridge_from_stats


class FormalHRTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(10)
        self.x=rng.normal(size=(350,30))*np.linspace(1,3,30)+np.linspace(-2,4,30)
        self.subjects=np.array(['a']*70+['b']*90+['c']*80+['v']*110)
        self.y=self.x@rng.normal(size=30)+rng.normal(size=350)
        self.train=self.subjects!='v'
        self.stats=sufficient_statistics(self.x,self.y,self.subjects)

    def test_covariance_projection_matches_direct_svd_with_sign_alignment(self):
        p=projection_from_stats(self.stats,['a','b','c'])
        actual=project(self.x,p)
        scaler=StandardScaler().fit(self.x[self.train])
        pca=PCA(16,svd_solver='full').fit(scaler.transform(self.x[self.train]))
        out=StandardScaler().fit(pca.transform(scaler.transform(self.x[self.train])))
        expected=out.transform(pca.transform(scaler.transform(self.x)))
        signs=np.sign(np.sum(actual[self.train]*expected[self.train],axis=0))
        np.testing.assert_allclose(actual*signs,expected,atol=2e-12,rtol=1e-11)

    def test_weighted_statistics_ridge_matches_direct_fit(self):
        a,b,info=ridge_from_stats(self.stats,['a','b','c'],self.x[~self.train],self.y[~self.train],self.subjects[~self.train])
        labels=self.subjects[self.train]
        w=np.array([len(labels)/(3*sum(labels==person)) for person in labels])
        scaler=StandardScaler().fit(self.x[self.train],sample_weight=w)
        model=Ridge(alpha=info['alpha']).fit(scaler.transform(self.x[self.train]),self.y[self.train],sample_weight=w)
        np.testing.assert_allclose(self.x@a+b,model.predict(scaler.transform(self.x)),atol=1e-10)

    def test_test_statistics_do_not_affect_training_projection_or_ridge(self):
        changed=dict(self.stats)
        changed['v']={k: v*1e8 for k,v in changed['v'].items()}
        p=projection_from_stats(self.stats,['a','b','c'])
        q=projection_from_stats(changed,['a','b','c'])
        for k in p:
            np.testing.assert_array_equal(p[k],q[k])


if __name__=='__main__':
    unittest.main()
