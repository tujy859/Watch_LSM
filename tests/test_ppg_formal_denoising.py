"""Check synthetic-pair construction and inference-only normalization rules."""
import unittest

import numpy as np
import torch

from scripts.ppg_formal_denoising import (
    synthetic_pair,noisy_normalization,finalized_arrays,metrics,_network,
)
from scripts.audit_ppg_formal_denoising import metrics as audit_metrics


class FormalDenoisingTests(unittest.TestCase):
    def test_noise_energy_matches_declared_snr_and_seed_is_identity_stable(self):
        reference=np.sin(2*np.pi*np.arange(1000)/125)+3
        for snr in (-5.,5.,15.):
            noisy,clean=synthetic_pair(reference,snr,"person-record-start-level"+str(snr))
            measured=10*np.log10(np.mean(clean.astype(float)**2)/np.mean((noisy-clean).astype(float)**2))
            self.assertAlmostEqual(measured,snr,places=5)
            a,b=synthetic_pair(reference,snr,"person-record-start-level"+str(snr))
            np.testing.assert_array_equal(a,noisy); np.testing.assert_array_equal(b,clean)

    def test_reference_change_cannot_change_inference_normalization(self):
        noisy=np.sin(np.arange(1000)[None,:]/20)+np.arange(1000)[None,:]/300
        reference=np.sin(np.arange(1000)[None,:]/20)
        args=(["p"],[{"subject":"p"}],[60.])
        a,_=finalized_arrays(noisy,reference,*args)
        b,_=finalized_arrays(noisy,reference*100+90,*args)
        for key in ("x125","x64","baseline","noisy_mean","noisy_std"):
            np.testing.assert_array_equal(a[key],b[key])
        self.assertFalse(np.array_equal(a["target"],b["target"]))

    def test_noisy_input_local_mean_and_scale_only(self):
        x=np.array([[1.,2.,3.,4.],[10.,12.,14.,16.]])
        z,mean,std=noisy_normalization(x)
        np.testing.assert_allclose(z.mean(axis=1),0,atol=1e-7)
        np.testing.assert_allclose(z.std(axis=1),1,atol=1e-7)
        np.testing.assert_array_equal(mean,x.mean(axis=1))
        np.testing.assert_array_equal(std,x.std(axis=1))
        with self.assertRaisesRegex(ValueError,"nonconstant"):
            noisy_normalization(np.zeros((1,512)))

    def test_metrics_use_fixed_central_region_without_gain_fit(self):
        x=np.sin(2*np.pi*np.arange(512)[None,:]/64).repeat(2,axis=0)
        y=x.copy(); y[:,:64]+=100; y[:,448:]-=100
        m=metrics(x,y,np.array([60.,60.]))
        np.testing.assert_array_equal(m["nrmse"],0)
        m=metrics(x,x*2,np.array([60.,60.]))
        np.testing.assert_allclose(m["nrmse"],1)
        np.testing.assert_allclose(m["correlation"],1)

    def test_separate_audit_metric_formula_matches_known_gain_error(self):
        x=np.sin(2*np.pi*np.arange(512)[None,:]/64)
        m=audit_metrics(x,x*1.5,np.array([60.]))
        np.testing.assert_allclose(m["nrmse"],.5)
        np.testing.assert_allclose(m["snr_db"],20*np.log10(2))
        np.testing.assert_allclose(m["correlation"],1)

    def test_residual_head_initially_preserves_fixed_filter_baseline(self):
        torch.set_num_threads(2)
        model=_network("SmallCNN").eval()
        with torch.no_grad():
            correction=model(torch.randn(2,1,512))
        torch.testing.assert_close(correction,torch.zeros_like(correction),rtol=0,atol=0)


if __name__=="__main__":
    unittest.main()
