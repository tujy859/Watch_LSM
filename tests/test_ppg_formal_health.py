"""Guard participant isolation, common observation, and label-free fitting."""
import unittest

import numpy as np

from scripts.ppg_formal_health import (
    ALPHAS, PROTOCOL, fit_head, fit_projection, person_folds, prepared_group,
)
from scripts.audit_ppg_formal_health import holm,paired_inference,independent_input_rebuild


class FormalHealthTests(unittest.TestCase):
    def test_five_test_folds_partition_people_and_inner_validation_is_disjoint(self):
        people = [f"p{i:03}" for i in range(60)]
        folds = person_folds(people)
        self.assertEqual(folds, person_folds(people[::-1]))
        self.assertEqual(sorted(sum((x["test"] for x in folds), [])), people)
        for fold in folds:
            self.assertEqual([len(fold[k]) for k in ("train","validation","test")], [36,12,12])
            self.assertFalse(set(fold["train"]) & set(fold["validation"]))
            self.assertFalse(set(fold["train"]) & set(fold["test"]))

    def test_duplicate_people_cannot_enter_fold_builder(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            person_folds([str(i) for i in range(20)]+["1"])

    def test_pca_and_missing_scaling_are_training_only(self):
        raw=np.random.default_rng(1).normal(size=(30,12)); train=np.arange(20)
        reduced,state=fit_projection(raw,train)
        changed=raw.copy(); changed[20:]+=1000
        reduced2,state2=fit_projection(changed,train)
        for key in state:
            np.testing.assert_array_equal(state[key],state2[key])
        np.testing.assert_array_equal(reduced[:20],reduced2[:20])
        x=np.array([[1.,np.nan],[3.,np.nan],[1000.,8.]])
        z,prep=prepared_group(x,np.array([0,1]),8)
        np.testing.assert_array_equal(prep["imputation_median"],[2.,0.])
        self.assertEqual(z.shape,(3,8))
        np.testing.assert_array_equal(z[:,4:],0)

    def test_test_labels_cannot_select_linear_regularization_or_prediction(self):
        rng=np.random.default_rng(2); z=rng.normal(size=(30,10)); y=z[:,0]*3+rng.normal(size=30)
        train=np.arange(18); val=np.arange(18,24); test=np.arange(24,30)
        a,_,meta=fit_head(z,y,train,val,test,"linear",42)
        y2=y.copy(); y2[test]+=1000
        b,_,meta2=fit_head(z,y2,train,val,test,"linear",42)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(meta["alpha"],meta2["alpha"])
        self.assertIn(meta["alpha"],ALPHAS)
        self.assertLess(meta["independent_prediction_max_difference"],1e-9)

    def test_mlp_uses_same_width_and_test_labels_do_not_choose_epoch(self):
        rng=np.random.default_rng(3); z=rng.normal(size=(30,10)); y=z[:,0]*3+rng.normal(size=30)
        train=np.arange(18); val=np.arange(18,24); test=np.arange(24,30)
        a,_,meta=fit_head(z,y,train,val,test,"mlp",42)
        y2=y.copy(); y2[test]-=999
        b,_,meta2=fit_head(z,y2,train,val,test,"mlp",42)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(meta["epochs"],meta2["epochs"])
        self.assertEqual(meta["trainable_head_parameters"],193)
        self.assertLess(meta["independent_prediction_max_difference"],1e-5)

    def test_overlapping_person_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"Overlapping"):
            fit_head(np.ones((10,2)),np.arange(10),[0,1,2],[2,3],[4,5],"linear",42)

    def test_holm_correction_preserves_original_model_order(self):
        np.testing.assert_allclose(holm([.04,.001,.02]),[.04,.003,.04])
        with self.assertRaisesRegex(ValueError,"Invalid"):
            holm([1.2])

    def test_paired_inference_identical_methods_has_zero_difference(self):
        result=paired_inference(np.zeros(20))
        self.assertEqual(result["difference_mae"],0.)
        self.assertEqual(result["ci_low"],0.)
        self.assertEqual(result["p_two_sided"],1.)

    def test_person_cnn_pools_clips_before_prediction_and_is_order_invariant(self):
        import torch
        from scripts.ppg_formal_health_cnn import PersonCNN
        torch.set_num_threads(2)
        with torch.random.fork_rng():
            torch.manual_seed(9)
            model=PersonCNN(4).eval()
            x=torch.randn(2,3,1,128); clinical=torch.randn(2,4)
            with torch.no_grad():
                a=model(x,clinical); b=model(x[:,[2,0,1]],clinical)
        self.assertEqual(tuple(a.shape),(2,))
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        with self.assertRaisesRegex(ValueError,"person"):
            model(x.reshape(6,1,128),clinical)

    def test_auxiliary_age_target_is_removed_from_covariates(self):
        from scripts.ppg_formal_age import age_arrays
        source={k:np.zeros((2,3)) for k in ("subjects","traditional","windows","waveform_subjects","stats")}
        source["covariate_names"]=["age_years","male_indicator","bmi_kg_m2"]
        source["covariates"]=np.array([[29.,1.,21.],[61.,0.,25.]])
        result=age_arrays(source)
        np.testing.assert_array_equal(result["targets"],[[29.],[61.]])
        np.testing.assert_array_equal(result["covariates"],[[1.,21.],[0.,25.]])

    def test_audit_rejects_age_column_in_saved_predictor_inputs(self):
        data={"covariates":np.array([[1.,21.],[0.,25.],[1.,24.],[0.,28.]]),
              "traditional":np.arange(4.)[:,None]}
        train=np.array([0,1,2]); row={"inputs":"demographics"}
        z,state=prepared_group(data["covariates"],train,8)
        saved={**state,"projected_features":np.zeros((4,8)),"prepared_features":z}
        self.assertLess(independent_input_rebuild(data,row,saved,train),1e-12)
        saved["prepared_features"]=z.copy();saved["prepared_features"][:,-1]=[29.,61.,35.,70.]
        self.assertGreater(independent_input_rebuild(data,row,saved,train),1.)


if __name__=="__main__":
    unittest.main()
