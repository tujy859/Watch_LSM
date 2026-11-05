"""Guard point-label semantics and label-blind frame selection."""
import unittest
import numpy as np
from scripts.ppg_formal_artifact import counts,local_features,person_f1,select_threshold,subset
from scripts.summarize_ppg_formal_artifact import exact_signflip,holm,history_check,point_metrics,validation_selection


class ArtifactProtocolTests(unittest.TestCase):
    def test_frame_selection_is_even_per_person_and_never_crosses(self):
        subjects=np.repeat(['a','b','c'],[12,4,7])
        ix=subset(subjects,['a','b'],3)
        np.testing.assert_array_equal(ix,[0,5,11,12,13,15])
        self.assertEqual(set(subjects[ix]),{'a','b'})

    def test_false_positives_in_clean_points_are_counted(self):
        value=counts([0,0,1,1],[.9,.1,.9,.1])
        self.assertEqual([value[k] for k in ('tp','fp','fn','tn')],[1,1,1,1])
        self.assertEqual(value['f1'],.5)

    def test_people_receive_equal_weight_not_point_counts(self):
        y=np.array([[1,1],[1,1],[1,1]])
        p=np.array([[0,0],[1,1],[1,1]])
        self.assertEqual(person_f1(y,p,np.array(['a','b','b']),.5),.5)

    def test_local_features_keep_flatline_finite_and_shape(self):
        x=np.ones((2,512));x[1]=np.sin(np.arange(512)/10)
        z=local_features(x)
        self.assertEqual(z.shape,(2,512,29));self.assertTrue(np.isfinite(z).all())

    def test_threshold_tie_is_nearest_half(self):
        self.assertEqual(select_threshold(np.ones((2,384)),np.ones((2,384)),np.array(['a','b'])),.5)

    def test_exact_signflip_keeps_people_as_the_unit(self):
        self.assertEqual(exact_signflip([1,1,1]), .25)
        self.assertEqual(exact_signflip([1,-1]), 1.)
        self.assertEqual(exact_signflip([0,0,0]), 1.)

    def test_holm_is_monotone_in_sorted_order_and_restores_input_order(self):
        np.testing.assert_allclose(holm([.04,.001,.03,.9]), [.09,.004,.09,.9])

    def test_independent_metrics_count_clean_false_positives(self):
        value=point_metrics([0,0,1,1],[.9,.1,.9,.1],.5)
        self.assertEqual([value[k] for k in ('tp','fp','fn','tn')],[1,1,1,1])
        self.assertEqual(value['f1'],.5)
        self.assertEqual(value['balanced_accuracy'],.5)

    def test_independent_validation_ties_keep_nearest_half(self):
        threshold,score=validation_selection(np.ones((2,3)),np.ones((2,3)),np.array(['a','b']))
        self.assertEqual((threshold,score),(.5,1.))

    def test_epoch_check_rejects_a_nonbest_saved_model(self):
        info=dict(history=[dict(epoch=i,validation_f1=1/i,threshold=.5) for i in range(1,7)],
                  best_epoch=1,validation_f1=1.,threshold=.5)
        history_check(info)
        info.update(best_epoch=2,validation_f1=.5)
        with self.assertRaisesRegex(ValueError,'validation-best'):
            history_check(info)

    def test_epoch_check_rejects_premature_stopping(self):
        info=dict(history=[dict(epoch=i,validation_f1=1/i,threshold=.5) for i in range(1,4)],
                  best_epoch=1,validation_f1=1.,threshold=.5)
        with self.assertRaisesRegex(ValueError,'patience'):
            history_check(info)


if __name__=='__main__':unittest.main()
