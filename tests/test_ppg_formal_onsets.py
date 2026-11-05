"""Regression guards for formal onset selection and tiny-cohort inference."""
import copy
import unittest

import numpy as np

from scripts.audit_ppg_formal_onsets import verify_selection
from scripts.audit_ppg_formal_hr import exact_sign_p
from scripts.audit_ppg_formal_onsets_independent import matches,variability


class FormalOnsetAuditTests(unittest.TestCase):
    def setUp(self):
        self.rows={'subject':np.array(['S1','S1','S2','S2','S3'])}
        self.job={'train':['S1'],'validation':['S2'],'test':['S3']}
        self.thresholds=[.2,.5]
        self.fit={'train_frame_indices':[0,1],'validation_frame_indices':[2,3],
                  'train_frames':2,'validation_frames':2,'best_epoch':1,
                  'threshold':.2,'validation_f1_50ms':.9,
                  'history':[{'epoch':1,'threshold':.2,'validation_f1_50ms':.9,
                              'threshold_scores':{'0.2':.9,'0.5':.9}},
                             {'epoch':2,'threshold':.5,'validation_f1_50ms':.9,
                              'threshold_scores':{'0.2':.8,'0.5':.9}}]}

    def test_first_threshold_and_epoch_win_ties(self):
        verify_selection(self.fit,self.job,self.rows,self.thresholds)

    def test_test_person_frame_cannot_enter_training(self):
        bad=copy.deepcopy(self.fit)
        bad['train_frame_indices']=[0,4]
        with self.assertRaisesRegex(ValueError,'frame identities'):
            verify_selection(bad,self.job,self.rows,self.thresholds)

    def test_later_tied_epoch_is_rejected(self):
        bad=copy.deepcopy(self.fit)
        bad.update(best_epoch=2,threshold=.5)
        with self.assertRaisesRegex(ValueError,'Epoch'):
            verify_selection(bad,self.job,self.rows,self.thresholds)

    def test_nonmaximal_and_nonfinite_threshold_scores_rejected(self):
        for scores in ({'0.2':.8,'0.5':.9},{'0.2':np.nan,'0.5':.9}):
            bad=copy.deepcopy(self.fit)
            bad['history'][0]['threshold_scores']=scores
            with self.assertRaisesRegex(ValueError,'Threshold'):
                verify_selection(bad,self.job,self.rows,self.thresholds)

    def test_seven_people_cannot_pass_eight_test_holm_family(self):
        self.assertEqual(exact_sign_p(np.ones(7)),.015625)
        self.assertEqual(8*exact_sign_p(np.ones(7)),.125)

    def test_independent_matches_keep_boundary_and_count_no_event_twice(self):
        self.assertEqual(matches([1.,1.08],[1.04],.05),1)
        self.assertEqual(matches([1.,2.],[1.05,2.05],.05),2)
        self.assertEqual(matches([1.,2.],[],.05),0)
        self.assertEqual(matches([1.,1.08],[1.04,1.12],.05),2)

    def test_independent_variability_preserves_missing_beat_interval(self):
        values=variability([0.,1.,2.,4.])
        self.assertAlmostEqual(values['mean_pp_ms'],4000/3)
        self.assertAlmostEqual(values['sdpp_ms'],1000/np.sqrt(3))
        self.assertAlmostEqual(values['rmssd_ms'],1000/np.sqrt(2))


if __name__=='__main__':
    unittest.main()
