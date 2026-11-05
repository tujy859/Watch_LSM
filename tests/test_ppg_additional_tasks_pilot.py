"""Guard person isolation and reference semantics in the task pilots."""
import unittest
import numpy as np
from scripts.ppg_additional_tasks_pilot import manual_rate, roles_for, split_people, fit_one, metric
from scripts.prepare_ppg_expansion_assets import checked_destination
from scripts.ppg_denoising_data_pilot import waveform_metrics


class AdditionalTaskPilotTests(unittest.TestCase):
    def test_breath_samples_are_one_based_with_half_open_boundaries(self):
        # Sample 4000 is 31.992 sec and must not enter the 32-sec window.
        self.assertAlmostEqual(manual_rate([4000,4001,5001,6001],32,49),7.5)
        # Right endpoint at sample 6001 (48 sec) is excluded, leaving only two breaths.
        self.assertTrue(np.isnan(manual_rate([4000,4001,5001,6001],32,48)))

    def test_invalid_breath_reference(self):
        self.assertTrue(np.isnan(manual_rate([1,1,251],0,10)))
        self.assertTrue(np.isnan(manual_rate([1,251],0,10)))

    def test_real_person_group_not_record_identifier(self):
        split=split_people([f'p{i}' for i in range(8)]+['p0']*3)
        subjects=['p0','p0','p4','p6']
        roles=roles_for(subjects,split)
        self.assertEqual(roles.tolist(),['train','train','validation','test'])

    def test_test_labels_do_not_change_fit(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(24,3))
        y=x[:,0]+.1*rng.normal(size=24)
        subjects=np.repeat(['a','b','c','d','e','f'],4)
        roles=np.repeat(['train','train','train','validation','test','test'],4)
        p,info=fit_one(x,y,subjects,roles,'regression')
        changed=y.copy();changed[roles=='test']+=10000
        p2,info2=fit_one(x,changed,subjects,roles,'regression')
        np.testing.assert_array_equal(p,p2)
        self.assertEqual(info,info2)

    def test_metric_weights_people_equally(self):
        self.assertEqual(metric(np.array([0,0,0]),np.array([10,0,0]),np.array(['a','b','b']),'regression'),5)

    def test_download_destination_rejects_escape(self):
        with self.assertRaises(ValueError):
            checked_destination('../outside')

    def test_waveform_metrics_are_paired(self):
        result=waveform_metrics(np.array([-1.,0.,1.]),np.array([-1.,.5,1.]))
        self.assertAlmostEqual(result['mae'],1/6)
        self.assertAlmostEqual(result['snr_db'],10*np.log10(8))


if __name__=='__main__':
    unittest.main()
