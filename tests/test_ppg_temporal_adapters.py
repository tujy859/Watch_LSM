import unittest
import torch
from scripts.ppg_temporal_adapters import make_head, PAPAGEI_CHANNELS
from scripts.audit_ppg_formal_hr import exact_sign_p


class TemporalAdapterTests(unittest.TestCase):
    def test_dense_heads_have_fixed_comparable_capacity(self):
        a=make_head('AnyPPG')
        p=make_head('PaPaGei-P')
        self.assertEqual(sum(x.numel() for x in a.parameters()),26737)
        self.assertEqual(sum(x.numel() for x in p.parameters()),27367)
        self.assertEqual(sum(x.numel() for x in make_head('AnyPPG-random-frozen').parameters()),26737)

    def test_papagei_head_preserves_dense_clock_and_backpropagates(self):
        p=make_head('PaPaGei-S')
        stages=[torch.randn(2,c,t) for c,t in zip(PAPAGEI_CHANNELS,(1000,250,63,16,4,2))]
        out=p(stages)
        self.assertEqual(tuple(out.shape),(2,512))
        out.square().mean().backward()
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in p.parameters()))

    def test_missing_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'six'):
            make_head('PaPaGei-P')([torch.zeros(1,32,10)])

    def test_exact_sign_flip_probability(self):
        self.assertEqual(exact_sign_p([1.,1.,1.]),.25)
        self.assertEqual(exact_sign_p([1.,-1.]),1.)


if __name__=='__main__':
    unittest.main()
