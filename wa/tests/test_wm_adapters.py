import unittest
import torch
from torch import nn
from wa.wm.adapters import polar_features,GoalMetaQueryAdapter

class QueryFixture(nn.Module):
    def __init__(self):
        super().__init__();self.q=nn.Parameter(torch.randn(64,24)*.02)
    def forward(self):return self.q

class ContractTests(unittest.TestCase):
    def test_polar_wrap_and_zero_distance(self):
        p=torch.tensor([[0.,torch.pi],[0.,-torch.pi]])
        f,v=polar_features(p,torch.ones(2,dtype=torch.bool),torch.zeros(2))
        self.assertTrue(v.all());torch.testing.assert_close(f[0],f[1],atol=1e-6,rtol=0)

    def test_invalid_and_stale_are_finite_zero(self):
        f,v=polar_features(torch.full((2,2),float('nan')),torch.tensor([False,True]),torch.tensor([float('nan'),1.]))
        self.assertFalse(v.any());self.assertTrue(torch.equal(f,torch.zeros_like(f)))

    def test_bad_valid_rejected(self):
        for p in [torch.tensor([[-1.,0.]]),torch.tensor([[1.,float('nan')]])]:
            with self.assertRaises(ValueError):polar_features(p,torch.tensor([True]),torch.zeros(1))

    def test_modes_masks_and_causality(self):
        torch.manual_seed(7);torch.set_num_threads(2)
        model=GoalMetaQueryAdapter(QueryFixture(),backbone_dim=24).eval()
        batch=dict(visual=torch.randn(1,2,256,384),template=torch.randn(1,256,384),template_valid=torch.tensor([True]),polar=torch.tensor([[2.,.3]]),uwb_valid=torch.tensor([True]),age_s=torch.zeros(1),times=torch.tensor([[-.1,0.]]))
        with torch.no_grad():
            mixed,_=model(**batch)
            self.assertEqual(len(mixed),28);self.assertEqual(mixed[-1].shape,(1,64,24))
            image=dict(batch,uwb_valid=torch.tensor([False]))
            a,_=model(**image)
            b,_=model(**dict(image,polar=torch.full((1,2),float('nan')),age_s=torch.tensor([float('nan')])))
            torch.testing.assert_close(a[-1],b[-1],atol=0,rtol=0)
            point=dict(batch,template_valid=torch.tensor([False]))
            c,_=model(**point)
            d,_=model(**dict(point,template=torch.full_like(batch['template'],float('nan'))))
            torch.testing.assert_close(c[-1],d[-1],atol=0,rtol=0)
            self.assertGreater((mixed[-1]-a[-1]).abs().mean().item(),0)
        with self.assertRaises(ValueError):model(**dict(batch,times=torch.tensor([[-.1,.1]])))
        with self.assertRaises(ValueError):model(**dict(point,uwb_valid=torch.tensor([False])))

if __name__=='__main__':unittest.main()
