import unittest
import torch
from wa.model import Policy

class PolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        cls.model = Policy().eval()
        cls.rgb = torch.randn(1,3,3,224,224)
        cls.crop = torch.randn(1,3,224,224)
        cls.point = torch.tensor([[2.,.3,0.]])
        cls.times = torch.tensor([[-1.,-.5,0.]])
    def call(self, rgb=None, crop=None, point=None, iv=True,pv=True):
        return self.model(self.rgb if rgb is None else rgb,self.crop if crop is None else crop,
                          self.point if point is None else point,self.times,
                          torch.tensor([iv]),torch.tensor([pv]))
    def test_shape_and_invalid_point_invariance(self):
        with torch.no_grad():
            a=self.call(pv=False)
            b=self.call(point=torch.full((1,3),float("nan")),pv=False)
        self.assertEqual(a["xy"].shape,(1,7,2))
        torch.testing.assert_close(a["xy"],b["xy"],atol=0,rtol=0)
    def test_point_mode_ignores_template(self):
        with torch.no_grad():
            a=self.call(iv=False)
            b=self.call(crop=torch.zeros_like(self.crop),iv=False)
        torch.testing.assert_close(a["xy"],b["xy"],atol=0,rtol=0)
    def test_causal_prefix(self):
        altered=self.rgb.clone(); altered[:,-1]*=3
        with torch.no_grad():
            a=self.call()["states"][:,:2]
            b=self.call(rgb=altered)["states"][:,:2]
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    def test_prompt_and_world_gradients(self):
        self.model.train()
        out=self.model(self.rgb,self.crop,self.point,self.times,torch.tensor([True]),torch.tensor([True]),
                       future=self.rgb[:,-1],labels=torch.zeros(1,7,2),world_weight=.1)
        out["loss"].backward()
        for module in (self.model.identity,self.model.geometry,self.model.dynamics,self.model.action):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.encoder.parameters()))
        self.model.zero_grad();self.model.eval()
