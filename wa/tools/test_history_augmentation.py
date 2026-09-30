import unittest
import torch
from wa.wm.history_augmentation import repeat_current_history,verify_preserved

class HistoryTests(unittest.TestCase):
 def batch(self):
  b={'rgb':torch.arange(8*4*3*2*2).reshape(8,4,3,2,2).float(),'times':torch.tensor([[-1.5,-1.,-.5,0.]]).repeat(8,1)}
  for k in ('template','polar','pose','geometry','wm_rgb','commands','proprio','future'):b[k]=torch.randn(8,2)
  return b
 def test_zero_identity_rng(self):
  b=self.batch();state=torch.random.get_rng_state();self.assertIs(repeat_current_history(b,0),b);self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
 def test_all_repeat_preserve(self):
  b=self.batch();old=b['rgb'].clone();a=repeat_current_history(b,1)
  self.assertTrue(torch.equal(a['rgb'],b['rgb'][:,-1:].expand_as(b['rgb'])))
  self.assertTrue(torch.equal(a['times'],torch.zeros_like(b['times'])))
  self.assertTrue(verify_preserved(b,a));self.assertTrue(torch.equal(old,b['rgb']))
 def test_reproducible(self):
  b=self.batch();a=repeat_current_history(b,.25,torch.Generator().manual_seed(7));c=repeat_current_history(b,.25,torch.Generator().manual_seed(7));self.assertTrue(torch.equal(a['rgb'],c['rgb']))
 def test_invalid(self):
  for p in [-1,2,float('nan')]:
   with self.assertRaises(ValueError):repeat_current_history(self.batch(),p)
  b=self.batch();b['times'][0,0]=1
  with self.assertRaises(ValueError):repeat_current_history(b,1)
if __name__=='__main__':unittest.main()
