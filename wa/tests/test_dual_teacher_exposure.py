import unittest
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from wa.wm.recovery_mix import RecoveryMix
from wa.tools.audit_dual_teacher_exposure import exposure

class ExposureTests(unittest.TestCase):
    def test_matches_real_dataloader(self):
        for base, count, repeat in ((101,3,4), (110,17,1), (100,40,2)):
            report, counts = exposure(base,count,repeat)
            mix = RecoveryMix(list(range(base)),list(range(base,base+count)),repeat)
            actual = []
            for rank in range(8):
                sampler = DistributedSampler(mix,num_replicas=8,rank=rank,
                    shuffle=True,seed=42,drop_last=True)
                sampler.set_epoch(1)
                for batch in DataLoader(mix,batch_size=2,sampler=sampler,drop_last=True):
                    actual.extend(batch.tolist())
            observed = np.bincount([i-base for i in actual if i>=base],minlength=count)
            np.testing.assert_array_equal(counts,observed)
            self.assertEqual(report['actual_total'],len(actual))
            self.assertEqual(report['actual_base'],sum(i<base for i in actual))
            self.assertLess(report['dropped'],16)

    def test_deterministic(self):
        a,x=exposure(101,17,3)
        b,y=exposure(101,17,3)
        self.assertEqual(a,b)
        np.testing.assert_array_equal(x,y)

    def test_invalid(self):
        for args in ((0,3,1),(3,0,1),(3,3,0),(3,3,33)):
            with self.assertRaises(ValueError):exposure(*args)

if __name__=='__main__':unittest.main()
