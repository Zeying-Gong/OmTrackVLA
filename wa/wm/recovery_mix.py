"""Explicit bounded recovery exposure; original data and heldout are unchanged."""
from torch.utils.data import Dataset

class RecoveryMix(Dataset):
    def __init__(self,base,recovery,repeats):
        if not isinstance(repeats,int) or isinstance(repeats,bool) or not 1<=repeats<=32:
            raise ValueError('recovery repeats must be explicit integer in [1,32]')
        if not len(base) or not len(recovery):raise ValueError('empty source')
        self.base,self.recovery,self.repeats=base,recovery,repeats
        self.exposure=dict(base_windows=len(base),recovery_unique_windows=len(recovery),
            recovery_repeats=repeats,recovery_exposures=len(recovery)*repeats,
            total_windows=len(self),recovery_fraction=len(recovery)*repeats/len(self),
            note='Pre-DDP exposure; shuffled DistributedSampler/drop_last may discard final rows.')
    def __len__(self):return len(self.base)+len(self.recovery)*self.repeats
    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        if index<len(self.base):return self.base[index]
        return self.recovery[(index-len(self.base))%len(self.recovery)]

def test():
    from collections import Counter
    from torch.utils.data import DistributedSampler
    data=RecoveryMix(list(range(101)),[101,102,103],4)
    assert Counter(data[i] for i in range(len(data)))==Counter({**{i:1 for i in range(101)},101:4,102:4,103:4})
    shards=[list(DistributedSampler(data,num_replicas=8,rank=r,shuffle=True,seed=42,drop_last=True)) for r in range(8)]
    assert len(set(sum(shards,[])))==sum(map(len,shards))==112
    assert shards[0]==list(DistributedSampler(data,num_replicas=8,rank=0,shuffle=True,seed=42,drop_last=True))
    for bad in (0,-1,33,1.5,True):
        try:RecoveryMix([0],[1],bad)
        except ValueError:pass
        else:raise AssertionError(bad)
    print('PASS exact exposure; no extra keys; deterministic nonoverlapping 8-rank sampling; invalid repeats rejected')
if __name__=='__main__':test()
