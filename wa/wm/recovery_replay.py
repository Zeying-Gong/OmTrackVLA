"""Fail-closed action-prefix replay with causal teacher warm-up.

Collector must use training scenes and identical seed/config/episode state.
RGB equality is necessary, not sufficient: dynamic-state replay checks remain
required in the simulator collector before any branch is admitted to training.
"""
import hashlib
import numpy as np

def rgb_hash(rgb):
    a=np.ascontiguousarray(rgb)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()

def backward_candidates(failure_step, gap=5):
    if failure_step < 1 or gap < 1:
        raise ValueError('positive failure step and gap required')
    return list(range(failure_step-1,-1,-gap)) + ([] if (failure_step-1)%gap==0 else [0])

class ReplayThenTeacher:
    """Teacher observes every prefix frame; its prefix actions are discarded."""
    def __init__(self,teacher,prefix,takeover_step,rgb_key):
        if not 0<=takeover_step<=len(prefix):
            raise ValueError('takeover outside recorded prefix')
        self.teacher=teacher;self.prefix=prefix;self.takeover=takeover_step
        self.rgb_key=rgb_key;self.step=0;self.last_trajectory=None
    def __getattr__(self,key):
        return getattr(self.teacher,key)
    def reset(self,*args,**kwargs):
        self.step=0;self.last_trajectory=None
        return self.teacher.reset(*args,**kwargs)
    def act(self,observations,detector,episode_id,instruction=None):
        t=self.step
        if t<=self.takeover and t<len(self.prefix):
            if rgb_hash(observations[self.rgb_key])!=self.prefix[t]['rgb_sha256']:
                raise RuntimeError(f'REPLAY_DIVERGED at step {t}')
        expert=np.asarray(self.teacher.act(observations,detector,episode_id,instruction),dtype=float)
        if expert.shape!=(3,) or not np.isfinite(expert).all():
            raise RuntimeError('invalid expert action')
        if getattr(self.teacher,'reply_error',None) is not None:
            raise RuntimeError('teacher transport fallback invalidates branch')
        action=np.asarray(self.prefix[t]['action'],dtype=float) if t<self.takeover else expert
        if action.shape!=(3,) or not np.isfinite(action).all() or (abs(action)>1).any():
            raise RuntimeError('invalid normalized action')
        self.last_trajectory=None if t<self.takeover else getattr(self.teacher,'last_trajectory',None)
        self.step+=1
        return action.tolist()

def test():
    class Teacher:
        def __init__(self):self.calls=0
        def act(self,*args):self.calls+=1;return [.2,0,0]
        def reset(self,*args):self.calls=0
    rgb=np.zeros((2,2,3),dtype=np.uint8)
    prefix=[dict(rgb_sha256=rgb_hash(rgb),action=[-.1,0,0]) for _ in range(3)]
    teacher=Teacher();agent=ReplayThenTeacher(teacher,prefix,2,'rgb')
    assert agent.act({'rgb':rgb},None,0)==[-.1,0,0]
    assert agent.act({'rgb':rgb},None,0)==[-.1,0,0]
    assert agent.act({'rgb':rgb},None,0)==[.2,0,0] and teacher.calls==3
    agent.reset()
    try:agent.act({'rgb':rgb+1},None,0)
    except RuntimeError:pass
    else:raise AssertionError('divergence must abort')
    assert backward_candidates(12)==[11,6,1,0]
    print('PASS prefix replay; causal teacher warmup; takeover; reset; divergence; candidates')
if __name__=='__main__':test()
