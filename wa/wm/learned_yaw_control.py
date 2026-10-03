"""Controller ablation: retain learned yaw and legacy translation guard.
Not a unit correction or an adopted safety improvement. Simulator evaluation only.
"""
import numpy as np
from evt_text_action_v3.control import target_control

def learned_yaw_control(raw, target, scales, action_dt=.048, integration_dt=.025):
    raw=np.asarray(raw,dtype=float)
    if raw.shape!=(3,) or not np.isfinite(raw).all() or np.any(abs(raw)>1):
        raise ValueError('expected finite clipped pose action')
    legacy,info=target_control(raw,target,scales,action_dt,integration_dt)
    final=list(legacy)
    final[2]=float(raw[2])
    assert np.array_equal(np.asarray(final[:2]),np.asarray(legacy[:2]))
    assert np.isfinite(final).all() and np.max(np.abs(final))<=1
    return final,dict(info,raw_pose_action=raw.tolist(),legacy_guard_action=legacy,
        learned_yaw_retained=True,ablation_only=True)

def test():
    rng=np.random.default_rng(42)
    for _ in range(10000):
        raw=rng.uniform(-1,1,3);radius=rng.uniform(.05,5);theta=rng.uniform(-np.pi,np.pi)
        target=[np.log(radius),np.cos(theta),np.sin(theta)]
        dt=float(rng.uniform(.025,.15))
        legacy,_=target_control(raw,target,[15,10,6.28],dt,.025)
        final,info=learned_yaw_control(raw,target,[15,10,6.28],dt,.025)
        assert final[:2]==legacy[:2] and final[2]==raw[2] and info['legacy_guard_action']==legacy
    for bad in ([0,0,1.01],[0,float('nan'),0],[0,0]):
        try:learned_yaw_control(bad,[0,1,0],[15,10,6.28])
        except ValueError:pass
        else:raise AssertionError(bad)
    print('PASS 10000 translation invariants; exact learned yaw; bounds/invalid inputs')
if __name__=='__main__':test()
