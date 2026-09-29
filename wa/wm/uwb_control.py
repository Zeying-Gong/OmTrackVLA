"""Experimental measured-UWB guard; does not replace learned translation planning.

Hypothesis test, NOT a proven collision-avoidance controller. Keeps the original
1.5m approach limit and yaw gain/cap; removes the zero far-heading correction.
Only currently measured polar range/bearing and original actuator constants.
"""
import numpy as np
from evt_text_action_v3.control import target_control

def measured_guard(base_action,polar,scales,action_dt,integration_dt):
    r,bearing=map(float,polar)
    if not np.isfinite([r,bearing]).all() or r<0:raise ValueError('invalid UWB')
    result,info=target_control(base_action,[np.log(max(r,1e-6)),np.cos(bearing),np.sin(bearing)],scales,action_dt,integration_dt)
    blend=max(.5,float(np.clip((3.-r)/1.5,0,1)))
    desired=float(np.clip(2*bearing,-1.5,1.5))*action_dt/(integration_dt*scales[2])
    result[2]=float(np.clip((1-blend)*base_action[2]+blend*desired,-1,1))
    info.update(source='ideal_simulated_uwb',heading_blend=blend,experimental_controller='uwb_heading_v1')
    return result,info
