"""Explicit simulator UWB sensor, not a policy access to simulator state.

Ideal polar samples reproduce the TRAINING simulation approximation. This is
not real UWB validation. No future position, path, text or semantic ID is emitted.
Robot coordinates: +X forward, -Z left; bearing positive left in radians.
"""
import numpy as np

def polar_from_world(robot_position, rotation_world_from_body, target_position):
    p=np.asarray(robot_position,dtype=float);r=np.asarray(rotation_world_from_body,dtype=float);t=np.asarray(target_position,dtype=float)
    if p.shape!=(3,) or t.shape!=(3,) or r.shape!=(3,3) or not all(np.isfinite(v).all() for v in (p,r,t)):
        raise ValueError('invalid UWB sensor transform')
    local=(t-p)@r
    return [float(np.hypot(local[0],local[2])),float(np.arctan2(-local[2],local[0]))]

class SimulatedUWB:
    def __init__(self,env):self.env=env
    def sample(self,timestamp_s):
        sim=self.env.sim
        robot=sim.agents_mgr[1].articulated_agent
        target=sim.agents_mgr[0].articulated_agent
        return dict(polar=polar_from_world(robot.base_pos,robot.sim_obj.transformation.rotation(),target.base_pos),
                    timestamp_s=float(timestamp_s),valid=True,source='ideal_simulated_uwb',range_noise_m=0.,bearing_noise_rad=0.,delay_s=0.)
