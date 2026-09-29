"""Reuse the original WLA v3 SE2 controller; never forward text or later target hints."""
import base64,io
import numpy as np
from PIL import Image
from trained_agent import DA3EVTAgent
from evt_text_action_v3.eval_agent import WLAAgent
from evt_text_action_v3.control import pose_action,target_control

class WAAgent(WLAAgent):
    def act(self,observations,detector,episode_id,instruction=None,timestamp_s=None):
        if self.policy_failure_reason is not None:self.sim_step+=1;return [0.,0.,0.]
        rgb=np.asarray(observations['agent_1_articulated_agent_jaw_rgb'])[...,:3]
        box=None
        if self.sim_step==0:
            try:
                box=DA3EVTAgent._bbox(detector)
                x1,y1,x2,y2=box
                if not np.isfinite(box).all() or not 0<=x1<x2<=rgb.shape[1] or not 0<=y1<y2<=rgb.shape[0]:raise ValueError('invalid RGB bbox')
            except (ValueError,RuntimeError) as e:
                self.initialization_valid=False;self.policy_failure_reason='invalid_initialization_bbox: '+str(e);self.sim_step+=1;return [0.,0.,0.]
        stream=io.BytesIO();Image.fromarray(rgb.astype(np.uint8)).save(stream,format='PNG')
        prediction=self.rpc('/predict',dict(rgb_png=base64.b64encode(stream.getvalue()).decode(),timestamp_s=timestamp_s,initial_bbox=box))
        if self.last_time is not None:
            self.action_dt=float(timestamp_s)-self.last_time
            if not 0<self.action_dt<=.2:raise ValueError('invalid simulator action interval')
        self.last_time=float(timestamp_s)
        xy=np.asarray(prediction['xy'])
        action=pose_action(xy,prediction['yaw'],self.action_scales,self.action_dt,self.integration_dt)
        action,info=target_control(action,prediction['target_geometry'],self.action_scales,self.action_dt,self.integration_dt)
        self.last_trajectory=xy;self.diagnostics=dict(prediction,controller='learned_target_guard_v3',control_info=info,normalized_action=action,action_dt_s=self.action_dt)
        self.sim_step+=1;return action
