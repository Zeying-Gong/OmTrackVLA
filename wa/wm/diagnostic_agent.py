"""Mixed-mode sensor bridge. The policy RPC receives polar UWB, never world poses."""
import json,os
from wa.wm.eval_agent import WAAgent
from wa.wm.sim_uwb import SimulatedUWB

class DiagnosticAgent(WAAgent):
    def __init__(self,url,action_config,mode,trace):
        self.mode=mode;self.trace=trace;self.uwb_sensor=None
        super().__init__(url,action_config)
    def bind_environment(self,env):
        super().bind_environment(env)
        self.uwb_sensor=SimulatedUWB(env) if self.mode=='mixed' else None
    def rpc(self,path,data):
        if path=='/predict' and self.mode=='mixed':
            data=dict(data,uwb=self.uwb_sensor.sample(data['timestamp_s']))
        return super().rpc(path,data)
    def act(self,*args,**kwargs):
        result=super().act(*args,**kwargs)
        controller=os.environ.get('WA_DIAG_CONTROLLER','learned_target_guard_v3')
        if controller=='uwb_heading_v1' and not self.policy_failure_reason:
            from wa.wm.uwb_control import measured_guard
            from evt_text_action_v3.control import pose_action
            d=self.diagnostics
            raw=pose_action(d['xy'],d['yaw'],self.action_scales,self.action_dt,self.integration_dt)
            result,info=measured_guard(raw,d['uwb']['polar'],self.action_scales,self.action_dt,self.integration_dt)
            self.diagnostics=dict(d,controller=controller,control_info=info,normalized_action=result)
        elif controller not in ('learned_target_guard_v3','uwb_heading_v1'):raise ValueError('unknown controller')
        with self.trace.open('a') as f:
            f.write(json.dumps(dict(step=self.sim_step,mode=self.mode,diagnostics=self.diagnostics,
                failure=self.policy_failure_reason,action=result),allow_nan=False)+'\n')
        return result
