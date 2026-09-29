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
        self.diagnostic_env=env
        self.uwb_sensor=SimulatedUWB(env) if self.mode=='mixed' else None
    def rpc(self,path,data):
        if path=='/predict' and self.mode=='mixed':
            data=dict(data,uwb=self.uwb_sensor.sample(data['timestamp_s']))
        return super().rpc(path,data)
    def act(self,*args,**kwargs):
        # Observer-only evidence. Never include this object in predict RPC.
        observer_state=None
        if os.environ.get('WA_DIAG_OBSERVER_STATE')=='1':
            import numpy as np
            sim=self.diagnostic_env.sim
            robot=sim.agents_mgr[1].articulated_agent
            target=sim.agents_mgr[0].articulated_agent
            observer_state=dict(
                usage='offline_diagnosis_only_not_policy_input',
                timing='before_current_action_after_previous_action',
                robot_position_world=np.asarray(robot.base_pos,dtype=float).tolist(),
                robot_rotation_world_from_body=np.asarray(robot.sim_obj.transformation.rotation(),dtype=float).tolist(),
                target_position_world=np.asarray(target.base_pos,dtype=float).tolist())
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
                failure=self.policy_failure_reason,action=result,observer_state=observer_state),allow_nan=False)+'\n')
        return result
