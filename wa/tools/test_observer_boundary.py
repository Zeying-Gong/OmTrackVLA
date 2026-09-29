import sys,types,tempfile,json,os
from pathlib import Path
import numpy as np
class Base:
 def __init__(self,*args):self.policy_failure_reason=None;self.sim_step=0;self.diagnostics={};self.sent=[]
 def bind_environment(self,env):pass
 def rpc(self,path,data):self.sent.append(data);return {}
 def act(self,*args,**kwargs):
  self.rpc('/predict',dict(timestamp_s=0,rgb_png='test',initial_bbox=[0,0,1,1]));self.sim_step+=1
  return [.1,0,0]
m=types.ModuleType('wa.wm.eval_agent');m.WAAgent=Base;sys.modules['wa.wm.eval_agent']=m
from wa.wm.diagnostic_agent import DiagnosticAgent
T=types.SimpleNamespace
robot=T(base_pos=np.array([0,0,0]),sim_obj=T(transformation=T(rotation=lambda:np.eye(3))))
target=T(base_pos=np.array([2,0,0]))
env=T(sim=T(agents_mgr=[T(articulated_agent=target),T(articulated_agent=robot)]))
with tempfile.TemporaryDirectory() as d:
 records=[]
 for value in ('0','1'):
  os.environ['WA_DIAG_OBSERVER_STATE']=value
  a=DiagnosticAgent('',{},'mixed',Path(d)/value);a.bind_environment(env)
  action=a.act();record=json.loads((Path(d)/value).read_text())
  assert set(a.sent[0])=={'timestamp_s','rgb_png','initial_bbox','uwb'}
  records.append((action,a.sent,record))
 assert records[0][:2]==records[1][:2]
 assert records[0][2]['observer_state'] is None
 assert records[1][2]['observer_state']['robot_position_world']==[0,0,0]
 print('PASS: mock runtime observer on/off action and RPC identical; actual simulator rerun pending')
