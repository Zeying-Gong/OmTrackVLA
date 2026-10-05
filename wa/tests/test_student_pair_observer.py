import os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch
import numpy as np
from wa.wm.diagnostic_agent import DiagnosticAgent
from wa.wm.eval_agent import WAAgent
from wa.wm.recovery_replay import rgb_hash,dynamic_state

class ObserverTests(unittest.TestCase):
    def test_first_only_offline_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent=DiagnosticAgent.__new__(DiagnosticAgent)
            agent.sim_step=0;agent.mode='mixed';agent.trace=Path(tmp)/'trace.jsonl'
            agent.diagnostics={};agent.policy_failure_reason='test_no_policy'
            obj=N(transformation=np.eye(4),joint_positions=np.zeros(2))
            agent.diagnostic_env=N(sim=N(agents_mgr=[N(articulated_agent=N(sim_obj=obj))],get_world_time=lambda:0.))
            rgb=np.arange(48,dtype=np.uint8).reshape(3,4,4)
            observations={'agent_1_articulated_agent_jaw_rgb':rgb}
            before=rgb.copy()
            with patch.dict(os.environ,{'WA_STUDENT_EVAL':'evaluation_set_adaptation_v1','WA_DIAG_OBSERVER_STATE':'0','WA_DIAG_CONTROLLER':'learned_target_guard_v3'}),patch.object(WAAgent,'act',return_value=[0,0,0]) as act,patch.object(WAAgent,'rpc') as rpc:
                result=agent.act(observations,{},'1')
                self.assertEqual(result,[0,0,0]);act.assert_called_once_with(observations,{},'1');rpc.assert_not_called()
                self.assertEqual(agent.initial_pair_evidence,dict(rgb=rgb_hash(rgb),state=dynamic_state(agent.diagnostic_env)))
                np.testing.assert_array_equal(rgb,before)
                evidence=agent.initial_pair_evidence
                agent.sim_step=1;agent.act({'agent_1_articulated_agent_jaw_rgb':rgb+1},{},'1')
                self.assertIs(agent.initial_pair_evidence,evidence)

    def test_raw_hash_includes_alpha_and_shape(self):
        rgb=np.zeros((3,4,4),dtype=np.uint8);changed=rgb.copy();changed[...,3]=1
        self.assertNotEqual(rgb_hash(rgb),rgb_hash(changed))
        self.assertNotEqual(rgb_hash(rgb),rgb_hash(rgb[...,:3]))

if __name__=='__main__':unittest.main()
