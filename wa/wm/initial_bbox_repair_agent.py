"""Only first-frame bbox annotation replacement; parent policy/controller unchanged."""
from wa.wm.diagnostic_agent import DiagnosticAgent
from wa.wm.initial_bbox_repair import repaired_detector
class InitialBBoxRepairAgent(DiagnosticAgent):
    def __init__(self,*args,repair,**kwargs):
        self.initial_bbox_repair=repair;self.repair_applied=False
        super().__init__(*args,**kwargs)
    def act(self,observations,detector,episode_id,instruction=None,timestamp_s=None):
        if self.sim_step==0:
            if str(episode_id)!=self.initial_bbox_repair["key"].split("/")[-1]:
                raise ValueError("repair episode mismatch")
            detector=repaired_detector(detector,observations["agent_1_articulated_agent_jaw_rgb"],self.initial_bbox_repair)
            self.repair_applied=True
        return super().act(observations,detector,episode_id,instruction,timestamp_s)
