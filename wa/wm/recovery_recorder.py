"""Executed-pose recovery records; failed expert branches never positive labels."""
import json
import numpy as np
from scripts.da3.teacher_recorder import TeacherRecorder
from wa.wm.recovery_replay import rgb_hash,dynamic_state

class RecoveryRecorder(TeacherRecorder):
    def __init__(self,root,metadata,environment_provider,takeover_step=None):
        if metadata.get('partition')!='train':
            raise ValueError('recovery collection requires training partition')
        super().__init__(root,metadata)
        self.environment_provider=environment_provider
        self.takeover_step=takeover_step
        self.replay=[]
    def observe(self,obs,detector,robot,human,episode,step,frequency,world_time=None):
        super().observe(obs,detector,robot,human,episode,step,frequency,world_time)
        self.replay.append(dict(step=int(step),
            rgb_sha256=rgb_hash(obs['agent_1_articulated_agent_jaw_rgb']),
            dynamic_state=dynamic_state(self.environment_provider())))
    def record_action(self,step,action,predicted):
        prediction=None if predicted is None else np.asarray(predicted).tolist()
        super().record_action(step,[float(x) for x in action],prediction)
        assert self.replay[-1]['step']==step
        self.replay[-1]['action']=[float(x) for x in action]
        self.actions[-1]['owner']='teacher' if self.takeover_step is not None and step>=self.takeover_step else 'student'
    def finish(self,result):
        super().finish(result)
        raw=json.loads((self.root/'windows.json').read_text())
        (self.root/'raw_windows.json').write_text(json.dumps(raw))
        (self.root/'replay.json').write_text(json.dumps(self.replay))
        success=bool(result.get('success',False)) and not bool(result.get('collision',False))
        # Complete 0.7s supervision must start at or after teacher takeover.
        admitted=[w for w in raw if success and (self.takeover_step is None or w['current_index']>=self.takeover_step)]
        (self.root/'windows.json').write_text(json.dumps(admitted))
        audit=dict(episode_success=success,takeover_step=self.takeover_step,
                   raw_windows=len(raw),candidate_positive_windows=len(admitted),
                   training_eligible=False,
                   pending='repeat-success verification and full split/state/label audit',
                   positive_source='student_success' if self.takeover_step is None else 'successful_teacher_suffix',
                   failed_branch_positive_windows=0 if not success else None)
        (self.root/'admission.json').write_text(json.dumps(audit,indent=2))
        done=json.loads((self.root/'complete.json').read_text())
        done.update(windows=len(admitted),training_eligible=False,blocker=audit['pending'])
        (self.root/'complete.json').write_text(json.dumps(done,indent=2))
