"""Observer-only video/telemetry; never supplies inputs to either policy."""
import json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw

class ReviewRecorder:
    def __init__(self,directory,method):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.method=method;self.writer=None;self.frames=0
        self.log=(self.directory/'steps.jsonl').open('x')
    def observe(self,obs,detector,robot,target,episode,step,fps,timestamp):
        front=np.asarray(obs['agent_1_articulated_agent_jaw_rgb'])[...,:3]
        third=np.asarray(obs['agent_1_third_rgb'])[...,:3]
        self.canvas=Image.new('RGB',(768,432));self.canvas.paste(Image.fromarray(front).resize((384,384)),(0,48));self.canvas.paste(Image.fromarray(third).resize((384,384)),(384,48))
        p=np.asarray(robot.base_pos,dtype=float);r=np.asarray(robot.sim_obj.transformation.rotation(),dtype=float);t=np.asarray(target.base_pos,dtype=float)
        local=(t-p)@r
        self.row=dict(step=int(step),timestamp_s=float(timestamp),robot_position_world=p.tolist(),robot_rotation_world_from_body=r.tolist(),target_position_world=t.tolist(),polar=[float(np.hypot(local[0],local[2])),float(np.arctan2(-local[2],local[0]))],usage='observer_only_not_policy_input')
        if self.writer is None:self.writer=imageio.get_writer(str(self.directory/'review.mp4'),fps=20,codec='libx264',quality=7,macro_block_size=16)
    def record_action(self,step,action,trajectory):
        assert int(step)==self.row['step']
        self.row['action']=np.asarray(action,dtype=float).tolist()
        self.row['trajectory']=None if trajectory is None else np.asarray(trajectory).tolist()
        self.log.write(json.dumps(self.row,allow_nan=False)+'\n');self.log.flush()
        d=ImageDraw.Draw(self.canvas);r,b=self.row['polar'];a=self.row['action']
        d.text((8,4),f'{self.method} step={step} sim_t={self.row["timestamp_s"]:.3f}s range={r:.2f}m bearing={b:.2f}rad',fill='white')
        d.text((8,23),f'action={a[0]:+.3f},{a[1]:+.3f},{a[2]:+.3f} | FRONT / OBSERVER THIRD | fixed20fps; timestamps authoritative',fill='white')
        self.writer.append_data(np.asarray(self.canvas));self.frames+=1
    def finish(self,result):
        if self.writer:self.writer.close()
        self.log.close()
        (self.directory/'complete.json').write_text(json.dumps(dict(frames=self.frames,result=result,video_timebase='20fps frame playback; actual variable simulator timestamps in overlay/steps.jsonl',policy_inputs_unchanged=True),indent=2))

def install():
    import trained_agent
    original=trained_agent.evaluate_agent
    def wrapped(config,dataset,save_path,*args,**kwargs):
        ep=dataset.episodes[0];scene=Path(ep.scene_id).name.split('.')[0]
        directory=Path(save_path)/'_review'/scene/str(ep.episode_id)
        assert kwargs.get('recorder') is None
        kwargs['recorder']=ReviewRecorder(directory,'WA RGB+BBox+idealUWB')
        return original(config,dataset,save_path,*args,**kwargs)
    trained_agent.evaluate_agent=wrapped
