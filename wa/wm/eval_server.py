"""JEPA-trained WA inference RPC; image-only closed-loop diagnostic, no text/GT stream."""
import argparse,base64,io,json,time,traceback
from collections import deque
from http.server import HTTPServer,BaseHTTPRequestHandler
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from wa.wm.training import JointRobotModel
from wa.wm.loaders import sha
from wa.wm.robot_data import CONTRACT

def preprocess(image):
    a=np.asarray(image.convert('RGB').resize((224,224)),dtype=np.float32).copy()/255.
    return (torch.from_numpy(a).permute(2,0,1)-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]

class Session:
    def __init__(self,model,mode='image',noise_mode='random'):
        self.model=model;self.mode=mode;self.noise_mode=noise_mode;self.reset()
    def reset(self):self.frames=deque(maxlen=40);self.times=deque(maxlen=40);self.template=None;self.step=0;self.template_valid=False
    def predict(self,r):
        required={'rgb_png','timestamp_s','initial_bbox'}
        if self.mode=='mixed':required.add('uwb')
        if set(r)!=required:raise ValueError('only RGB/time/one-time bbox and declared UWB allowed')
        if self.mode not in ('image','mixed'):raise ValueError('unsupported mode')
        t=float(r['timestamp_s'])
        if not np.isfinite(t) or (self.times and t<=self.times[-1]):raise ValueError('invalid time')
        rgb=Image.open(io.BytesIO(base64.b64decode(r['rgb_png']))).convert('RGB')
        if self.template is None:
            box=r['initial_bbox']
            if box is None and self.mode=='mixed':
                self.template=torch.zeros(3,224,224,device='cuda')
            else:
                if box is None or len(box)!=4 or not np.isfinite(box).all():raise ValueError('initial bbox missing')
                x1,y1,x2,y2=box
                if not 0<=x1<x2<=rgb.width or not 0<=y1<y2<=rgb.height:raise ValueError('initial bbox outside RGB')
                self.template=preprocess(rgb.crop(box)).cuda();self.template_valid=True
        elif r['initial_bbox'] is not None:raise ValueError('later bbox forbidden')
        self.frames.append(preprocess(rgb));self.times.append(t)
        times=np.asarray(self.times);idx=np.clip(np.searchsorted(times,t+np.array([-1.5,-1.,-.5,0.]),side='right')-1,0,len(times)-1)
        batch=dict(rgb=torch.stack([self.frames[int(i)] for i in idx])[None].cuda(),template=self.template[None],
                   polar=torch.zeros(1,2,device='cuda'),times=torch.tensor(times[idx]-t,dtype=torch.float32,device='cuda')[None])
        batch['template_valid']=torch.tensor([self.template_valid],device='cuda')
        if self.mode=='mixed':
            u=r['uwb']
            if set(u)!={'polar','timestamp_s','valid','source','range_noise_m','bearing_noise_rad','delay_s'} or u['source']!='ideal_simulated_uwb' or u['valid'] is not True:raise ValueError('invalid declared UWB')
            age=t-float(u['timestamp_s']);polar=np.asarray(u['polar'],dtype=np.float32)
            if polar.shape!=(2,) or not np.isfinite(polar).all() or polar[0]<0 or not 0<=age<=.5:raise ValueError('invalid/stale UWB')
            if any(float(u[k])!=0 for k in ('range_noise_m','bearing_noise_rad','delay_s')):raise ValueError('ideal profile mismatch')
            batch['polar']=torch.tensor(polar,device='cuda')[None];batch['uwb_age_s']=torch.tensor([age],device='cuda')
        torch.cuda.synchronize();start=time.monotonic()
        with torch.inference_mode():
            noise=torch.randn((1,7,4),generator=torch.Generator(device='cuda').manual_seed(7+self.step),device='cuda')
            if self.noise_mode=='zero':noise.zero_()
            pose,target=self.model.predict(batch,torch.full((1,),2 if self.mode=='mixed' else 0,dtype=torch.long,device='cuda'),noise)
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        pose=pose[0].float().cpu().numpy();target=target[0].float().cpu().numpy()
        if not np.isfinite(pose).all() or not np.isfinite(target).all():raise ValueError('nonfinite prediction')
        result=dict(xy=pose[:,:2].tolist(),yaw=np.arctan2(pose[:,2],pose[:,3]).tolist(),target_geometry=target.tolist(),
                    frame_times_s=times[idx].tolist(),inference_s=elapsed,prediction_step=self.step,inputs='RGB+first_frame_bbox'+('+ideal_simulated_uwb' if self.mode=='mixed' else ''),world_predictor_inference=False,
                    mode=self.mode,noise_mode=self.noise_mode,template_valid=self.template_valid,uwb=r.get('uwb'))
        self.step+=1;return result

def main():
    p=argparse.ArgumentParser()
    for k in ['root','encoder-weight','wla-source','wla-checkpoint','checkpoint','ready']:p.add_argument('--'+k,required=True)
    p.add_argument('--mode',choices=['image','mixed'],default='image')
    p.add_argument('--noise-mode',choices=['random','zero'],default='random')
    p.add_argument('--developer-check',action='store_true');a=p.parse_args()
    if a.developer_check and __import__('os').environ.get('MD_AK_JOB_ID'):raise ValueError('no cluster smoke')
    torch.set_num_threads(2);torch.manual_seed(7);torch.cuda.set_device(0)
    model=JointRobotModel(a.root,a.encoder_weight,a.wla_source,a.wla_checkpoint,'jepa')
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if ckpt['step']<1 or ckpt['kind']!='jepa' or ckpt['contract']!=CONTRACT:raise ValueError('wrong trained checkpoint')
    expected={k for k in model.state_dict() if not k.startswith('encoder.')}
    if set(ckpt['model'])!=expected:raise ValueError('checkpoint keys mismatch')
    result=model.load_state_dict(ckpt['model'],strict=False)
    if result.unexpected_keys or any(not k.startswith('encoder.') for k in result.missing_keys):raise ValueError('partial load')
    trained_step=int(ckpt['step'])
    del ckpt
    model.cuda().eval().requires_grad_(False);session=Session(model,a.mode,a.noise_mode)
    metadata=dict(checkpoint=a.checkpoint,checkpoint_sha256=sha(a.checkpoint),step=trained_step,gpu=torch.cuda.get_device_name(0),mode=a.mode,noise_mode=a.noise_mode,sampling_steps=4,seed='7+step',text_used=False,world_predictor_inference=False)
    if a.developer_check:
        stream=io.BytesIO();Image.new('RGB',(224,224)).save(stream,format='PNG')
        r=dict(rgb_png=base64.b64encode(stream.getvalue()).decode(),timestamp_s=0.,initial_bbox=[20,20,120,200])
        if a.mode=='mixed':r['uwb']=dict(polar=[2.,.1],timestamp_s=0.,valid=True,source='ideal_simulated_uwb',range_noise_m=0.,bearing_noise_rad=0.,delay_s=0.)
        first=session.predict(r);r.update(timestamp_s=.05,initial_bbox=None);second=session.predict(r)
        Path(a.ready).write_text(json.dumps(dict(status='DEVELOPER_INTERFACE_PASS_NOT_BENCHMARK',metadata=metadata,first=first,second=second),indent=2));return
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                r=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path=='/reset':
                    if r:raise ValueError('reset payload must be empty')
                    session.reset();value={'reset':True}
                elif self.path=='/predict':value=session.predict(r)
                else:raise ValueError('unknown endpoint')
                data=json.dumps(value,allow_nan=False).encode();self.send_response(200)
            except Exception as e:
                traceback.print_exc();data=json.dumps({'error':str(e)}).encode();self.send_response(500)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=HTTPServer(('127.0.0.1',0),Handler);metadata['url']='http://127.0.0.1:'+str(server.server_port)
    Path(a.ready).write_text(json.dumps(metadata,indent=2));print('WA_SERVER_READY',flush=True);server.serve_forever()

if __name__=='__main__':main()
