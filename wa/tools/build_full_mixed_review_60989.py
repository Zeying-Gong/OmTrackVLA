import concurrent.futures, hashlib, html, json, subprocess
from pathlib import Path
from wa.wm.full_mixed_resume import load_plan, verify_new_rows
from wa.wm.full_mixed_contract import summarize, validate_ready

R=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
J=Path('/data/nas_ray/project/md-ak/users/zeying.gong')
D=J/'job_60989/task_71912/wa_full_mixed_resume_8gpu_v1'
OLD=J/'job_60885/task_71808/wa_full_mixed_learned_yaw_v2'
OUT=R/'artifacts/full_mixed_review_60989'
manifest=json.loads((R.parent/'WLA-EVT-20260925/evt_full_20260926/manifest.json').read_text())
plan=load_plan(R/'artifacts/full_mixed_resume_8gpu_20261004.json',manifest,'66bb0c34f3d44b0a238911e21b21a29502cc85d7076e9a59b2f9048aef1ed33b')
new=[]
for i in range(8):
    q=D/f'shard_{i:02d}'
    validate_ready(json.loads((q/'server_ready.json').read_text()))
    part=[json.loads(x) for x in (q/'episodes.jsonl').read_text().splitlines()]
    assert json.loads((q/'COMPLETE.json').read_text())['episodes']==len(part)==len(plan['lanes'][i])
    assert {(x['task'],x['key']) for x in part}=={(x['task'],x['key']) for x in plan['lanes'][i]}
    for x in part:x['artifact_root']=str(q)
    new.extend(part)
verify_new_rows(new,plan)
rows=plan['completed_rows']+new
combined=[json.loads(x) for x in (D/'combined_episodes.jsonl').read_text().splitlines()]
assert combined==rows
summary=json.loads((D/'summary.json').read_text())
assert summarize(rows,manifest)['metrics_percent']==summary['metrics_percent']
assert all(r['policy_init_valid'] or r['success']==0 for r in rows)

def audit(r):
    root=Path(r['artifact_root']);task=r['task'];key=r['key']
    photo=root/task/'_live'/key/'step_0000.jpg'
    assert hashlib.sha256(photo.read_bytes()).hexdigest()==r['initial_rgb_sha256'],str(photo)
    video=root/task/'_review'/key/'review.mp4'
    assert video.stat().st_size>0
    probe=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,width,height','-of','json',str(video)],capture_output=True,text=True,check=True)
    meta=json.loads(probe.stdout)
    assert float(meta['format']['duration'])>0 and any(s.get('codec_type')=='video' for s in meta['streams'])
    base=OLD if root.is_relative_to(OLD) else D
    prefix='parent' if base==OLD else 'continuation'
    return dict(r,video=prefix+'/'+str(video.relative_to(base)),duration=float(meta['format']['duration']))

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    checked=list(pool.map(audit,rows))
print('AUDIT_PASS',len(checked),flush=True)
OUT.mkdir(exist_ok=False)
(OUT/'parent').symlink_to(OLD,target_is_directory=True)
(OUT/'continuation').symlink_to(D,target_is_directory=True)
(OUT/'episodes.json').write_text(json.dumps(checked))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
audit_report={'status':'PASS','episodes':4215,'reused':2238,'new':1977,'initial_rgb_sha256_verified':4215,'video_metadata_verified':4215,'video_check_scope':'ffprobe metadata/duration; not every frame decoded','invalid_init_count':sum(not x['policy_init_valid'] for x in rows),'combined_sha256':hashlib.sha256((D/'combined_episodes.jsonl').read_bytes()).hexdigest(),'summary_sha256':hashlib.sha256((D/'summary.json').read_bytes()).hexdigest()}
(OUT/'audit.json').write_text(json.dumps(audit_report,indent=2))
table=''.join('<tr><td>'+t.upper()+'</td>'+''.join('<td>'+str(round(m[k],3))+'</td>' for k in ('SR','TR','CR','invalid_init_count'))+'</tr>' for t,m in summary['metrics_percent'].items())
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>WA 全量闭环 · 4215</title>
<style>body{font:16px system-ui;background:#101827;color:#e7edf7;max-width:1150px;margin:32px auto;padding:0 18px}table{border-collapse:collapse}th,td{padding:12px 25px;border-bottom:1px solid #445}select,input,button{font:inherit;padding:8px;margin:5px;background:#202c40;color:white;border:1px solid #789;border-radius:5px}article{padding:15px;border:1px solid #456;border-radius:8px;margin:18px 0}video{width:100%;max-height:480px}a{color:#91caff}.warn{color:#ffd28a}</style>
<h1>WA 全量混合闭环 · 4215 episodes</h1><p>60989 / 71912 八卡续评 + 60885 / 71808 冻结产物 · 60502 checkpoint step 45900</p>
<p class="warn">RGB + 首帧BBox + 理想模拟极坐标UWB（噪声0、延迟0），无文本。既有validation包含开发/确认样本，不是未见test。无全量LightNav配对结果，不宣称超过LightNav。</p>
<table><thead><tr><th>任务</th><th>SR %</th><th>TR %</th><th>CR %</th><th>初始化失败</th></tr></thead><tbody>TABLE</tbody></table>
<p>每类1405条；初始化失败保留分母。TR按参考步数归一（每类52条缺参考步数，按既定实现回退实际步数）；CR为曾距目标人&lt;0.5m，不是门框/一般障碍物接触。JEPA仅训练辅助，无在线MPC。</p>
<p><a href="summary.json">完整指标</a> · <a href="audit.json">真实性审计</a> · <a href="episodes.json">逐例记录与视频出处</a></p>
<select id="task"><option value="">全部任务</option><option>stt</option><option>dt</option><option>at</option></select>
<select id="outcome"><option value="">全部结果</option><option value="success">成功</option><option value="failure">失败</option><option value="collision">HumanCollision</option><option value="invalid">初始化失败</option></select><input id="search" placeholder="场景 / episode 搜索"><button id="prev">上一页</button><button id="next">下一页</button><p id="count"></p><main id="items"></main>
<script>
let rows=[],page=0;const task=document.getElementById('task'),outcome=document.getElementById('outcome'),search=document.getElementById('search'),items=document.getElementById('items');
function render(){const f=rows.filter(r=>(!task.value||r.task===task.value)&&r.key.includes(search.value)&&(!outcome.value||(outcome.value==='success'&&r.success===1)||(outcome.value==='failure'&&!r.success)||(outcome.value==='collision'&&r.collision===1)||(outcome.value==='invalid'&&!r.policy_init_valid)));page=Math.max(0,Math.min(page,Math.ceil(f.length/12)-1));items.replaceChildren();document.getElementById('count').textContent=`匹配 ${f.length} 条 · 第 ${page+1} / ${Math.max(1,Math.ceil(f.length/12))} 页`;
for(const r of f.slice(page*12,page*12+12)){const a=document.createElement('article'),h=document.createElement('h2'),p=document.createElement('p'),v=document.createElement('video'),l=document.createElement('a');h.textContent=r.task.toUpperCase()+' · '+r.key;p.textContent=`成功 ${r.success} / HumanCollision ${r.collision} / 初始化有效 ${r.policy_init_valid} / ${r.status}`;v.controls=true;v.preload='none';v.src=r.video;l.href=r.video;l.download='review.mp4';l.textContent='下载原始视频';a.append(h,p,v,l);items.append(a)}}
for(const el of [task,outcome,search])el.addEventListener('input',()=>{page=0;render()});document.getElementById('prev').onclick=()=>{page--;render()};document.getElementById('next').onclick=()=>{page++;render()};fetch('episodes.json').then(r=>r.json()).then(r=>{rows=r.sort((a,b)=>a.task.localeCompare(b.task)||a.key.localeCompare(b.key));render()}).catch(e=>document.getElementById('count').textContent='加载失败 '+e);
</script></html>'''.replace('TABLE',table)
(OUT/'index.html').write_text(page)
print(json.dumps(summary,indent=2),flush=True)
print(OUT,flush=True)
