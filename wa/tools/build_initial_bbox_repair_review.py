import concurrent.futures, hashlib, html, json, subprocess, sys
from pathlib import Path
from wa.wm.initial_bbox_repair import load_plan,merge_rows,KEYS
from wa.wm.full_mixed_contract import summarize,validate_ready
R=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
D=Path(sys.argv[1])
OUT=R/('artifacts/initial_bbox_repair_review_'+D.parents[1].name)
manifest=json.loads((R.parent/'WLA-EVT-20260925/evt_full_20260926/manifest.json').read_text())
plan=load_plan(R/'artifacts/initial_bbox_repair_v2/plan.json','6535f7b9a53e399ba7f2323c2d7f72d223146ee77f090223eb260f5885f6269a')
new=[]
for i in range(2):
    q=D/f'shard_{i:02d}'
    validate_ready(json.loads((q/'server_ready.json').read_text()))
    part=[json.loads(x) for x in (q/'episodes.jsonl').read_text().splitlines()]
    assert json.loads((q/'COMPLETE.json').read_text())['episodes']==len(part)==len(plan['lanes'][i])
    for x in part:x['artifact_root']=str(q)
    new.extend(part)
rows=merge_rows(plan,new)
combined=[json.loads(x) for x in (D/'combined_episodes.jsonl').read_text().splitlines()]
assert rows==combined
summary=json.loads((D/'summary.json').read_text())
assert summarize(rows,manifest)['metrics_percent']==summary['metrics_percent']
baseline={(r['task'],r['key']):r for r in map(json.loads,Path(plan['baseline']).read_text().splitlines())}
assert all(r['initial_rgb_sha256']==baseline[(r['task'],r['key'])]['initial_rgb_sha256'] for r in rows)
assert sum(r==baseline[(r['task'],r['key'])] for r in rows if (r['task'],r['key']) not in KEYS)==4208
roots={v:'source_'+str(i) for i,v in enumerate(sorted({r['artifact_root'] for r in rows}))}

def audit(r):
    root=Path(r['artifact_root']);task=r['task'];key=r['key']
    photo=root/task/'_live'/key/'step_0000.jpg'
    assert hashlib.sha256(photo.read_bytes()).hexdigest()==r['initial_rgb_sha256'],str(photo)
    video=root/task/'_review'/key/'review.mp4'
    assert video.stat().st_size>0
    probe=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,width,height','-of','json',str(video)],capture_output=True,text=True,check=True)
    meta=json.loads(probe.stdout)
    assert float(meta['format']['duration'])>0 and any(s.get('codec_type')=='video' for s in meta['streams'])
    return dict(r,video=roots[str(root)]+'/'+str(video.relative_to(root)),duration=float(meta['format']['duration']))

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    checked=list(pool.map(audit,rows))
print('AUDIT_PASS',len(checked),flush=True)
OUT.mkdir(exist_ok=False)
for root,label in roots.items():(OUT/label).symlink_to(root,target_is_directory=True)
(OUT/'episodes.json').write_text(json.dumps(checked))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
audit_report={'status':'PASS','episodes':4215,'reused_unchanged':4208,'new':7,'old_initial_rgb_pairs':4215,'repaired_mp3d':2163,'unaffected_hm3d':2052,'initial_rgb_sha256_verified':4215,'video_metadata_verified':4215,'video_check_scope':'ffprobe metadata/duration; not every frame decoded','invalid_init_count':sum(not x['policy_init_valid'] for x in rows),'combined_sha256':hashlib.sha256((D/'combined_episodes.jsonl').read_bytes()).hexdigest(),'summary_sha256':hashlib.sha256((D/'summary.json').read_bytes()).hexdigest()}
(OUT/'replacement_comparison.json').write_text(json.dumps([{"task":r["task"],"key":r["key"],"before":baseline[(r["task"],r["key"])],"after":r} for r in new],indent=2))
(OUT/'audit.json').write_text(json.dumps(audit_report,indent=2))
table=''.join('<tr><td>'+t.upper()+'</td>'+''.join('<td>'+str(round(m[k],3))+'</td>' for k in ('SR','TR','CR','invalid_init_count'))+'</tr>' for t,m in summary['metrics_percent'].items())
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>WA 全量闭环 · 4215</title>
<style>body{font:16px system-ui;background:#101827;color:#e7edf7;max-width:1150px;margin:32px auto;padding:0 18px}table{border-collapse:collapse}th,td{padding:12px 25px;border-bottom:1px solid #445}select,input,button{font:inherit;padding:8px;margin:5px;background:#202c40;color:white;border:1px solid #789;border-radius:5px}article{padding:15px;border:1px solid #456;border-radius:8px;margin:18px 0}video{width:100%;max-height:480px}a{color:#91caff}.warn{color:#ffd28a}</style>
<h1>WA 全量混合闭环 · 4215 episodes</h1><p>61257 / 72334 仅7条初始化框补测 + 4208条原样保留 · 60502 checkpoint step 45900</p>
<p><b>本次只修复7条首帧GT框。其他4208条原样保留；后续语义评分未改，不能据此声称所有可见性误差已修复。</b></p><p>此前MP3D语义PLY旋转修复：2163条受影响MP3D均为修复后结果，2052条未受影响HM3D复用；不是4215条全部重跑。旧协议81.138790%保留，此变化不是训练提升。</p><p class="warn">RGB + 首帧GT BBox + 理想模拟极坐标UWB（噪声0、延迟0），无文本。既有validation包含开发/确认样本，不是未见test。无全量LightNav配对结果，不宣称超过LightNav。</p>
<table><thead><tr><th>任务</th><th>SR %</th><th>TR %</th><th>CR %</th><th>初始化失败</th></tr></thead><tbody>TABLE</tbody></table>
<p>每类1405条；初始化失败保留分母。TR按参考步数归一（每类52条缺参考步数，按既定实现回退实际步数）；CR为曾距目标人&lt;0.5m，不是门框/一般障碍物接触。JEPA仅训练辅助，无在线MPC。</p>
<p><a href="summary.json">完整指标</a> · <a href="replacement_comparison.json">7条修复前后明细</a> · <a href="audit.json">真实性审计</a> · <a href="episodes.json">逐例记录与视频出处</a></p>
<select id="task"><option value="">全部任务</option><option>stt</option><option>dt</option><option>at</option></select>
<select id="outcome"><option value="">全部结果</option><option value="success">成功</option><option value="failure">失败</option><option value="collision">HumanCollision</option><option value="invalid">初始化失败</option><option value="repaired">本次7条补测</option></select><input id="search" placeholder="场景 / episode 搜索"><button id="prev">上一页</button><button id="next">下一页</button><p id="count"></p><main id="items"></main>
<script>
let rows=[],page=0;const task=document.getElementById('task'),outcome=document.getElementById('outcome'),search=document.getElementById('search'),items=document.getElementById('items');
function render(){const f=rows.filter(r=>(!task.value||r.task===task.value)&&r.key.includes(search.value)&&(!outcome.value||(outcome.value==='success'&&r.success===1)||(outcome.value==='failure'&&!r.success)||(outcome.value==='collision'&&r.collision===1)||(outcome.value==='invalid'&&!r.policy_init_valid)||(outcome.value==='repaired'&&r.initialization_repair)));page=Math.max(0,Math.min(page,Math.ceil(f.length/12)-1));items.replaceChildren();document.getElementById('count').textContent=`匹配 ${f.length} 条 · 第 ${page+1} / ${Math.max(1,Math.ceil(f.length/12))} 页`;
for(const r of f.slice(page*12,page*12+12)){const a=document.createElement('article'),h=document.createElement('h2'),p=document.createElement('p'),v=document.createElement('video'),l=document.createElement('a');h.textContent=r.task.toUpperCase()+' · '+r.key;p.textContent=`成功 ${r.success} / HumanCollision ${r.collision} / 初始化有效 ${r.policy_init_valid} / ${r.status} / ${r.initialization_repair?'本次首帧框修复':'保留原结果'}`;v.controls=true;v.preload='none';v.src=r.video;l.href=r.video;l.download='review.mp4';l.textContent='下载原始视频';a.append(h,p,v,l);items.append(a)}}
for(const el of [task,outcome,search])el.addEventListener('input',()=>{page=0;render()});document.getElementById('prev').onclick=()=>{page--;render()};document.getElementById('next').onclick=()=>{page++;render()};fetch('episodes.json').then(r=>r.json()).then(r=>{rows=r.sort((a,b)=>a.task.localeCompare(b.task)||a.key.localeCompare(b.key));render()}).catch(e=>document.getElementById('count').textContent='加载失败 '+e);
</script></html>'''.replace('TABLE',table)
page=page.replace("61257 / 72334",D.parents[1].name.removeprefix("job_")+" / "+D.parent.name.removeprefix("task_"))
(OUT/'index.html').write_text(page)
print(json.dumps(summary,indent=2),flush=True)
print(OUT,flush=True)
