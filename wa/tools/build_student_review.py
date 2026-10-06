"""Publish a separate, audited student result browser; never reuse baseline rows."""
import argparse
import concurrent.futures
import math
import html
import json
import subprocess
from pathlib import Path
from urllib.parse import quote
from wa.tools.merge_student_partitions import digest, read_partitions
from wa.wm.full_mixed_contract import MANIFEST_SHA, summarize
from wa.wm.student_eval_finalize import finalize

def audit_media(row, ffprobe):
    root=Path(row['artifact_root']); task=row['task']; key=row['key']
    photo=root/task/'_live'/key/'step_0000.jpg'
    video=root/task/'_review'/key/'review.mp4'
    if digest(photo)!=row['initial_rgb_sha256'] or video.stat().st_size<=0:
        raise ValueError('missing or changed media: '+str(video))
    raw=subprocess.run([ffprobe,'-v','error','-show_entries',
        'format=duration:stream=codec_type,width,height','-of','json',str(video)],
        check=True,capture_output=True,text=True,timeout=60)
    meta=json.loads(raw.stdout)
    duration=float(meta['format']['duration'])
    if not math.isfinite(duration) or duration<=0 or not any(s.get('codec_type')=='video' for s in meta['streams']):
        raise ValueError('invalid video: '+str(video))
    return str(video.relative_to(root)),float(meta['format']['duration'])

def main():
    p=argparse.ArgumentParser()
    for name in ('merged','manifest','teachers','ffprobe','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args(); merged=Path(a.merged); out=Path(a.output)
    if out.exists():raise ValueError('refusing to overwrite review')
    if digest(a.manifest)!=MANIFEST_SHA:raise ValueError('manifest changed')
    summary_path=merged/'summary.json'; combined_path=merged/'combined_episodes.jsonl'
    saved_hashes={str(f):digest(f) for f in (summary_path,combined_path)}
    saved=json.loads(summary_path.read_text()); manifest=json.loads(Path(a.manifest).read_text())
    combined=[json.loads(s) for s in combined_path.read_text().splitlines() if s]
    if len(combined)!=4215 or saved.get('new_episodes')!=4215 or saved.get('reused_baseline_episodes')!=0:
        raise ValueError('complete new student evaluation required')
    contract=dict(checkpoint_sha=combined[0]['checkpoint_sha256'],step=combined[0]['checkpoint_step'])
    rows,source_hashes=read_partitions(saved['partition_roots'],manifest,contract)
    if rows!=combined:raise ValueError('merged rows changed')
    report=summarize(rows,manifest,**contract); comparison,pairs=finalize(rows,manifest,a.teachers)
    if report['metrics_percent']!=saved['metrics_percent'] or comparison['superiority']!=saved['superiority']:
        raise ValueError('saved metrics differ from recomputation')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        media=list(pool.map(lambda r:audit_media(r,a.ffprobe),rows))
    for f,h in dict(source_hashes,**saved_hashes).items():
        if digest(f)!=h:raise ValueError('source changed during review audit')
    roots={r:'source_'+str(i) for i,r in enumerate(sorted({r['artifact_root'] for r in rows}))}
    records=[]
    for r,(video,duration) in zip(rows,media):
        records.append(dict(task=r['task'],key=r['key'],success=r['success'],collision=r['collision'],
            policy_init_valid=r['policy_init_valid'],duration=duration,
            video=roots[r['artifact_root']]+'/'+quote(video,safe='/')))
    out.mkdir(parents=True,exist_ok=False)
    for root,label in roots.items():(out/label).symlink_to(root,target_is_directory=True)
    (out/'summary.json').write_text(json.dumps(saved,indent=2,allow_nan=False))
    (out/'episodes.json').write_text(json.dumps(records,allow_nan=False))
    (out/'audit.json').write_text(json.dumps(dict(status='PASS',episodes=4215,
        initial_pairs=pairs['episodes'],first_jpeg_hashes=4215,video_metadata=4215,
        video_scope='metadata and duration, not every frame decoded',source_hashes=source_hashes,
        merged_hashes=saved_hashes),allow_nan=False))
    table=''.join('<tr><td>'+html.escape(r['task'])+'</td><td>'+html.escape(r['key'])+'</td><td>'+str(r['success'])+'</td><td>'+str(r['collision'])+'</td><td><a href="'+html.escape(r['video'],quote=True)+'">视频</a></td></tr>' for r in records)
    page='<!doctype html><meta charset="utf-8"><title>WA 双教师集内适配评测</title><h1>WA 新学生 · 4215条全新闭环</h1><p>评测集内训练后评测，不是未见测试集泛化。WA使用RGB+首帧GT框+理想UWB，无文本；LightNav使用RGB+文本。CR仅表示距目标人曾小于0.5m，不是通用障碍物碰撞率。JEPA仅训练辅助。</p>'
    page+='<p>权重SHA '+html.escape(contract['checkpoint_sha'])+' · step '+str(contract['step'])+'</p><pre>'+html.escape(json.dumps(dict(metrics=saved['metrics_percent'],comparison=comparison['superiority']),indent=2))+'</pre>'
    page+='<p><a href="summary.json">完整指标及TR定义</a> · <a href="audit.json">审计</a> · <a href="episodes.json">视频索引</a></p><table><tr><th>任务</th><th>样本</th><th>成功</th><th>HumanCollision</th><th>回放</th></tr>'+table+'</table>'
    (out/'index.html').write_text(page)
    print(json.dumps(dict(status='REVIEW_AUDITED',episodes=4215,output=str(out))))

if __name__=='__main__':main()
