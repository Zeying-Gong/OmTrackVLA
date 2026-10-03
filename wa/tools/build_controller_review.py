"""Build a read-only paired video review from complete controller evaluation."""
import argparse,html,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    sources={'Candidate WA':Path(a.candidate),'WA 60509':Path('/data/nas_ray/project/md-ak/users/zeying.gong/job_60509/task_71388/wa_recovery_mix_eval_v1'),'LightNav':Path('/data/nas_ray/project/md-ak/users/zeying.gong/job_59846/task_70708/wa_lightnav_review_v1/lightnav')}
    data={}
    for name,root in sources.items():
        rows={}
        for f in sorted(root.glob('*/episodes.jsonl')):
            for line in f.read_text().splitlines():
                e=json.loads(line);key=(e['task'],e['key'])
                assert key not in rows
                video=f.parent/e['task']/'_review'/e['key']/'review.mp4'
                assert video.is_file()
                e['video']=str(video.relative_to(root));rows[key]=e
        assert len(rows)==24
        data[name]=rows
    keys=set(data['Candidate WA']);assert all(set(x)==keys for x in data.values())
    assert all(len({rows[k]['initial_rgb_sha256'] for rows in data.values()})==1 for k in keys)
    out=Path(a.output);out.mkdir(exist_ok=False)
    for i,(name,root) in enumerate(sources.items()):(out/f'media{i}').symlink_to(root,target_is_directory=True)
    head=' · '.join(f'{name}: {sum(e["success"] for e in rows.values()):.0f}/24; collisions {sum(e["collision"] for e in rows.values()):.0f}/24' for name,rows in data.items())
    parts=['<!doctype html><meta charset="utf-8"><title>WA controller comparison</title><style>body{font:16px sans-serif;background:#101827;color:#eee;margin:24px}section{border-top:1px solid #567;padding:18px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}video{width:100%}.bad{color:#ffadad}.good{color:#9eebad}a{color:#9cf}select{padding:8px}</style><h1>WA controller comparison — same 60502 checkpoint</h1><p>'+html.escape(head)+'</p><p>Both WA columns use identical model weights, RGB+BBox+ideal simulated UWB, no text. Candidate retains learned yaw; WA 60509 uses legacy target-heading override. Translation guard and benchmark physics unchanged. LightNav uses RGB+text, different modality. Development set only, not product/generalization acceptance. Videos20fps: compare simulator timestamps, not playback time.</p><label>Filter <select onchange="document.querySelectorAll(\'section\').forEach(x=>x.hidden=this.value!==\'all\'&&!x.dataset.tags.includes(this.value))"><option value="all">All24</option><option value="changed">Changed success</option><option value="door">Doorway</option><option value="failed">Candidate failures</option></select></label>']
    for k in sorted(keys):
        e=data['Candidate WA'][k];old=data['WA 60509'][k]
        tags=('changed ' if e['success']!=old['success'] else '')+('failed ' if not e['success'] else '')+('door ' if k[1]=='VLzqgDo317F/89' else '')
        parts.append(f'<section data-tags="{tags}"><h2>{html.escape(" / ".join(k))}</h2><div class="grid">')
        for i,(name,rows) in enumerate(data.items()):
            e=rows[k];url=f'media{i}/'+e['video'];cls='good' if e['success'] else 'bad'
            parts.append(f'<div><h3>{html.escape(name)}</h3><p class="{cls}">{html.escape(e["status"])} · success {e["success"]} · collision {e["collision"]}</p><video controls preload="none" src="{html.escape(url)}"></video><a href="{html.escape(url)}" download>Download MP4</a></div>')
        parts.append('</div></section>')
    (out/'index.html').write_text(''.join(parts))
    print(out,head)
if __name__=='__main__':main()
