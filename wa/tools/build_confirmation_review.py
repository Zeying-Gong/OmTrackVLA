"""Build strict two-column confirmation video review; never changes evaluation data."""
import argparse
import html
import json
from pathlib import Path
from wa.tools.compare_controller import load_method, rates, require

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wa', type=Path, required=True)
    p.add_argument('--lightnav', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path, default=Path('wa/wm/mixed_diagnostic_plan_v1.json'))
    a = p.parse_args()
    methods = {}
    roots = {'WA 60770': a.wa.resolve(), 'LightNav 60771': a.lightnav.resolve()}
    for label, root in roots.items():
        methods[label] = load_method(root, candidate=label.startswith('WA'))[0]
    keys = set(methods['WA 60770'])
    plan = json.loads(a.plan.read_text())['confirmation']
    require(keys == {(t, k) for t, values in plan.items() for k in values}, 'wrong confirmation set')
    require(set(methods['LightNav 60771']) == keys, 'method key mismatch')
    require(all(methods['WA 60770'][k]['initial_rgb_sha256'] == methods['LightNav 60771'][k]['initial_rgb_sha256'] for k in keys), 'initial RGB mismatch')
    for rows in methods.values():
        require(all(Path(e['video_path']).is_file() for e in rows.values()), 'missing video')
    a.output.mkdir(exist_ok=False)
    for i, root in enumerate(roots.values()):
        (a.output / f'media{i}').symlink_to(root, target_is_directory=True)
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WA / LightNav 独立确认</title><style>body{font:16px/1.5 system-ui,sans-serif;background:#101827;color:#eee;margin:24px auto;max-width:1400px;padding:0 16px}section{border-top:1px solid #567;padding:18px 0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}video{width:100%;background:#060a10}.bad{color:#ffadad}.good{color:#9eebad}a{color:#9cf}select,button{font:inherit;padding:8px;margin-right:8px}th,td{text-align:left;padding:8px 22px 8px 0}@media(max-width:640px){.grid{grid-template-columns:1fr}body{margin:12px 0}th,td{padding-right:8px}}</style><h1>WA / LightNav · 独立确认 24 条</h1><p>预先固定的另一组 8 场景，每场景 STT / DT / AT 各 1 条；与开发集不交叉。24 个 episode key 与初始 RGB 全部匹配。两项任务已完成；未挑选成功样本。</p><table><thead><tr><th>任务</th><th>WA 成功 / 碰撞</th><th>LightNav 成功 / 碰撞</th></tr></thead><tbody>']
    for task in ['stt','dt','at','all']:
        cells = []
        for rows in methods.values():
            r = rates([e for k,e in rows.items() if task == 'all' or k[0] == task])
            cells.append(f"<td>{r['successful']}/{r['episodes']} · {r['collisions']}/{r['episodes']}</td>")
        parts.append(f"<tr><th>{task.upper()}</th>{''.join(cells)}</tr>")
    parts.append('</tbody></table><p>WA：RGB + 首帧 BBox + 理想模拟极坐标 UWB（零噪声、零延迟），无文本。LightNav：RGB + 文本。输入不同；此结果不是纯视觉同模态公平对比，也不代表产品安全或广泛泛化达标。</p><p>WA 保留模型学到的 yaw，平移 guard 和物理约束不变，使用 60502 原权重，无新训练。录像 20fps；两种方法的每步仿真时间不同，播放时间不是严格物理时间对齐。</p><label>显示 <select id="filter"><option value="all">全部 24 条</option><option value="wa_only">仅 WA 成功</option><option value="ln_only">仅 LightNav 成功</option><option value="wa_fail">WA 失败</option><option value="collision">任一碰撞</option></select></label><span id="count" aria-live="polite">24 / 24</span>')
    def sort_key(k):
        w,l = methods['WA 60770'][k],methods['LightNav 60771'][k]
        return (w['success'] == l['success'], k)
    for key in sorted(keys, key=sort_key):
        w,l = methods['WA 60770'][key],methods['LightNav 60771'][key]
        tags = ['all']
        if w['success'] and not l['success']: tags.append('wa_only')
        if l['success'] and not w['success']: tags.append('ln_only')
        if not w['success']: tags.append('wa_fail')
        if w['collision'] or l['collision']: tags.append('collision')
        parts.append(f'<section data-tags="{",".join(tags)}"><h2>{html.escape(" / ".join(key))}</h2><div class="grid">')
        for i,(label, rows) in enumerate(methods.items()):
            e = rows[key]
            url = f'media{i}/' + str(Path(e['video_path']).relative_to(roots[label]))
            css = 'good' if e['success'] else 'bad'
            parts.append(f'<div><h3>{label}</h3><p class="{css}">{html.escape(e["status"])} · 成功 {e["success"]} · 碰撞 {e["collision"]} · 初始化有效 {e["policy_init_valid"]}</p><video controls preload="metadata" src="{html.escape(url)}"></video><a href="{html.escape(url)}" download>下载 MP4</a></div>')
        parts.append('</div></section>')
    parts.append('<script>document.getElementById("filter").addEventListener("change",function(){let n=0;document.querySelectorAll("section").forEach(s=>{s.hidden=!s.dataset.tags.split(",").includes(this.value);if(!s.hidden)n++;});document.getElementById("count").textContent=n+" / 24";});</script></html>')
    (a.output/'index.html').write_text(''.join(parts), encoding='utf-8')
    print(json.dumps({'output':str(a.output),'episodes':len(keys),'videos':48,'initial_rgb_matched':24}))

if __name__ == '__main__':
    main()
