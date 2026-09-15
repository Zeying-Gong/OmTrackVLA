"""Plot audited raw observations and label/prediction boxes; no inference."""
from pathlib import Path
import hashlib
import json
import tarfile
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

from semantic_labels import array_sha256, instance_label

root = Path(__file__).resolve().parent
extracted = root / 'preview_raw'
extracted.mkdir(exist_ok=False)
with tarfile.open(root / 'preview.tar.gz') as archive:
    for member in archive.getmembers():
        if not member.isfile() or extracted.resolve() not in (extracted / member.name).resolve().parents:
            raise ValueError('unsafe archive member')
    archive.extractall(extracted)
metadata = json.loads((extracted / '.codex_upload/perception_preview_v1/preview.json').read_text())
font = FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')
fig, axes = plt.subplots(2, 2, figsize=(11, 11.8))
for ax, row in zip(axes.flat, metadata['rows']):
    label = row['label']
    rgb_path, panoptic_path = extracted / row['rgb_path'], extracted / row['panoptic_path']
    assert hashlib.sha256(rgb_path.read_bytes()).hexdigest() == label['rgb_file']['sha256']
    assert hashlib.sha256(panoptic_path.read_bytes()).hexdigest() == label['raw_panoptic_file']['sha256']
    rgb = np.asarray(Image.open(rgb_path))
    raw_panoptic = np.load(panoptic_path, allow_pickle=False)
    panoptic = raw_panoptic[..., 0] if raw_panoptic.ndim == 3 else raw_panoptic
    assert array_sha256(rgb) == label['rgb_array_sha256']
    assert array_sha256(panoptic) == label['panoptic_array_sha256']
    assert instance_label(panoptic, label['target']['semantic_id_label_side_only']) == label['target']
    ax.imshow(rgb)
    target = label['target']
    if target['bbox_label_valid']:
        x1, y1, x2, y2 = target['bbox_xyxy_inclusive']
        ax.add_patch(Rectangle((x1-.5, y1-.5), x2-x1+1, y2-y1+1,
                               fill=False, edgecolor='#ffd400', linewidth=2.5))
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = row['policy']['predicted_bbox_xyxy_norm']
    ax.add_patch(Rectangle((x1*w-.5, y1*h-.5), (x2-x1)*w+1, (y2-y1)*h+1,
                           fill=False, edgecolor='#28ee77', linewidth=2, linestyle='--'))
    name = row['run_id'].split('_128steps')[0].upper()
    visibility = row['policy']['visibility_probability']
    status = '目标仍有像素' if target['visible'] else '目标无可见像素'
    ax.set_title(f'{name} · 输入观测 {row["observation_step"]}\n'
                 f'{status}；预测可见性 {visibility:.3f} < 0.90，动作归零',
                 fontproperties=font, fontsize=11, pad=10)
    ax.set_xlim(-.5, w-.5)
    ax.set_ylim(h-.5, -.5)
    ax.axis('off')
fig.suptitle('低置信停车的实际输入画面', fontproperties=font, fontsize=19, y=.975)
fig.text(.5, .943, '黄色实线：原目标真值框    绿色虚线：模型预测框（当前已被停车门停用）',
         ha='center', fontproperties=font, fontsize=11)
fig.text(.5, .025, '固定训练源的开发诊断；真值仅用于标签/审计。无目标像素不区分遮挡与出视野，阈值保持不变。',
         ha='center', fontproperties=font, fontsize=10)
fig.subplots_adjust(top=.90, bottom=.075, hspace=.18, wspace=.08)
output = root / 'visibility_bbox_diagnostic.png'
fig.savefig(output, dpi=150, facecolor='white')
plt.close(fig)
print(json.dumps({'status': 'audited_plot_created', 'path': str(output),
                  'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))
