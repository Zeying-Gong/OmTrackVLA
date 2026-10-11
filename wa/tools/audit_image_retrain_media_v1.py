"""Versioned stored-media/trace review for the 63395 image-only retrain.

Run only after all three 63933/63934/63935 partitions and a fresh frozen
4215-row merge are complete. No simulator, GPU, live output or frozen source
is written. A new report directory contains corrected derived wording; the
original frozen merge summary and its historical wording remain unchanged.
"""
import argparse
import concurrent.futures
import hashlib
import html
import inspect
import json
import math
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

from wa.tools.audit_image_retrain_results_v1 import (
    NEW_AUDIT_SHA, NEW_CHECKPOINT, NEW_EVAL_CONFIGS, NEW_PARTITIONS, NEW_SHA,
    OLD_AUDIT_SHA, PROJECT, STEP, corrected_new_summary, validate_training_audits,
)
from wa.tools.audit_student_goal import file_sha, merge_hashes
from wa.tools.compare_uwb_modes import (
    load_bundle, output_path, validate_bundle, video_reference,
)
from wa.wm.dual_teacher_metrics import summarize_teachers
from wa.wm.full_mixed_contract import MANIFEST_SHA
from wa.wm.student_eval_finalize import TEACHER_SHA, load_teachers

SCHEMA = 'wa.image-retrain-media-trace-audit.v1'
VIDEO_DIMENSIONS = [768, 432]
FRONT_DIMENSIONS = (384, 384)
SNAPSHOT_INTERVAL = 100
# Eight real STT episodes / 13 snapshots: MAE 2.77–4.87, p95 8–15.
# Eight mismatched video/JPEG pairs: MAE 47–75, p95 127–170.
# These bounds leave >3x authentic compression headroom; they are not
# proof of simulator or RPC provenance for unsampled frames.
MAX_SNAPSHOT_MAE = 16.0
MAX_SNAPSHOT_P95 = 45.0
CAPTION = (
    '这是 Job 63395 的图像输入重训权重：从同一 59866 父模型独立续训一轮。'
    '闭环推理使用 RGB 与首帧 GT 框，无 UWB、文本或在线世界预测器。'
    '与直接移除 UWB 输入的 62256 权重分开报告。'
)
OLD_TRAIN_AUDIT = PROJECT/'artifacts/failure_state_training_audit_62256_v2.json'
NEW_TRAIN_AUDIT = PROJECT/'artifacts/failure_state_image_only_training_audit_63395_v1.json'
EVAL_SOURCE = PROJECT/'source_student62256_eval24_v1'
EVAL_COMMIT = '5a23a982c7ef01f7fdec58faf27f6ea623ed9eeb'
ACTUAL_FRAME_WRITER = PROJECT.parent/'OmTrackVLA-da3-polar-20260924/trained_agent.py'
ACTUAL_FRAME_WRITER_SHA = '2b5c0faeff3e633588b9af39a14472055ae263291bed3840ff218a680f2fe3a1'
REVIEW_RECORDER = EVAL_SOURCE/'wa/wm/review_recorder.py'
REVIEW_RECORDER_SHA = '1a079b5336bc09568fae84b2c4e04328e700536e5c5370b401ed77c2d025afaf'
METRIC_FIELDS = (
    'status', 'success', 'collision', 'finish', 'following_rate',
    'following_step', 'total_step', 'policy_init_valid',
)
POSITION_FIELDS = (
    'robot_position_world', 'robot_rotation_world_from_body',
    'target_position_world',
)


def _reject_constant(value):
    raise ValueError('nonfinite JSON constant: '+value)


def json_lines(path):
    data = path.read_bytes()
    if not data:
        raise ValueError('empty JSONL: '+str(path))
    rows = [json.loads(line, parse_constant=_reject_constant)
            for line in data.splitlines()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError('non-object JSONL row: '+str(path))
    return rows, hashlib.sha256(data).hexdigest()


def checked_file(root, path):
    root = Path(root)
    path = Path(path)
    if root.resolve() != root or root not in path.resolve().parents:
        raise ValueError('artifact escaped canonical shard root: '+str(path))
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError('symlink in persisted artifact path: '+str(current))
        current = current.parent
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError('missing or empty persisted artifact: '+str(path))
    return path.stat()


def probe_video(video, ffprobe):
    result = subprocess.run(
        [ffprobe, '-v', 'error', '-show_entries',
         'format=duration:stream=codec_type,width,height', '-of', 'json', str(video)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    metadata = json.loads(result.stdout, parse_constant=_reject_constant)
    duration = float(metadata['format']['duration'])
    streams = [stream for stream in metadata['streams']
               if stream.get('codec_type') == 'video'
               and type(stream.get('width')) is int and stream['width'] > 0
               and type(stream.get('height')) is int and stream['height'] > 0]
    if not math.isfinite(duration) or duration <= 0 or not streams:
        raise ValueError('invalid video metadata: '+str(video))
    return duration, [streams[0]['width'], streams[0]['height']]


def decode_video(video, ffmpeg):
    """Decode every frame once, retaining only frames 0,100,200,... as RGB."""
    frame_bytes = VIDEO_DIMENSIONS[0]*VIDEO_DIMENSIONS[1]*3
    command = [
        str(ffmpeg), '-nostdin', '-threads', '1', '-v', 'error', '-xerror',
        '-err_detect', 'explode', '-i', str(video), '-filter_complex',
        f"[0:v]split=2[all][snap];[snap]select='not(mod(n,{SNAPSHOT_INTERVAL}))'[selected]",
        '-map', '[all]', '-vsync', '0', '-an', '-f', 'null', '-',
        '-map', '[selected]', '-vsync', '0', '-an', '-pix_fmt', 'rgb24',
        '-f', 'rawvideo', 'pipe:1', '-progress', 'pipe:2', '-nostats',
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True,
                                timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError('full video decode failed: '+str(video)) from exc
    progress = result.stderr.decode('utf-8', errors='replace').splitlines()
    counts = [int(line[6:]) for line in progress
              if re.fullmatch(r'frame=[0-9]+', line)]
    if not progress or progress[-1] != 'progress=end' or not counts:
        raise ValueError('full video decode lacks terminal frame count: '+str(video))
    if len(result.stdout) == 0 or len(result.stdout) % frame_bytes:
        raise ValueError('periodic RGB snapshots have invalid byte count: '+str(video))
    samples = [result.stdout[offset:offset+frame_bytes]
               for offset in range(0, len(result.stdout), frame_bytes)]
    return counts[-1], samples


def compare_snapshot(photo, decoded_frame):
    """Compare saved raw-camera JPEG with the video's lower-left front panel."""
    if len(decoded_frame) != VIDEO_DIMENSIONS[0]*VIDEO_DIMENSIONS[1]*3:
        raise ValueError('decoded video frame has invalid RGB byte count')
    with Image.open(photo) as image:
        if image.size != FRONT_DIMENSIONS:
            raise ValueError('saved front JPEG dimensions differ: '+str(photo))
        reference = np.asarray(image.convert('RGB'), dtype=np.int16)
    frame = np.frombuffer(decoded_frame, dtype=np.uint8).reshape(
        VIDEO_DIMENSIONS[1], VIDEO_DIMENSIONS[0], 3)
    panel = frame[48:432, 0:384].astype(np.int16)
    difference = np.abs(reference-panel)
    mae = float(difference.mean())
    p95 = float(np.percentile(difference, 95))
    if (not math.isfinite(mae) or not math.isfinite(p95)
            or mae > MAX_SNAPSHOT_MAE or p95 > MAX_SNAPSHOT_P95):
        raise ValueError('saved JPEG disagrees with decoded video frame: '+str(photo))
    return dict(mae=mae, p95_abs_error=p95)


def _valid_action(action):
    return (isinstance(action, list) and len(action) == 3
            and all(type(value) in (int, float) and math.isfinite(value)
                    for value in action))


def audit_episode(row, ffprobe, probe=None):
    """Check one validated row against all persisted video, JPEG and trace data."""
    video_reference(row)  # Reject path traversal in task/key before joining paths.
    if (row.get('mode'), row.get('checkpoint_sha256'), row.get('checkpoint_step')) != (
            'image', NEW_SHA, STEP):
        raise ValueError('foreign image episode identity')
    root = Path(row['artifact_root'])
    task, key = row['task'], row['key']
    live_dir = root/task/'_live'/key
    photo = live_dir/'step_0000.jpg'
    review = root/task/'_review'/key
    video = review/'review.mp4'
    steps_file = review/'steps.jsonl'
    complete_file = review/'complete.json'
    trace_file = root/task/(key.replace('/', '_')+'.trace.jsonl')
    files = (photo, video, steps_file, complete_file, trace_file)
    before = {str(path): checked_file(root, path) for path in files}
    photo_sha = file_sha(photo)
    if photo_sha != row.get('initial_rgb_sha256'):
        raise ValueError('first JPEG differs from episode SHA: '+str(photo))
    complete_bytes = complete_file.read_bytes()
    complete = json.loads(complete_bytes, parse_constant=_reject_constant)
    traces, trace_sha = json_lines(trace_file)
    steps, steps_sha = json_lines(steps_file)
    total = row.get('total_step')
    frames = complete.get('frames')
    if (type(total) not in (int, float) or not math.isfinite(total)
            or total <= 0 or int(total) != total
            or type(frames) is not int or frames != total
            or len(traces) != frames or len(steps) != frames):
        raise ValueError('trace/review/episode step counts differ: '+str(trace_file))
    if complete.get('policy_inputs_unchanged') is not True:
        raise ValueError('review recorder lacks policy-input declaration')
    if not str(complete.get('video_timebase', '')).startswith('20fps frame playback'):
        raise ValueError('unexpected review video timebase')
    recorded = complete.get('result')
    if not isinstance(recorded, dict) or any(
            recorded.get(name) != row.get(name) for name in METRIC_FIELDS):
        raise ValueError('review completion differs from episode result')
    predictions = 0
    failures = 0
    failure_reason = None
    for index, (trace, step) in enumerate(zip(traces, steps)):
        if (trace.get('step') != index+1 or step.get('step') != index
                or trace.get('mode') != 'image'):
            raise ValueError('trace/review step or mode mismatch: '+str(trace_file))
        action = trace.get('action')
        if not _valid_action(action) or action != step.get('action'):
            raise ValueError('trace/review action mismatch: '+str(trace_file))
        observer = trace.get('observer_state')
        if (not isinstance(observer, dict)
                or observer.get('usage') != 'offline_diagnosis_only_not_policy_input'
                or observer.get('timing') != 'before_current_action_after_previous_action'
                or step.get('usage') != 'observer_only_not_policy_input'
                or any(observer.get(name) != step.get(name)
                       for name in POSITION_FIELDS)):
            raise ValueError('observer-only trace/review state mismatch: '+str(trace_file))
        diagnostics = trace.get('diagnostics')
        if not isinstance(diagnostics, dict):
            raise ValueError('missing trace diagnostics')
        failure = trace.get('failure')
        if failure is None:
            if (diagnostics.get('mode') != 'image'
                    or diagnostics.get('noise_mode') != 'zero'
                    or diagnostics.get('inputs') != 'RGB+first_frame_bbox'
                    or diagnostics.get('uwb') is not None
                    or diagnostics.get('world_predictor_inference') is not False):
                raise ValueError('trace contradicts image/no-UWB runtime contract')
            predictions += 1
        else:
            # The frozen WAAgent can return an all-zero action without an RPC
            # only when the first-frame GT box is invalid. It persists the same
            # failure and empty diagnostics on subsequent no-op steps.
            if (row.get('policy_init_valid') is not False
                    or row.get('success') != 0
                    or not isinstance(failure, str)
                    or not failure.startswith('invalid_initialization_bbox: ')
                    or len(failure) <= len('invalid_initialization_bbox: ')
                    or diagnostics != {} or action != [0., 0., 0.]
                    or (failure_reason is not None and failure != failure_reason)):
                raise ValueError('invalid-init failure exemption does not apply')
            failure_reason = failure
            failures += 1
    if row.get('policy_init_valid') is True:
        if failures or predictions == 0:
            raise ValueError('valid-init episode lacks an image prediction trace')
    elif row.get('policy_init_valid') is False:
        if failures != frames or predictions:
            raise ValueError('invalid-init episode has a prediction trace')
    else:
        raise ValueError('episode policy_init_valid is not boolean')

    duration, dimensions = (probe or probe_video)(video, ffprobe)
    if dimensions != VIDEO_DIMENSIONS:
        raise ValueError('review video dimensions differ from frozen recorder')
    if not math.isfinite(duration) or duration <= 0 or abs(duration-frames/20) > 0.1:
        raise ValueError('video duration differs from recorded 20fps frames')
    decoded_frames, decoded_snapshots = decode_video(
        video, Path(ffprobe).with_name('ffmpeg'))
    if decoded_frames != frames:
        raise ValueError('decoded video frame count differs from recorder steps')
    indices = list(range(0, frames, SNAPSHOT_INTERVAL))
    if len(decoded_snapshots) != len(indices):
        raise ValueError('decoded periodic frame count differs from recorder steps')
    expected_photos = [live_dir/f'step_{index:04d}.jpg' for index in indices]
    if {path.name for path in live_dir.glob('step_*.jpg')} != {
            path.name for path in expected_photos}:
        raise ValueError('saved periodic JPEG set differs from recorder cadence')
    snapshots = []
    for index, path, decoded in zip(indices, expected_photos, decoded_snapshots):
        if index:
            before[str(path)] = checked_file(root, path)
        digest = photo_sha if index == 0 else file_sha(path)
        similarity = compare_snapshot(path, decoded)
        snapshots.append(dict(step=index, sha256=digest, **similarity))
    video_sha = file_sha(video)
    for path in (*files, *expected_photos[1:]):
        after = path.stat()
        prior = before[str(path)]
        if (after.st_size, after.st_mtime_ns) != (prior.st_size, prior.st_mtime_ns):
            raise ValueError('media changed during audit: '+str(path))
    return dict(task=task, key=key, artifact_root=str(root),
                video_path=str(video), video_size=before[str(video)].st_size,
                video_sha256=video_sha, duration_s=duration,
                video_dimensions=dimensions, video_decoded_frames=decoded_frames,
                frame_snapshots=snapshots,
                first_jpeg_sha256=photo_sha, trace_sha256=trace_sha,
                review_steps_sha256=steps_sha,
                complete_sha256=hashlib.sha256(complete_bytes).hexdigest(),
                steps=frames, prediction_steps=predictions,
                invalid_init_noop_steps=failures)

def check_eval_source():
    head = subprocess.run(
        ['git', '-C', str(EVAL_SOURCE), 'rev-parse', 'HEAD'],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    dirty = subprocess.run(
        ['git', '-C', str(EVAL_SOURCE), 'status', '--porcelain'],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout
    if head != EVAL_COMMIT or dirty:
        raise ValueError('frozen evaluation checkout commit/status differs')
    return head


def corrected_summary(saved):
    result = corrected_new_summary(saved)
    result['review_caption'] = CAPTION
    result['derived_review_scope'] = (
        'Original frozen merge summary is unchanged; this is a derived '
        'image-only retrain interpretation backed by the separate training audit.'
    )
    return result


def render_index(summary, records, roots):
    labels = {root: 'source_'+str(index)
              for index, root in enumerate(sorted(roots))}
    table = []
    for record in records:
        video = Path(record['video_path'])
        relative = labels[record['artifact_root']]+'/'+quote(
            str(video.relative_to(record['artifact_root'])), safe='/')
        table.append(
            '<tr><td>'+html.escape(record['task'])+'</td><td>'
            +html.escape(record['key'])+'</td><td><a href="'
            +html.escape(relative, quote=True)+'">视频</a></td></tr>'
        )
    page = (
        '<!doctype html><meta charset="utf-8"><title>WA image-only retrain '
        'media/trace review</title><h1>WA 图像输入重训 · 4215 条闭环结果</h1><p>'
        +html.escape(CAPTION)+'</p><p>存储证据审计：4215 个首帧 JPEG 哈希、'
        '每帧视频解码与精确帧数、每100步所存前视 JPEG 对齐、视频 SHA、'
        '动作 trace 和 observer 回放记录。非快照帧未与模拟器图像逐像素配对；'
        '评测集含训练适配样本，'
        '结果不代表未见测试集泛化。</p><p><a href="summary.json">修正解释与指标</a>'
        ' · <a href="audit.json">审计范围</a> · <a href="episodes.json">逐例索引</a></p>'
        '<pre>'+html.escape(json.dumps(summary['metrics_percent'], indent=2))
        +'</pre><table><tr><th>任务</th><th>样本</th><th>回放</th></tr>'
        +''.join(table)+'</table>'
    )
    return page, labels


def diagnose(merged, manifest_path, teacher_path, ffprobe):
    if Path(ffprobe) != Path('/usr/bin/ffprobe'):
        raise ValueError('expected pinned /usr/bin/ffprobe')
    ffmpeg = Path(ffprobe).with_name('ffmpeg')
    if not ffmpeg.is_file():
        raise ValueError('expected /usr/bin/ffmpeg for full video decode')
    merged = Path(merged)
    if not merged.is_absolute() or merged.resolve() != merged or merged.is_symlink():
        raise ValueError('merged root must be canonical and non-symlink')
    if file_sha(manifest_path) != MANIFEST_SHA or file_sha(teacher_path) != TEACHER_SHA:
        raise ValueError('frozen manifest or teacher selections changed')
    if file_sha(OLD_TRAIN_AUDIT) != OLD_AUDIT_SHA or file_sha(NEW_TRAIN_AUDIT) != NEW_AUDIT_SHA:
        raise ValueError('pinned terminal training audit changed')
    if file_sha(NEW_CHECKPOINT) != NEW_SHA:
        raise ValueError('image-only checkpoint bytes changed')
    provenance = validate_training_audits(
        json.loads(OLD_TRAIN_AUDIT.read_text()),
        json.loads(NEW_TRAIN_AUDIT.read_text()),
    )
    eval_commit = check_eval_source()
    if (file_sha(ACTUAL_FRAME_WRITER) != ACTUAL_FRAME_WRITER_SHA
            or file_sha(REVIEW_RECORDER) != REVIEW_RECORDER_SHA):
        raise ValueError('actual JPEG/video frame-writer source changed')
    config_hashes = {}
    for task, expected in NEW_EVAL_CONFIGS.items():
        path = PROJECT/'checkout/wa/jobs'/('student63395_'+task+'_image_4090_v1.yaml')
        if file_sha(path) != expected:
            raise ValueError('submitted evaluation YAML changed: '+str(path))
        config_hashes[str(path)] = expected
    summary = json.loads((merged/'summary.json').read_text())
    expected_roots = {task: str(root) for task, root in NEW_PARTITIONS.items()}
    if summary.get('partition_roots') != expected_roots:
        raise ValueError('merged rows do not bind all three new Job/Task roots')
    manifest = json.loads(Path(manifest_path).read_text())
    teachers = load_teachers(teacher_path)
    teacher_metrics = summarize_teachers(teachers, manifest)['metrics_percent']
    bundle = load_bundle(merged, manifest, dict(checkpoint_sha=NEW_SHA, step=STEP), 'image')
    validate_bundle(bundle, manifest, dict(checkpoint_sha=NEW_SHA, step=STEP),
                    'image', teachers, teacher_metrics)
    rows = bundle['rows']
    if len(rows) != 4215:
        raise ValueError('full 4215 rows required')
    for row in rows:
        task = row['task']
        root = Path(row['artifact_root'])
        if (task not in NEW_PARTITIONS or root.parent != NEW_PARTITIONS[task]
                or root.name not in {'shard_'+str(i).zfill(2) for i in range(8)}):
            raise ValueError('row media points outside pinned Job/Task shards')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        media = list(pool.map(lambda row: audit_episode(row, ffprobe), rows))
    source_hashes = merge_hashes(
        bundle['hashes'], config_hashes,
        {str(Path(manifest_path).resolve()): MANIFEST_SHA,
         str(Path(teacher_path).resolve()): TEACHER_SHA,
         str(OLD_TRAIN_AUDIT): OLD_AUDIT_SHA,
         str(NEW_TRAIN_AUDIT): NEW_AUDIT_SHA,
         str(NEW_CHECKPOINT): NEW_SHA,
         str(Path(ffprobe).resolve()): file_sha(ffprobe),
         str(ffmpeg.resolve()): file_sha(ffmpeg),
         str(ACTUAL_FRAME_WRITER): ACTUAL_FRAME_WRITER_SHA,
         str(REVIEW_RECORDER): REVIEW_RECORDER_SHA},
        {str(Path(inspect.getfile(fn)).resolve()): file_sha(inspect.getfile(fn))
         for fn in (diagnose, load_bundle, validate_bundle,
                    validate_training_audits, corrected_new_summary, file_sha)},
    )
    for path, sha in source_hashes.items():
        if file_sha(path) != sha:
            raise ValueError('source changed during media/trace audit: '+path)
    derived = corrected_summary(bundle['summary'])
    report = dict(
        schema=SCHEMA, status='PASS_STORED_MEDIA_TRACE_REVIEW',
        episodes=4215, checkpoint_sha256=NEW_SHA, checkpoint_step=STEP,
        eval_source_commit=eval_commit, merged_root=str(merged),
        frame_writer_contract=(
            'Actual benchmark trained_agent.py writes pre-action _live JPEG at step '
            '0/100/... from jaw RGB; frozen review_recorder.py uses the same obs for '
            'the corresponding front panel. Both writer hashes are pinned in source_hashes.'),
        partition_roots=expected_roots, training_provenance=provenance,
        first_jpeg_sha256_verified=4215, video_metadata_duration_verified=4215,
        video_full_decode_and_exact_count_verified=4215,
        video_sha256_recorded=4215,
        periodic_front_jpeg_frames_matched=sum(len(item['frame_snapshots']) for item in media),
        trace_review_step_action_verified=4215,
        observer_only_state_alignment_verified=4215,
        video_scope=('All stored MP4 frames decoded with exact frame count; every available '
                     'step-0/100/... JPEG compared to the corresponding video front panel '
                     'with recorded SHA256 and compression-tolerant error. Other frames '
                     'decoded but not pixel-paired with independently saved camera frames.'),
        trace_scope='Persisted JSONL and recorder consistency; no simulator/RPC replay',
        limits=[
            'Stored files can agree without proving the actual GPU placement, simulator '
            'physics, RPC payloads, camera frames between saved snapshots, or causal isolation.',
            'The server-ready and trace declarations support image/no-UWB/no-text/no-online-WM '
            'admission, but are not a replay or an independent runtime observation.',
            'Evaluation-set adaptation; not unseen-test generalization. Failed episodes '
            'remain in each 1405 denominator.',
        ],
        source_hashes=source_hashes,
    )
    return report, derived, media, teachers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('merged', 'manifest', 'teacher-selections', 'ffprobe', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    protected = [args.merged, args.manifest, args.teacher_selections,
                 OLD_TRAIN_AUDIT, NEW_TRAIN_AUDIT, NEW_CHECKPOINT,
                 *NEW_PARTITIONS.values()]
    destination = output_path(args.output, protected)
    if (destination.parent != PROJECT/'artifacts' or
            re.fullmatch(r'student63395_image_media_trace_review_[0-9]{8}_v[0-9]+',
                         destination.name) is None):
        raise ValueError('review output must be a new versioned original-NAS artifact')
    report, summary, records, teachers = diagnose(
        args.merged, args.manifest, args.teacher_selections, args.ffprobe)
    protected += [row['branches'][branch]['artifact_root']
                  for row in teachers for branch in ('lightnav', 'oracle')]
    destination = output_path(destination, protected)
    page, labels = render_index(summary, records,
                                {item['artifact_root'] for item in records})
    destination.mkdir(parents=True, exist_ok=False)
    for root, label in labels.items():
        (destination/label).symlink_to(root, target_is_directory=True)
    (destination/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    (destination/'episodes.json').write_text(json.dumps(records, allow_nan=False))
    (destination/'index.html').write_text(page)
    report['derived_summary_sha256'] = file_sha(destination/'summary.json')
    report['episode_inventory_sha256'] = file_sha(destination/'episodes.json')
    report['index_sha256'] = file_sha(destination/'index.html')
    (destination/'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(status=report['status'], episodes=4215,
        output=str(destination), audit_sha256=file_sha(destination/'audit.json'))))


if __name__ == '__main__':
    main()
