"""Read-only real-schema smoke of an isolated loader candidate, NumPy only."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parent
candidate = Path(r'C:\Users\59783\Desktop\OmTrackVLA_takeover\omtrackvla\data\perception_sidecar.py').read_bytes()
payload = {'base64': base64.b64encode(candidate).decode(), 'sha256': hashlib.sha256(candidate).hexdigest()}
remote_code = '''
import base64, hashlib, importlib.util, json, os, pathlib, sys, traceback
import numpy as np
repo = pathlib.Path('/data/nfs/share/wam_tracking/OmTrackVLA')
stage = repo/'.codex_upload/perception_sidecar_loader_review_v1'
stage.mkdir(exist_ok=False)
payload = json.load(sys.stdin)
data = base64.b64decode(payload['base64'], validate=True)
assert hashlib.sha256(data).hexdigest() == payload['sha256']
path = stage/'perception_sidecar.py'
with path.open('xb') as stream: stream.write(data)
original = repo/'omtrackvla/data/perception_sidecar.py'
before = hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else None
os.environ['CUDA_VISIBLE_DEVICES'] = ''
report = {'candidate_sha256': payload['sha256'], 'original_runtime_module_sha256_before': before,
          'gpu_started': False, 'model_loaded': False, 'optimizer_created': False}
try:
    spec = importlib.util.spec_from_file_location('isolated_perception_sidecar', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    store = module.PerceptionSidecarStore(
        verification_path=repo/'outputs/takeover/perception_sidecar_independent_admission_v1/verification.json',
        verification_sha256='aec0690b2d2a3727d7e3ba80afce07670876568e49ed2028e48d6545d70cfc5d',
        plan_path=repo/'.codex_upload/perception_label_collection_v1/frozen_plan.json',
        plan_file_sha256='dd9fcd6b00249a802410f1638216f56a64af51ef998fd1fc9d6566317e080780', artifact_root=repo)
    assert len(store.sample_paths) == len(set(store.sample_paths)) == 19
    sample_path = repo/'outputs/takeover/long_prefix_collection_v1/stt_0000_128steps__anchor027/sample/sample.json'
    supervision = store.supervision_for(sample_path, learning_steps=4, for_optimizer=False)
    labels = supervision.as_numpy()
    assert supervision.sequence_length == 28 and supervision.learning_start == 24
    assert set(labels) == {'target_visible', 'visibility_label_valid', 'target_bbox', 'bbox_label_valid'}
    assert labels['target_visible'].dtype == labels['target_bbox'].dtype == np.float32
    assert labels['visibility_label_valid'].dtype == labels['bbox_label_valid'].dtype == np.bool_
    assert labels['target_bbox'].shape == (28,4)
    assert all(labels[key].shape == (28,) for key in labels if key != 'target_bbox')
    assert labels['target_visible'][24:].tolist() == [1.,1.,1.,0.]
    assert labels['visibility_label_valid'][24:].tolist() == [True]*4
    assert labels['bbox_label_valid'][24:].tolist() == [True,True,True,False]
    assert not labels['visibility_label_valid'][:24].any() and not labels['bbox_label_valid'][:24].any()
    assert not labels['target_visible'][:24].any() and not labels['target_bbox'][:24].any()
    assert np.isfinite(labels['target_bbox']).all() and not labels['target_bbox'][27].any()
    # Compare full-precision raw label semantics after the documented float32 conversion.
    plan = json.loads((repo/'.codex_upload/perception_label_collection_v1/frozen_plan.json').read_text())
    entry = next(e for e in plan['entries'] if e['run_id'] == 'stt_0000_128steps')
    label_path = pathlib.Path(entry['output_dir'])/'labels.jsonl'
    raw_bytes = label_path.read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == supervision.labels_sha256
    raw = [json.loads(line) for line in raw_bytes.decode().splitlines()]
    for k in (24,25,26):
        assert np.array_equal(labels['target_bbox'][k], np.asarray(raw[k]['target']['bbox_xyxy_norm'], dtype=np.float32))
    rejected = False
    try:
        store.supervision_for(sample_path, for_optimizer=True)
    except module.PerceptionSidecarError:
        rejected = True
    assert rejected
    assert 'torch' not in sys.modules
    report.update(status='real_numpy_schema_smoke_passed', admitted_sample_count=19,
        sample_path=str(sample_path), sample_sha256=supervision.sample_sha256,
        sequence_length=28, learning_start=24, labels_sha256=supervision.labels_sha256,
        suffix={key:value[24:].tolist() for key,value in labels.items()},
        original_prefix_labels_invalid=True, optimizer_access_rejected=True, torch_imported=False)
except Exception as error:
    report.update(status='real_numpy_schema_smoke_failed', error=f'{type(error).__name__}: {error}',
                  traceback=traceback.format_exc())
after = hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else None
report.update(original_runtime_module_sha256_after=after, original_runtime_module_unchanged=before==after)
assert before == after
with (stage/'REAL_SCHEMA_SMOKE_RESULT.json').open('x') as stream: json.dump(report,stream,indent=2)
print(json.dumps(report))
'''
spec = importlib.util.spec_from_file_location('loader_smoke_access', r'C:\Users\59783\Desktop\OmTrackVLA\.codex_remote.py')
access = importlib.util.module_from_spec(spec); spec.loader.exec_module(access)
transport = access.connect()
try:
    channel = transport.open_session(timeout=30); channel.settimeout(120)
    channel.exec_command('/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python -B -c '+shlex.quote(remote_code))
    channel.sendall(json.dumps(payload).encode()); channel.shutdown_write()
    stdout=channel.makefile('rb').read(); stderr=channel.makefile_stderr('rb').read()
    if channel.recv_exit_status(): raise RuntimeError(stderr.decode(errors='replace'))
    report=json.loads(stdout)
    (ROOT/'perception_loader_real_schema_smoke.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))
finally:
    transport.close()
