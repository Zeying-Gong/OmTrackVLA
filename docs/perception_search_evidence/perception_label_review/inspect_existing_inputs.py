"""Local read-only summary of frozen collection inputs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / 'evidence_v4/outputs/takeover/long_prefix_admission_v2/manifest.json'
C = ROOT / 'evidence_v4/outputs/takeover/long_prefix_collection_v1/frozen_collection_plan.json'
manifest = json.loads(M.read_text(encoding='utf-8'))
collection = json.loads(C.read_text(encoding='utf-8'))
print('collection keys', list(collection))
for key in collection:
    if key not in ('entries', 'code_artifacts', 'source_selection_audits'):
        value = collection[key]
        if len(str(value)) < 1600: print(key, json.dumps(value))
print('manifest identities')
for row in manifest['samples']:
    print(row['partition_role'], row['identity'], row['artifacts']['rollout']['path'])
source_root = ROOT / 'evidence_longrollout_and_profile/outputs/takeover/phase2_train4_long_rollouts_v1'
for path in source_root.glob('*/result.json'):
    value = json.loads(path.read_text(encoding='utf-8'))
    print('SOURCE', path.parent.name, {k:value.get(k) for k in ['task','dataset_index','scene_id','episode_id','assigned_humanoid_semantic_ids','initialization']})
    print('SUMMARY', value['summary'])
    launch = json.loads((path.parent/'launch_contract.json').read_text())
    print('SIM', launch['run'].get('original_sim_configuration'))
    print('LAST METRIC', {k:v for k,v in value['steps'][-1]['evaluation_only_after_action'].items() if k!='render_audit'})
    print('RENDER_KEYS', list(value['steps'][0]['evaluation_only_after_action']['render_audit']))
    print('CONFIG_INVENTORY', [s for s in launch.get('sources',[]) if 'inventory' in s['path']])
