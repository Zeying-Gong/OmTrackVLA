"""Read-only CPU probes of verifier completeness; never used as a collection plan."""
import copy
import json
from pathlib import Path
import tempfile

from collection_contract import EXPECTED, STAGE, artifact, canonical_sha, load, validate_source, verify_plan


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'evidence_longrollout_and_profile/outputs/takeover/phase2_train4_long_rollouts_v1'


def seal(plan):
    plan.pop('plan_sha256', None)
    plan['plan_sha256'] = canonical_sha(plan)
    return plan


def outcome(plan):
    try:
        verify_plan(seal(plan), verify_files=True)
        return 'accepted'
    except (ValueError, KeyError, TypeError) as exc:
        return 'rejected: ' + str(exc)


def main():
    with tempfile.TemporaryDirectory(prefix='omtrack-label-contract-review-') as temporary:
        repo = Path(temporary).resolve()
        root = repo / 'outputs/takeover/sidecar'
        roles = {value[2]: 'train' for value in EXPECTED.values()}
        entries, refs = [], []
        for index, run_id in enumerate(EXPECTED):
            directory = SOURCE / (run_id if run_id == 'stt_0000_128steps' else run_id + '_gpu3_v2')
            input_dir = repo / run_id
            input_dir.mkdir()
            references = {}
            for key, name in [('source', 'result.json'), ('source_launch', 'launch_contract.json'),
                              ('source_status', 'execution_status.json')]:
                path = input_dir / name
                path.write_bytes((directory / name).read_bytes())
                references[key] = artifact(path)
                refs.append(references[key])
            source = load(references['source']['path'])
            identity = validate_source(source, load(references['source_status']['path']),
                                       load(references['source_launch']['path']), run_id,
                                       references['source']['sha256'], roles)
            entries.append(dict(identity, **references, run_id=run_id,
                                permanent_partition_role='train', output_dir=str(root / run_id),
                                prefix_samples=[{'unverified_fake_prefix': True}] * ([5, 5, 5, 4][index])))
        plan = dict(stage=STAGE, physical_gpu=3, maximum_parallel_collectors=1,
                    formal_training_eligible=False, optimizer_input_allowed=False, test_locked_used=False,
                    repository=str(repo), output_root=str(root), permanent_scene_roles=roles,
                    excluded_legacy_train_samples=[{}] * 4, held_out_sample_count=4,
                    entries=entries, artifacts=refs)
        results = {'real_sources_but_19_fake_duplicate_prefixes': outcome(copy.deepcopy(plan))}
        missing = copy.deepcopy(plan)
        missing['artifacts'] = []
        results['all_actual_artifacts_omitted_from_manifest'] = outcome(missing)
        path = Path(entries[0]['source']['path'])
        changed = load(path)
        changed['steps'][0]['evaluation_only_after_action']['gt_distance_m'] += 100.
        path.write_text(json.dumps(changed), encoding='utf-8')
        results['changed_source_with_original_recorded_sha_and_no_artifact_pins'] = outcome(missing)
        results['same_changed_source_with_original_artifact_pins'] = outcome(copy.deepcopy(plan))
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
