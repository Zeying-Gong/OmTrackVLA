"""Verify a frozen label-side replay plan; only --execute starts GPU collection.

There is no model/controller construction, oracle planning, or training path.
Each worker executes every original source action exactly once. Raw semantic
observations are labels/audit only and never choose an action.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from collection_contract import (artifact, assignments, atomic_status, canonical_sha,
    compare_observation, compare_policy_rgb, digest, finite_number, load, matrix, require, verify_plan, write_new)
from semantic_labels import array_sha256, label_observation


class HabitatReplay:
    """Lazy simulator adapter reproducing probe_next007_recovery_sequence_v2 reset/replay."""
    def __init__(self, entry, plan):
        self.entry, self.plan, self.env = entry, plan, None

    def __enter__(self):
        repo = Path(self.plan['repository'])
        require(Path.cwd().resolve() == repo.resolve(), 'worker must run from the frozen repository')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '3'
                and os.environ.get('MAGNUM_CUDA_DEVICE') == '0', 'worker must use physical GPU 3 only')
        sys.path.insert(0, str(repo/'habitat-lab'))
        sys.path.insert(0, str(repo))
        import habitat
        import evt_bench  # noqa: F401 -- existing structured configuration registration
        from habitat.config import read_write
        from habitat.datasets import make_dataset
        from omegaconf import OmegaConf
        from omtrackvla.evaluation import end_to_end_closed_loop as runtime
        self.runtime = runtime
        config = runtime.configure(habitat.get_config(str(Path(self.entry['entry_config']['path']).relative_to(repo))),
            self.entry['resolved_config']['habitat']['simulator']['scene_dataset'])
        with read_write(config):
            for key in (runtime.RGB_KEY, runtime.PANOPTIC_KEY):
                if key not in config.habitat.gym.obs_keys:
                    config.habitat.gym.obs_keys.append(key)
        resolved = OmegaConf.to_container(config, resolve=True)
        require(canonical_sha(resolved) == self.entry['resolved_config_sha256'],
                'current resolved config differs from original source before environment creation')
        dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
        index = self.entry['dataset_index']
        require(0 <= index < len(dataset.episodes), 'dataset index out of range')
        episode = dataset.episodes[index]
        require(str(episode.episode_id) == self.entry['episode_id']
                and str(episode.scene_id) == self.entry['scene_id'], 'resolved dataset identity mismatch')
        dataset.episodes = [episode]
        self.env = habitat.TrackEnv(config=config, dataset=dataset)
        return self

    def reset(self):
        self.reset_task_observations = self.env.reset()
        self.assigned = self.runtime._assign_unique_humanoid_semantic_ids(self.env)
        require(self.assigned == assignments(self.entry['assigned_humanoid_semantic_ids']),
                'reset assigned semantic IDs differ from original source')
        require(str(self.env.current_episode.episode_id) == self.entry['episode_id']
                and str(self.env.current_episode.scene_id) == self.entry['scene_id'], 'reset switched episode')

    def capture(self, step):
        # Both arrays come from this single observation dictionary. Never render
        # RGB and semantic sensors through separate calls or advancing actions.
        time_before = float(self.env.sim.get_world_time())
        camera_before = self.runtime._camera_transform(self.env.sim)
        observations = self.env.sim.get_sensor_observations()
        rgb = np.asarray(observations[self.runtime.RGB_KEY])[..., :3].copy()
        panoptic = np.asarray(observations[self.runtime.PANOPTIC_KEY]).copy()
        camera_after = self.runtime._camera_transform(self.env.sim)
        time_after = float(self.env.sim.get_world_time())
        robot = self.env.sim.agents_mgr[1].articulated_agent
        target = self.env.sim.agents_mgr[0].articulated_agent
        distance = float(np.linalg.norm(np.asarray(robot.base_pos, dtype=np.float64)
                                       - np.asarray(target.base_pos, dtype=np.float64)))
        init = None
        if step == 0:
            bbox, origin = self.runtime.initialization_bbox(
                self.reset_task_observations, observations, self.assigned[0])
            init = dict(source=origin, bbox_xyxy=None if bbox is None else list(bbox),
                bbox_xyxy_norm=None if bbox is None else list(self.runtime.normalized_bbox(bbox, rgb.shape)),
                environment_step=0, used_frames=[0], robot_action_before_initialization='fixed_zero_hold')
        return dict(rgb=rgb, panoptic=panoptic, assigned_ids=dict(self.assigned),
            world_time_before_render_s=time_before, world_time_s=time_after,
            camera_before_render=camera_before, camera_after_render=camera_after,
            gt_distance_m=distance, terminal=bool(self.env.episode_over), initialization=init)

    def step(self, action):
        require(not self.env.episode_over, 'episode ended before next frozen source action')
        self.env.step({'action': self.runtime.ACTION_NAMES, 'action_args': {'agent_1_base_vel': list(action)}})

    def __exit__(self, exc_type, exc, tb):
        if self.env is not None:
            self.env.close()


def collect_episode(entry, plan, output, backend_factory=HabitatReplay):
    """Collect all reset..terminal observations; mismatches remain failed evidence."""
    output = Path(output)
    frames_dir = output/'observations'
    frames_dir.mkdir(exist_ok=False)
    source = load(entry['source']['path'])
    actions = [[float(row['policy']['action'][key]) for key in ('forward', 'lateral', 'yaw')]
               for row in source['steps']]
    require(canonical_sha(actions) == entry['saved_actions_sha256'], 'frozen action stream changed')
    progress = dict(run_id=entry['run_id'], status='collecting', observations_saved=0,
        expected_observations=entry['observation_count'], completed_source_actions=0,
        failed_check_count=0, checked_prefix_observations=0, natural_terminal_reproduced=False,
        formal_training_eligible=False, optimizer_input_allowed=False, test_locked_used=False,
        model_loaded=False, gt_used_only_on_label_or_audit_side=True)
    atomic_status(output/'worker_progress.json', progress)
    previous_time = None
    previous_rgb = None
    checks_total = 0
    with (output/'labels.jsonl').open('x', encoding='utf-8') as labels_stream:
        with backend_factory(entry, plan) as backend:
            backend.reset()
            for step in range(len(actions) + 1):
                capture = backend.capture(step)
                rgb, raw_panoptic = capture['rgb'], capture['panoptic']
                require(capture['assigned_ids'] == assignments(entry['assigned_humanoid_semantic_ids']),
                        'per-frame semantic assignment changed')
                expected_terminal = step == len(actions)
                label = label_observation(rgb=rgb, panoptic=raw_panoptic,
                    assigned_humanoid_semantic_ids=capture['assigned_ids'], environment_step=step,
                    world_time_s=capture['world_time_s'], terminal_observation=capture['terminal'])
                # Store raw shape and dtype unchanged. helper hashes normalized
                # HxW panoptic; preserve the distinct raw-array hash as well.
                rgb_path = frames_dir/f'rgb_{step:04d}.png'
                panoptic_path = frames_dir/f'panoptic_{step:04d}.npy'
                with rgb_path.open('xb') as stream:
                    Image.fromarray(rgb).save(stream, format='PNG')
                with panoptic_path.open('xb') as stream:
                    np.save(stream, raw_panoptic, allow_pickle=False)
                checks = compare_observation(source, step, rgb, capture['camera_after_render'],
                    capture['gt_distance_m'], label['target']['visible'], capture['world_time_s'],
                    entry['prefix_samples'], plan['replay_tolerance'], plan['world_time_tolerance_s'])
                rgb_checks, rgb_statistics = compare_policy_rgb(source, step, rgb, previous_rgb,
                    plan.get('rgb_statistics_tolerance_0_255', 1e-5))
                checks.extend(rgb_checks)
                before, after = capture['world_time_before_render_s'], capture['world_time_s']
                require(finite_number(before) and before >= 0, 'invalid pre-render world time')
                camera_delta = float(np.abs(matrix(np.asarray(capture['camera_before_render']).tolist())
                    - matrix(np.asarray(capture['camera_after_render']).tolist())).max())
                checks.extend([
                    {'kind': 'single_render_worldtime_stable', 'passed': before == after,
                     'before_s': before, 'after_s': after},
                    {'kind': 'single_render_camera_stable', 'passed': camera_delta <= plan['replay_tolerance'],
                     'maximum_absolute_error': camera_delta},
                    {'kind': 'worldtime_strictly_increasing', 'passed': previous_time is None or after > previous_time},
                    {'kind': 'exact_natural_terminal_step', 'passed': capture['terminal'] == expected_terminal,
                     'expected': expected_terminal, 'actual': capture['terminal']},
                ])
                if step == 0:
                    checks.append({'kind': 'same_original_initialization',
                                   'passed': capture['initialization'] == entry['initialization'],
                                   'actual': capture['initialization'], 'expected': entry['initialization']})
                label.update(rgb_file={'path': str(rgb_path.relative_to(output)), 'sha256': digest(rgb_path)},
                    raw_panoptic_file={'path': str(panoptic_path.relative_to(output)), 'sha256': digest(panoptic_path),
                        'raw_array_sha256': array_sha256(raw_panoptic), 'shape': list(raw_panoptic.shape),
                        'dtype': str(raw_panoptic.dtype)},
                    source_audit={'source_result_sha256': entry['source']['sha256'],
                        'capture_evidence': {
                            'world_time_before_render_s': before, 'world_time_after_render_s': after,
                            'camera_before_render': np.asarray(capture['camera_before_render']).tolist(),
                            'camera_after_render': np.asarray(capture['camera_after_render']).tolist(),
                            'terminal_observation': capture['terminal'],
                            'assigned_humanoid_semantic_ids': {str(k):v for k,v in capture['assigned_ids'].items()},
                            'initialization': capture['initialization'],
                        },
                        'action_just_replayed': None if step == 0 else actions[step - 1],
                        'camera_after_render': np.asarray(capture['camera_after_render']).tolist(),
                        'gt_distance_m': capture['gt_distance_m'],
                        'source_policy_input_rgb_statistics': rgb_statistics,
                        'rgb_equal_previous': None if previous_rgb is None else bool(np.array_equal(rgb, previous_rgb)),
                        'checks': checks, 'passed': all(v['passed'] for v in checks)},
                    formal_training_eligible=False, optimizer_input_allowed=False)
                labels_stream.write(json.dumps(label, sort_keys=True, allow_nan=False) + '\n')
                labels_stream.flush()
                os.fsync(labels_stream.fileno())
                checks_total += len(checks)
                progress['observations_saved'] += 1
                progress['completed_source_actions'] = step
                progress['failed_check_count'] += sum(not v['passed'] for v in checks)
                progress['checked_prefix_observations'] += sum(v['kind'] == 'stored_prefix_rgb_and_worldtime' for v in checks)
                progress['natural_terminal_reproduced'] = bool(expected_terminal and capture['terminal'])
                atomic_status(output/'worker_progress.json', progress)
                previous_time, previous_rgb = after, rgb.copy()
                if step < len(actions):
                    require(not capture['terminal'], 'early natural termination prevents completing the frozen action stream')
                    backend.step(actions[step])
            require(progress['observations_saved'] == entry['observation_count'], 'terminal frame missing')
    progress.update(status='sidecar_collected_pending_independent_admission'
                    if progress['failed_check_count'] == 0 else 'failed_alignment',
                    total_checks=checks_total, labels_sha256=digest(output/'labels.jsonl'))
    write_new(output/'worker_result.json', progress)
    atomic_status(output/'worker_progress.json', progress)
    return progress


def worker(entry, plan):
    output = Path(entry['output_dir'])
    launch = load(output/'launch_contract.json')
    require(launch['plan_sha256'] == plan['plan_sha256'] and launch['entry'] == entry,
            'worker requires the original batch launch contract')
    for ref in plan['artifacts']:
        artifact(ref['path'], ref['sha256'])
    result = collect_episode(entry, plan, output)
    return 0 if result['status'] == 'sidecar_collected_pending_independent_admission' else 2


def initial_batch(plan):
    return dict(stage=plan['stage'], plan_sha256=plan['plan_sha256'], status='pending',
        expected_source_count=4, expected_observation_count=370, formal_training_eligible=False,
        optimizer_input_allowed=False, test_locked_used=False,
        entries=[dict(run_id=e['run_id'], output_dir=e['output_dir'], expected_observations=e['observation_count'],
                      status='pending', exit_code=None) for e in plan['entries']])


def execute_plan(plan_path, plan):
    """One child at a time; native crashes cannot erase the fixed four-entry denominator."""
    output = Path(plan['output_root'])
    output.mkdir(parents=False, exist_ok=False)
    batch = initial_batch(plan)
    batch['status'] = 'running'
    atomic_status(output/'batch_status.json', batch)
    write_new(output/'frozen_plan.json', plan)
    cancelled = False
    process = None
    def interrupt(signum, frame):
        raise KeyboardInterrupt('batch interrupted')
    previous_handlers = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for index, entry in enumerate(plan['entries']):
            row = batch['entries'][index]
            if cancelled:
                row.update(status='cancelled_before_start')
                continue
            directory = Path(entry['output_dir'])
            try:
                directory.mkdir(exist_ok=False)
                env = dict(os.environ, **plan['environment'], CUDA_VISIBLE_DEVICES='3',
                    OMTRACKVLA_EGL_VENDOR_JSON=str(directory/'nvidia_egl.json'), PYTHONDONTWRITEBYTECODE='1')
                command = ['bash', str(Path(plan['repository'])/'scripts/run_egl.sh'), plan['python']['path'],
                    '-B', str(Path(plan['bundle'])/'collect_perception_sidecar.py'), '--plan', str(plan_path),
                    '--execute', '--worker-entry', str(index)]
                write_new(directory/'launch_contract.json', dict(plan_sha256=plan['plan_sha256'],
                    entry=entry, command_argv=command, environment_overrides={k:env[k] for k in
                        (*plan['environment'], 'CUDA_VISIBLE_DEVICES', 'OMTRACKVLA_EGL_VENDOR_JSON')},
                    formal_training_eligible=False, optimizer_input_allowed=False))
                row['status'] = 'running'
                atomic_status(output/'batch_status.json', batch)
                started = time.monotonic()
                with (directory/'collector.log').open('xb') as log:
                    process = subprocess.Popen(command, cwd=plan['repository'], env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = process.wait(timeout=plan['maximum_worker_wall_seconds'])
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        code = process.returncode
                        row['error'] = 'worker timeout; no retry or replacement'
                process = None
                row.update(exit_code=code, elapsed_wall_s=time.monotonic()-started)
                result_path = directory/'worker_result.json'
                if result_path.is_file():
                    row['worker_result'] = load(result_path)
                    row['worker_result_sha256'] = digest(result_path)
                row['status'] = ('sidecar_collected_pending_independent_admission'
                    if code == 0 and 'error' not in row
                    and row.get('worker_result', {}).get('status') == 'sidecar_collected_pending_independent_admission'
                    else 'failed_retained')
            except KeyboardInterrupt:
                cancelled = True
                row.update(status='cancelled', error='batch interrupted; no retry or replacement')
                if process is not None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    row['exit_code'] = process.returncode
                    process = None
            except Exception as error:
                row.update(status='failed_retained', error=f'{type(error).__name__}: {error}')
            atomic_status(output/'batch_status.json', batch)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        for row in batch['entries']:
            if row['status'] in ('pending', 'running'):
                row['status'] = 'not_completed_retained'
        passed = sum(r['status'] == 'sidecar_collected_pending_independent_admission' for r in batch['entries'])
        batch.update(status='sidecars_pending_independent_admission' if passed == 4 else 'failed_batch_retained',
            successful_source_count=passed, all_four_sources_retained=True,
            observed_frames=sum(r.get('worker_result', {}).get('observations_saved', 0) for r in batch['entries']))
        atomic_status(output/'batch_status.json', batch)
    return 0 if batch['successful_source_count'] == 4 else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-entry', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    plan_path = args.plan.resolve(strict=True)
    plan = load(plan_path)
    report = verify_plan(plan)
    if args.worker_entry is not None:
        require(args.execute and 0 <= args.worker_entry < 4, 'worker requires explicit execute and a frozen index')
        return worker(plan['entries'][args.worker_entry], plan)
    print(json.dumps(report, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    require(not Path(plan['output_root']).exists(), 'refusing to overwrite existing sidecar output')
    return execute_plan(plan_path, plan)


if __name__ == '__main__':
    raise SystemExit(main())
