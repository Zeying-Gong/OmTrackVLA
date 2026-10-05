"""Explicit in-set dataset; original dataset admission remains strict."""
import json
from pathlib import Path
from wa.data import sha
from wa.wm.robot_data import RobotWorldData, CONTRACT
from wa.wm.dual_teacher_selection import EXPERIMENT

class DualTeacherData(RobotWorldData):
    def __init__(self, cache, *, development=False):
        cache = Path(cache)
        manifest = json.loads((cache/'complete.json').read_text())
        if manifest['experiment'] != EXPERIMENT:
            raise ValueError('not an explicit in-set cache')
        if manifest['development_only'] and not development:
            raise ValueError('development cache cannot enter formal training')
        release_path = Path(manifest['source_release'])
        if sha(release_path) != manifest['source_sha256']:
            raise ValueError('paired release changed')
        release = json.loads(release_path.read_text())
        if not release['paired_outcomes_validated']:
            raise ValueError('unverified paired outcomes')
        if not development and release['expected'] != 4215:
            raise ValueError('complete paired collection required')
        for name, expected in manifest['files'].items():
            if sha(cache/name) != expected: raise ValueError('cache changed: '+name)
        inventory = cache/'source_image_hashes.json'
        if sha(inventory) != manifest['source_image_inventory_sha256']:
            raise ValueError('image inventory changed')
        self.image_hashes = json.loads(inventory.read_text())
        self.admissions = {str(Path(e['branch']).resolve()): e for e in release['teacher_demonstrations']}
        index = cache/'index'
        audit = json.loads((index/'audit.json').read_text())
        if audit['contract'] != CONTRACT or audit['experiment'] != EXPERIMENT:
            raise ValueError('index contract mismatch')
        if audit['cache_complete_sha256'] != sha(cache/'complete.json'):
            raise ValueError('index cache mismatch')
        if audit['splits']['train']['index_sha256'] != sha(index/'train_valid.npy'):
            raise ValueError('row index changed')
        super().__init__(cache, 'train', index)

    def validate_identity(self, meta, root):
        admission = self.admissions.get(str(root.resolve()))
        if admission is None: raise ValueError('unreleased episode')
        for name, expected in admission['hashes'].items():
            if sha(root/name) != expected: raise ValueError('teacher artifact changed')
        for name, expected in self.image_hashes[str(root.resolve())].items():
            if sha(root/name) != expected: raise ValueError('teacher image changed')
        if meta.get('experiment') != EXPERIMENT or meta.get('partition') != 'evaluation_adaptation':
            raise ValueError('unmarked in-set data')
        if not meta.get('camera_alignment_verified'):
            raise ValueError('unaligned camera')
        if meta.get('initial_bbox_status') not in ('VERIFIED_CONFIG_AND_SEMANTIC', 'VERIFIED_FROZEN_FIRST_RGB_REPAIR'):
            raise ValueError('unverified initial identity')
        if (meta['task'], meta['key'], meta['teacher']) != (admission['task'], admission['key'], admission['teacher']):
            raise ValueError('teacher identity mismatch')
        result = json.loads((root/'result.json').read_text())
        if not result['success'] or result['collision']:
            raise ValueError('failed branch is not a demonstration')
