"""Check persisted paired start evidence without confusing RGBA and RGB hashes."""
import hashlib
import json
from pathlib import Path
from PIL import Image
from wa.wm.recovery_replay import check_dynamic_state

def verify_pair_evidence(row):
    pair = row['pair']
    starts, images, hashes = [], [], {}
    for teacher in ('lightnav', 'oracle'):
        root = Path(row['branches'][teacher]['artifact_root'])
        start_path = root / 'pair_start.json'
        start_bytes = start_path.read_bytes()
        start = json.loads(start_bytes)
        if start['rgb'] != pair['initial_rgb_sha256']:
            raise ValueError('persisted start RGB hash differs')
        observations_path = root / 'observations.json'
        observations_bytes = observations_path.read_bytes()
        first = json.loads(observations_bytes)[0]
        if first['sim_step'] != 0 or first['timestamp_s'] != 0:
            raise ValueError('first observation is not start state')
        frame_path = root / first['frame']
        frame_bytes = frame_path.read_bytes()
        with Image.open(frame_path) as image:
            if image.mode != 'RGB':raise ValueError('expected persisted RGB image')
            images.append((image.size, image.tobytes()))
        starts.append(start)
        hashes[teacher] = {str(p): hashlib.sha256(data).hexdigest() for p,data in
            ((start_path,start_bytes),(observations_path,observations_bytes),(frame_path,frame_bytes))}
    check_dynamic_state(starts[0]['state'], starts[1]['state'])
    state_sha = hashlib.sha256(json.dumps(starts[0]['state'], sort_keys=True).encode()).hexdigest()
    if state_sha != pair['takeover_state_sha256']:
        raise ValueError('canonical start state hash differs')
    if images[0] != images[1]:raise ValueError('persisted teacher RGB pixels differ')
    # Runtime hashes include the original sensor's nonconstant alpha channel.
    # Saved PNGs deliberately contain only RGB; their hash must NOT be equated.
    for files in hashes.values():
        for name, expected in files.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError('pair evidence changed during verification')
    return hashes
