"""Read-only source discovery/download for the bounded simulator yaw review."""
import importlib.util
from pathlib import Path
import shlex
import sys

HELPER = Path(r'C:\Users\59783\Desktop\OmTrackVLA\.codex_remote.py')
ROOT = '/data/nfs/share/wam_tracking/OmTrackVLA'
spec = importlib.util.spec_from_file_location('omtrack_remote_helper', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
transport = helper.connect()
try:
    if len(sys.argv) == 1:
        command = "grep -R -n -E 'class BaseVel|class BaseVelocity|class TrackEnv|ac_freq_ratio|ctrl_freq|agent_1_base_vel' " + shlex.quote(ROOT + '/habitat-lab/habitat') + " --include='*.py'"
        channel = transport.open_session(timeout=30)
        channel.exec_command(command)
        sys.stdout.buffer.write(channel.makefile('rb').read())
        sys.stderr.buffer.write(channel.makefile_stderr('rb').read())
        if channel.recv_exit_status():
            raise RuntimeError('read-only source discovery failed')
    else:
        import hashlib
        for relative in sys.argv[1:]:
            if relative.startswith('/') or '..' in Path(relative).parts:
                raise ValueError('repository relative paths only')
            local = Path(__file__).parent/'yaw_source_snapshot'/relative
            local.parent.mkdir(parents=True, exist_ok=True)
            channel = transport.open_session(timeout=30)
            channel.exec_command('cat ' + shlex.quote(ROOT+'/'+relative))
            data = channel.makefile('rb').read()
            error = channel.makefile_stderr('rb').read()
            if channel.recv_exit_status():
                raise RuntimeError(error.decode(errors='replace'))
            if data.startswith(b'cat: ') and b'No such file or directory' in data:
                print(relative, 'MISSING_REMOTE_FILE')
                continue
            local.write_bytes(data)
            print(relative, len(data), hashlib.sha256(data).hexdigest())
finally:
    transport.close()
