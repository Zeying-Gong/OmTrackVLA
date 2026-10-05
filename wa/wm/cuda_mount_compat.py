"""Process-only workaround for a CUDA bind-mount path reported by /proc/maps.
Never substitutes a different runtime: map_files bytes must equal candidate bytes.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import sys

def resolve(name):
    if not isinstance(name, str):
        return name
    prefix = '/cuda-12.8/targets/x86_64-linux/lib/'
    if not name.startswith(prefix) or '/' in name[len(prefix):]:
        return name
    if not Path(name).name.startswith('libcudart.so.') or Path(name).exists():
        return name
    candidate = Path('/usr/local') / name.lstrip('/')
    try:
        expected = hashlib.sha256(candidate.read_bytes()).digest()
        for line in Path('/proc/self/maps').read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or fields[5] != name:
                continue
            mapped = Path('/proc/self/map_files') / fields[0]
            if hashlib.sha256(mapped.read_bytes()).digest() != expected:
                raise RuntimeError('CUDA mount runtime byte mismatch')
            print('CUDA_MOUNT_ALIAS_VERIFIED', name, '->', str(candidate),
                  'sha256=' + expected.hex(), file=sys.stderr, flush=True)
            return str(candidate)
    except OSError:
        # No readable evidence: retain original path and its original failure.
        return name
    return name

def install():
    if getattr(ctypes.CDLL.__init__, '_wa_cuda_mount_compat', False):
        return
    original = ctypes.CDLL.__init__
    def checked(self, name, *args, **kwargs):
        return original(self, resolve(name), *args, **kwargs)
    checked._wa_cuda_mount_compat = True
    ctypes.CDLL.__init__ = checked

if __name__ == '__main__':
    install()
    import flashinfer.comm
    print('FLASHINFER_CUDA_MOUNT_IMPORT_PASS')
