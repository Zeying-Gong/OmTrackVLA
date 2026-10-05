"""Unmodified official teacher transport with opt-in library mount resolution."""
import runpy
from pathlib import Path
from wa.wm.cuda_mount_compat import install

if __name__ == '__main__':
    install()
    transport = Path(__file__).resolve().parents[4] / 'WLA-EVT-20260925/lightnav_transport_20260927/server_transport.py'
    runpy.run_path(str(transport), run_name='__main__')
