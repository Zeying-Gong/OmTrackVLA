"""Run every selected episode once across allocated A800 lanes."""
import os,sys,subprocess,concurrent.futures,json
from pathlib import Path
S=Path(__file__).resolve().parents[2]
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1').split(',')
assert len(devices)==2
def lane(i):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[i],WA_RECOVERY_SLOT=str(i),WA_RECOVERY_SHARDS='2')
 subprocess.run([sys.executable,str(S/'wa/tools/recovery_dev_launch.py'),str(out/f'lane{i}'),'--formal'],env=env,check=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lane,range(2)))
counts=[json.loads((out/f'lane{i}/collection/COMPLETE.json').read_text()) for i in range(2)]
assert all(x['episodes']==x['expected'] and not x['audit_only'] for x in counts)
assert sum(x['episodes'] for x in counts)==96
(out/'COMPLETE.json').write_text(json.dumps(dict(episodes=96,training_release=False,pending='independent branch and label audit')))
