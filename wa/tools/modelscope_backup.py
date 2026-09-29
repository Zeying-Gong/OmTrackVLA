"""Private, resumable NAS -> ModelScope -> NAS archive transfer; no credentials in arguments."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
from modelscope_hub import HubApi

REPO = 'a597836509/wa-evt-jepa-private-backup-20260929'
ROOT = Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
NAS = Path('/data/nas_ray')

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)

def private(api):
    r=api.get_repo(REPO,'dataset')
    if not r.private or r.license.lower()!='other':raise RuntimeError('private/other repository required')

def retry(fn):
    for n in range(6):
        try:return fn()
        except Exception as e:
            print('network operation failed',type(e).__name__,'attempt',n+1,flush=True)
            if n==5:raise
            time.sleep(30)

def upload(api,out):
    inventory=ROOT/'artifacts/h100_migration_v1/file_sizes.tsv'
    files=[]
    for line in inventory.read_text().splitlines():
        size,name=line.split('\t',1)
        files.append(('home/zeying.gong/'+name,int(size)))
    for folder in [ROOT/'artifacts/robot_transition_audit_v2',ROOT/'dependencies/wla_v1/src']:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
                files.append((str(p.relative_to(NAS)),p.stat().st_size))
    cp=NAS/'project/md-ak/users/zeying.gong/job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt'
    files.append((str(cp.relative_to(NAS)),cp.stat().st_size))
    if len({n for n,_ in files})!=len(files):raise ValueError('duplicate backup path')
    chunks=[];current=[];size=0
    for name,n in files:
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or '\n' in name:raise ValueError('unsafe archive path')
        if current and size+n>1024**3:chunks.append(current);current=[];size=0
        current.append((name,n));size+=n
    if current:chunks.append(current)
    plan={'repo':REPO,'file_count':len(files),'source_bytes':sum(n for _,n in files),'shards':len(chunks),'inventory_sha256':digest(inventory)}
    save(out/'plan.json',plan)
    manifest_path=out/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(plan,complete=False,archives=[])
    for key in ('file_count','source_bytes','shards','inventory_sha256'):
        if manifest[key]!=plan[key]:raise ValueError('backup plan changed')
    for index,chunk in enumerate(chunks):
        name=f'shard-{index:04d}.tar';archive=out/name
        if index<len(manifest['archives']):
            if digest(archive)!=manifest['archives'][index]['sha256']:raise ValueError('local archive changed')
            continue
        if not archive.exists():
            listing=out/f'shard-{index:04d}.files'
            listing.write_text(''.join(n+'\n' for n,_ in chunk))
            for n,expected in chunk:
                p=NAS/n
                if p.is_symlink() or not p.is_file() or p.stat().st_size!=expected:raise ValueError('source mismatch '+n)
            partial=archive.with_suffix('.tar.partial')
            subprocess.run(['tar','--format=posix','--verbatim-files-from','-C',str(NAS),'-T',str(listing),'-cf',str(partial)],check=True)
            os.replace(partial,archive)
        record={'file':name,'bytes':archive.stat().st_size,'sha256':digest(archive),'files':len(chunk)}
        private(api)
        retry(lambda:api.upload_file(REPO,'dataset',archive,'archives/'+name,disable_tqdm=True))
        manifest['archives'].append(record);save(manifest_path,manifest)
        retry(lambda:api.upload_file(REPO,'dataset',manifest_path,'manifest.json',disable_tqdm=True))
        print('UPLOADED',index+1,'/',len(chunks),record,flush=True)
    manifest['complete']=True;save(manifest_path,manifest)
    retry(lambda:api.upload_file(REPO,'dataset',manifest_path,'manifest.json',disable_tqdm=True))
    print('BACKUP_COMPLETE',plan,flush=True)

def download(api,out):
    state_path=out/'verified.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'repo':REPO,'archives':[],'complete':False}
    done={r['file']:r for r in state['archives']}
    while True:
        private(api)
        try:
            p=api.download_file(REPO,'dataset','manifest.json',local_dir=out/'metadata',force=True)
        except Exception as e:
            if type(e).__name__ not in ('NotFoundError','EntryNotFoundError','FileNotFoundError'):raise
            print('WAITING_FOR_FIRST_MANIFEST',flush=True);time.sleep(30);continue
        manifest=json.loads(Path(p).read_text())
        for r in manifest['archives']:
            if r['file'] in done:
                if done[r['file']]!=r:raise ValueError('remote archive changed')
                continue
            p=retry(lambda:api.download_file(REPO,'dataset','archives/'+r['file'],local_dir=out,expected_sha256=r['sha256']))
            if Path(p).stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise ValueError('download integrity mismatch')
            state['archives'].append(r);done[r['file']]=r;save(state_path,state)
            print('VERIFIED',len(done),'/',manifest['shards'],r['file'],flush=True)
        if manifest['complete']:
            if len(done)!=manifest['shards']:raise ValueError('incomplete manifest')
            state['complete']=True;save(state_path,state);print('DOWNLOAD_COMPLETE',flush=True);return
        time.sleep(30)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['upload','download']);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    api=HubApi();private(api)
    (upload if a.mode=='upload' else download)(api,out)
