import hashlib,json,tempfile,unittest
from pathlib import Path
from wa.tests.test_student_eval_partition import PartitionTests
from wa.wm.student_eval_partition import write_partition
from wa.tools.merge_student_partitions import read_partitions

class MergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.manifest,rows,self.contract=PartitionTests().fixture()
        self.roots={}
        for task in ('stt','dt','at'):
            root=Path(self.tmp.name)/task;root.mkdir();self.roots[task]=root;part=[]
            for i in range(8):
                dest=root/f'shard_{i:02d}';dest.mkdir()
                keys={e['key'] for e in self.manifest['tasks'][task]['episodes'] if e['shard']==i}
                local=[r for r in rows if r['task']==task and r['key'] in keys]
                (dest/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in local))
                (dest/'COMPLETE.json').write_text(json.dumps(dict(episodes=len(local))))
                ready=dict(checkpoint_sha256='a'*64,step=59065,mode='mixed',noise_mode='zero',sampling_steps=4,text_used=False,world_predictor_inference=False)
                (dest/'server_ready.json').write_text(json.dumps(ready))
                part.extend(dict(r,artifact_root=str(dest)) for r in local)
            write_partition(root,part,self.manifest,self.contract,task)
    def test_complete(self):
        rows,hashes=read_partitions(self.roots,self.manifest,self.contract)
        self.assertEqual(len(rows),4215);self.assertEqual(len(hashes),78)
    def test_missing_partition(self):
        (self.roots['at']/'PARTITION_COMPLETE.json').unlink()
        with self.assertRaises(FileNotFoundError):read_partitions(self.roots,self.manifest,self.contract)
    def test_wrong_ready(self):
        p=self.roots['stt']/'shard_00/server_ready.json';r=json.loads(p.read_text());r['step']=1;p.write_text(json.dumps(r))
        with self.assertRaises(ValueError):read_partitions(self.roots,self.manifest,self.contract)
    def test_changed_rows(self):
        p=self.roots['dt']/'combined_episodes.jsonl';p.write_text(p.read_text()+'\n')
        with self.assertRaises(ValueError):read_partitions(self.roots,self.manifest,self.contract)
    def test_forged_combined_disagrees_shards(self):
        root=self.roots['at'];p=root/'combined_episodes.jsonl'
        rows=[json.loads(s) for s in p.read_text().splitlines()];rows[0]['artifact_root']='wrong'
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        q=root/'PARTITION_COMPLETE.json';r=json.loads(q.read_text());r['combined_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();q.write_text(json.dumps(r))
        with self.assertRaises(ValueError):read_partitions(self.roots,self.manifest,self.contract)
    def test_duplicate_roots(self):
        roots=dict(self.roots,at=self.roots['dt'])
        with self.assertRaises(ValueError):read_partitions(roots,self.manifest,self.contract)
if __name__=='__main__':unittest.main()
