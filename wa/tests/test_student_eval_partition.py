import copy,json,tempfile,unittest
from pathlib import Path
from wa.wm.student_eval_partition import task_scope,write_partition
from wa.wm.student_eval_finalize import validate_shard
from wa.wm.student_eval_contract import REPAIR_SHA
from wa.wm.initial_bbox_repair import KEYS,VERSION

class PartitionTests(unittest.TestCase):
    def fixture(self):
        manifest=dict(tasks={});rows=[]
        for task in ('stt','dt','at'):
            keys=[k for t,k in sorted(KEYS) if t==task]
            keys += ['scene/'+str(i) for i in range(1405-len(keys))]
            manifest['tasks'][task]=dict(episodes=[dict(key=k,shard=i%8) for i,k in enumerate(keys)])
            for key in keys:
                r=dict(task=task,key=key,mode='mixed',checkpoint_sha256='a'*64,checkpoint_step=59065,semantic_protocol='mp3d_semantic_ply_v1')
                if (task,key) in KEYS:r.update(initialization_repair=VERSION,initialization_repair_plan_sha256=REPAIR_SHA)
                rows.append(r)
        return manifest,rows,dict(checkpoint_sha='a'*64,step=59065)
    def test_scope(self):
        self.assertEqual(task_scope({}),('stt','dt','at'))
        for t in ('stt','dt','at'):
            self.assertEqual(task_scope(dict(WA_STUDENT_EVAL='evaluation_set_adaptation_v1',WA_EVAL_TASK=t)),(t,))
        for env in (dict(WA_EVAL_TASK='stt'),dict(WA_STUDENT_EVAL='evaluation_set_adaptation_v1',WA_EVAL_TASK='all')):
            with self.assertRaises(ValueError):task_scope(env)
    def test_24_lanes_disjoint_full_coverage(self):
        manifest,rows,c=self.fixture();seen=set()
        for task in ('stt','dt','at'):
            for i in range(8):
                keys={e['key'] for e in manifest['tasks'][task]['episodes'] if e['shard']==i}
                part=[r for r in rows if r['task']==task and r['key'] in keys]
                validate_shard(part,dict(episodes=len(part)),manifest,i,(task,))
                ids={(r['task'],r['key']) for r in part}
                self.assertFalse(seen & ids);seen |= ids
        self.assertEqual(len(seen),4215)
    def test_write_and_foreign(self):
        manifest,rows,c=self.fixture()
        for task in ('stt','dt','at'):
            part=[r for r in rows if r['task']==task]
            with tempfile.TemporaryDirectory() as d:
                write_partition(d,part,manifest,c,task)
                report=json.loads((Path(d)/'PARTITION_COMPLETE.json').read_text())
                self.assertEqual(report['episodes'],1405);self.assertTrue(report['global_pair_audit_pending'])
                with self.assertRaises(ValueError):write_partition(d,part[:-1],manifest,c,task)
                foreign=dict(part[0],task='at' if task!='at' else 'stt')
                with self.assertRaises(ValueError):write_partition(d,part[1:]+[foreign],manifest,c,task)

if __name__=='__main__':unittest.main()
