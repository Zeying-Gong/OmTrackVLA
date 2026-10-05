import copy
import unittest
from wa.tests.test_dual_teacher_selection import branch
from wa.wm.dual_teacher_selection import select_teacher
from wa.wm.dual_teacher_metrics import summarize_teachers

def fixture():
    rows=[]
    for task in ('stt','dt','at'):
        for key in ('scene/1','scene/2'):
            a,b=branch('lightnav'),branch('oracle')
            for x in (a,b):
                x.update(task=task,key=key)
                x['result'].update(following_step=4,total_step=5)
            row=select_teacher(a,b);row['branches']=dict(lightnav=a,oracle=b);rows.append(row)
    manifest=dict(tasks={t:dict(episodes=[dict(key='scene/1'),dict(key='scene/2')]) for t in ('stt','dt','at')},reference_steps={'scene/1':15})
    return rows,manifest

class TeacherMetricsTests(unittest.TestCase):
    def test_reference_normalization(self):
        rows,m=fixture();r=summarize_teachers(rows,m)['metrics_percent']['stt']['lightnav']
        self.assertEqual(r['TR'],40);self.assertEqual(r['macro_TR'],80)
        self.assertEqual(r['reference_missing'],1);self.assertEqual(r['episodes'],2)
    def test_coverage(self):
        rows,m=fixture()
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):summarize_teachers(bad,m)
    def test_selection_tampering(self):
        rows,m=fixture();rows[0]['selected_teacher']='oracle'
        with self.assertRaises(ValueError):summarize_teachers(rows,m)
    def test_invalid_counts(self):
        for value in (-1,float('nan'),True,6):
            rows,m=fixture();r=rows[0]['branches']['lightnav']['result'];r['following_step']=value
            rows[0]['results']['lightnav']=copy.deepcopy(r)
            with self.assertRaises(ValueError):summarize_teachers(rows,m)
    def test_invalid_retained(self):
        rows,m=fixture();a=rows[0]['branches']['lightnav'];a['result'].update(success=False,policy_init_valid=False,following_step=0,following_rate=0)
        rows[0].update(select_teacher(a,rows[0]['branches']['oracle']))
        r=summarize_teachers(rows,m)['metrics_percent']['stt']['lightnav']
        self.assertEqual(r['episodes'],2);self.assertEqual(r['SR'],50);self.assertEqual(r['invalid_init_count'],1)
