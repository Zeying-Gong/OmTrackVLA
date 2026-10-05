import copy
import unittest
from wa.wm.dual_teacher_partitions import partition, validate

class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(task=t, key=str(i)) for t in ('stt','dt','at') for i in range(47)]
    def test_balanced_exact_cover(self):
        for count in (1,2,4,8):
            groups=partition(self.rows,count)
            self.assertTrue(validate(self.rows,groups))
            sizes=[len(lane) for group in groups for lane in group]
            self.assertLessEqual(max(sizes)-min(sizes),1)
    def test_order_independent_no_mutation(self):
        before=copy.deepcopy(self.rows)
        self.assertEqual(partition(self.rows,4),partition(self.rows[::-1],4))
        self.assertEqual(self.rows,before)
    def test_duplicate_remainder(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            partition(self.rows+self.rows[:1],4)
    def test_duplicate_assignment(self):
        g=partition(self.rows,4);g[0][0].append(g[1][0][0])
        with self.assertRaisesRegex(ValueError,'duplicate'):validate(self.rows,g)
    def test_missing_assignment(self):
        g=partition(self.rows,4);g[0][0].pop()
        with self.assertRaisesRegex(ValueError,'missing'):validate(self.rows,g)
    def test_foreign_assignment(self):
        g=partition(self.rows,4);g[0][0][0]={'task':'at','key':'foreign'}
        with self.assertRaisesRegex(ValueError,'unexpected'):validate(self.rows,g)
    def test_lane_count(self):
        g=partition(self.rows,4);g[0].pop()
        with self.assertRaisesRegex(ValueError,'eight'):validate(self.rows,g)
    def test_count_and_idle_guards(self):
        for count in (0,9,True,1.5):
            with self.assertRaises(ValueError):partition(self.rows,count)
        with self.assertRaisesRegex(ValueError,'idle'):partition(self.rows[:3],1)
    def test_invalid_item(self):
        with self.assertRaisesRegex(ValueError,'invalid'):
            partition([dict(task='other',key='1')]*32,4)

if __name__=='__main__':unittest.main()
