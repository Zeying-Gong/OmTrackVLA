import unittest
from wa.wm.full_mixed_contract import allocated_devices
class AllocationTests(unittest.TestCase):
    def test_unset_environment_uses_actual_two(self):
        self.assertEqual(allocated_devices(None,2),["0","1"])
    def test_explicit_mapping(self):
        self.assertEqual(allocated_devices("3,5",2),["3","5"])
    def test_uuid_mapping(self):
        self.assertEqual(allocated_devices("GPU-a,GPU-b",2),["GPU-a","GPU-b"])
    def test_invalid(self):
        for raw,count in [(None,0),(None,9),("",2),("0,0",2),("0,1,2",2),("0",2)]:
            with self.assertRaises(ValueError): allocated_devices(raw,count)
