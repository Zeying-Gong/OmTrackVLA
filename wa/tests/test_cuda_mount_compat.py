import unittest
from unittest.mock import patch
from pathlib import Path
from wa.wm.cuda_mount_compat import resolve

class CudaMountTests(unittest.TestCase):
    source = '/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12.8.90'
    def call(self, mapped=b'ELF same', maps=True, readable=True, exists=False):
        def read(p):
            if str(p).startswith('/proc/self/map_files'):
                if not readable: raise PermissionError('no evidence')
                return mapped
            return b'ELF same'
        line = '100-200 r-xp 00000000 00:42 12 ' + self.source if maps else ''
        with patch.object(Path, 'exists', return_value=exists), \
             patch.object(Path, 'read_bytes', read), \
             patch.object(Path, 'read_text', return_value=line):
            return resolve(self.source)
    def test_same_mapped_bytes(self):
        self.assertEqual(self.call(), '/usr/local' + self.source)
    def test_mismatch_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'byte mismatch'):
            self.call(mapped=b'other runtime')
    def test_unreadable_not_substituted(self):
        self.assertEqual(self.call(readable=False), self.source)
    def test_not_mapped_not_substituted(self):
        self.assertEqual(self.call(maps=False), self.source)
    def test_existing_path_untouched(self):
        self.assertEqual(self.call(exists=True), self.source)
    def test_other_library_untouched(self):
        for name in (None, 'libcudart.so.12', '/cuda-12.8/lib/libcudart.so.12',
                     '/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12',
                     self.source + '/../other'):
            self.assertEqual(resolve(name), name)

if __name__ == '__main__':
    unittest.main()
