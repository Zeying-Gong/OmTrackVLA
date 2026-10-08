"""Bounded wrapper tests; no model, GPU, large assets or uploads."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from wa.tools import pack_best61609_evt_source as module

class SourceInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'empty.py'
        self.source.write_bytes(b'')

    def test_empty_file_retained_and_historical_pin(self):
        digest = hashlib.sha256(b'').hexdigest()
        result = module.entries_for([(str(self.source),'evt_bench')], {str(self.source):digest})
        self.assertEqual(result[0]['size'],0)
        self.assertTrue(result[0]['historical_pin'])
        self.assertEqual(result[0]['archive_member'],str(self.source)[1:])

    def test_cache_excluded_but_regular_tree_retained(self):
        nested = self.root/'nested'; nested.mkdir()
        (nested/'value.yaml').write_bytes(b'x: 1\n')
        cache = self.root/'__pycache__'; cache.mkdir()
        (cache/'ignored.pyc').write_bytes(b'bytecode')
        self.assertEqual(set(module.tree_files(self.root)), {self.source,nested/'value.yaml'})

    def test_file_and_directory_symlinks_rejected(self):
        for target in (self.source,self.root):
            link=self.root/'link'; link.symlink_to(target)
            with self.assertRaisesRegex(ValueError,'symlink'):
                module.tree_files(self.root)
            link.unlink()

    def test_special_file_rejected_without_open(self):
        os.mkfifo(self.root/'fifo')
        with self.assertRaisesRegex(ValueError,'special file'):
            module.tree_files(self.root)

    def test_duplicate_missing_and_wrong_historical_hash(self):
        source=str(self.source)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            module.entries_for([(source,'evt_bench')]*2,{})
        with self.assertRaisesRegex(ValueError,'missing'):
            module.entries_for([(source,'evt_bench')],{str(self.root/'absent'):'0'*64})
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            module.entries_for([(source,'evt_bench')],{source:'0'*64})

    def test_unsafe_path_and_link_rejected(self):
        with self.assertRaises(ValueError):
            module.entries_for([(str(self.root)+'/../unsafe','evt_bench')],{})
        link=self.root/'linked'; link.symlink_to(self.source)
        with self.assertRaises(OSError):
            module.entries_for([(str(link),'evt_bench')],{})

    def test_obvious_credential_pattern_rejected(self):
        self.source.write_bytes(b'-----BEGIN PRIVATE KEY-----')
        with self.assertRaisesRegex(ValueError,'credential'):
            module.entries_for([(str(self.source),'evt_bench')],{})

    def test_pin_list_identity_checked(self):
        folder=self.root/'checkout/wa/wm'; folder.mkdir(parents=True)
        name='list.sha256'; (folder/name).write_bytes(b'changed')
        with patch.object(module,'ROOT',self.root),patch.object(module,'PIN_LISTS',{name:'0'*64}):
            with self.assertRaisesRegex(ValueError,'dependency list changed'):
                module.pinned_dependencies()

    def test_wrapper_failure_and_no_overwrite(self):
        from wa.tools import pack_best61609_assets as helper
        (self.root/'artifacts').mkdir()
        output=self.root/'artifacts/new'
        # Synthetic control-flow fixture only; production inventory/hash functions tested above.
        entries=[dict(source=str(self.root/str(i)),archive_member='fixture/'+str(i),
                      size=16816185 if i==0 else 0,sha256='0'*64,package='fixture') for i in range(373)]
        with patch.object(module,'ROOT',self.root),patch.object(module,'OUTPUT',output), \
             patch.object(module,'HELPERS',{}),patch.object(module,'selected_paths',return_value=[]), \
             patch.object(module,'pinned_dependencies',return_value={}), \
             patch.object(module,'entries_for',return_value=entries), \
             patch.object(helper,'pack_archive'),patch.object(helper,'verify_archive',side_effect=ValueError('verification failed')):
            with self.assertRaisesRegex(ValueError,'verification failed'):
                module.main()
            self.assertTrue((output/'failed.json').is_file())
            self.assertFalse((output/'complete.json').exists())
            original=(output/'manifest.json').read_bytes()
            with self.assertRaises(FileExistsError):
                module.main()
            self.assertEqual((output/'manifest.json').read_bytes(),original)

    def test_success_metadata_and_postpack_inventory_change(self):
        from wa.tools import pack_best61609_assets as helper
        (self.root/'artifacts').mkdir()
        entries=[dict(source=str(self.root/str(i)),archive_member='fixture/'+str(i),
                      size=16816185 if i==0 else 0,sha256='0'*64,package='fixture') for i in range(373)]
        for changed in (False,True):
            with self.subTest(changed=changed):
                output=self.root/'artifacts'/('changed' if changed else 'success')
                snapshots=[[],[('added.py','fixture')] if changed else []]
                with patch.object(module,'ROOT',self.root),patch.object(module,'OUTPUT',output), \
                     patch.object(module,'HELPERS',{}),patch.object(module,'selected_paths',side_effect=snapshots), \
                     patch.object(module,'pinned_dependencies',return_value={}), \
                     patch.object(module,'entries_for',return_value=entries), \
                     patch.object(helper,'pack_archive'),patch.object(helper,'verify_archive',return_value={'files':373}):
                    if changed:
                        with self.assertRaisesRegex(ValueError,'inventory changed'):
                            module.main()
                        self.assertTrue((output/'failed.json').exists())
                        self.assertFalse((output/'complete.json').exists())
                    else:
                        module.main()
                        result=json.loads((output/'complete.json').read_text())
                        self.assertEqual((result['source_config_files'],result['xvfb_debs']),(339,34))
                        self.assertFalse(result['uploaded'])
                        self.assertFalse(result['h100_validated'])
                        self.assertFalse(result['entire_snapshot_historically_verified'])
                        manifest=json.loads((output/'manifest.json').read_text())
                        self.assertEqual(manifest['archive_member_mode'],'0600')
                        self.assertEqual(manifest['executable_restore_allowlist'],[str(module.BENCH/'scripts/runtime/run_xvfb.sh')[1:]])

if __name__ == '__main__':
    unittest.main()
