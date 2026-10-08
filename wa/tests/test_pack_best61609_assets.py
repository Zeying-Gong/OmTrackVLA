import hashlib
import io
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock
from wa.tools.pack_best61609_assets import (
    canonical_absolute, normalize_entries, open_source,
    pack_archive, verify_archive, write_json_new, build_bundle)

class PackAssetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.json'
        self.source.write_bytes(b'{ "unaltered": 1.00 }\n')
        self.entry = {'source': str(self.source), 'sha256': hashlib.sha256(self.source.read_bytes()).hexdigest(),
                      'size': self.source.stat().st_size, 'package': 'evidence'}
        self.entries = normalize_entries([self.entry])
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, self.fd)

    def test_roundtrip_original_bytes_and_archive_sha(self):
        pack_archive(self.fd, 'x.tar', self.entries)
        result = verify_archive(self.fd, 'x.tar', self.entries)
        self.assertEqual(result['files'], 1)
        self.assertEqual(result['payload_bytes'], self.entry['size'])
        self.assertEqual(result['sha256'], hashlib.sha256((self.root/'x.tar').read_bytes()).hexdigest())
        with tarfile.open(self.root/'x.tar') as tf:
            self.assertEqual(tf.getnames(), [str(self.source)[1:]])
            self.assertEqual(tf.extractfile(tf.getmembers()[0]).read(), self.source.read_bytes())

    def test_no_overwrite_archive(self):
        pack_archive(self.fd, 'x.tar', self.entries)
        original = (self.root/'x.tar').read_bytes()
        with self.assertRaises(FileExistsError):
            pack_archive(self.fd, 'x.tar', self.entries)
        self.assertEqual((self.root/'x.tar').read_bytes(), original)

    def test_source_hash_rejects_and_keeps_partial(self):
        self.source.write_bytes(b'X' * self.entry['size'])
        with self.assertRaisesRegex(ValueError, 'source hash differs'):
            pack_archive(self.fd, 'x.tar', self.entries)
        self.assertTrue((self.root/'x.tar').is_file())

    def test_size_rejects(self):
        self.source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'source size differs'):
            pack_archive(self.fd, 'x.tar', self.entries)

    def test_direct_symlink_rejected(self):
        link = self.root/'link'
        link.symlink_to(self.source)
        with self.assertRaises(OSError):
            with open_source(str(link)):
                self.fail('followed symlink')

    def test_ancestor_symlink_rejected(self):
        link = self.root/'linked'
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(OSError):
            with open_source(str(link/self.source.name)):
                self.fail('followed ancestor symlink')

    def test_nonregular_fifo_rejected_without_block(self):
        fifo = self.root/'fifo'
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, 'not regular'):
            with open_source(str(fifo)):
                self.fail('accepted fifo')

    def test_noncanonical(self):
        for s in ('relative', '//tmp/a', '/tmp/../a', '/tmp//a', '/tmp/a/'):
            with self.subTest(s=s), self.assertRaises(ValueError):
                canonical_absolute(s)

    def test_manifest_rejects_duplicate_extra_invalid(self):
        for value in ([self.entry, self.entry], [dict(self.entry, sha256='wrong')],
                      [dict(self.entry, package='unknown')], [dict(self.entry, size=True)],
                      [dict(self.entry, extra=1)], []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_entries(value)

    def test_archive_wrong_hash(self):
        pack_archive(self.fd, 'x.tar', self.entries)
        bad = [dict(self.entries[0], sha256='0'*64)]
        with self.assertRaisesRegex(ValueError, 'content hash'):
            verify_archive(self.fd, 'x.tar', bad)

    def test_archive_extra_or_duplicate_rejected(self):
        for name in ('unlisted', self.entries[0]['archive_member']):
            with tarfile.open(self.root/'bad.tar', 'w') as tf:
                for member_name in (self.entries[0]['archive_member'], name):
                    ti = tarfile.TarInfo(member_name)
                    ti.size = self.entry['size']
                    ti.mode = 0o600
                    tf.addfile(ti, io.BytesIO(self.source.read_bytes()))
            with self.assertRaisesRegex(ValueError, 'trailing'):
                verify_archive(self.fd, 'bad.tar', self.entries)

    def test_trailing_hidden_data_rejected(self):
        pack_archive(self.fd, 'x.tar', self.entries)
        with (self.root/'x.tar').open('ab') as stream:
            stream.write(b'secret trailing data')
        with self.assertRaisesRegex(ValueError, 'trailing'):
            verify_archive(self.fd, 'x.tar', self.entries)

    def test_json_no_overwrite(self):
        info = write_json_new(self.fd, 'complete.json', {'status': 'example'})
        self.assertEqual(info['sha256'], hashlib.sha256((self.root/'complete.json').read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            write_json_new(self.fd, 'complete.json', {'status': 'replace'})

    def test_cli_output_scope_rejected_before_manifest(self):
        for path in (self.root/'output', Path('/data/nas_ray/elsewhere'), Path('/tmp')):
            with self.assertRaisesRegex(ValueError, 'bounded best61609'):
                build_bundle(path)

    def test_member_padding_rejects_hidden_bytes(self):
        pack_archive(self.fd, 'x.tar', self.entries)
        with (self.root/'x.tar').open('r+b') as stream:
            stream.seek(512 + self.entry['size'])
            stream.write(b'X')
        with self.assertRaisesRegex(ValueError, 'member padding'):
            verify_archive(self.fd, 'x.tar', self.entries)

    def test_truncated_zero_footer_rejected(self):
        pack_archive(self.fd, 'x.tar', self.entries)
        with (self.root/'x.tar').open('r+b') as stream:
            stream.truncate((self.root/'x.tar').stat().st_size - 1)
        with self.assertRaisesRegex(ValueError, 'trailing'):
            verify_archive(self.fd, 'x.tar', self.entries)

    def test_canonical_long_pax_header(self):
        entries = [dict(self.entries[0], archive_member='data/' + 'longname'*30 + '/file')]
        pack_archive(self.fd, 'x.tar', entries)
        result = verify_archive(self.fd, 'x.tar', entries)
        self.assertEqual(result['files'], 1)

    def test_json_sync_error_before_publication(self):
        real_fsync = os.fsync
        def sync(fd):
            if fd == self.fd:
                raise OSError('injected directory sync failure')
            return real_fsync(fd)
        with mock.patch('wa.tools.pack_best61609_assets.os.fsync', side_effect=sync):
            with self.assertRaises(OSError):
                write_json_new(self.fd, 'complete.json', {'status': 'PASS'})
        self.assertFalse((self.root/'complete.json').exists())
        self.assertFalse(list(self.root.glob('.*.tmp')))

    def test_json_complete_visible_only_at_link(self):
        real_link = os.link
        def link(*args, **kwargs):
            self.assertFalse((self.root/'complete.json').exists())
            return real_link(*args, **kwargs)
        with mock.patch('wa.tools.pack_best61609_assets.os.link', side_effect=link):
            write_json_new(self.fd, 'complete.json', {'status': 'PASS'})
        self.assertFalse(list(self.root.glob('.*.tmp')))

if __name__ == '__main__':
    unittest.main()
