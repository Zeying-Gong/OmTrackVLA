import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

spec = importlib.util.spec_from_file_location('exchange', Path(__file__).parents[1] / 'tools/exchange.py')
exchange = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exchange)

class ExchangeTest(unittest.TestCase):
    def test_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'environment.json').write_text('{}')
            (root / 'checkpoint.pt').write_bytes(b'not exported')
            bundle = root / 'result.zip'
            expected = exchange.pack(root, bundle)
            self.assertEqual(exchange.verify(bundle), expected)
            self.assertEqual(expected['kind'], 'preflight_only')
            with self.assertRaises(FileExistsError):
                exchange.pack(root, bundle)

    def test_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / 'bad.zip'
            with zipfile.ZipFile(bundle, 'w') as archive:
                archive.writestr('environment.json', '{}')
                archive.writestr('manifest.json', json.dumps(dict(schema_version=1,
                    files={'environment.json': 'wrong hash'})))
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                exchange.verify(bundle)

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / 'bad.zip'
            with zipfile.ZipFile(bundle, 'w') as archive:
                archive.writestr('../escape', 'bad')
            with self.assertRaises(ValueError):
                exchange.verify(bundle)
