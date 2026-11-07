"""Portable publishing integrity and pre-import stop checks, standard library only."""
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from scripts.fetch_training_bundle import extract_verified


class PortableTrainingTests(unittest.TestCase):
    def test_training_denied_before_jax_import(self):
        code = "from scripts.portable_training import require_training_authorization; require_training_authorization()"
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Persistent user stop', result.stderr)
        self.assertNotIn('No module named', result.stderr)

    def test_archive_bad_hash_and_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bundle.tar.gz'
            with tarfile.open(path, 'w:gz') as tar:
                item = tarfile.TarInfo('data/portable-training-20261107/../../escape')
                item.size = 1
                tar.addfile(item, io.BytesIO(b'x'))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                extract_verified(path, Path(folder)/'out', '0'*64)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                extract_verified(path, Path(folder)/'out', digest)

    def test_archive_links_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bundle.tar.gz'
            with tarfile.open(path, 'w:gz') as tar:
                item = tarfile.TarInfo('data/portable-training-20261107/link')
                item.type = tarfile.SYMTYPE
                item.linkname = 'outside'
                tar.addfile(item)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                extract_verified(path, Path(folder)/'out', digest)

    def test_archive_extract_and_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bundle.tar.gz'
            with tarfile.open(path, 'w:gz') as tar:
                item = tarfile.TarInfo('data/portable-training-20261107/ok.json')
                item.size = 2
                tar.addfile(item, io.BytesIO(b'{}'))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            out = Path(folder)/'out'
            self.assertEqual(extract_verified(path, out, digest), 1)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                extract_verified(path, out, digest)


if __name__ == '__main__':
    unittest.main()
