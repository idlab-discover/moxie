"""External pin checks stay independent of live Moxie source changes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dependencies import verify_external_dependencies


class ExternalDependenciesTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root/'source.mojo').write_text('original')
        self.manifest = self.root/'DEPENDENCIES.json'
        self.data = {'projects': [
            {'id': 'numojo', 'path': '.', 'revision': 'a'*40,
             'assets': {'numojo/source.mojo': 'source.mojo'},
             'files': {'source.mojo': hashlib.sha256(b'original').hexdigest()}},
            {'id': 'moxie', 'path': 'live-moxie', 'files': {'base.mojo': 'old'}},
        ]}
        self.git = patch('dependencies.subprocess.check_output', return_value='a'*40+'\n').start()
        self.addCleanup(patch.stopall)

    def verify(self):
        self.manifest.write_text(json.dumps(self.data))
        verify_external_dependencies(self.manifest)

    def test_accepts_pinned_external_and_live_moxie(self):
        self.verify()
        (self.root/'live-moxie').mkdir()
        (self.root/'live-moxie/base.mojo').write_text('changed live source')
        self.verify()

    def test_rejects_revision_drift(self):
        self.git.return_value = 'b'*40
        with self.assertRaisesRegex(ValueError, 'revision drift'):
            self.verify()

    def test_rejects_working_source_drift(self):
        (self.root/'source.mojo').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'content drift'):
            self.verify()

    def test_rejects_missing_source(self):
        (self.root/'source.mojo').unlink()
        with self.assertRaisesRegex(ValueError, 'missing NuMojo source'):
            self.verify()

    def test_rejects_missing_pin(self):
        del self.data['projects'][0]['revision']
        with self.assertRaisesRegex(ValueError, 'missing NuMojo revision'):
            self.verify()

    def test_rejects_unpinned_asset(self):
        self.data['projects'][0]['assets']['new'] = 'new.mojo'
        with self.assertRaisesRegex(ValueError, 'missing NuMojo asset pins'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
