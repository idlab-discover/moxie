"""Exercise moving HEAD resolution without modifying the real dependency."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import dependencies


class HeadResolutionTest(unittest.TestCase):
    def test_external_head_advances_and_includes_uncommitted_edits(self):
        self.check_head_resolution(False)

    def test_moxie_head_advances_and_includes_uncommitted_edits(self):
        self.check_head_resolution(True)

    def check_head_resolution(self, is_moxie):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'family'
            source = Path(directory) / 'external-owner'
            root.mkdir()
            source.mkdir()

            def git(*args):
                return subprocess.check_output(
                    ['git', '-C', str(source), *args], stderr=subprocess.PIPE)

            git('init')
            git('config', 'user.email', 'test@example.com')
            git('config', 'user.name', 'Dependency test')
            module = source / 'base.mojo'
            module.write_text('first commit\n')
            git('add', 'base.mojo')
            git('commit', '-m', 'First')
            lock = root / 'DEPENDENCIES.lock.json'
            lock.write_text(json.dumps({'schema': 1, 'overrides': [], 'projects': [{
                'id': 'assembly', 'path': '../external-owner', 'revision': 'obsolete-pin',
                'files': {'base.mojo': 'obsolete-hash'}, 'assets': {}, 'exports': None,
                'modules': {'moxie.base': {'file': 'base.mojo'}},
            }]}))
            with patch.object(dependencies, 'ROOT', root), patch.object(dependencies, 'MOXIE', source if is_moxie else root / 'unused-moxie'), patch.object(dependencies, 'moxie_head', side_effect=lambda: git('rev-parse', 'HEAD').decode().strip()):
                first = git('rev-parse', 'HEAD').decode().strip()
                module.write_text('uncommitted\n')
                effective, resolved = dependencies.prepare(root, lock)
                self.assertEqual(resolved, source)
                self.assertEqual((resolved / 'base.mojo').read_text(), 'uncommitted\n')
                self.assertEqual(json.loads(effective.read_text())['projects'][0]['revision'], first)
                git('add', 'base.mojo')
                git('commit', '-m', 'Second')
                effective, resolved = dependencies.prepare(root, lock)
                self.assertNotEqual(json.loads(effective.read_text())['projects'][0]['revision'], first)
                self.assertEqual((resolved / 'base.mojo').read_text(), 'uncommitted\n')
                self.assertEqual(json.loads(effective.read_text())['projects'][0]['files']['base.mojo'],
                                 dependencies.digest(module.read_bytes()))
                git('rm', 'base.mojo')
                git('commit', '-m', 'Remove')
                with self.assertRaisesRegex(ValueError, 'missing dependency source'):
                    dependencies.prepare(root, lock)


if __name__ == '__main__':
    unittest.main()
