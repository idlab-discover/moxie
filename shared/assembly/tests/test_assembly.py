"""Failure tests use disposable sources and never mutate dependency owners."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from assemble import AssemblyError, assemble, digest, plan, read_json
from consume import consume


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'builds')
        self.root = Path(self.temporary.name)
        source = ROOT / 'tests/fixtures/consumer_b/DEPENDENCIES.lock.json'
        self.lock = json.loads(source.read_text())
        for project in self.lock['projects']:
            project['path'] = str((source.parent / project['path']).resolve())
        self.path = self.root / 'lock.json'

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        self.path.write_text(json.dumps(self.lock))
        return self.path

    def execute(self):
        return plan(self.root, self.save())

    def fails(self, message):
        with self.assertRaisesRegex(AssemblyError, message):
            self.execute()

    def project(self, name):
        return next(p for p in self.lock['projects'] if p['id'] == name)

    def local(self):
        (self.root / 'module.mojo').write_text('def marker() -> Int:\n    return 1\n')
        p = {'id': 'local', 'path': str(self.root), 'revision': self.project('moxie')['revision'],
             'files': {'module.mojo': digest((self.root / 'module.mojo').read_bytes())},
             'modules': {'moxie.local': {'file': 'module.mojo'}}, 'assets': {},
             'exports': {'schema': 1, 'packages': {}}}
        self.lock['projects'].append(p)
        return p

    def test_real_baseline_deduplication_and_override(self):
        result = self.execute()
        receipt = json.loads(result['ASSEMBLY.json'])
        self.assertEqual(receipt['modules']['moxie.base'], ['dummy-family', 'moxie'])
        self.assertEqual(receipt['modules']['moxie.dummy'], ['consumer'])
        self.assertNotIn(b'DummyClassifier', result['moxie/__init__.mojo'])
        self.assertIn(b'replacement_marker', result['moxie/__init__.mojo'])
        self.assertEqual(result['moxie/_dense.mojo'], (ROOT.parent / 'numerics/src/moxie/_dense.mojo').read_bytes())
        self.assertNotIn('moxie/_sparse.mojo', result)

    def test_deterministic_independent_of_project_order(self):
        first = self.execute()
        self.lock['projects'].reverse()
        second = self.execute()
        # Receipt records input order/provenance; generated executable sources do not.
        self.assertEqual({k: v for k, v in first.items() if k != 'ASSEMBLY.json'},
                         {k: v for k, v in second.items() if k != 'ASSEMBLY.json'})

    def test_collision_requires_override(self):
        self.lock['overrides'] = []
        self.fails('module collision: moxie.dummy')

    def test_identical_nonbaseline_collision(self):
        p = copy.deepcopy(self.project('numerics'))
        p['id'] = 'duplicate'
        self.lock['projects'].append(p)
        self.fails('module collision')

    def test_false_baseline(self):
        self.project('moxie')['modules']['moxie.base']['baseline'] = '0' * 40
        self.fails('Git verification failed')

    def test_export_collision(self):
        self.project('consumer')['exports']['packages']['moxie']['BaseEstimator'] = 'moxie.probe_b'
        self.fails('export collision')

    def test_missing_export_target(self):
        self.project('consumer')['exports']['packages']['moxie']['Absent'] = 'moxie.absent'
        self.fails('export lacks module ownership')

    def test_matching_baseline_exports_deduplicate(self):
        self.project('dummy-family')['exports']['packages']['moxie']['BaseEstimator'] = 'moxie.base'
        output = self.execute()
        self.assertEqual(output['moxie/__init__.mojo'].count(b'import BaseEstimator\n'), 1)

    def test_asset_directory_collision(self):
        p = self.local()
        p['assets'] = {'licenses/a': 'module.mojo', 'licenses/a/b': 'module.mojo'}
        self.fails('file/directory collision')

    def test_invalid_export_manifest(self):
        self.project('consumer')['exports']['schema'] = 2
        self.fails('unsupported exports schema')

    def test_manifest_injection(self):
        self.project('consumer')['exports']['packages']['moxie']['x\nimport bad'] = 'moxie.probe_b'
        self.fails('invalid export symbol')

    def test_invalid_module_name(self):
        p = self.local()
        p['modules'] = {'moxie...escape': {'file': 'module.mojo'}}
        self.fails('invalid module')

    def test_missing_file(self):
        p = self.local()
        (self.root / 'module.mojo').unlink()
        self.fails('missing file')

    def test_source_drift(self):
        self.local()
        (self.root / 'module.mojo').write_text('changed')
        self.fails('source drift')

    def test_git_revision_drift(self):
        self.project('numerics')['revision'] = '0' * 40
        self.fails('revision drift')

    def test_path_traversal(self):
        p = self.local()
        for name in ('../escape', '/tmp/escape', 'a/../../escape', 'a\\escape', './module.mojo', 'a//b'):
            with self.subTest(name=name):
                p['files'] = {name: '0' * 64}
                self.fails('path traversal|invalid relative path')

    def test_source_symlink_escape(self):
        p = self.local()
        (self.root / 'module.mojo').unlink()
        (self.root / 'module.mojo').symlink_to(ROOT / 'tools/assemble.py')
        self.fails('source symlink escape')

    def test_unpinned_module(self):
        self.project('consumer')['files'] = {}
        self.fails('unpinned file')

    def test_override_must_name_exact_losers(self):
        self.lock['overrides'][0]['from'].append('numerics')
        self.fails('override owners mismatch')

    def test_unused_override(self):
        self.lock['overrides'][0]['module'] = 'moxie.absent'
        self.fails('module collision|unused overrides')

    def test_missing_package_manifest(self):
        self.project('moxie')['exports']['packages'].pop('moxie.utils')
        self.fails('missing package manifest')

    def test_module_package_collision(self):
        self.project('consumer')['exports']['packages']['moxie.probe_b'] = {}
        self.fails('module/package collision')

    def test_asset_collision(self):
        p = copy.deepcopy(self.project('numojo'))
        p['id'] = 'duplicate'
        self.lock['projects'].append(p)
        self.fails('asset collision')

    def test_unknown_schema_and_fields(self):
        self.lock['schema'] = 2
        self.fails('unsupported lock schema')
        self.lock['schema'] = 1
        self.lock['typo'] = True
        self.fails('invalid fields')

    def test_duplicate_json_keys(self):
        with self.assertRaisesRegex(AssemblyError, 'duplicate JSON key'):
            read_json('{"schema": 1, "schema": 1}')

    def test_stale_output_removed_and_invalid_input_preserves_output(self):
        target = self.root / 'builds/deps'
        assemble(self.root, self.save(), target)
        (target / 'stale.mojo').write_text('stale')
        assemble(self.root, self.save(), target)
        self.assertFalse((target / 'stale.mojo').exists())
        receipt = (target / 'ASSEMBLY.json').read_bytes()
        self.project('numerics')['revision'] = '0' * 40
        with self.assertRaisesRegex(AssemblyError, 'revision drift'):
            assemble(self.root, self.save(), target)
        self.assertEqual((target / 'ASSEMBLY.json').read_bytes(), receipt)

    def test_output_escape(self):
        with self.assertRaisesRegex(AssemblyError, 'output must be'):
            assemble(self.root, self.save(), self.root / 'src')
        (self.root / 'builds').symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaisesRegex(AssemblyError, 'symlink escape'):
            assemble(self.root, self.save(), self.root / 'builds/deps')

    def test_tool_pin_verified_before_execution(self):
        self.project('assembly')['files']['tools/assemble.py'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'tool drift'):
            consume(self.root, self.save(), ROOT / 'tools/assemble.py')


if __name__ == '__main__':
    unittest.main()
