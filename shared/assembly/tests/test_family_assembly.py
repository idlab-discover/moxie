"""Family independence contracts using disposable Git repositories."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from assemble_family import assemble


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE).decode().strip()


def commit(root):
    git(root, 'add', '.')
    git(root, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')
    return git(root, 'rev-parse', 'HEAD')


class FamilyAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.family, self.external = self.root / 'family', self.root / 'external'
        for root in [self.family, self.external]:
            root.mkdir()
            git(root, 'init', '-q')
            (root / '.gitignore').write_text('builds/\n')
            (root / 'module.mojo').write_text('def value() -> Int:\n    return 1\n')
            commit(root)
        self.lock = {'schema': 1, 'projects': [self.entry(self.family, 'owned'), self.entry(self.external, 'dependency')], 'overrides': []}
        self.path = self.family / 'DEPENDENCIES.lock.json'

    def tearDown(self):
        self.temp.cleanup()

    def entry(self, root, name):
        return {'id': name, 'path': str(root), 'revision': git(root, 'rev-parse', 'HEAD'),
                'files': {'module.mojo': hashlib.sha256((root / 'module.mojo').read_bytes()).hexdigest()},
                'modules': {'moxie.' + name: {'file': 'module.mojo'}}, 'assets': {},
                'exports': {'schema': 1, 'packages': {'moxie': {}}}}

    def run_assembly(self):
        self.path.write_text(json.dumps(self.lock))
        return assemble(self.family, self.path, self.family / 'builds/deps')

    def test_working_source_changes_without_repinning(self):
        self.run_assembly()
        original = self.path.read_bytes()
        data = 'def value() -> Int:\n    return 2\n'
        (self.family / 'module.mojo').write_text(data)
        output = self.run_assembly()
        self.assertEqual((output / 'moxie/owned.mojo').read_text(), data)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertIn('development', json.loads((output / 'ASSEMBLY.json').read_text()))

    def test_external_advance_uses_exact_old_bytes_on_cold_build(self):
        expected = (self.external / 'module.mojo').read_bytes()
        (self.external / 'module.mojo').write_text('changed external source')
        commit(self.external)
        output = self.run_assembly()
        self.assertEqual((output / 'moxie/dependency.mojo').read_bytes(), expected)

    def test_cached_sibling_can_be_absent(self):
        self.run_assembly()
        self.external.rename(self.root / 'moved')
        self.run_assembly()

    def test_unrecoverable_external_hash_rejected_without_losing_output(self):
        output = self.run_assembly()
        before = (output / 'ASSEMBLY.json').read_bytes()
        self.lock['projects'][1]['files']['module.mojo'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'source drift'):
            self.run_assembly()
        self.assertEqual((output / 'ASSEMBLY.json').read_bytes(), before)

    def test_module_conflicts_still_rejected(self):
        self.lock['projects'][0]['modules'] = {'moxie.dependency': {'file': 'module.mojo'}}
        with self.assertRaisesRegex(ValueError, 'module collision'):
            self.run_assembly()

    def test_shared_baseline_still_immutable(self):
        self.lock['projects'][0]['modules']['moxie.owned']['baseline'] = git(self.family, 'rev-parse', 'HEAD')
        (self.family / 'module.mojo').write_text('changed shared baseline')
        with self.assertRaisesRegex(ValueError, 'baseline drift'):
            self.run_assembly()

    def test_child_consumes_live_family_engine(self):
        child = self.family / 'algorithms' / 'example'
        child.mkdir(parents=True)
        self.path = child / 'DEPENDENCIES.lock.json'
        self.path.write_text(json.dumps(self.lock))
        (self.family / 'module.mojo').write_text('family engine edit')
        output = assemble(child, self.path, child / 'builds/deps')
        self.assertEqual((output / 'moxie/owned.mojo').read_text(), 'family engine edit')

    def test_metadata_only_external_project_has_a_valid_snapshot_root(self):
        nested = self.external / 'shared/readiness'
        nested.mkdir(parents=True)
        self.lock['projects'].append({
            'id': 'readiness', 'path': str(nested),
            'revision': git(self.external, 'rev-parse', 'HEAD'),
            'files': {}, 'modules': {}, 'assets': {},
            'exports': {'schema': 1, 'packages': {}}})
        self.run_assembly()

    def test_upstream_metadata_is_not_a_build_gate(self):
        self.lock['projects'][1]['files']['AGENTS.md'] = '0' * 64
        self.lock['projects'][1]['files']['DEPENDENCIES.lock.json'] = '0' * 64
        self.run_assembly()

    def test_existing_optimize_asset_namespace(self):
        self.lock['projects'][1]['assets'] = {'mojo_optimize/probe.mojo': 'module.mojo'}
        output = self.run_assembly()
        self.assertEqual((output / 'mojo_optimize/probe.mojo').read_bytes(),
                         (self.external / 'module.mojo').read_bytes())

    def test_balsa_asset_namespace(self):
        self.lock['projects'][1]['assets'] = {'balsa/__init__.mojo': 'module.mojo'}
        output = self.run_assembly()
        self.assertEqual((output / 'balsa/__init__.mojo').read_bytes(),
                         (self.external / 'module.mojo').read_bytes())
        self.lock['projects'][1]['assets'] = {'unexpected/probe.mojo': 'module.mojo'}
        with self.assertRaisesRegex(ValueError, 'unsupported asset namespace'):
            self.run_assembly()

    def test_two_content_versions_at_one_revision_remain_available(self):
        first = (self.external / 'module.mojo').read_bytes()
        self.run_assembly()
        first_pin = self.lock['projects'][1]['files']['module.mojo']
        second = b'different uncommitted external bytes'
        (self.external / 'module.mojo').write_bytes(second)
        self.lock['projects'][1]['files']['module.mojo'] = hashlib.sha256(second).hexdigest()
        self.run_assembly()
        self.external.rename(self.root / 'moved')
        self.lock['projects'][1]['files']['module.mojo'] = first_pin
        output = self.run_assembly()
        self.assertEqual((output / 'moxie/dependency.mojo').read_bytes(), first)

    def test_legacy_tool_is_unchanged(self):
        from assemble_family import legacy
        self.assertTrue(callable(legacy().plan))


if __name__ == '__main__':
    unittest.main()
