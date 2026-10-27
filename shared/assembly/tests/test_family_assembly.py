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
        self.path = self.family / 'DEPENDENCIES.json'

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
        self.assertIn('files', json.loads((output / 'ASSEMBLY.json').read_text()))

    def test_external_edits_are_used_on_every_build(self):
        self.run_assembly()
        expected = b'changed external source'
        (self.external / 'module.mojo').write_bytes(expected)
        output = self.run_assembly()
        self.assertEqual((output / 'moxie/dependency.mojo').read_bytes(), expected)
        commit(self.external)
        self.run_assembly()

    def test_missing_sibling_is_reported_without_cached_fallback(self):
        output = self.run_assembly()
        before = (output / 'ASSEMBLY.json').read_bytes()
        self.external.rename(self.root / 'moved')
        with self.assertRaisesRegex(ValueError, 'missing dependency directory'):
            self.run_assembly()
        self.assertEqual((output / 'ASSEMBLY.json').read_bytes(), before)

    def test_old_hashes_and_revisions_are_not_gates(self):
        self.lock['projects'][1]['files']['module.mojo'] = '0' * 64
        self.lock['projects'][1]['revision'] = '0' * 40
        self.run_assembly()

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
        self.path = child / 'DEPENDENCIES.json'
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
        self.lock['projects'][1]['files']['DEPENDENCIES.json'] = '0' * 64
        self.run_assembly()

    def test_existing_optimize_asset_namespace(self):
        self.lock['projects'][1]['assets'] = {'mojo_optimize/probe.mojo': 'module.mojo'}
        output = self.run_assembly()
        self.assertEqual((output / 'mojo_optimize/probe.mojo').read_bytes(),
                         (self.external / 'module.mojo').read_bytes())

    def test_snappy_asset_namespace(self):
        self.lock['projects'][1]['assets'] = {'mojo_snappy/__init__.mojo': 'module.mojo'}
        output = self.run_assembly()
        self.assertEqual((output / 'mojo_snappy/__init__.mojo').read_bytes(),
                         (self.external / 'module.mojo').read_bytes())

    def test_snappy_release_asset(self):
        binary = b"\x00compiled-snappy\xff"
        (self.external / 'release.mojoc').write_bytes(binary)
        self.lock['projects'][1]['assets'] = {'mojo_snappy.mojoc': 'release.mojoc'}
        output = self.run_assembly()
        self.assertEqual((output / 'mojo_snappy.mojoc').read_bytes(), binary)
        self.lock['projects'][1]['assets'] = {'unexpected.mojoc': 'release.mojoc'}
        with self.assertRaisesRegex(ValueError, 'unsupported asset namespace'):
            self.run_assembly()

    def test_balsa_asset_namespace(self):
        self.lock['projects'][1]['assets'] = {'balsa/__init__.mojo': 'module.mojo'}
        output = self.run_assembly()
        self.assertEqual((output / 'balsa/__init__.mojo').read_bytes(),
                         (self.external / 'module.mojo').read_bytes())
        self.lock['projects'][1]['assets'] = {'unexpected/probe.mojo': 'module.mojo'}
        with self.assertRaisesRegex(ValueError, 'unsupported asset namespace'):
            self.run_assembly()

    def test_manifest_needs_no_revision_or_hashes(self):
        for project in self.lock['projects']:
            project.pop('files')
            project.pop('revision')
        self.run_assembly()

    def test_entrypoints_share_the_same_validator(self):
        from assemble import plan as first
        from assemble_family import plan as second
        self.assertIs(first, second)


if __name__ == '__main__':
    unittest.main()
