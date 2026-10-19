"""Tree contracts, source ownership, isolated imports and pinned-source parity."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests/tree'
DEPS = ROOT / 'builds/deps'
OUTPUT = ROOT / 'builds/tree'
FLAGS = ('-O3', '--fp-mode=contract=off')


def run(*args):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def sources():
    return sorted(TESTS.glob('*.mojo')) + sorted(TESTS.glob('storage/*.mojo')) + sorted(TESTS.glob('criteria/*.mojo'))


def executable(source):
    return OUTPUT / source.relative_to(TESTS).with_suffix('')


def compile_storage(source, output):
    run('mojo', 'build', *FLAGS, '-I', DEPS, source, '-o', output)


def build(run):
    version = run("mojo", "--version", capture_output=True, text=True).stdout
    if not version.startswith("Mojo 1.0.0 "):
        raise RuntimeError(f"Expected Mojo 1.0.0, got {version.strip()}")
    for source in sources():
        binary = executable(source)
        binary.parent.mkdir(parents=True, exist_ok=True)
        run('mojo', 'build', *FLAGS, '-I', DEPS, '-I', TESTS, source, '-o', binary)
    for name in ('tree', 'tree_forest_members'):
        binary = ROOT / f'builds/examples/{name}'
        binary.parent.mkdir(parents=True, exist_ok=True)
        run('mojo', 'build', *FLAGS, '-I', DEPS, ROOT / f'examples/{name}.mojo', '-o', binary)


def ownership(deps=DEPS):
    receipt = json.loads((deps / 'ASSEMBLY.json').read_text())
    for source in (ROOT / 'src/moxie/tree').glob('*.mojo'):
        if source.stem == '__init__':
            continue
        assert receipt['modules']['moxie.tree.' + source.stem] == ['moxie']
        assert (deps / 'moxie/tree' / source.name).read_bytes() == source.read_bytes()
    assert not (deps / 'moxie/tree/_api.mojo').exists()
    manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
    assert all('moxie-tree' not in p['path'] for p in manifest['projects'])


def isolated(run):
    for _ in range(2):
        with tempfile.TemporaryDirectory(prefix='moxie-tree-consumer-') as directory:
            consumer = Path(directory)
            manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
            for project in manifest['projects']:
                project['path'] = str((ROOT / project['path']).resolve())
            for name in ('a', 'b'):
                manifest['projects'].append(dict(
                    id='consumer_' + name, path=str(TESTS / 'consumers' / name),
                    modules={'moxie.tree.consumer_' + name: {'file': 'src/moxie/tree/consumer_' + name + '.mojo'}},
                    assets={}, exports={'schema': 1, 'packages': {'moxie.tree': {
                        'consumer_' + name: 'moxie.tree.consumer_' + name}}}))
            run('git', 'init', '-q', consumer)
            (consumer / '.gitignore').write_text('builds/\n')
            lock = consumer / 'DEPENDENCIES.json'
            lock.write_text(json.dumps(manifest))
            deps = consumer / 'builds/deps'
            command = [sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
                       '--consumer', consumer, '--lock', lock, '--output', deps]
            run(*command)
            ownership(deps)
            # Independent include roots exercise all estimators, categorical and typed labels,
            # probabilities, pruning and native exports using only Moxie's manifest.
            for fixture in ('consumers/a/main.mojo', 'test_extra.mojo', 'test_labels.mojo',
                            'test_mixed_tables.mojo', 'test_export.mojo'):
                source = consumer / 'main.mojo'
                source.write_bytes((TESTS / fixture).read_bytes())
                run('mojo', 'build', *FLAGS, '-I', deps, source, '-o', consumer / 'probe')
                run(consumer / 'probe')
            before = (deps / 'ASSEMBLY.json').read_bytes()
            for kind in ('missing', 'module', 'export'):
                bad = json.loads(json.dumps(manifest))
                owner = next(p for p in bad['projects'] if p['id'] == 'moxie')
                if kind == 'missing':
                    owner['modules']['moxie.tree._records']['file'] = 'missing.mojo'
                elif kind == 'module':
                    bad['projects'][-1]['modules']['moxie.tree._records'] = bad['projects'][-1]['modules']['moxie.tree.consumer_b']
                else:
                    bad['projects'][-1]['exports']['packages']['moxie.tree']['DecisionTreeClassifier'] = 'moxie.tree.consumer_b'
                lock.write_text(json.dumps(bad))
                result = subprocess.run(list(map(str, command)), capture_output=True, text=True)
                assert result.returncode and any(s in result.stderr.lower() for s in ('missing file', 'collision', 'conflict')), result.stderr
                assert (deps / 'ASSEMBLY.json').read_bytes() == before


def native_test(run):
    ownership()
    for source in sources():
        if source.stem.startswith('test_') or (
            source.parent.name in ('storage', 'criteria')
            and source.stem not in ('parity_driver', 'benchmark')
        ):
            run(executable(source))
    for name in ('tree', 'tree_forest_members'):
        run(ROOT / f'builds/examples/{name}')
    for name, marker in [('missing_predict', "required function 'predict'"),
                         ('dense_path', 'decision_path'), ('regressor_probability', 'predict_proba')]:
        result = subprocess.run(['mojo', 'build', '-I', str(DEPS),
            str(TESTS / 'compile_fail' / f'{name}.mojo'), '-o', str(OUTPUT / 'invalid')],
            capture_output=True, text=True)
        (OUTPUT / f'{name}.log').write_text(result.stderr)
        assert result.returncode and marker in result.stderr, result.stderr
    run(sys.executable, TESTS / 'storage/negative.py')


def test(run):
    native_test(run)
    isolated(run)
    print('PASS: tree contracts, storage, criteria, exports, isolated consumers and source ownership')


def parity(run):
    ownership()
    python = OUTPUT / 'oracle-env/bin/python'
    if not python.is_file():
        raise RuntimeError('Run pixi run tree-prepare-oracle first')
    # Verify source provenance before every comparison suite, including member bitwise checks.
    for source in [TESTS / 'parity.py', *sorted(TESTS.glob('*_parity.py'))]:
        run(python, source)
    for engine in ('storage', 'criteria'):
        run(sys.executable, TESTS / engine / 'parity.py')


if __name__ == '__main__':
    import check
    action = sys.argv[1]
    if action in ('build', 'test', 'parity'):
        check.run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
                  '--consumer', ROOT, '--lock', ROOT / 'DEPENDENCIES.json', '--output', DEPS)
        check.run('mojo', 'precompile', DEPS / 'moxie', '-I', DEPS, '-o', ROOT / 'builds/moxie.mojoc')
        build(check.run)
    if action in ('test', 'test-only'):
        test(check.run)
    elif action == 'native-only':
        native_test(check.run)
    elif action == 'isolated-only':
        isolated(check.run)
    elif action in ('parity', 'parity-only'):
        parity(check.run)
    elif action != 'build':
        raise ValueError(action)
