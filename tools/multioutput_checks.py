"""Multioutput gates against Moxie's live assembled sources."""
from pathlib import Path
import copy
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests/multioutput'
DEPS = ROOT / 'builds/deps'
MODULES = ('multioutput', '_multioutput', '_multioutput_contracts')


def sources():
    return sorted(TESTS.glob('test_*.mojo'))


def ownership(deps=DEPS):
    receipt = json.loads((deps / 'ASSEMBLY.json').read_text())
    for name in MODULES:
        assert receipt['modules']['moxie.' + name] == ['moxie']
        assert (deps / f'moxie/{name}.mojo').read_bytes() == (ROOT / f'src/moxie/{name}.mojo').read_bytes()
    for name, owner in [('_dense', 'numerics'), ('_random', 'random'), ('base', 'moxie')]:
        assert receipt['modules']['moxie.' + name] == [owner]
    assert (deps / 'moxie/base.mojo').read_bytes() == (ROOT / 'src/moxie/base.mojo').read_bytes()
    manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
    assert all('moxie-multioutput' not in p['path'] for p in manifest['projects'])


def isolated(run):
    manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
    for project in manifest['projects']:
        project['path'] = str((ROOT / project['path']).resolve())
    for name in ('a', 'b'):
        manifest['projects'].append(dict(
            id=f'consumer-{name}', path=str(TESTS / 'consumers' / name),
            modules={f'moxie.consumer_{name}': {'file': f'src/moxie/consumer_{name}.mojo'}},
            assets={}, exports={'schema': 1, 'packages': {
                'moxie': {f'Consumer{name.upper()}': f'moxie.consumer_{name}'}}}))
    for name in ('a', 'b'):
        with tempfile.TemporaryDirectory(prefix=f'moxie-multioutput-{name}-') as folder:
            folder = Path(folder)
            run('git', 'init', '-q', folder)
            (folder / '.gitignore').write_text('builds/\n')
            lock = folder / 'DEPENDENCIES.json'
            lock.write_text(json.dumps(manifest))
            deps = folder / 'builds/deps'
            command = [sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
                       '--consumer', folder, '--lock', lock, '--output', deps]
            run(*command)
            ownership(deps)
            for source in [TESTS / 'test_estimators.mojo', ROOT / 'examples/multioutput.mojo']:
                (folder / source.name).write_bytes(source.read_bytes())
            source = folder / 'main.mojo'
            source.write_text((TESTS / 'consumers' / name / 'main.mojo').read_text().replace(
                'def main() raises:', 'from test_estimators import main as wrappers\nfrom multioutput import main as production\n\ndef main() raises:\n    wrappers()\n    production()'))
            run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', deps,
                '-I', folder, source, '-o', folder / 'probe')
            run(folder / 'probe')
            before = (deps / 'ASSEMBLY.json').read_bytes()
            for kind in ('module', 'export'):
                bad = copy.deepcopy(manifest)
                extra = copy.deepcopy(bad['projects'][-1])
                extra['id'] = 'conflicting-owner'
                if kind == 'export':
                    extra['modules'] = {'moxie.other_probe': {'file': 'src/moxie/consumer_b.mojo'}}
                    extra['exports']['packages']['moxie']['ConsumerB'] = 'moxie.other_probe'
                bad['projects'].append(extra)
                lock.write_text(json.dumps(bad))
                result = subprocess.run(list(map(str, command)), capture_output=True, text=True)
                assert result.returncode and any(word in result.stderr.lower()
                    for word in ('conflict', 'duplicate', 'collision')), result.stderr
                assert (deps / 'ASSEMBLY.json').read_bytes() == before


def test(run, executable):
    ownership()
    for source in sources():
        run(executable(source))
    run(executable(ROOT / 'examples/multioutput.mojo'))
    for name, marker in [('missing_predict', "required function 'predict'"),
                         ('missing_probability', 'ProbabilityPredictor'),
                         ('missing_weighted_fit', 'WeightedSupervisedEstimator'),
                         ('missing_incremental', 'IncrementalRegressor'),
                         ('missing_decision', 'DecisionFunction')]:
        result = subprocess.run(['mojo', 'build', '-I', str(DEPS), '-I', str(TESTS),
                                 str(TESTS / 'compile_fail' / f'{name}.mojo'),
                                 '-o', str(ROOT / 'builds/invalid-multioutput')],
                                capture_output=True, text=True)
        (ROOT / f'builds/multioutput-{name}.log').write_text(result.stdout + result.stderr)
        assert result.returncode and marker in result.stdout + result.stderr, result.stderr
    isolated(run)
    print('PASS: multioutput contracts, live ownership, isolated consumers and negative checks')


def parity(run):
    ownership()
    run(ROOT / 'builds/reference-env/bin/python', TESTS / 'parity.py')


if __name__ == '__main__':
    import check
    action = sys.argv[1]
    if action in ('build', 'test', 'parity'):
        check.run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
                  '--consumer', ROOT, '--lock', ROOT / 'DEPENDENCIES.json', '--output', DEPS)
        check.run('mojo', 'precompile', DEPS / 'moxie', '-I', DEPS, '-o', ROOT / 'builds/moxie.mojoc')
        for source in sources() + [ROOT / 'examples/multioutput.mojo']:
            binary = check.executable(source)
            binary.parent.mkdir(parents=True, exist_ok=True)
            check.run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', DEPS,
                      '-I', TESTS, source, '-o', binary)
    if action in ('test', 'test-only'):
        test(check.run, check.executable)
    elif action in ('parity', 'parity-only'):
        parity(check.run)
    elif action != 'build':
        raise ValueError(action)
