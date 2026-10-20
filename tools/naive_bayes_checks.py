"""Naive Bayes contracts, live-source ownership and reference comparisons."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests/naive_bayes'
DEPS = ROOT / 'builds/deps'
MODULES = ('naive_bayes', '_naive_bayes_common', '_naive_bayes_dense',
           '_nb_gaussian', '_nb_counts', '_nb_bernoulli', '_nb_categorical')


def sources():
    return sorted(TESTS.glob('*.mojo')) + sorted(TESTS.glob('gaussian/*.mojo')) + [
        TESTS / 'counts/benchmark.mojo', TESTS / 'counts/kernels.mojo', TESTS / 'counts/tie_reproducer.mojo',
        TESTS / 'bernoulli/dependency_contract.mojo',
    ]


def ownership():
    receipt = json.loads((DEPS / 'ASSEMBLY.json').read_text())
    for name in MODULES:
        assert receipt['modules']['moxie.' + name] == ['moxie']
        assert (DEPS / f'moxie/{name}.mojo').read_bytes() == (ROOT / f'src/moxie/{name}.mojo').read_bytes()
    assert not (DEPS / 'moxie/_naive_bayes_legacy.mojo').exists()
    manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
    assert all('moxie-naive_bayes' not in project['path'] for project in manifest['projects'])


def isolated_once(run):
    with tempfile.TemporaryDirectory(prefix='moxie-naive-bayes-') as folder:
        consumer = Path(folder)
        manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
        for project in manifest['projects']:
            project['path'] = str((ROOT / project['path']).resolve())
        for name in ('a', 'b'):
            manifest['projects'].append(dict(
                id='consumer-' + name, path=str(TESTS / 'consumers'),
                modules={'moxie.nb_consumer_' + name: {'file': 'nb_consumer_' + name + '.mojo'}},
                assets={}, exports={'schema': 1, 'packages': {'moxie': {}}}))
        run('git', 'init', '-q', consumer)
        (consumer / '.gitignore').write_text('builds/\n')
        lock = consumer / 'DEPENDENCIES.json'
        lock.write_text(json.dumps(manifest))
        output = consumer / 'builds/deps'
        command = [sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
                   '--consumer', consumer, '--lock', lock, '--output', output]
        run(*command)
        for fixture in ('consumers/main.mojo', 'test_public.mojo'):
            source = consumer / 'consumer.mojo'
            source.write_bytes((TESTS / fixture).read_bytes())
            binary = consumer / 'consumer'
            run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', output, source, '-o', binary)
            run(binary)
        receipt = (output / 'ASSEMBLY.json').read_bytes()
        for kind in ('missing', 'module', 'export'):
            bad = json.loads(json.dumps(manifest))
            if kind == 'missing':
                bad['projects'][0]['modules']['moxie.naive_bayes']['file'] = 'missing.mojo'
            elif kind == 'module':
                duplicate = dict(bad['projects'][-1], id='duplicate')
                bad['projects'].append(duplicate)
            else:
                for project in bad['projects'][-2:]:
                    project['exports']['packages']['moxie']['check_model'] = next(iter(project['modules']))
            lock.write_text(json.dumps(bad))
            result = subprocess.run(list(map(str, command)), capture_output=True, text=True)
            assert result.returncode and any(marker in result.stderr.lower() for marker in
                ('missing file', 'conflict', 'duplicate', 'collision')), result.stderr
            assert (output / 'ASSEMBLY.json').read_bytes() == receipt


def isolated(run):
    # Independent assemblies exercise both producer/consumer contributions.
    for _ in range(2):
        isolated_once(run)


def test(run, executable):
    ownership()
    for source in sources():
        if source.stem != 'benchmark':
            run(executable(source))
    for name in ('naive_bayes', 'naive_bayes_counts'):
        run(executable(ROOT / f'examples/{name}.mojo'))
    for name, marker in [('missing_predict', "required function 'predict'"),
                          ('list_to_dense', 'validate_matrix'),
                          ('dense_normalization', 'log_normalize')]:
        result = subprocess.run(['mojo', 'build', '-I', str(DEPS),
            str(TESTS / 'compile_fail' / f'{name}.mojo'), '-o', str(ROOT / 'builds/invalid-naive-bayes')],
            capture_output=True, text=True)
        (ROOT / f'builds/naive-bayes-{name}.log').write_text(result.stderr)
        assert result.returncode and marker in result.stderr, result.stderr
    isolated(run)
    print('PASS: Naive Bayes contracts, ownership, external consumers and rejection checks')


def parity(run):
    ownership()
    python = os.environ.get('SKLEARN_PYTHON', str(ROOT / 'builds/naive-bayes-reference-env/bin/python'))
    if not Path(python).is_file():
        raise RuntimeError('Install the pinned local scikit-learn in builds/naive-bayes-reference-env or set SKLEARN_PYTHON; see docs/naive_bayes/README.md')
    for relative in ('check_sklearn.py', 'gaussian/parity.py', 'counts/parity.py'):
        run(python, TESTS / relative)


if __name__ == '__main__':
    import check
    action = sys.argv[1]
    if action in ('build', 'test', 'parity'):
        check.run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
            '--consumer', ROOT, '--lock', ROOT / 'DEPENDENCIES.json', '--output', DEPS)
        check.run('mojo', 'precompile', DEPS / 'moxie', '-I', DEPS, '-o', ROOT / 'builds/moxie.mojoc')
        for source in sources() + [ROOT / 'examples/naive_bayes.mojo', ROOT / 'examples/naive_bayes_counts.mojo']:
            binary = check.executable(source)
            binary.parent.mkdir(parents=True, exist_ok=True)
            check.run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', DEPS,
                      '-I', ROOT / 'examples', source, '-o', binary)
    if action in ('test', 'test-only'):
        test(check.run, check.executable)
    elif action in ('parity', 'parity-only'):
        parity(check.run)
    elif action != 'build':
        raise ValueError(action)
