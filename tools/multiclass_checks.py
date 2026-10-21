"""Multiclass gates against Moxie's live assembled sources."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests/multiclass'
DEPS = ROOT / 'builds/deps'


def sources():
    return sorted(TESTS.glob('test_*.mojo'))


def ownership():
    receipt = json.loads((DEPS / 'ASSEMBLY.json').read_text())
    for source in (ROOT / 'src/moxie').glob('*multiclass*.mojo'):
        module = 'moxie.' + source.stem
        assert receipt['modules'][module] == ['moxie']
        assert (DEPS / 'moxie' / source.name).read_bytes() == source.read_bytes()
    assert receipt['modules']['moxie._labels'] == ['labels']
    assert receipt['modules']['moxie._random'] == ['random']
    assert receipt['modules']['moxie.preprocessing._label'] == ['preprocessing']
    assert (DEPS / 'moxie/preprocessing/_label.mojo').read_bytes() == (ROOT / 'src/moxie/preprocessing/_label.mojo').read_bytes()
    manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
    assert all('moxie-multiclass' not in p['path'] for p in manifest['projects'])


def isolated(run):
    with tempfile.TemporaryDirectory(prefix='moxie-multiclass-consumer-') as folder:
        consumer = Path(folder)
        manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
        for project in manifest['projects']:
            project['path'] = str((ROOT / project['path']).resolve())
        lock = consumer / 'DEPENDENCIES.json'
        lock.write_text(json.dumps(manifest))
        run('git', 'init', '-q', consumer)
        (consumer / '.gitignore').write_text('builds/\n')
        output = consumer / 'builds/deps'
        run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
            '--consumer', consumer, '--lock', lock, '--output', output)
        source = consumer / 'consumer.mojo'
        source.write_bytes((TESTS / 'test_public.mojo').read_bytes())
        for name in ('ovr_fixtures.mojo', 'ovr_incremental_fixtures.mojo'):
            (consumer / name).write_bytes((TESTS / name).read_bytes())
        binary = consumer / 'consumer'
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', output, '-I', consumer, source, '-o', binary)
        run(binary)


def test(run, executable):
    ownership()
    for source in sources():
        run(executable(source))
    for name in ['multiclass_estimator', 'multiclass_one_vs_rest', 'multiclass_one_vs_one', 'multiclass_output_code']:
        run(executable(ROOT / f'examples/{name}.mojo'))
    for name, marker in [
        ('output_code_plain_regressor', 'component needs decision_function or predict_proba'),
        ('ovo_missing_partial_fit', "invalid call to 'partial_fit': violated constraint"),
        ('ovr_missing_partial_fit', "invalid call to 'partial_fit': violated constraint"),
        ('ovr_regressor_no_decision', "invalid call to 'decision_function': violated constraint"),
        ('ovr_regressor_no_proba', "invalid call to 'predict_proba': violated constraint"),
        ('output_code_mutable_rng', "value passed to 'random_state' cannot be converted from 'RandomState' to 'Optional[Int]'"),
        ('output_code_component_target', 'OutputCodeClassifier component Input must be NDArray[dtype]'),
        ('output_code_no_proba', "has no attribute 'predict_proba'"),
        ('output_code_no_decision', "has no attribute 'decision_function'"),
        ('missing_predict', "required function 'predict' is not implemented"),
        ('ovr_missing_proba', "invalid call to 'predict_proba': violated constraint"),
        ('ovr_missing_decision', "invalid call to 'decision_function': violated constraint"),
        ('ovr_component_target', 'component Input must be NDArray[dtype]'),
        ('ovo_no_proba', "has no attribute 'predict_proba'"),
        ('ovo_component_target', 'OneVsOneClassifier component Input must be NDArray[dtype]'),
        ('output_code_no_partial_fit', "has no attribute 'partial_fit'"),
    ]:
        result = subprocess.run(['mojo', 'build', '-I', str(DEPS), '-I', str(TESTS),
                                 str(TESTS / 'compile_fail' / f'{name}.mojo'),
                                 '-o', str(ROOT / 'builds/invalid-multiclass')],
                                capture_output=True, text=True)
        (ROOT / f'builds/multiclass-{name}.log').write_text(result.stdout + result.stderr)
        assert result.returncode and marker in result.stdout + result.stderr, result.stderr
    isolated(run)
    print('PASS: multiclass contracts, source ownership, isolated consumer and negative checks')


def parity(run):
    ownership()
    for name in ('ovr', 'ovr_partial_fit', 'ovo', 'ovo_partial_fit', 'output_code', 'string_labels', 'pairwise', 'production'):
        run(ROOT / 'builds/reference-env/bin/python', TESTS / f'{name}_parity.py')


if __name__ == '__main__':
    import check
    action = sys.argv[1]
    if action in ('build', 'test', 'parity'):
        check.run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
                  '--consumer', ROOT, '--lock', ROOT / 'DEPENDENCIES.json', '--output', DEPS)
        check.run('mojo', 'precompile', DEPS / 'moxie', '-I', DEPS, '-o', ROOT / 'builds/moxie.mojoc')
        for source in sources() + sorted((ROOT / 'examples').glob('multiclass_*.mojo')):
            binary = check.executable(source)
            binary.parent.mkdir(parents=True, exist_ok=True)
            check.run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', DEPS,
                      '-I', TESTS, '-I', ROOT / 'examples', source, '-o', binary)
    if action in ('test', 'test-only'):
        test(check.run, check.executable)
    elif action in ('parity', 'parity-only'):
        parity(check.run)
    elif action != 'build':
        raise ValueError(action)
