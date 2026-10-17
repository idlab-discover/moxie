"""Isotonic regression gates against Moxie's live assembled sources."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests/isotonic'
DEPS = ROOT / 'builds/deps'


def sources():
    return sorted(TESTS.glob('*.mojo'))


def ownership():
    receipt = json.loads((DEPS / 'ASSEMBLY.json').read_text())
    source = ROOT / 'src/moxie/isotonic.mojo'
    assert receipt['modules']['moxie.isotonic'] == ['moxie']
    assert (DEPS / 'moxie/isotonic.mojo').read_bytes() == source.read_bytes()
    manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
    assert all('moxie-isotonic' not in p['path'] for p in manifest['projects'])


def isolated(run):
    with tempfile.TemporaryDirectory(prefix='moxie-isotonic-consumer-') as folder:
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
        binary = consumer / 'consumer'
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', output, source, '-o', binary)
        run(binary)


def test(run, executable):
    ownership()
    for source in sources():
        if source.stem not in ('parity_runner', 'benchmark'):
            run(executable(source))
    run(executable(ROOT / 'examples/isotonic.mojo'))
    for name, marker in [('mixed_dtype', 'cannot be converted')]:
        result = subprocess.run(['mojo', 'build', '-I', str(DEPS),
                                 str(TESTS / 'compile_fail' / f'{name}.mojo'),
                                 '-o', str(ROOT / 'builds/invalid-isotonic')],
                                capture_output=True, text=True)
        (ROOT / f'builds/isotonic-{name}.log').write_text(result.stdout + result.stderr)
        assert result.returncode and marker in result.stdout + result.stderr, result.stderr
    isolated(run)
    print('PASS: isotonic contracts, source ownership, isolated consumer and negative checks')


def parity(run):
    ownership()
    run(sys.executable, TESTS / 'check_parity.py')


if __name__ == '__main__':
    import check
    action = sys.argv[1]
    if action in ('build', 'test', 'parity'):
        check.run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
                  '--consumer', ROOT, '--lock', ROOT / 'DEPENDENCIES.json', '--output', DEPS)
        check.run('mojo', 'precompile', DEPS / 'moxie', '-I', DEPS, '-o', ROOT / 'builds/moxie.mojoc')
        for source in sources() + [ROOT / 'examples/isotonic.mojo']:
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
