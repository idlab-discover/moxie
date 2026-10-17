"""Kernel feature-map gates against Moxie's assembled, owned sources."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests/kernel_approximation'
DEPS = ROOT / 'builds/deps'
ALGORITHMS = ('fourier', 'additive_chi2', 'polynomial', 'nystroem')
MODULES = ('kernel_approximation', '_kernel_approximation_contracts', '_kernel_features',
           '_kernel_fourier', '_kernel_additive', '_kernel_polynomial', '_kernel_nystroem')


def sources():
    return [p for p in sorted(TESTS.rglob('*.mojo'))
            if 'negative' not in p.parts and p.name != 'kernel_contract_fixture.mojo']


def ownership():
    receipt = json.loads((DEPS / 'ASSEMBLY.json').read_text())
    for name in MODULES:
        assert receipt['modules']['moxie.' + name] == ['moxie'], name
        assert (DEPS / f'moxie/{name}.mojo').read_bytes() == (ROOT / f'src/moxie/{name}.mojo').read_bytes(), name
    assert (DEPS / 'mojo_linalg/fft.mojo').read_bytes() == (ROOT.parent / 'mojo-linalg/src/mojo_linalg/fft.mojo').read_bytes()
    assert not (DEPS / 'moxie/_sparse.mojo').exists()
    assert not any(name.startswith('moxie.consumer_') for name in receipt['modules'])
    assert receipt['modules']['moxie._kernel_ridge._kernels'] == ['moxie']


def isolated(run):
    # An external consumer gets only Moxie's live manifest and its own input.
    # No family project, generated package, or source import directory is used.
    with tempfile.TemporaryDirectory(prefix='moxie-kernel-consumer-') as folder:
        consumer = Path(folder)
        manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
        for project in manifest['projects']:
            project['path'] = str((ROOT / project['path']).resolve())
            assert 'moxie-kernel_approximation' not in project['path']
        lock = consumer / 'DEPENDENCIES.json'
        lock.write_text(json.dumps(manifest))
        run('git', 'init', '-q', consumer)
        (consumer / '.gitignore').write_text('builds/\n')
        output = consumer / 'builds/deps'
        assembler = ROOT / 'shared/assembly/tools/assemble_family.py'
        run(sys.executable, assembler, '--consumer', consumer, '--lock', lock, '--output', output)
        consumer_sources = [TESTS / 'test_public.mojo', *sorted((TESTS / 'consumers').glob('*.mojo'))]
        for source in consumer_sources:
            (consumer / source.name).write_bytes(source.read_bytes())
        for source in consumer_sources:
            local = consumer / source.name
            binary = consumer / source.stem
            run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', output, '-I', consumer, local, '-o', binary)
            run(binary)
        before = (output / 'ASSEMBLY.json').read_bytes()
        for case in ('missing-source', 'module-conflict', 'export-conflict'):
            bad = json.loads(json.dumps(manifest))
            owner = next(p for p in bad['projects'] if p['id'] == 'moxie')
            if case == 'missing-source':
                owner['modules']['moxie._kernel_features']['file'] = 'src/moxie/missing.mojo'
            elif case == 'module-conflict':
                bad['projects'].append(dict(id='conflict', path=str(ROOT), modules={
                    'moxie._kernel_features': {'file': 'src/moxie/_kernel_features.mojo'}}, assets={}, exports=None))
            else:
                bad['projects'].append(dict(id='export-conflict', path=str(ROOT),
                    modules={'moxie.consumer_a': {'file': 'tests/kernel_approximation/consumers/consumer_a.mojo'}},
                    assets={}, exports={'schema': 1, 'packages': {'moxie': {'BaseEstimator': 'moxie.consumer_a'}}}))
            lock.write_text(json.dumps(bad))
            result = subprocess.run([sys.executable, str(assembler), '--consumer', str(consumer),
                                     '--lock', str(lock), '--output', str(output)], capture_output=True, text=True)
            diagnostic = {'missing-source': 'missing', 'module-conflict': 'module collision',
                          'export-conflict': 'export collision'}[case]
            assert result.returncode and diagnostic in result.stderr, (case, result.stderr)
            assert (output / 'ASSEMBLY.json').read_bytes() == before, case


def test(run, executable):
    ownership()
    for source in sources():
        if source.stem != 'evaluate':
            run(executable(source))
    run(executable(ROOT / 'examples/kernel_approximation.mojo'))
    negative = [('missing_predict', "required function 'predict' is not implemented")]
    for name in ('sparse', 'predict', 'predict_skewed', 'predict_additive',
                 'predict_polynomial', 'predict_nystroem', 'partial_fit'):
        negative.append((name, '_sparse' if name == 'sparse' else ('predict' if name.startswith('predict') else name)))
    for name, diagnostic in negative:
        result = subprocess.run(['mojo', 'build', '-I', str(DEPS), str(TESTS / 'negative' / f'{name}.mojo'),
                                 '-o', str(ROOT / 'builds/invalid-kernel')], capture_output=True, text=True)
        (ROOT / f'builds/kernel-{name}.log').write_text(result.stderr)
        assert result.returncode and diagnostic in result.stderr, result.stderr
    isolated(run)
    print('PASS: kernel contracts, source ownership, isolated consumers and negative checks')


def parity(run):
    ownership()
    for algorithm in ALGORITHMS:
        (ROOT / 'docs/private' / algorithm).mkdir(parents=True, exist_ok=True)
        run(sys.executable, TESTS / algorithm / 'parity.py')


if __name__ == '__main__':
    import check
    action = sys.argv[1]
    if action in ('build', 'test', 'parity'):
        check.run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
                  '--consumer', ROOT, '--lock', ROOT / 'DEPENDENCIES.json', '--output', DEPS)
        for source in sources() + [ROOT / 'examples/kernel_approximation.mojo']:
            binary = check.executable(source)
            binary.parent.mkdir(parents=True, exist_ok=True)
            check.run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', DEPS,
                      '-I', TESTS, '-I', TESTS / 'consumers', source, '-o', binary)
    if action in ('test', 'test-only'):
        test(check.run, check.executable)
    elif action in ('parity', 'parity-only'):
        parity(check.run)
    elif action != 'build':
        raise ValueError(action)
