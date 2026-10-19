"""Gaussian-process validation against Moxie's live assembled sources."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests/gaussian_process'
DEPS = ROOT / 'builds/deps'
FLAGS = ('-O3', '--fp-mode=contract=off')
EXAMPLES = [ROOT / 'examples/gaussian_process.mojo', ROOT / 'examples/gaussian_process_kernels.mojo']


def sources():
    return [p for p in sorted(TESTS.rglob('*.mojo'))
            if not {'negative', 'consumers'}.intersection(p.relative_to(TESTS).parts)]


def ownership(deps=DEPS):
    receipt = json.loads((deps / 'ASSEMBLY.json').read_text())
    for source in (ROOT / 'src/moxie/gaussian_process').glob('*.mojo'):
        if source.name == '__init__.mojo':
            continue
        module = 'moxie.gaussian_process.' + source.stem
        assert receipt['modules'][module] == ['moxie'], module
        assert (deps / 'moxie/gaussian_process' / source.name).read_bytes() == source.read_bytes()
    manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
    for project in manifest['projects']:
        assert 'moxie-gaussian_process' not in project['path']
        if project['id'] in ('gaussian-process-linalg', 'gaussian-process-optimize', 'random', 'numerics'):
            for target, source in project['assets'].items():
                assert (deps / target).read_bytes() == (ROOT / project['path'] / source).read_bytes()
            for module, entry in project['modules'].items():
                assert receipt['modules'][module] == [project['id']], module
                target = Path(*module.split('.')).with_suffix('.mojo')
                assert (deps / target).read_bytes() == (ROOT / project['path'] / entry['file']).read_bytes()


def isolated(run):
    with tempfile.TemporaryDirectory(prefix='moxie-gp-consumer-') as folder:
        workspace = Path(folder)
        manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
        for project in manifest['projects']:
            project['path'] = str((ROOT / project['path']).resolve())
        assembler = ROOT / 'shared/assembly/tools/assemble_family.py'
        outputs = []
        for name in ('a', 'b'):
            consumer = workspace / name
            consumer.mkdir()
            lock = consumer / 'DEPENDENCIES.json'
            lock.write_text(json.dumps(manifest))
            run('git', 'init', '-q', consumer)
            (consumer / '.gitignore').write_text('builds/\n')
            deps = consumer / 'builds/deps'
            run(sys.executable, assembler, '--consumer', consumer, '--lock', lock, '--output', deps)
            ownership(deps)
            for source in (TESTS / 'consumers').glob('*.mojo'):
                (consumer / source.name).write_bytes(source.read_bytes())
            binary = consumer / 'probe'
            run('mojo', 'build', *FLAGS, '-I', deps, '-I', consumer,
                consumer / 'main.mojo', '-o', binary)
            outputs.append(run(binary, capture_output=True, text=True).stdout)
            # The regression contracts exercise optimized/disabled fitting, composed
            # kernels, multi-target uncertainty, likelihood gradients and sampling.
            source = consumer / 'regression.mojo'
            source.write_bytes((TESTS / 'regression/contracts.mojo').read_bytes())
            run('mojo', 'build', *FLAGS, '-I', deps, source, '-o', consumer / 'regression')
            run(consumer / 'regression')
        assert outputs[0] == outputs[1]
        assert (workspace / 'a/builds/deps/moxie/_dense.mojo').stat().st_ino != (workspace / 'b/builds/deps/moxie/_dense.mojo').stat().st_ino
        before = (deps / 'ASSEMBLY.json').read_bytes()
        for case in ('missing', 'ownership', 'export'):
            bad = json.loads(json.dumps(manifest))
            owner = next(p for p in bad['projects'] if p['id'] == 'moxie')
            if case == 'missing':
                owner['modules']['moxie.gaussian_process._gpr']['file'] = 'src/moxie/missing.mojo'
            else:
                bad['projects'].append(dict(id='conflict', path=str(ROOT), assets={},
                    modules={'moxie.gaussian_process._gpr' if case == 'ownership' else 'moxie.consumer_a':
                             {'file': 'tests/gaussian_process/consumers/consumer_a.mojo'}},
                    exports=None if case == 'ownership' else {'schema': 1, 'packages': {
                        'moxie.gaussian_process': {'GaussianProcessRegressor': 'moxie.consumer_a'}}}))
            lock.write_text(json.dumps(bad))
            result = subprocess.run([sys.executable, str(assembler), '--consumer', str(consumer),
                                     '--lock', str(lock), '--output', str(deps)], capture_output=True, text=True)
            diagnostic = {'missing': 'missing file:', 'ownership': 'module collision:', 'export': 'export collision:'}[case]
            assert result.returncode and diagnostic in result.stderr, result.stderr
            assert (deps / 'ASSEMBLY.json').read_bytes() == before


def test(run, executable):
    ownership()
    for source in sources() + EXAMPLES:
        if source.stem not in ('benchmark', 'gpr_driver', 'kernel_driver', 'pairwise_driver'):
            run(executable(source))
    for name, diagnostic in [('missing_predict', "required function 'predict' is not implemented"),
                             ('missing_kernel', 'diag'),
                             ('classification', 'GaussianProcessClassifier')]:
        result = subprocess.run(['mojo', 'build', '-I', str(DEPS), str(TESTS / 'negative' / f'{name}.mojo'),
                                 '-o', str(ROOT / 'builds/invalid-gp')], capture_output=True, text=True)
        assert result.returncode and diagnostic in result.stderr, result.stderr
    isolated(run)
    print('PASS: Gaussian-process contracts, ownership, isolated consumers and assembly rejection')


def parity(run):
    ownership()
    for script in ('kernels/parity.py', 'kernels/pairwise_parity.py', 'regression/parity.py'):
        run(sys.executable, TESTS / script)


if __name__ == '__main__':
    import check
    action = sys.argv[1]
    if action in ('build', 'test', 'parity'):
        check.run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
                  '--consumer', ROOT, '--lock', ROOT / 'DEPENDENCIES.json', '--output', DEPS)
        for source in sources() + EXAMPLES:
            binary = check.executable(source)
            binary.parent.mkdir(parents=True, exist_ok=True)
            check.run('mojo', 'build', *FLAGS, '-I', DEPS, source, '-o', binary)
    if action in ('test', 'test-only'):
        test(check.run, check.executable)
    elif action in ('parity', 'parity-only'):
        parity(check.run)
    elif action != 'build':
        raise ValueError(action)
