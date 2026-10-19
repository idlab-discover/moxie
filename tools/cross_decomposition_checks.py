"""Build and validate Moxie's dense cross-decomposition implementations."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests/cross_decomposition'
DEPS = ROOT / 'builds/deps'
FLAGS = ('-O3', '--fp-mode=contract=off')
EXAMPLES = [ROOT / 'examples' / f'{name}.mojo' for name in
            ('cross_decomposition', 'cross_decomposition_svd', 'cross_decomposition_blocks')]


def sources():
    return [p for p in sorted(TESTS.rglob('*.mojo'))
            if not {'negative', 'consumers'}.intersection(p.relative_to(TESTS).parts)]


def ownership(deps=DEPS):
    receipt = json.loads((deps / 'ASSEMBLY.json').read_text())
    for source in (ROOT / 'src/moxie/cross_decomposition').glob('*.mojo'):
        if source.stem == '__init__':
            continue
        module = 'moxie.cross_decomposition.' + source.stem
        assert receipt['modules'][module] == ['moxie'], module
        assert (deps / 'moxie/cross_decomposition' / source.name).read_bytes() == source.read_bytes()
    public = (deps / 'moxie/cross_decomposition/__init__.mojo').read_text()
    assert '_api' not in public
    for name in ('PLSRegression', 'PLSCanonical', 'CCA', 'PLSSVD'):
        assert name in public
    assert receipt['modules']['moxie._dense'] == ['numerics']
    assert (deps / 'moxie/_dense.mojo').read_bytes() == (ROOT / 'shared/numerics/src/moxie/_dense.mojo').read_bytes()
    for project in json.loads((ROOT / 'DEPENDENCIES.json').read_text())['projects']:
        assert 'moxie-cross_decomposition' not in project['path']
        if project['id'] == 'mojo-linalg':
            for name in ('svd', '_storage', '_spectral'):
                target = f'mojo_linalg/{name}.mojo'
                source = ROOT / project['path'] / project['assets'][target]
                assert (deps / target).read_bytes() == source.read_bytes()


def isolated(run):
    with tempfile.TemporaryDirectory(prefix='moxie-cross-decomposition-') as folder:
        workspace = Path(folder)
        manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
        for project in manifest['projects']:
            project['path'] = str((ROOT / project['path']).resolve())
        assembler = ROOT / 'shared/assembly/tools/assemble_family.py'
        for name in ('a', 'b'):
            consumer = workspace / name
            consumer.mkdir()
            run('git', 'init', '-q', consumer)
            (consumer / '.gitignore').write_text('builds/\n')
            lock = consumer / 'DEPENDENCIES.json'
            lock.write_text(json.dumps(manifest))
            deps = consumer / 'builds/deps'
            run(sys.executable, assembler, '--consumer', consumer, '--lock', lock, '--output', deps)
            ownership(deps)
            binary = consumer / 'probe'
            run('mojo', 'build', *FLAGS, '-I', deps, TESTS / 'public.mojo', '-o', binary)
            run(binary)
        assert (workspace / 'a/builds/deps/moxie/cross_decomposition/_pls.mojo').stat().st_ino != (workspace / 'b/builds/deps/moxie/cross_decomposition/_pls.mojo').stat().st_ino
        before = (deps / 'ASSEMBLY.json').read_bytes()
        for case in ('module', 'export'):
            bad = json.loads(json.dumps(manifest))
            bad['projects'].append(dict(id='conflict', path=str(ROOT), assets={},
                modules={'moxie.cross_decomposition._pls' if case == 'module' else 'moxie.consumer_probe':
                         {'file': 'tests/cross_decomposition/public.mojo'}},
                exports=None if case == 'module' else {'schema': 1, 'packages': {
                    'moxie.cross_decomposition': {'PLSRegression': 'moxie.consumer_probe'}}}))
            lock.write_text(json.dumps(bad))
            result = subprocess.run([sys.executable, str(assembler), '--consumer', str(consumer),
                                     '--lock', str(lock), '--output', str(deps)], capture_output=True, text=True)
            assert result.returncode and case + ' collision' in result.stderr, result.stderr
            assert (deps / 'ASSEMBLY.json').read_bytes() == before


def test(run, executable):
    ownership(DEPS)
    for source in sources() + EXAMPLES:
        if source.stem not in ('benchmark', 'parity_runner'):
            run(executable(source))
    for capability, diagnostic in [('predict', "'PLSSVD' value has no attribute 'predict'"),
                                   ('score', "'PLSSVD' value has no attribute 'score'"),
                                   ('inverse_transform', "'PLSSVD' value has no attribute 'inverse_transform'"),
                                   ('missing_engine_method', "required function 'rotations'")]:
        source = TESTS / 'negative' / f'{capability}.mojo'
        result = subprocess.run(['mojo', 'build', '-I', str(DEPS), str(source),
                                 '-o', str(ROOT / 'builds/invalid-cross-decomposition')],
                                capture_output=True, text=True)
        assert result.returncode and diagnostic in result.stderr, result.stderr
    isolated(run)
    print('PASS: cross-decomposition contracts, public consumers, ownership and exclusions')


def parity(run):
    ownership(DEPS)
    for child in ('pls', 'plssvd'):
        run(sys.executable, TESTS / child / 'parity.py')


def build(run, executable):
    for source in sources() + EXAMPLES:
        binary = executable(source)
        binary.parent.mkdir(parents=True, exist_ok=True)
        run('mojo', 'build', *FLAGS, '-I', DEPS, source, '-o', binary)


if __name__ == '__main__':
    import check
    action = sys.argv[1]
    if action in ('build', 'test', 'parity'):
        check.run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
                  '--consumer', ROOT, '--lock', ROOT / 'DEPENDENCIES.json', '--output', DEPS)
        build(check.run, check.executable)
    if action in ('test', 'test-only'):
        test(check.run, check.executable)
    elif action in ('parity', 'parity-only'):
        parity(check.run)
    elif action != 'build':
        raise ValueError(action)
