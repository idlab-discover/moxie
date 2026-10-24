"""Validate the supported neural-network subset in the full Moxie assembly."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

FAMILY = Path(__file__).resolve().parents[1]
CORE = FAMILY.parents[1]
sys.path.insert(0, str(CORE / 'tools'))
from compiler_limits import compiler_command
DEPS = CORE / 'builds/deps'
RBM = FAMILY / 'algorithms/rbm'
ENGINE = FAMILY / 'shared/mlp_engine'
FLAGS = ['-O3', '--fp-mode=contract=off', '-I', str(DEPS)]


def run(*args, cwd=CORE):
    subprocess.run(compiler_command(args), cwd=cwd, check=True)


def assemble():
    run(sys.executable, CORE / 'shared/assembly/tools/assemble_family.py',
        '--consumer', CORE, '--lock', CORE / 'DEPENDENCIES.json', '--output', DEPS)
    manifest = json.loads((CORE / 'DEPENDENCIES.json').read_text())
    receipt = json.loads((DEPS / 'ASSEMBLY.json').read_text())
    for project in manifest['projects']:
        owner = (CORE / project['path']).resolve()
        assert not any(part in ('moxie-neural_network', 'retired-moxie-neural_network') for part in owner.parts)
        if not project['id'].startswith('neural_network.'):
            continue
        assert owner.is_relative_to(FAMILY)
        for name, spec in project['modules'].items():
            assert receipt['modules'][name] == [project['id']]
            assembled = DEPS / Path(*name.split('.')).with_suffix('.mojo')
            assert assembled.read_bytes() == (owner / spec['file']).read_bytes()
        for destination, source in project['assets'].items():
            assert (DEPS / destination).read_bytes() == (owner / source).read_bytes()
    print('PASS neural-network core ownership, source bytes and license assets', flush=True)


def build():
    assemble()
    run('mojo', 'precompile', DEPS / 'moxie', '-I', DEPS, '-o', FAMILY / 'builds/core.mojoc')
    sources = list((RBM / 'tests').glob('test_*.mojo'))
    sources += [RBM / 'tests/kernel_dump.mojo', RBM / 'tests/estimator_dump.mojo', RBM / 'tests/lifecycle_dump.mojo',
                RBM / 'examples/rbm.mojo', ENGINE / 'tests/parameters.mojo',
                ENGINE / 'tests/parameter_dump.mojo', ENGINE / 'examples/parameters.mojo']
    for source in sources:
        owner = RBM if source.is_relative_to(RBM) else ENGINE
        (owner / 'builds').mkdir(parents=True, exist_ok=True)
        target = owner / 'builds' / (source.stem if source.parent.name == 'tests' else 'example')
        run('mojo', 'build', *FLAGS, '-I', RBM / 'tests', source, '-o', target)


def test():
    for owner, names in [(RBM, ['test_kernels', 'test_estimator', 'test_composition', 'example']),
                         (ENGINE, ['parameters', 'example'])]:
        for name in names:
            run(owner / 'builds' / name)
    run(sys.executable, FAMILY / 'tools/check.py', 'test')


def parity():
    for path in [RBM / 'tests/kernel_parity.py', RBM / 'tests/estimator_parity.py', RBM / 'tests/lifecycle_parity.py',
                 ENGINE / 'tests/parameter_parity.py']:
        run(sys.executable, path)
    print('PASS supported subset parity; MLP training and Float32 partial_fit remain deferred', flush=True)


def independence():
    assemble()
    # This compiler receives only materialized dependencies and copied probes.
    with tempfile.TemporaryDirectory(prefix='neural-network-consumer-') as directory:
        root = Path(directory)
        shutil.copytree(DEPS, root / 'deps')
        for name in ['estimator_dump.mojo', 'kernel_dump.mojo']:
            shutil.copy2(RBM / 'tests' / name, root / name)
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', root / 'deps',
            '-I', root, root / 'estimator_dump.mojo', '-o', root / 'probe', cwd=root)
        actual = subprocess.check_output([root / 'probe'], cwd=root)
        expected = subprocess.check_output([RBM / 'builds/estimator_dump'])
        assert actual == expected
    print('PASS isolated public consumer with byte-identical lifecycle results', flush=True)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    action = sys.argv[1]
    if action in ('build', 'test', 'parity', 'all'):
        build()
    if action in ('test', 'all'):
        test()
    if action in ('parity', 'all'):
        parity()
    if action in ('independence', 'all'):
        independence()
    if action not in ('build', 'test', 'parity', 'independence', 'all'):
        raise ValueError(action)
