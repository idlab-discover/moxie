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
DEPS = FAMILY / 'builds/deps'
RBM = FAMILY / 'algorithms/rbm'
ENGINE = FAMILY / 'shared/mlp_engine'
FLAGS = ['-O3', '--fp-mode=contract=off', '-I', str(DEPS)]
RBM_TESTS = ('test_estimator', 'test_kernels', 'test_composition')
RBM_PARITY = ('kernel', 'estimator', 'lifecycle')
MLP_TESTS = ('test_mlp', 'test_mlp_composition', 'mlp_dump')
MLP_INCREMENTAL_TESTS = ('test_mlp_incremental', 'mlp_incremental_dump')


def run(*args, cwd=CORE):
    subprocess.run(compiler_command(args), cwd=cwd, check=True)


def assemble():
    run(sys.executable, CORE / 'shared/assembly/tools/assemble_family.py',
        '--consumer', FAMILY, '--lock', CORE / 'DEPENDENCIES.json', '--output', DEPS)
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


def build_public_mlp(names=MLP_TESTS):
    for name in names:
        run('mojo', 'build', *FLAGS, '-I', FAMILY / 'tests', FAMILY / 'tests' / f'{name}.mojo',
            '-o', FAMILY / 'builds' / name)


def test_public_mlp():
    for name in MLP_TESTS[:2]:
        run(FAMILY / 'builds' / name)


def parity_public_mlp():
    run(sys.executable, FAMILY / 'tests/mlp_parity.py')


def build_rbm():
    (RBM / 'builds').mkdir(parents=True, exist_ok=True)
    for name in (*RBM_TESTS, *(f'{case}_dump' for case in RBM_PARITY)):
        run('mojo', 'build', *FLAGS, '-I', RBM / 'tests', RBM / 'tests' / f'{name}.mojo',
            '-o', RBM / 'builds' / name)
    run('mojo', 'build', *FLAGS, RBM / 'examples/rbm.mojo', '-o', RBM / 'builds/example')


def parity_rbm():
    for name in RBM_PARITY:
        run(sys.executable, RBM / f'tests/{name}_parity.py')


def build():
    assemble()
    run('mojo', 'precompile', DEPS / 'moxie', '-I', DEPS, '-o', FAMILY / 'builds/core.mojoc')
    build_rbm()
    sources = [ENGINE / 'tests/parameters.mojo', ENGINE / 'tests/parameter_dump.mojo',
               ENGINE / 'tests/objective_dump.mojo', ENGINE / 'tests/lbfgs_dump.mojo',
               ENGINE / 'tests/stochastic_dump.mojo', ENGINE / 'examples/parameters.mojo']
    (ENGINE / 'builds').mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = ENGINE / 'builds' / (source.stem if source.parent.name == 'tests' else 'example')
        run('mojo', 'build', *FLAGS, '-I', RBM / 'tests', source, '-o', target)
    build_public_mlp(MLP_TESTS + MLP_INCREMENTAL_TESTS)


def test():
    for owner, names in [(RBM, (*RBM_TESTS, 'example')),
                         (ENGINE, ['parameters', 'example'])]:
        for name in names:
            run(owner / 'builds' / name)
    subprocess.run([ENGINE / 'builds/objective_dump'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([ENGINE / 'builds/lbfgs_dump'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([ENGINE / 'builds/stochastic_dump'], check=True, stdout=subprocess.DEVNULL)
    test_public_mlp()
    run(FAMILY / 'builds/test_mlp_incremental')
    run(sys.executable, FAMILY / 'tools/check.py', 'test')


def parity():
    parity_rbm()
    for path in [ENGINE / 'tests/parameter_parity.py', ENGINE / 'tests/objective_parity.py',
                 ENGINE / 'tests/lbfgs_parity.py', ENGINE / 'tests/stochastic_parity.py']:
        run(sys.executable, path)
    parity_public_mlp()
    run(sys.executable, FAMILY / 'tests/mlp_incremental_parity.py')
    print('PASS supported neural-network subset parity', flush=True)


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
    if action == 'rbm':
        assemble()
        build_rbm()
        for name in (*RBM_TESTS, 'example'):
            run(RBM / 'builds' / name)
        parity_rbm()
    if action == 'mlp':
        assemble()
        build_public_mlp()
        test_public_mlp()
        parity_public_mlp()
    if action == 'mlp-incremental':
        assemble()
        build_public_mlp(MLP_INCREMENTAL_TESTS)
        run(FAMILY / 'builds/test_mlp_incremental')
        run(sys.executable, FAMILY / 'tests/mlp_incremental_parity.py')
    if action in ('build', 'test', 'parity', 'all'):
        build()
    if action in ('test', 'all'):
        test()
    if action in ('parity', 'all'):
        parity()
    if action in ('independence', 'all'):
        independence()
    if action not in ('build', 'test', 'parity', 'independence', 'all', 'rbm', 'mlp', 'mlp-incremental'):
        raise ValueError(action)
