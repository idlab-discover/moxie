"""Validate integrated linear models, ownership and the local sklearn oracle."""
from pathlib import Path
import json
import os
import subprocess
import sys

from compiler_limits import compiler_command

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / 'families/linear_model'
DEPS = ROOT / 'builds/deps'


def run(*args):
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    subprocess.run(compiler_command(args), cwd=ROOT, env=env, check=True)


def ownership():
    manifest = json.loads((ROOT / 'DEPENDENCIES.json').read_text())
    receipt = json.loads((DEPS / 'ASSEMBLY.json').read_text())
    retired = ROOT.parent / 'moxie-linear_model'
    for project in manifest['projects']:
        source = (ROOT / project['path']).resolve()
        assert source != retired and not source.is_relative_to(retired), source
    for source in (ROOT / 'src/moxie/linear_model').glob('*.mojo'):
        if source.stem == '__init__':
            continue
        module = 'moxie.linear_model.' + source.stem
        assert receipt['modules'][module] == ['moxie'], (module, receipt['modules'][module])
        assert (DEPS / 'moxie/linear_model' / source.name).read_bytes() == source.read_bytes()
    for name in ('_api', '_api_types'):
        assert not (DEPS / 'moxie/linear_model' / f'{name}.mojo').exists()
    assert (DEPS / 'licenses/linear_model-scikit-learn.txt').read_bytes() == (
        FAMILY / 'LICENSES/scikit-learn.txt').read_bytes()
    linalg = next(p for p in manifest['projects'] if p['id'] == 'linear-model-linalg')
    for module, spec in linalg['modules'].items():
        assert receipt['modules'][module] == ['linear-model-linalg']
        assert (DEPS / (module.replace('.', '/') + '.mojo')).read_bytes() == (
            ROOT / linalg['path'] / spec['file']).read_bytes()
    print('PASS: core owns linear_model; private linalg bytes match their external owner; no standalone dependency', flush=True)


def public_namespace():
    run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
        '--consumer', ROOT, '--lock', ROOT / 'DEPENDENCIES.json', '--output', DEPS)
    ownership()
    run('mojo', 'precompile', DEPS / 'moxie', '-I', DEPS, '-o', ROOT / 'builds/linear_model-public.mojoc')
    for relative in ('examples/dense_estimators.mojo', 'tests/test_fluent.mojo', 'tests/test_composition.mojo',
                     'algorithms/bayesian/examples/bayesian_ridge.mojo'):
        source = FAMILY / relative
        binary = ROOT / 'builds/linear_model-public' / source.stem
        binary.parent.mkdir(parents=True, exist_ok=True)
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', DEPS, source, '-o', binary)
        run(binary)
    print('PASS: all supported estimators compile and execute through the complete Moxie namespace', flush=True)


def main(action):
    if action not in ('build', 'test', 'parity', 'start', 'namespace', 'test-only', 'parity-only'):
        raise SystemExit('expected build, test, parity, start or namespace')
    if action.endswith('-only'):
        ownership()
    else:
        public_namespace()
    if action != 'namespace':
        run(sys.executable, FAMILY / 'tools/check.py', action)


if __name__ == '__main__':
    main(sys.argv[1])
