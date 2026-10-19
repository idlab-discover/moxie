"""Build Moxie's assembled namespace and run its native contracts."""
from pathlib import Path
import json
import os
import subprocess
import sys

import feature_selection_checks
from compiler_limits import compiler_command

import tree_checks
import mixture_checks
import cross_decomposition_checks
import gaussian_process_checks
import multioutput_checks
import multiclass_checks
import pipeline_checks
import frozen_checks
import calibration_checks
import isotonic_checks
import random_projection_checks
import discriminant_analysis_checks
import kernel_approximation_checks

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / 'builds/deps'


def run(*args, **kwargs):
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return subprocess.run(compiler_command(args), cwd=ROOT, env=env, check=True, **kwargs)


def executable(source):
    return ROOT / 'builds' / source.relative_to(ROOT).with_suffix('')


def build():
    feature_selection_checks.main("build")
    run(sys.executable, ROOT / 'families/covariance/tools/check.py', 'build')
    version = run('mojo', '--version', capture_output=True, text=True).stdout
    if not version.startswith('Mojo 1.0.0 '):
        raise RuntimeError(f'Expected Mojo 1.0.0, got {version.strip()}')
    metadata = list(Path(sys.prefix, 'conda-meta').glob('max-core-*.json'))
    if len(metadata) != 1 or json.loads(metadata[0].read_text())['version'] != '26.5.0':
        raise RuntimeError('Expected MAX 26.5.0')
    (ROOT / 'docs/private/dummy.estimators').mkdir(parents=True, exist_ok=True)
    run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
        '--consumer', ROOT, '--lock', ROOT / 'DEPENDENCIES.json', '--output', DEPS)
    run('mojo', 'precompile', DEPS / 'moxie', '-I', DEPS, '-o', ROOT / 'builds/moxie.mojoc')
    tree_checks.build(run)
    run(sys.executable, ROOT / 'families/preprocessing/tools/check.py', 'build')
    run(sys.executable, ROOT / 'families/kernel_ridge/tools/check.py', 'build')
    sources = sorted((ROOT / 'tests').glob('*.mojo'))
    sources += sorted((ROOT / 'tests/dummy').glob('*.mojo'))
    sources += compose_sources()
    sources += mixture_checks.sources()
    sources += cross_decomposition_checks.sources()
    sources += gaussian_process_checks.sources()
    sources += multioutput_checks.sources()
    sources += multiclass_checks.sources()
    sources += pipeline_checks.sources()
    sources += frozen_checks.sources()
    sources += kernel_approximation_checks.sources()
    sources += discriminant_analysis_checks.sources()
    sources += isotonic_checks.sources()
    sources += calibration_checks.sources()
    sources += random_projection_checks.sources()
    sources += sorted((ROOT / 'examples').glob('*.mojo'))
    for source in sources:
        binary = executable(source)
        binary.parent.mkdir(parents=True, exist_ok=True)
        extra = ('-I', ROOT / 'tests/multioutput') if source.is_relative_to(ROOT / 'tests/multioutput') else ()
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', DEPS, *extra,
            '-I', ROOT / 'tests/multiclass', '-I', ROOT / 'tests/discriminant_analysis', '-I', ROOT / 'tests/kernel_approximation', '-I', ROOT / 'tests/kernel_approximation/consumers', '-I', ROOT / 'tests/frozen', '-I', ROOT / 'examples/frozen',
            '-I', ROOT / 'tests/pipeline', '-I', ROOT / 'examples', '-I', ROOT / 'examples/calibration_support', '-I', ROOT / 'tests/compose/support', source, '-o', binary)


def compose_sources():
    return [source for source in sorted((ROOT / 'tests/compose').rglob('*.mojo'))
            if not {'support', 'negative'}.intersection(source.parts)]


def compose_test():
    receipt = json.loads((DEPS / 'ASSEMBLY.json').read_text())
    for module, owners in receipt['modules'].items():
        if module.startswith('moxie.compose.'):
            assert owners == ['moxie'], (module, owners)
    for source in compose_sources():
        if not source.stem.endswith('parity_driver'):
            run(executable(source))
    for name in ['compose_contracts', 'column_transformer', 'transformed_target']:
        run(executable(ROOT / f'examples/{name}.mojo'))
    for name, message in [('missing_predict', "required function 'predict'"),
                          ('missing_blocks', 'concatenate')]:
        result = subprocess.run(
            ['mojo', 'build', '-I', str(DEPS),
             str(ROOT / f'tests/compose/negative/{name}.mojo'),
             '-o', str(ROOT / 'builds/invalid-compose')],
            cwd=ROOT, capture_output=True, text=True,
        )
        (ROOT / f'builds/compose-{name}.log').write_text(result.stderr)
        if result.returncode == 0 or message not in result.stderr:
            raise RuntimeError(f'Unexpected negative compilation result: {result.stderr}')


def parity():
    feature_selection_checks.main("parity-only")
    tree_checks.parity(run)
    mixture_checks.parity(run)
    cross_decomposition_checks.parity(run)
    gaussian_process_checks.parity(run)
    multioutput_checks.parity(run)
    multiclass_checks.parity(run)
    random_projection_checks.parity(run)
    isotonic_checks.parity(run)
    calibration_checks.parity(run)
    discriminant_analysis_checks.parity(run)
    kernel_approximation_checks.parity(run)
    run(sys.executable, ROOT / 'families/covariance/tools/check.py', 'parity-only')
    run(sys.executable, ROOT / 'families/kernel_ridge/tools/check.py', 'parity-only')
    frozen_checks.parity(run)
    run(sys.executable, ROOT / 'families/preprocessing/tools/check.py', 'parity-only')
    for name in ["parity.py", "union_parity.py", "compose_parity.py"]:
        run(sys.executable, ROOT / "tests/pipeline" / name)
    run(sys.executable, ROOT / 'tests/dummy/parity.py')
    for relative in ['columns/parity.py', 'columns/named_parity.py', 'target/parity.py',
                     'models/parity.py', 'pipeline/parity.py']:
        run(sys.executable, ROOT / 'tests/compose' / relative)


def test():
    feature_selection_checks.main("test-only")
    tree_checks.test(run)
    mixture_checks.test(run, executable)
    cross_decomposition_checks.test(run, executable)
    gaussian_process_checks.test(run, executable)
    multioutput_checks.test(run, executable)
    multiclass_checks.test(run, executable)
    random_projection_checks.test(run, executable)
    isotonic_checks.test(run, executable)
    calibration_checks.test(run, executable)
    discriminant_analysis_checks.test(run, executable)
    kernel_approximation_checks.test(run, executable)
    run(sys.executable, ROOT / 'families/covariance/tools/check.py', 'test-only')
    run(sys.executable, ROOT / 'families/kernel_ridge/tools/check.py', 'test-only')
    frozen_checks.test(run, executable)
    run(sys.executable, ROOT / 'families/preprocessing/tools/check.py', 'test-only')
    pipeline_checks.test(run, executable)
    compose_test()
    for source in sorted((ROOT / 'tests').glob('*.mojo')):
        run(executable(source))
    run('bash', ROOT / 'tests/check_compile_fail.sh')
    run(sys.executable, ROOT / 'tests/serialization_wire.py')
    run(sys.executable, ROOT / 'tests/categorical_parity.py')
    for name in ['contracts', 'dtypes', 'serialization']:
        run(executable(ROOT / f'tests/dummy/{name}.mojo'))
    for name, message in [('sparse', '_sparse'), ('incremental', 'partial_fit')]:
        result = subprocess.run(
            ['mojo', 'build', '-I', str(DEPS), str(ROOT / f'tests/dummy/negative/{name}.mojo'),
             '-o', str(ROOT / 'builds/invalid-dummy')],
            cwd=ROOT, capture_output=True, text=True,
        )
        (ROOT / f'builds/dummy-{name}.log').write_text(result.stderr)
        if result.returncode == 0 or message not in result.stderr:
            raise RuntimeError(f'Unexpected negative compilation result for {name}: {result.stderr}')
    print('Dummy sparse and incremental APIs correctly rejected')


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'build':
        build()
    elif action == 'test':
        test()
    elif action == 'parity':
        parity()
    else:
        raise ValueError(action)
