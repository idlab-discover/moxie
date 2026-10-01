"""Build Moxie's assembled namespace and run its native contracts."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / 'builds/deps'


def run(*args, **kwargs):
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return subprocess.run([str(arg) for arg in args], cwd=ROOT, env=env, check=True, **kwargs)


def executable(source):
    return ROOT / 'builds' / source.relative_to(ROOT).with_suffix('')


def build():
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
    sources = sorted((ROOT / 'tests').glob('*.mojo'))
    sources += sorted((ROOT / 'tests/dummy').glob('*.mojo'))
    sources += sorted((ROOT / 'examples').glob('*.mojo'))
    for source in sources:
        binary = executable(source)
        binary.parent.mkdir(parents=True, exist_ok=True)
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', DEPS,
            '-I', ROOT / 'examples', source, '-o', binary)


def test():
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
    else:
        raise ValueError(action)
