"""Build and validate integrated KernelRidge against live Moxie sources."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ['-O3', '--fp-mode=contract=off']


def environment(root=ROOT):
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


def run(*args, root=ROOT, capture=False):
    return subprocess.run([str(a) for a in args], cwd=root, env=environment(root),
                          check=True, text=True, capture_output=capture)


def assemble():
    moxie = ROOT.parents[1]
    run(sys.executable, moxie / 'shared/assembly/tools/assemble_family.py',
        '--consumer', ROOT, '--lock', moxie / 'DEPENDENCIES.json',
        '--output', ROOT / 'builds/deps')


def compile(source, output, root=ROOT):
    run('mojo', 'build', *FLAGS, '-I', ROOT / 'builds/deps',
        '-I', ROOT / 'examples', source, '-o', output, root=root)


def build():
    assemble()
    run('mojo', 'precompile', ROOT / 'builds/deps/moxie', '-I', ROOT / 'builds/deps',
        '-o', ROOT / 'builds/moxie.mojoc')
    compile(ROOT / 'examples/kernel_ridge.mojo', ROOT / 'builds/example')
    compile(ROOT / 'examples/callable_kernel.mojo', ROOT / 'builds/callable_example')


def main():
    mode = sys.argv[1]
    if mode not in ('test-only', 'parity-only'):
        build()
    mode = mode.removesuffix('-only')
    if mode == 'test':
        run(sys.executable, ROOT / 'tests/test_build.py')
        for name in ('contracts', 'capabilities', 'family', 'kernel_ridge', 'numerics', 'dtypes', 'serialization', 'api', 'fluent', 'composition'):
            executable = ROOT / 'builds' / ('test_' + name)
            compile(ROOT / 'tests' / ('test_' + name + '.mojo'), executable)
            run(executable)
        for name, diagnostic in (
            ('immutable_mutation', "operand mutability 'False' doesn't match expected mutability 'True'"),
            ('escaping_borrow', 'cannot return reference with incompatible origin'),
            ('missing_predict', "required function 'predict' is not implemented"),
            ('mismatched_kernel_adapter', 'adapter kernel type mismatch'),
        ):
            result = subprocess.run(
                ['mojo', 'build', *FLAGS, '-I', str(ROOT / 'builds/deps'),
                 '-I', str(ROOT / 'examples'),
                 str(ROOT / 'tests/compile_fail' / (name + '.mojo')),
                 '-o', str(ROOT / 'builds' / name)],
                cwd=ROOT, env=environment(), capture_output=True, text=True,
            )
            (ROOT / 'builds' / (name + '.log')).write_text(result.stdout + result.stderr)
            if result.returncode == 0 or diagnostic not in result.stderr:
                raise RuntimeError(f'Compile-time contract was not rejected: {name}')
    elif mode == 'parity':
        run(sys.executable, ROOT / 'tests/check_api_reference.py')
        run(sys.executable, ROOT / 'tests/check_parity.py')
        run(sys.executable, ROOT / 'tests/check_dtype_parity.py')
        run(sys.executable, ROOT / 'tests/check_numerics_parity.py')
        run(sys.executable, ROOT / 'tests/check_dense64_parity.py')
    elif mode == 'start':
        run(ROOT / 'builds/example')
        run(ROOT / 'builds/callable_example')
    elif mode != 'build':
        raise ValueError(mode)


if __name__ == '__main__':
    main()
