"""Assemble live Moxie HEAD sources with the current shared assembler, then check."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT.parents[3]/'shared/assembly'
CORE = ROOT.parents[3]
sys.path.insert(0, str(CORE / 'tools'))
from compiler_limits import compiler_command

FLAGS = ['-O3', '--fp-mode=contract=off']

def environment(consumer):
    env = os.environ.copy()
    for key, folder in {'PIXI_CACHE_DIR':'.pixi/cache', 'XDG_CACHE_HOME':'.pixi/xdg',
                        'MODULAR_CACHE_DIR':'.pixi/modular', 'TMPDIR':'builds/tmp'}.items():
        path = consumer/folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    return env

def run(*args, consumer=ROOT, check=True, capture=False):
    return subprocess.run(compiler_command(args), cwd=consumer, env=environment(consumer),
                          check=check, capture_output=capture, text=True)

def assembly(consumer, manifest=None, check=True):
    # Current shared assembler; reads live declared sources on every build.
    return run(sys.executable, ASSEMBLY/'tools/assemble_family.py', '--consumer', consumer,
               '--lock', manifest or consumer/'DEPENDENCIES.json',
               '--output', consumer/'builds/deps', consumer=consumer, check=check,
               capture=not check)

def compile_file(consumer, source, target):
    run('mojo', 'build', *FLAGS, '-I', consumer/'builds/deps', '-I', ROOT/'examples',
        source, '-o', consumer/'builds'/target, consumer=consumer)

def main():
    action = sys.argv[1]
    assert run('mojo', '--version', capture=True).stdout.startswith('Mojo 1.0.0 ')
    metadata = list(Path(sys.prefix, 'conda-meta').glob('max-core-*.json'))
    assert len(metadata) == 1 and json.loads(metadata[0].read_text())['version'] == '26.5.0'
    assembly(ROOT)
    compile_file(ROOT, ROOT/'tests/test_kernels.mojo', 'test_kernels')
    compile_file(ROOT, ROOT/'tests/kernel_dump.mojo', 'kernel_dump')
    compile_file(ROOT, ROOT/'tests/test_estimator.mojo', 'test_estimator')
    compile_file(ROOT, ROOT/'examples/rbm.mojo', 'example')
    compile_file(ROOT, ROOT/'tests/repro_float32_partial_fit.mojo', 'repro_float32_partial_fit')
    if action == 'test':
        run(ROOT/'builds/test_kernels')
        run(ROOT/'builds/test_estimator')
        rejected = run('mojo', 'build', *FLAGS, '-I', ROOT/'builds/deps',
                       ROOT/'tests/compile_no_sparse.mojo', '-o', ROOT/'builds/invalid',
                       check=False, capture=True)
        assert rejected.returncode and '_sparse' in rejected.stderr
        print('PASS unsupported sparse import is rejected')
    elif action == 'parity':
        run(sys.executable, ROOT/'tests/kernel_parity.py')
        raise RuntimeError('Private kernel checks passed; BernoulliRBM public estimator parity is not implemented')
    elif action == 'example':
        run(ROOT/'builds/example')
    elif action != 'build':
        raise ValueError(action)
    print('PASS', action, '(implemented subset; see docs/scope.md)')

if __name__ == '__main__':
    main()
