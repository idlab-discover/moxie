"""Build/test/parity orchestration; never rewrites dependency pins."""
from pathlib import Path
import json
import subprocess
import sys
from assemble import assemble
from consume import consume

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([str(a) for a in args], check=True, cwd=ROOT)


def compile(source, target, includes):
    args = ['mojo', 'build', '-O3']
    for include in includes:
        args += ['-I', str(include)]
    run(*args, source, '-o', target)


def build():
    version = subprocess.check_output(['mojo', '--version'], text=True)
    if not version.startswith('Mojo 1.0.0'):
        raise RuntimeError(f'compiler mismatch: {version}')
    metadata = list(Path(sys.prefix, 'conda-meta').glob('max-core-*.json'))
    if len(metadata) != 1 or json.loads(metadata[0].read_text())['version'] != '26.5.0':
        raise RuntimeError('MAX mismatch')
    assemble(ROOT, ROOT / 'DEPENDENCIES.lock.json', ROOT / 'builds/deps')
    # Compile every owned runtime module as a package, including the empty owner API.
    run('mojo', 'precompile', ROOT / 'src/moxie', '-o', ROOT / 'builds/owner.mojoc')
    for letter in ('a', 'b'):
        consumer = ROOT / 'tests/fixtures' / f'consumer_{letter}'
        consume(consumer, consumer / 'DEPENDENCIES.lock.json', ROOT / 'tools/assemble.py')
        compile(consumer / 'main.mojo', consumer / 'builds/probe', [consumer / 'builds/deps'])
        run('mojo', 'precompile', consumer / 'builds/deps/moxie', '-I', consumer / 'builds/deps',
            '-o', consumer / 'builds/moxie.mojoc')
    compile(ROOT / 'examples/consumer.mojo', ROOT / 'builds/example', [ROOT / 'tests/fixtures/consumer_a/builds/deps'])
    compile(ROOT / 'tests/direct.mojo', ROOT / 'builds/direct',
            [ROOT.parent / 'numerics/src', ROOT.parents[2] / '../NuMojo'])
    compile(ROOT / 'tests/direct_parent.mojo', ROOT / 'builds/direct_parent', [ROOT.parents[1] / 'src'])


def main():
    action = sys.argv[1]
    build()
    if action == 'test':
        run(sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v')
        for letter in ('a', 'b'):
            run(ROOT / 'tests/fixtures' / f'consumer_{letter}' / 'builds/probe')
        run(ROOT / 'builds/example')
    elif action == 'parity':
        run(ROOT / 'builds/direct_parent')
        expected = subprocess.check_output([ROOT / 'builds/direct'])
        for letter in ('a', 'b'):
            consumer = ROOT / 'tests/fixtures' / f'consumer_{letter}'
            actual = subprocess.check_output([consumer / 'builds/probe'])
            if actual != expected:
                raise RuntimeError(f'consumer {letter} differs: {actual!r} != {expected!r}')
        print('PASS: exact native assembly/direct-import parity, Float32 and Float64; no estimator parity claim')


if __name__ == '__main__':
    main()
