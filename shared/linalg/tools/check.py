"""Locked assembly, compilation and validation for shared covariance solves."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT.parent / 'assembly'


def run(*args):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def assemble(consumer):
    run(sys.executable, ASSEMBLY / 'tools/consume.py', '--consumer', consumer,
        '--lock', consumer / 'DEPENDENCIES.lock.json',
        '--tool', ASSEMBLY / 'tools/assemble.py')


def build():
    assert subprocess.check_output(['mojo', '--version'], text=True).startswith('Mojo 1.0.0 ')
    metadata = list(Path(sys.prefix, 'conda-meta').glob('max-core-*.json'))
    assert len(metadata) == 1 and json.loads(metadata[0].read_text())['version'] == '26.5.0'
    assemble(ROOT)
    run('mojo', 'precompile', '-I', ROOT / 'builds/deps', ROOT / 'src/moxie', '-o', ROOT / 'builds/linalg.mojoc')
    run('mojo', 'build', '-O3', '-I', ROOT / 'builds/deps', ROOT / 'tests/contracts.mojo', '-o', ROOT / 'builds/contracts')
    run('mojo', 'build', '-O3', '-I', ROOT / 'builds/deps', ROOT / 'tests/driver.mojo', '-o', ROOT / 'builds/driver')
    run('mojo', 'build', '-O3', '-I', ROOT / 'builds/deps', ROOT / 'tests/benchmark.mojo', '-o', ROOT / 'builds/benchmark')
    for name in ('a', 'b'):
        consumer = ROOT / 'tests/consumers' / name
        assemble(consumer)
        run('mojo', 'build', '-O3', '-I', consumer / 'builds/deps', consumer / 'main.mojo', '-o', consumer / 'builds/probe')


def main():
    build()
    if sys.argv[1] == 'test':
        run(ROOT / 'builds/contracts')
        rejected = subprocess.run(['mojo', 'build', '-I', str(ROOT/'builds/deps'),
                                   str(ROOT/'tests/negative/dtype.mojo'), '-o', str(ROOT/'builds/unsupported')],
                                  capture_output=True, text=True)
        assert rejected.returncode != 0 and 'Float32 or Float64' in rejected.stderr
        print('PASS unsupported dtype rejected explicitly')
        for name in ('a', 'b'):
            run(ROOT / 'tests/consumers' / name / 'builds/probe')
    elif sys.argv[1] == 'parity':
        run(sys.executable, ROOT / 'tests/parity.py')


if __name__ == '__main__':
    main()
