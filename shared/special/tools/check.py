"""Build current shared sources; Python is validation tooling only."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT.parent / 'assembly'


def run(*args):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def assemble(consumer):
    run(sys.executable, ASSEMBLY/'tools/consume.py', '--consumer', consumer,
        '--lock', consumer/'DEPENDENCIES.lock.json', '--tool', ASSEMBLY/'tools/assemble.py')


def verify_toolchain():
    assert subprocess.check_output(['mojo','--version'],text=True).startswith('Mojo 1.0.0 ')
    metadata = list(Path(sys.prefix,'conda-meta').glob('max-core-*.json'))
    assert len(metadata) == 1 and json.loads(metadata[0].read_text())['version'] == '26.5.0'


def build():
    verify_toolchain()
    assemble(ROOT)
    run('mojo','precompile','-I',ROOT/'builds/deps',ROOT/'src/moxie','-o',ROOT/'builds/special.mojoc')
    for name in ('contracts','driver','benchmark'):
        run('mojo','build','-O3','-I',ROOT/'builds/deps',ROOT/f'tests/{name}.mojo','-o',ROOT/f'builds/{name}')
    for name in ('a','b'):
        consumer=ROOT/'tests/consumers'/name
        assemble(consumer)
        run('mojo','build','-O3','-I',consumer/'builds/deps',consumer/'main.mojo','-o',consumer/'builds/probe')


def main():
    if sys.argv[1] == 'libm-accuracy':
        verify_toolchain()
        assemble(ROOT)
        run('mojo','build','-O3','-I',ROOT/'builds/deps',
            ROOT/'tests/libm_driver.mojo','-o',ROOT/'builds/libm_driver')
        run(sys.executable,ROOT/'tests/libm_accuracy.py')
        return
    build()
    if sys.argv[1] == 'test':
        run(ROOT/'builds/contracts')
        for name in ('a','b'):
            run(ROOT/'tests/consumers'/name/'builds/probe')
    elif sys.argv[1] == 'parity':
        run(sys.executable,ROOT/'tests/parity.py')


if __name__ == '__main__':
    main()
