"""Build and run serialization contracts with the supported codec toolchain."""
import argparse
from pathlib import Path
import subprocess
import sys

from compiler_limits import compiler_command

ROOT = Path(__file__).resolve().parents[1]
CONSUMER = ROOT / 'tests'


def run(*args):
    subprocess.run(compiler_command(args), cwd=CONSUMER, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codec', choices=('raw', 'snappy', 'all'), default='all')
    args = parser.parse_args()
    version = subprocess.check_output(['mojo', '--version'], text=True).strip()
    if not version.startswith('Mojo 1.0.0 '):
        raise SystemExit(
            f'Unsupported serialization toolchain: {version}. '
            'Use the root locked Pixi environment (Mojo 1.0.0); '
            'mojo-snappy 0.2.0 build hb0f4dca_0 is not compatible with Mojo 1.2.'
        )
    deps = CONSUMER / 'builds/deps'
    run(sys.executable, ROOT / 'shared/assembly/tools/assemble_family.py',
        '--consumer', CONSUMER, '--lock', ROOT / 'DEPENDENCIES.json', '--output', deps)
    names = ('serialization', 'serialization_snappy') if args.codec == 'all' else (
        'serialization' if args.codec == 'raw' else 'serialization_snappy',
    )
    for name in names:
        binary = CONSUMER / 'builds' / f'test_{name}'
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', deps,
            ROOT / 'tests' / f'test_{name}.mojo', '-o', binary)
        run(binary)
    if args.codec in ('raw', 'all'):
        run(sys.executable, ROOT / 'tests/serialization_wire.py')


if __name__ == '__main__':
    main()
