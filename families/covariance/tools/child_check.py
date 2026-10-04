"""Shared build/test/parity driver for covariance algorithm children.

Each algorithm's tools/check.py supplies its native sources and parity script.
Moxie's root manifest supplies the complete live namespace in the algorithm's
ignored builds/deps directory.

Environment:
  MOXIE_MOJO_FLAGS    extra space-separated Mojo compiler flags
  MOXIE_PARITY_SEEDS  number of seeded parity datasets per shape (default 3)
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

FAMILY = Path(__file__).resolve().parents[1]
FLAGS = ['-O3', '--fp-mode=contract=off', *os.environ.get('MOXIE_MOJO_FLAGS', '').split()]


def environment(root):
    env = os.environ.copy()
    for key, folder in {'XDG_CACHE_HOME': '.pixi/cache', 'PIXI_CACHE_DIR': '.pixi/pixi-cache',
                        'MODULAR_CACHE_DIR': '.pixi/modular-cache', 'TMPDIR': 'builds/tmp'}.items():
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


def run(root, *args, capture=False, check=True):
    return subprocess.run([str(a) for a in args], cwd=root, env=environment(root), check=check,
                          text=True, capture_output=capture)


def assemble(root):
    moxie = FAMILY.parents[1]
    run(root, sys.executable, moxie / 'shared/assembly/tools/assemble_family.py',
        '--consumer', root, '--lock', moxie / 'DEPENDENCIES.json',
        '--output', root / 'builds/deps')


def build(root, sources):
    """Verify toolchain, assemble, precompile the package and compile `sources`."""
    assert run(root, 'mojo', '--version', capture=True).stdout.startswith('Mojo 1.0.0 ')
    metadata = list(Path(sys.prefix, 'conda-meta').glob('max-core-*.json'))
    assert len(metadata) == 1 and json.loads(metadata[0].read_text())['version'] == '26.5.0'
    assemble(root)
    run(root, 'mojo', 'precompile', root / 'builds/deps/moxie', '-I', root / 'builds/deps', '-o', root / 'builds/moxie.mojoc')
    for source in sources:
        run(root, 'mojo', 'build', *FLAGS, '-I', root / 'builds/deps', '-I', FAMILY / 'tests/support', root / source, '-o', root / 'builds' / Path(source).stem)


def main(root, sources, tests, parity):
    mode = sys.argv[1] if len(sys.argv) == 2 else ''
    if mode not in ('build', 'test', 'parity'):
        raise SystemExit('usage: check.py build|test|parity')
    build(root, sources)
    if mode == 'test':
        for name in tests:
            run(root, root / 'builds' / name)
    elif mode == 'parity':
        run(root, sys.executable, root / parity)
