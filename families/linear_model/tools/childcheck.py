"""Verified assembly and native build/test/parity steps shared by family children.

Each child's tools/check.py calls `main(ROOT, native)`; `native` lists the
child's Mojo programs. Every build verifies the Mojo 1.0.0 / MAX 26.5.0
toolchain, reassembles `builds/deps` from the live sources declared in the
child's DEPENDENCIES.lock.json and precompiles it.

Environment: MOXIE_THREADS (default 1) sets OMP/OpenBLAS threads for the
Python oracle; MOXIE_MOJO_FLAGS overrides the default `-O3 --fp-mode=contract=off`.
"""
from pathlib import Path
import json
import os
import subprocess
import sys

FAMILY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
from compiler_limits import compiler_command
SUPPORT = FAMILY / 'tests/support'
FLAGS = os.environ.get('MOXIE_MOJO_FLAGS', '-O3 --fp-mode=contract=off').split()


def environment(root):
    env = os.environ.copy()
    for key, folder in {'XDG_CACHE_HOME': '.pixi/cache', 'PIXI_CACHE_DIR': '.pixi/pixi-cache',
                        'MODULAR_CACHE_DIR': '.pixi/modular-cache', 'TMPDIR': 'builds/tmp'}.items():
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    threads = os.environ.get('MOXIE_THREADS', '1')
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
    return env


def run(root, *args, capture=False):
    return subprocess.run(compiler_command(args), cwd=root, env=environment(root), check=True,
                          text=True, capture_output=capture)


def assemble(root):
    lock = json.loads((root / 'DEPENDENCIES.lock.json').read_text())
    pin = next(p for p in lock['projects'] if p['id'] == 'assembly')
    owner = (root / pin['path']).resolve()
    launcher = owner / 'tools/consume_family.py'
    run(root, sys.executable, launcher, '--consumer', root, '--lock', root / 'DEPENDENCIES.lock.json',
        '--tool', owner / 'tools/assemble_family.py')


def compile(root, source, output):
    run(root, 'mojo', 'build', *FLAGS, '-I', root / 'builds/deps', '-I', SUPPORT, source, '-o', output)


def build(root, native):
    version = run(root, 'mojo', '--version', capture=True).stdout
    assert version.startswith('Mojo 1.0.0 '), version
    metadata = list(Path(sys.prefix, 'conda-meta').glob('max-core-*.json'))
    assert len(metadata) == 1 and json.loads(metadata[0].read_text())['version'] == '26.5.0'
    assemble(root)
    run(root, 'mojo', 'precompile', root / 'builds/deps/moxie', '-I', root / 'builds/deps', '-o', root / 'builds/owned.mojoc')
    for source in native:
        compile(root, root / source, root / 'builds' / Path(source).stem)


def main(root, native, tests, parity):
    """native: sources to build; tests: executables (stems) run by `test`; parity: script."""
    mode = sys.argv[1]
    if mode not in ('build', 'test', 'parity', 'test-only', 'parity-only'):
        raise SystemExit('expected build, test or parity')
    if not mode.endswith('-only'):
        build(root, native)
    mode = mode.removesuffix('-only')
    if mode == 'test':
        for name in tests:
            run(root, root / 'builds' / name)
    elif mode == 'parity':
        run(root, sys.executable, root / parity)
