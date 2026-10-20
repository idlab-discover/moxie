"""Dependency/probe gates for an explicitly blocked, not implemented SGD engine."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    return subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def main():
    action = sys.argv[1]
    if action not in ('build', 'test', 'parity'):
        raise ValueError(action)
    for key, directory in {'XDG_CACHE_HOME': '.pixi/cache',
                           'MODULAR_CACHE_DIR': '.pixi/modular-cache',
                           'TMPDIR': 'builds/tmp'}.items():
        path = ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    pins = json.loads((ROOT / 'DEPENDENCIES.lock.json').read_text())
    pin = next(p for p in pins['projects'] if p['id'] == 'assembly')
    owner = (ROOT / pin['path']).resolve()
    launcher = owner / 'tools/consume_family.py'
    run([sys.executable, launcher, '--consumer', ROOT, '--lock',
         ROOT / 'DEPENDENCIES.lock.json', '--tool', owner / 'tools/assemble_family.py'])
    version = subprocess.check_output(['mojo', '--version'], text=True)
    assert version.startswith('Mojo 1.0.0 '), version
    run(['mojo', 'precompile', ROOT / 'builds/deps/moxie', '-I',
         ROOT / 'builds/deps', '-o', ROOT / 'builds/moxie.mojoc'])
    run(['mojo', 'build', '-O3', '--fp-mode=contract=off', '-I',
         ROOT / 'builds/deps', ROOT / 'tests/rng_probe.mojo',
         '-o', ROOT / 'builds/rng_probe'])
    if action == 'test':
        run([ROOT / 'builds/rng_probe'])
        print('PASS dependency probe only; SGD is not implemented')
    if action == 'parity':
        run([sys.executable, ROOT / 'tests/rng_parity.py'])


if __name__ == '__main__':
    main()
