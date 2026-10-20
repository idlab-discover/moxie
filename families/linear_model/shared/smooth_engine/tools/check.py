"""Dependency readiness and blocker probes; no engine-completion claim."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    action = sys.argv[1]
    if action not in ('build', 'test', 'parity'):
        raise SystemExit('expected build, test or parity')
    for key, rel in {'XDG_CACHE_HOME': '.pixi/cache',
                     'MODULAR_CACHE_DIR': '.pixi/modular-cache',
                     'PIXI_CACHE_DIR': '.pixi/pixi-cache',
                     'TMPDIR': 'builds/tmp'}.items():
        folder = ROOT / rel
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    pins = json.loads((ROOT / 'DEPENDENCIES.lock.json').read_text())
    owner = next(p for p in pins['projects'] if p['id'] == 'assembly')
    assembly = (ROOT / owner['path']).resolve()
    launcher = assembly / 'tools/consume_family.py'
    subprocess.run([sys.executable, str(launcher), '--consumer', str(ROOT),
                    '--lock', str(ROOT / 'DEPENDENCIES.lock.json'), '--tool',
                    str(assembly / 'tools/assemble_family.py')], check=True)
    version = subprocess.check_output(['mojo', '--version'], text=True)
    assert version.startswith('Mojo 1.0.0 '), version
    subprocess.run(['mojo', 'precompile', str(ROOT / 'builds/deps/moxie'),
                    '-I', str(ROOT / 'builds/deps'), '-o',
                    str(ROOT / 'builds/moxie.mojoc')], check=True)
    subprocess.run(['mojo', 'build', '-O3', '--fp-mode=contract=off',
                    '-I', str(ROOT / 'builds/deps'),
                    str(ROOT / 'tests/numerics_probe.mojo'), '-o',
                    str(ROOT / 'builds/numerics_probe')], check=True)
    if action == 'test':
        subprocess.run([sys.executable, str(ROOT / 'tests/reference_probe.py')], check=True)
    if action == 'parity':
        raise SystemExit('INCOMPLETE: no native smooth losses/optimizers to compare. '
                         'See docs/scope.md and docs/private/handoffs/.')
    print('PASS: dependency/probe ' + action + '; smooth engine remains unimplemented')


if __name__ == '__main__':
    main()
