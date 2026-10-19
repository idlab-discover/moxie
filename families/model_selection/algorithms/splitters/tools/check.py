"""Assemble live Moxie HEAD and family sources before compiling the splitter consumer."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    action = sys.argv[1]
    if action not in ('build', 'test', 'parity'):
        raise ValueError(action)
    env = os.environ.copy()
    for key, directory in {'XDG_CACHE_HOME': '.pixi/cache', 'PIXI_CACHE_DIR': '.pixi/pixi-cache', 'MODULAR_CACHE_DIR': '.pixi/modular-cache', 'TMPDIR': 'builds/tmp'}.items():
        path = ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    manifest = ROOT / 'DEPENDENCIES.json'
    tool = next(p for p in json.loads(manifest.read_text())['projects'] if p['id'] == 'assembly')
    owner = (ROOT / tool['path']).resolve()
    assert owner == (ROOT.parents[3] / 'shared/assembly').resolve(), owner
    launcher = owner / 'tools/consume_family.py'
    subprocess.run([sys.executable, str(launcher), '--consumer', str(ROOT), '--lock', str(manifest), '--tool', str(owner / 'tools/assemble_family.py')], env=env, check=True)
    subprocess.run(['mojo', 'precompile', '-I', str(ROOT / 'builds/deps'), str(ROOT / 'builds/deps/moxie'), '-o', str(ROOT / 'builds/package.mojoc')], env=env, check=True)
    sources = sorted((ROOT / 'tests').glob('*.mojo'))
    if not sources:
        raise RuntimeError('Splitter implementation and native contracts are not yet present')
    for source in sources + sorted((ROOT / 'examples').glob('*.mojo')):
        binary = ROOT / 'builds' / source.stem
        subprocess.run(['mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', str(ROOT / 'builds/deps'), str(source), '-o', str(binary)], env=env, check=True)
        if action == 'test':
            subprocess.run([str(binary)], env=env, check=True)
    if action == 'parity':
        subprocess.run([sys.executable, str(ROOT / 'tests/parity.py')], env=env, check=True)
    print('PASS', action)

if __name__ == '__main__':
    main()
