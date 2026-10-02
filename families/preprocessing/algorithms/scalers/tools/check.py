"""Live dependency assembly and explicit incomplete-task gates."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT.parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
from dependencies import assembly_command

def environment():
    env = os.environ.copy()
    for key, folder in {
        'XDG_CACHE_HOME': '.pixi/cache',
        'PIXI_CACHE_DIR': '.pixi/pixi-cache',
        'MODULAR_CACHE_DIR': '.pixi/modular-cache',
        'TMPDIR': 'builds/tmp',
    }.items():
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env

def run(*args):
    subprocess.run([str(a) for a in args], cwd=ROOT, env=environment(), check=True)

def family(*args):
    # The family environment carries the pinned sklearn oracle and GNU time.
    subprocess.run(['pixi', 'run', '--manifest-path', str(FAMILY / 'pixi.toml'), 'python',
                    str(FAMILY / 'tools/check.py'), *args], cwd=FAMILY, check=True)

def build():
    run(*assembly_command(ROOT))
    version = subprocess.check_output(['mojo', '--version'], env=environment(), text=True)
    assert version.startswith('Mojo 1.0.0 '), version
    metadata = list(Path(sys.prefix, 'conda-meta').glob('max-core-*.json'))
    assert len(metadata) == 1 and json.loads(metadata[0].read_text())['version'] == '26.5.0'
    run('mojo', 'precompile', ROOT / 'builds/deps/moxie', '-I', ROOT / 'builds/deps',
        '-o', ROOT / 'builds/moxie.mojoc')
    for source in sorted(ROOT.glob('tests/*.mojo')) + sorted(ROOT.glob('examples/*.mojo')):
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', ROOT / 'builds/deps',
            source, '-o', ROOT / 'builds' / source.stem)
    print('PASS build: live shared modules and local contract probes only', flush=True)

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode not in ('build', 'test', 'parity', 'memory'):
        raise SystemExit('expected build, test, parity, or memory')
    build()
    if mode == 'test':
        run(ROOT / 'builds/test_moments_contract')
        run(sys.executable, ROOT / 'tests/owner_update_parity.py')
        run(sys.executable, ROOT / 'tests/functional_statistics.py')
        run(sys.executable, ROOT / 'tests/unweighted_reduction.py')
    elif mode == 'parity':
        run(sys.executable, ROOT / 'tests/owner_update_parity.py', '--require-parity')
        run(sys.executable, ROOT / 'tests/functional_statistics.py', '--require-parity')
        run(sys.executable, ROOT / 'tests/unweighted_reduction.py', '--require-parity')
        # The eight public exports are assembled by the family package.
        family('parity', 'scalers')
    elif mode == 'memory':
        family('memory')
