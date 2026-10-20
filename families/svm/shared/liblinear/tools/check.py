"""Local build driver; assembles live Moxie HEAD with the current shared assembler."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT.parents[3] / 'shared/assembly'

sys.path.insert(0, str(ROOT.parents[3] / 'tools'))
from compiler_limits import compiler_command
def run(*args, check=True, capture=False):
    env = os.environ.copy()
    for key, directory in {'XDG_CACHE_HOME': '.pixi/cache', 'PIXI_CACHE_DIR': '.pixi/pixi-cache', 'MODULAR_CACHE_DIR': '.pixi/modular-cache', 'TMPDIR': 'builds/tmp'}.items():
        path = ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return subprocess.run(compiler_command(args), cwd=ROOT, env=env, check=check, capture_output=capture, text=True)

def build():
    lock = ROOT / 'DEPENDENCIES.json'
    tool = next(p for p in json.loads(lock.read_text())['projects'] if p['id'] == 'assembly')
    owner = (ROOT / tool['path']).resolve()
    assert owner == ASSEMBLY.resolve(), owner
    run(sys.executable, owner / 'tools/consume_family.py', '--consumer', ROOT, '--lock', lock, '--tool', owner / 'tools/assemble_family.py')
    run('mojo', 'precompile', '-I', ROOT / 'builds/deps', ROOT / 'builds/deps/moxie', '-o', ROOT / 'builds/moxie.mojoc')
    for path in sorted((ROOT / 'tests').glob('*.mojo')) + sorted((ROOT / 'examples').glob('*.mojo')):
        run('mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', ROOT / 'builds/deps', path, '-o', ROOT / 'builds' / path.stem)

if __name__ == '__main__':
    action = sys.argv[1]
    if action not in ('build', 'test', 'parity', 'test-only', 'parity-only'):
        raise ValueError(action)
    if not action.endswith('-only'): build()
    if action in ('test', 'test-only'):
        for path in sorted((ROOT / 'tests').glob('test_*.mojo')):
            run(ROOT / 'builds' / path.stem)
        negative = ROOT / 'builds/sparse_capability.mojo'
        negative.write_text('from moxie.svm._liblinear import LibLinear\ndef main() raises:\n    var engine = LibLinear()\n    _ = engine.solve_sparse()\n')
        proc = run('mojo', 'build', '-I', ROOT / 'builds/deps', negative, '-o', ROOT / 'builds/unsupported', check=False, capture=True)
        assert proc.returncode and 'solve_sparse' in proc.stderr, proc.stderr
        print('PASS sparse capability is unavailable')
    if action in ('parity', 'parity-only'):
        run(sys.executable, ROOT / 'tests/parity.py')
