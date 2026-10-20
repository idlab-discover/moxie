"""Assemble live Moxie HEAD sources and run native checks with local caches."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT.parents[3]/'shared/assembly'

CORE = ROOT.parents[3]
sys.path.insert(0, str(CORE / 'tools'))
from compiler_limits import compiler_command

def main():
    for key, name in {'PIXI_CACHE_DIR': '.pixi/cache', 'XDG_CACHE_HOME': '.pixi/xdg',
                      'MODULAR_CACHE_DIR': '.pixi/modular', 'TMPDIR': 'builds/tmp'}.items():
        folder = ROOT / name
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run([sys.executable, str(ASSEMBLY/'tools/assemble_family.py'), '--consumer', str(ROOT),
                    '--lock', str(ROOT/'DEPENDENCIES.json'), '--output', str(ROOT/'builds/deps')], check=True)
    action = sys.argv[1]
    if action not in ('build', 'test', 'parity'):
        raise ValueError(action)
    assert subprocess.check_output(['mojo', '--version'], text=True).startswith('Mojo 1.0.0 ')
    subprocess.run(compiler_command(['mojo', 'precompile', '-I', str(ROOT/'builds/deps'), str(ROOT/'builds/deps/moxie'), '-o', str(ROOT/'builds/moxie.mojoc')]), check=True)
    flags = ['-O3', '--fp-mode=contract=off', '-I', str(ROOT/'builds/deps')]
    subprocess.run(compiler_command(['mojo', 'build', *flags, str(ROOT/'tests/parameters.mojo'), '-o', str(ROOT/'builds/parameters')]), check=True)
    subprocess.run(compiler_command(['mojo', 'build', *flags, str(ROOT/'tests/parameter_dump.mojo'), '-o', str(ROOT/'builds/parameter_dump')]), check=True)
    subprocess.run(compiler_command(['mojo', 'build', *flags, str(ROOT/'examples/parameters.mojo'), '-o', str(ROOT/'builds/example')]), check=True)
    if action in ('test', 'parity'):
        subprocess.run([str(ROOT/'builds/parameters')], check=True)
    if action == 'parity':
        subprocess.run([sys.executable, str(ROOT/'tests/parameter_parity.py')], check=True)
        raise SystemExit('INCOMPLETE: forward/backprop and optimizer parity are not implemented')

if __name__ == '__main__':
    main()
