"""Build and validate against live dependency sources via the shared assembler."""
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
BUILDS = ROOT/'builds'
def env():
    result = os.environ.copy()
    for key, folder in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache','MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        path = ROOT/folder
        path.mkdir(parents=True, exist_ok=True)
        result[key] = str(path)
    result.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return result
def run(*args, **kw):
    return subprocess.run([str(a) for a in args], cwd=ROOT, env=env(), check=True, **kw)
def assemble():
    sys.path.insert(0, str(ROOT.parents[1]/'tools'))
    from assembly import assemble as assemble_live
    assemble_live(ROOT, ROOT/'DEPENDENCIES.json', env())

def compile(source, output):
    run('mojo','build','-O3','--fp-mode=contract=off','-I',BUILDS/'deps',source,'-o',output)
def build():
    assemble()
    run('mojo','precompile',ROOT/'builds/deps/moxie','-I',ROOT/'builds/deps','-o',ROOT/'builds/moxie.mojoc')
    for folder in ('tests','examples'):
        for p in sorted((ROOT/folder).glob('*.mojo')): compile(p, ROOT/'builds'/p.stem)
def main():
    action=sys.argv[1]
    build()
    if action=='test':
        for p in sorted((ROOT/'tests').glob('test_*.mojo')): run(ROOT/'builds'/p.stem)
    elif action=='parity':
        run(sys.executable,ROOT/'tests/parity.py')
        run(sys.executable,ROOT/'tests/integer_parity.py')
        run(sys.executable,ROOT/'tests/string_parity.py')
    elif action!='build': raise ValueError(action)
if __name__=='__main__': main()
