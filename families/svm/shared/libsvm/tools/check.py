"""Project gates assembling live Moxie HEAD with the current shared assembler."""
from pathlib import Path
import json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
for key,d in {'PIXI_CACHE_DIR':'.pixi/pixi-cache','XDG_CACHE_HOME':'.pixi/cache','MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
    p=ROOT/d;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0, str(ROOT.parents[3] / 'tools'))
from compiler_limits import compiler_command
def run(*args): subprocess.run(compiler_command(args),cwd=ROOT,check=True)
def build():
    manifest=json.loads((ROOT/'DEPENDENCIES.json').read_text())
    p=next(p for p in manifest['projects'] if p['id']=='assembly')
    tool=(ROOT/p['path']).resolve()
    assert tool==(ROOT.parents[3]/'shared/assembly').resolve(),tool
    run(sys.executable,tool/'tools/consume_family.py','--consumer',ROOT,'--lock',ROOT/'DEPENDENCIES.json','--tool',tool/'tools/assemble_family.py')
    assert subprocess.check_output(['mojo','--version'],text=True).startswith('Mojo 1.0.0 ')
    run('mojo','precompile','-I',ROOT/'builds/deps',ROOT/'builds/deps/moxie','-o',ROOT/'builds/moxie.mojoc')
    for folder in ('tests','examples'):
        for p in sorted((ROOT/folder).glob('*.mojo')):
            run('mojo','build','-O3','--fp-mode=contract=off','-I',ROOT/'builds/deps',p,'-o',ROOT/'builds'/p.stem)
if __name__=='__main__':
    action=sys.argv[1]
    if action not in ('build','test','parity','test-only','parity-only'): raise ValueError(action)
    if not action.endswith('-only'): build()
    if action in ('test','test-only'):
        tests=sorted((ROOT/'tests').glob('test_*.mojo'))
        if not tests: raise RuntimeError('No native tests found; refusing an empty test gate')
        for p in tests: run(ROOT/'builds'/p.stem)
    if action in ('parity','parity-only'): run(sys.executable,ROOT/'tests/parity.py')
