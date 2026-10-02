"""Live assembly and the four acceptance tasks."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
from dependencies import assembly_command
CONSUMERS=[ROOT/'tests/consumers'/n for n in ('a','b')]
def environment():
    e=os.environ.copy()
    for key,folder in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache','MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        p=ROOT/folder; p.mkdir(parents=True,exist_ok=True); e[key]=str(p)
    e.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    return e
def run(*args,**kwargs):
    return subprocess.run([str(a) for a in args],cwd=ROOT,env=environment(),check=True,**kwargs)
def assemble(c):
    run(*assembly_command(c))
def compile_file(source,dest,c=ROOT):
    run('mojo','build','-O3','--fp-mode=contract=off','-I',c/'builds/deps',source,'-o',dest)
def build():
    assert subprocess.check_output(['mojo','--version'],env=environment(),text=True).startswith('Mojo 1.0.0 ')
    meta=list(Path(sys.prefix,'conda-meta').glob('max-core-*.json'))
    assert len(meta)==1 and json.loads(meta[0].read_text())['version']=='26.5.0'
    assemble(ROOT)
    run('mojo','precompile',ROOT/'builds/deps/moxie','-I',ROOT/'builds/deps','-o',ROOT/'builds/moxie.mojoc')
    for source in sorted(ROOT.glob('tests/*.mojo'))+sorted(ROOT.glob('examples/*.mojo')):
        compile_file(source,ROOT/'builds'/source.stem)
    for c in CONSUMERS:
        if (c/'main.mojo').exists():
            assemble(c); compile_file(c/'main.mojo',c/'builds/probe',c)
    print('PASS build',flush=True)
if __name__=='__main__':
    mode=sys.argv[1]
    assert mode in ('build','test','parity','memory')
    build()
    if mode=='test':
        for source in sorted(ROOT.glob('tests/test_*.mojo')): run(ROOT/'builds'/source.stem)
        for c in CONSUMERS: run(c/'builds/probe')
        run(sys.executable,ROOT/'tests/assembly.py')
    elif mode in ('parity','memory'):
        run(sys.executable,ROOT/'tests'/f'{mode}.py')
        if mode=='parity': run(sys.executable,ROOT/'tests/functional_parity.py')
