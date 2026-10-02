"""Use the current foundation assembler and one generated import namespace."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
from dependencies import assembly_command
CONSUMERS=[ROOT/'tests/consumers'/n for n in ('a','b')]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def env(c=ROOT):
    e=os.environ.copy()
    for key,folder in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache','MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        p=c/folder;p.mkdir(parents=True,exist_ok=True);e[key]=str(p)
    e['PYTHONDONTWRITEBYTECODE']='1'
    e['OMP_NUM_THREADS']='1';e['OPENBLAS_NUM_THREADS']='1'
    return e
def run(*args,c=ROOT,check=True):
    result=subprocess.run([str(a) for a in args],cwd=c,env=env(c),text=True,capture_output=True)
    if check and result.returncode:
        print(result.stdout,result.stderr,file=sys.stderr);result.check_returncode()
    return result

def assemble(c):
    run(*assembly_command(c))
def compile_file(c,source,name):
    run('mojo','build','-O3','--fp-mode=contract=off','-I',c/'builds/deps',source,'-o',c/'builds'/name,c=c)
def build():
    assert run('mojo','--version').stdout.startswith('Mojo 1.0.0 ')
    for c in [ROOT]+CONSUMERS:
        assemble(c)
        run('mojo','precompile',c/'builds/deps/moxie','-I',c/'builds/deps','-o',c/'builds/moxie.mojoc',c=c)
        if c!=ROOT:compile_file(c,c/'main.mojo','probe')
    for p in list(ROOT.glob('tests/*.mojo'))+list(ROOT.glob('examples/*.mojo')):compile_file(ROOT,p,p.stem)
    print('PASS build')
def test():
    print(run(ROOT/'builds/contracts').stdout)
    for c in CONSUMERS:
        print(run(c/'builds/probe',c=c).stdout)
        report=json.loads((c/'builds/deps/ASSEMBLY.json').read_text())
        assert report['modules']['moxie.preprocessing._dense_transforms']==['transforms']
        assert sha(c/'builds/deps/moxie/preprocessing/_dense_transforms.mojo')==sha(ROOT/'src/moxie/preprocessing/_dense_transforms.mojo')
    for name in ('sparse','integer'):
        failed=run('mojo','build','-I',ROOT/'builds/deps',ROOT/'tests/compile_fail'/f'{name}.mojo','-o',ROOT/'builds/rejected',check=False)
        assert failed.returncode and ('normalize' in failed.stderr or 'Float32/64' in failed.stderr),failed.stderr
    print('PASS test')
if __name__=='__main__':
    build()
    mode=sys.argv[1]
    if mode=='test':test()
    elif mode in ('parity','memory'):
        print(run(sys.executable,ROOT/'tests'/f'{mode}.py').stdout)
    elif mode!='build':raise ValueError(mode)
