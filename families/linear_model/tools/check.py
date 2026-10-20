"""Family gates using current shared sources; no dependency writes."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
from compiler_limits import compiler_command
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def env(c):
    result=os.environ.copy()
    for key, folder in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache','MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        p=c/folder; p.mkdir(parents=True,exist_ok=True); result[key]=str(p)
    result['PYTHONDONTWRITEBYTECODE']='1'
    return result

def run(args,c=ROOT,ok=True):
    return subprocess.run(compiler_command(args),cwd=c,env=env(c),check=ok,text=True,capture_output=not ok)

def assemble(c,lock=None,ok=True):
    pins=json.loads((c/'DEPENDENCIES.lock.json').read_text())
    owner=next(p for p in pins['projects'] if p['id']=='assembly')
    tool=(c/owner['path']).resolve()
    return run([sys.executable,tool/'tools/consume_family.py','--consumer',c,'--lock',lock or c/'DEPENDENCIES.lock.json','--tool',tool/'tools/assemble_family.py'],c,ok)

def compile(source,out,c=ROOT):
    run(['mojo','build','-O3','--fp-mode=contract=off','-I',c/'builds/deps','-I',ROOT/'examples',source,'-o',out],c)

def main():
    action=sys.argv[1]
    assert action in ('build','test','start','parity','test-only','parity-only')
    build = not action.endswith('-only')
    mode = action.removesuffix('-only')
    version=run(['mojo','--version'],ok=False)
    assert version.returncode == 0 and version.stdout.startswith('Mojo 1.0.0 ')
    metadata=list(Path(sys.prefix,'conda-meta').glob('max-core-*.json'))
    assert len(metadata)==1 and json.loads(metadata[0].read_text())['version']=='26.5.0'
    consumers=[ROOT/'tests/consumers'/n for n in ('a','b')]
    for c in ([ROOT]+consumers if build else []):
        assemble(c)
        run(['mojo','precompile',c/'builds/deps/moxie','-I',c/'builds/deps','-o',c/'builds/moxie.mojoc'],c)
        if c!=ROOT:
            compile(c/'main.mojo',c/'builds/probe',c)
    if build:
        compile(ROOT/'examples/estimator.mojo',ROOT/'builds/example')
        compile(ROOT/'examples/dense_estimators.mojo',ROOT/'builds/dense_estimators')
        compile(ROOT/'examples/dense_ownership.mojo',ROOT/'builds/dense_ownership')
    for name in ['contracts','capabilities','family','plumbing','linear_base','fluent']:
        if build: compile(ROOT/f'tests/test_{name}.mojo',ROOT/f'builds/test_{name}')
        if mode=='test': run([ROOT/f'builds/test_{name}'])
    if mode=='start': run([ROOT/'builds/dense_estimators'])
    if mode in ('test', 'parity'): run([ROOT/'builds/dense_estimators'])
    if mode in ('build', 'test', 'parity'):
        for name in ('least_squares', 'ridge', 'elastic_net', 'bayesian'):
            child=ROOT/'algorithms'/name
            run([sys.executable, child/'tools/check.py', action],child)
    if mode in ('build', 'test', 'parity'):
        child=ROOT/'shared/coordinate_engine'
        run([sys.executable, child/'tools/check.py', action],child)
    if mode=='test':
        for c in consumers:
            run([c/'builds/probe'],c)
            report=json.loads((c/'builds/deps/ASSEMBLY.json').read_text())
            assert report['modules']['moxie.linear_model._contracts']==['moxie']
            assert report['modules']['moxie._dense']==['numerics']
            assert sha(c/'builds/deps/moxie/linear_model/_contracts.mojo')==sha(ROOT.parents[1]/'src/moxie/linear_model/_contracts.mojo')
        bad=run(['mojo','build','-I',ROOT/'builds/deps',ROOT/'tests/compile_fail/missing_predict.mojo','-o',ROOT/'builds/bad'],ok=False)
        assert bad.returncode and "required function 'predict' is not implemented" in bad.stderr
        assert (consumers[0]/'builds/deps/moxie/linear_model/_contracts.mojo').stat().st_ino != (consumers[1]/'builds/deps/moxie/linear_model/_contracts.mojo').stat().st_ino
        c=consumers[0]
        for mode in ['missing','module','export']:
            lock=json.loads((c/'DEPENDENCIES.lock.json').read_text())
            owner=next(p for p in lock['projects'] if p['id']=='probe_a')
            f=owner['modules']['moxie.linear_model.probe_a']['file']
            if mode=='missing': owner['modules']['moxie.linear_model.probe_a']['file']='src/missing.mojo'
            elif mode=='module': owner['modules']['moxie.linear_model._contracts']={'file':f}
            else: owner['exports']['packages']['moxie.linear_model']['LinearRegressor']='moxie.linear_model.probe_a'
            p=c/'bad-lock.json'
            before=sha(c/'builds/deps/ASSEMBLY.json')
            try:
                p.write_text(json.dumps(lock)); failure=assemble(c,p,False)
                assert failure.returncode,mode
                assert sha(c/'builds/deps/ASSEMBLY.json')==before
            finally: p.unlink(missing_ok=True)
        print('PASS: isolated consumers, missing sources, ownership/export conflicts and negative capability')
if __name__=='__main__': main()
