"""Live foundation assembly and native family checks."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
from dependencies import assembly_command
CONSUMERS=[ROOT/'tests/consumers'/n for n in ('a','b')]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def env(c):
    e=os.environ.copy()
    for key,folder in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache','MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        p=c/folder; p.mkdir(parents=True,exist_ok=True); e[key]=str(p)
    e['PYTHONDONTWRITEBYTECODE']='1'
    return e

def run(*args,c=ROOT,check=True):
    return subprocess.run([str(a) for a in args],cwd=c,env=env(c),check=check,text=True,capture_output=True)
def assemble(c):
    manifest = ROOT.parents[1]/'DEPENDENCIES.json' if c == ROOT else None
    run(*assembly_command(c, manifest), c=c)
def compile_file(c,source,name):
    run('mojo','build','-O3','--fp-mode=contract=off','-I',c/'builds/deps','-I',ROOT/'examples',source,'-o',c/'builds'/name,c=c)
def build():
    assert run('mojo','--version').stdout.startswith('Mojo 1.0.0 ')
    metadata=list(Path(sys.prefix,'conda-meta').glob('max-core-*.json'))
    assert len(metadata)==1 and json.loads(metadata[0].read_text())['version']=='26.5.0'
    for c in [ROOT]+CONSUMERS:
        assemble(c)
        run('mojo','precompile',c/'builds/deps/moxie','-I',c/'builds/deps','-o',c/'builds/moxie.mojoc',c=c)
        if c != ROOT: compile_file(c,c/'main.mojo','probe')
    for test in ROOT.glob('tests/test_*.mojo'): compile_file(ROOT,test,test.stem)
    for example in ROOT.glob('examples/*.mojo'): compile_file(ROOT,example,example.stem)
    print('PASS build: complete assembled package, contracts, examples, two consumers')
def test():
    receipt=json.loads((ROOT/'builds/deps/ASSEMBLY.json').read_text())
    assert receipt['modules']['moxie.pipeline']==['moxie']
    assert receipt['modules']['moxie.preprocessing._scalers']==['preprocessing']
    assert receipt['modules']['moxie.preprocessing._moments']==['statistics']
    assert receipt['modules']['moxie.preprocessing._dense_transforms']==['transforms']
    print(run(sys.executable, ROOT/'tests/live_dependencies.py').stdout, end='')
    print(run(sys.executable, ROOT/'tests/api_surface.py').stdout, end='')
    print(run(sys.executable, ROOT/'tests/deferred_surface.py', '--require-complete').stdout, end='')
    print(run(sys.executable, ROOT/'tools/sync_manifests.py', '--check').stdout, end='')
    for test in ROOT.glob('tests/test_*.mojo'):
        result=run(ROOT/'builds'/test.stem); print(result.stdout,end='')
    for c in CONSUMERS:
        print(run(c/'builds/probe',c=c).stdout,end='')
        report=json.loads((c/'builds/deps/ASSEMBLY.json').read_text())
        assert report['modules']['moxie._dense']==['numerics']
        for module,owner in [('moxie.probe_a','probe_a'),('moxie.probe_b','probe_b'),('moxie.preprocessing._validation','preprocessing')]:
            assert report['modules'][module]==[owner]
        assert sha(c/'builds/deps/moxie/_dense.mojo')==sha(ROOT.parents[1]/'shared/numerics/src/moxie/_dense.mojo')
    for file,expected in [('missing_predict.mojo',"required function 'predict' is not implemented"),('sparse_input.mojo','validate_dense')]:
        result=run('mojo','build','-I',ROOT/'builds/deps',ROOT/'tests/compile_fail'/file,'-o',ROOT/'builds/rejected',check=False)
        assert result.returncode and expected in result.stderr,result.stderr
    c=CONSUMERS[0]
    original=json.loads((c/'DEPENDENCIES.json').read_text())
    for mode in ['missing','module','export']:
        lock=json.loads(json.dumps(original))
        p=next(p for p in lock['projects'] if p['id']=='probe_a')
        if mode=='missing': p['modules']['moxie.probe_a']['file']='src/moxie/absent.mojo'
        elif mode=='module': p['modules']['moxie.probe_b']={'file':'src/moxie/probe_a.mojo'}
        else:
            p['exports']['packages']['moxie.preprocessing']={'Preprocessor':'moxie.probe_a'}
        bad=c/'bad-lock.json'; before=sha(c/'builds/deps/ASSEMBLY.json')
        try:
            bad.write_text(json.dumps(lock))
            result=run(*assembly_command(c, bad), c=c, check=False)
            assert result.returncode,result.stdout
            assert mode=='missing' and 'missing file' in result.stderr.lower() or mode!='missing' and 'collision' in result.stderr.lower(),result.stderr
            assert sha(c/'builds/deps/ASSEMBLY.json')==before
        finally: bad.unlink(missing_ok=True)
    print('PASS test: native contracts, composed isolated consumers, missing sources and ownership/export conflicts')
if __name__=='__main__':
    try:
        if not sys.argv[1].endswith('-only'): build()
        if sys.argv[1] in ('test', 'test-only'): test()
        elif sys.argv[1]=='start': print(run(ROOT/'builds/plumbing').stdout,end='')
        elif sys.argv[1] in ('parity', 'parity-only'):
            # Each script compiles its own native driver against builds/deps.
            selected = sys.argv[2:] or sorted(p.stem.removeprefix('test_') for p in ROOT.glob('tests/parity/test_*.py'))
            for group in selected:
                result=subprocess.run([sys.executable, str(ROOT/'tests/parity'/f'test_{group}.py')],cwd=ROOT,env=env(ROOT),text=True)
                if result.returncode: raise SystemExit(f'parity failed: {group}')
        elif sys.argv[1]=='memory':
            # Peak-RSS acceptance of public scaler operations (algorithms/scalers/docs/memory.md).
            result=subprocess.run([sys.executable, str(ROOT/'algorithms/scalers/tests/memory/acceptance.py'), str(ROOT/'builds/deps'), *sys.argv[2:]],cwd=ROOT,env=env(ROOT),text=True)
            if result.returncode: raise SystemExit('memory acceptance failed: scalers')
        elif sys.argv[1] not in ('build',): raise ValueError(sys.argv[1])
    except subprocess.CalledProcessError as e:
        print(e.stdout or '',e.stderr or '',file=sys.stderr); raise
