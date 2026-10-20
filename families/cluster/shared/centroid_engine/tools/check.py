"""Compile current Moxie and family sources; keep all output local."""
from pathlib import Path
import json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT.parents[3]/'shared/assembly'
CONSUMERS=[ROOT/'tests/consumers'/name for name in ('a','b')]
FLAGS=['-O3','--fp-mode=contract=off']
def environment(c):
    env=os.environ.copy()
    for key,folder in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache','MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        path=c/folder; path.mkdir(parents=True,exist_ok=True); env[key]=str(path)
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    return env
def run(*args,c=ROOT,capture=False,check=True):
    return subprocess.run(compiler_command(args),cwd=c,env=environment(c),check=check,text=True,capture_output=capture)
# Use the current Moxie namespace builder directly, without a pinned launcher.
sys.path.insert(0, str(ASSEMBLY / 'tools'))
from family_validator import assemble as compose_sources, AssemblyError
sys.path.insert(0, str(ASSEMBLY.parents[1] / 'tools'))
from compiler_limits import compiler_command

def assemble(c, manifest=None, check=True):
    try:
        compose_sources(c, manifest or c/'DEPENDENCIES.json', c/'builds/deps')
    except (AssemblyError, OSError, ValueError, TypeError) as exc:
        if check:
            raise
        return subprocess.CompletedProcess([], 1, '', str(exc))
    return subprocess.CompletedProcess([], 0, '', '')

def build(c):
    assert run('mojo','--version',capture=True).stdout.startswith('Mojo 1.0.0 ')
    assemble(c)
    run('mojo','precompile',c/'builds/deps/moxie','-I',c/'builds/deps','-o',c/'builds/moxie.mojoc',c=c)
    sources=sorted((c/'tests').glob('*.mojo'))+sorted((c/'examples').glob('*.mojo')) if c==ROOT else [c/'main.mojo']
    for path in sources:
        run('mojo','build',*FLAGS,'-I',c/'builds/deps',path,'-o',c/'builds'/path.stem,c=c)
    if c==ROOT:
        for consumer in CONSUMERS: build(consumer)
def test(c):
    if c!=ROOT:
        run(c/'builds/main',c=c)
        return
    for path in sorted((ROOT/'tests').glob('test_*.mojo')): run(ROOT/'builds'/path.stem)
    for consumer in CONSUMERS:
        test(consumer)
        report=json.loads((consumer/'builds/deps/ASSEMBLY.json').read_text())
        for module,owner in [('moxie.cluster._centroid_engine','centroid_engine'),('moxie.cluster._contracts','cluster'),('moxie._dense','numerics')]:
            assert report['modules'][module]==[owner]
        assert all('moxie.centroid_consumer_'+name in report['modules'] for name in ('a','b'))
    consumer=CONSUMERS[0]
    original=json.loads((consumer/'DEPENDENCIES.json').read_text())
    for mode in ('missing','conflict'):
        data=json.loads(json.dumps(original))
        engine=next(p for p in data['projects'] if p['id']=='centroid_engine')
        if mode=='missing': next(project for project in data['projects'] if project['id'] == 'moxie')['modules']['moxie.base']['file']='src/moxie/missing.mojo'
        else:
            extra=json.loads(json.dumps(engine));extra['id']='conflicting_engine';data['projects'].append(extra)
        path=consumer/'builds/bad-manifest.json'
        # Preserve manifest-relative paths while storing failure input in ignored builds.
        for project in data['projects']: project['path']=str((consumer/project['path']).resolve())
        path.write_text(json.dumps(data))
        before=(consumer/'builds/deps/ASSEMBLY.json').read_bytes()
        failure=assemble(consumer,path,check=False)
        assert failure.returncode,mode
        assert before==(consumer/'builds/deps/ASSEMBLY.json').read_bytes()
    failure=run('mojo','build','-I',ROOT/'builds/deps',ROOT/'tests/compile_fail/sparse.mojo','-o',ROOT/'builds/invalid',capture=True,check=False)
    assert failure.returncode and 'assign_sparse' in failure.stderr
    print('PASS: two isolated consumers, missing sources/ownership rejection, sparse negative capability')
def main():
    c=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT
    assert c in [ROOT,*CONSUMERS]
    build(c)
    if sys.argv[1]=='test': test(c)
    elif sys.argv[1]=='parity': run(sys.executable,ROOT/'tests/parity.py')
    elif sys.argv[1]!='build': raise ValueError(sys.argv[1])
if __name__=='__main__': main()
