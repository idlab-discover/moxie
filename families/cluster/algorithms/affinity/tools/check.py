"""Compile current Moxie and family sources; keep all output local."""
from pathlib import Path
import json, os, subprocess, sys
ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT.parents[3]/'shared/assembly'
CONSUMERS = [ROOT/'tests/consumers'/n for n in ('a','b')]
FLAGS = ['-O3','--fp-mode=contract=off']
def environment(c):
    env = os.environ.copy()
    for k,v in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache','MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        path=c/v; path.mkdir(parents=True,exist_ok=True); env[k]=str(path)
    env['PYTHONDONTWRITEBYTECODE']='1'
    return env

def run(*args,c=ROOT,capture=False,check=True):
    return subprocess.run(compiler_command(args),cwd=c,env=environment(c),text=True,capture_output=capture,check=check)

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

def compile_one(c,source,name):
    run('mojo','build',*FLAGS,'-I',c/'builds/deps','-I',ROOT/'examples',source,'-o',c/'builds'/name,c=c)

def build(c=ROOT):
    assert run('mojo','--version',capture=True).stdout.startswith('Mojo 1.0.0 ')
    metadata=list(Path(sys.prefix,'conda-meta').glob('max-core-*.json'))
    assert len(metadata)==1 and json.loads(metadata[0].read_text())['version']=='26.5.0'
    assemble(c)
    run('mojo','precompile',c/'builds/deps/moxie','-I',c/'builds/deps','-o',c/'builds/moxie.mojoc',c=c)
    if c==ROOT:
        for path in sorted((ROOT/'tests').glob('*.mojo')): compile_one(c,path,path.stem)
        for path in sorted((ROOT/'examples').glob('*.mojo')): compile_one(c,path,'example_'+path.stem)
        for child in CONSUMERS: build(child)
    else: compile_one(c,c/'main.mojo','probe')

def test(c=ROOT):
    if c != ROOT:
        run(c/'builds/probe',c=c)
        return
    for path in sorted((ROOT/'tests').glob('test_*.mojo')): run(ROOT/'builds'/path.stem)
    for child in CONSUMERS:
        test(child)
        report=json.loads((child/'builds/deps/ASSEMBLY.json').read_text())
        for module,owner in [('moxie._dense','numerics'),('moxie._random','random'),('moxie.cluster._contracts','cluster')]:
            assert report['modules'][module]==[owner]
        assert 'moxie.consumer_a' in report['modules'] and 'moxie.consumer_b' in report['modules']
    for source,expected in [('sparse.mojo','_sparse')]:
        proc=run('mojo','build','-I',ROOT/'builds/deps',ROOT/'tests/compile_fail'/source,'-o',ROOT/'builds/invalid',capture=True,check=False)
        assert proc.returncode and expected in proc.stderr,proc.stderr
    c=CONSUMERS[0]
    original=json.loads((c/'DEPENDENCIES.json').read_text())
    for mode in ('missing','module','export'):
        lock=json.loads(json.dumps(original))
        if mode=='missing':
            next(project for project in lock['projects'] if project['id'] == 'moxie')['modules']['moxie.base']['file']='src/moxie/missing.mojo'
        else:
            entry=json.loads(json.dumps(next(p for p in lock['projects'] if p['id']=='consumer_a')))
            entry['id']='conflict'
            if mode=='export':
                entry['modules']['moxie.conflict']=entry['modules'].pop('moxie.consumer_a')
                entry['exports']['packages']['moxie']['ConsumerA']='moxie.conflict'
            lock['projects'].append(entry)
        # Paths are relative to the consumer, so keep temporary manifest at its root.
        bad=c/'bad-manifest.json'; before=(c/'builds/deps/ASSEMBLY.json').read_bytes()
        try:
            bad.write_text(json.dumps(lock))
            proc=assemble(c,bad,check=False)
            assert proc.returncode, mode
            assert (c/'builds/deps/ASSEMBLY.json').read_bytes()==before
        finally: bad.unlink(missing_ok=True)
    print('PASS: isolated composition, missing sources and ownership/export rejection')

if __name__=='__main__':
    action=sys.argv[1]
    consumer=ROOT/'tests/consumers'/sys.argv[2] if len(sys.argv)>2 else ROOT
    assert consumer in [ROOT,*CONSUMERS]
    build(consumer)
    if action=='test': test(consumer)
    elif action=='start': run(ROOT/'builds/example_affinity')
    elif action=='parity': run(sys.executable,ROOT/'tests/parity.py')
    elif action!='build': raise ValueError(action)
