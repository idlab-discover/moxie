"""Consumer-local build and integration gate. Live shared sources."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
ASSEMBLY=ROOT.parent/'assembly'
CONSUMERS=[ROOT/'tests/consumers'/n for n in ('a','b')]
FLAGS=['-O3','--fp-mode=contract=off']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def environment(consumer):
    env=os.environ.copy()
    for key,folder in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache',
        'MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        path=consumer/folder; path.mkdir(parents=True,exist_ok=True); env[key]=str(path)
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['OMP_NUM_THREADS']='1'; env['OPENBLAS_NUM_THREADS']='1'
    return env

def run(*args, consumer=ROOT, capture=False):
    return subprocess.run([str(a) for a in args],cwd=consumer,env=environment(consumer),
        check=True,capture_output=capture,text=True)

def assemble(consumer):
    run(sys.executable,ASSEMBLY/'tools/consume.py','--consumer',consumer,
        '--lock',consumer/'DEPENDENCIES.lock.json','--tool',ASSEMBLY/'tools/assemble.py',consumer=consumer)

def compile_consumer(consumer):
    assemble(consumer)
    run('mojo','precompile',consumer/'builds/deps/moxie','-I',consumer/'builds/deps',
        '-o',consumer/'builds/assembled.mojoc',consumer=consumer)
    run('mojo','build',*FLAGS,'-I',consumer/'builds/deps',consumer/'main.mojo',
        '-o',consumer/'builds/probe',consumer=consumer)

def build():
    assert run('mojo','--version',capture=True).stdout.startswith('Mojo 1.0.0 ')
    metadata=list(Path(sys.prefix,'conda-meta').glob('max-core-*.json'))
    assert len(metadata)==1 and json.loads(metadata[0].read_text())['version']=='26.5.0'
    assemble(ROOT)
    run('mojo','precompile',ROOT/'builds/deps/moxie','-I',ROOT/'builds/deps','-o',ROOT/'builds/owned.mojoc')
    run('mojo','build',*FLAGS,'-I',ROOT/'builds/deps',ROOT/'examples/seeded.mojo','-o',ROOT/'builds/example')
    for c in CONSUMERS: compile_consumer(c)

def dependency_snapshot():
    roots=[]
    for p in json.loads((ROOT/'DEPENDENCIES.lock.json').read_text())['projects']:
        root=(ROOT/p['path']).resolve()
        if root==ROOT: continue
        if p['id']=='moxie': roots.extend([root/'src',root/'docs',root/'AGENTS.md'])
        else: roots.append(root)
    result={}
    for root in roots:
        if root.is_file():
            s=root.stat(); result[str(root)]=(s.st_size,s.st_mtime_ns); continue
        for parent,dirs,files in os.walk(root):
            dirs[:]=[d for d in dirs if d!='.git' and not (Path(parent)/d).is_symlink()]
            for name in files:
                f=Path(parent)/name
                s=f.lstat(); result[str(f)]=(s.st_size,s.st_mtime_ns)
    return result

def test():
    outputs=[run(c/'builds/probe',42,10,'stratified',1,0,2,consumer=c,capture=True).stdout for c in CONSUMERS]
    assert outputs[0]==outputs[1]
    for c in CONSUMERS:
        # Repeated integer calls reset, mutable state within a call advances.
        assert run(c/'builds/probe',42,10,'stratified',1,0,2,consumer=c,capture=True).stdout==outputs[0]
        bad=subprocess.run(['mojo','build','-I',str(c/'builds/deps'),str(ROOT/'tests/negative.mojo'),
            '-o',str(c/'builds/unsupported')],env=environment(c),cwd=c,capture_output=True,text=True)
        assert bad.returncode and '_random_choice_csc' in bad.stderr
        report=json.loads((c/'builds/deps/ASSEMBLY.json').read_text())
        assert report['modules']['moxie._random']==['random']
        assert report['modules']['moxie._dense']==['numerics']
        for key in ['XDG_CACHE_HOME','PIXI_CACHE_DIR','MODULAR_CACHE_DIR','TMPDIR']:
            assert Path(environment(c)[key]).is_relative_to(c)
        assert not (c/'builds/deps/moxie/_sparse.mojo').exists()
        assert sha(c/'builds/deps/moxie/_random.mojo')==sha(ROOT.parent/'random/src/moxie/_random.mojo')
        assert sha(c/'builds/deps/moxie/_dense.mojo')==sha(ROOT.parent/'numerics/src/moxie/_dense.mojo')
        assert not (c/'builds/deps').is_symlink()
    assert (CONSUMERS[0]/'builds/deps/moxie/_random.mojo').stat().st_ino != (CONSUMERS[1]/'builds/deps/moxie/_random.mojo').stat().st_ino
    # A missing source must fail without changing valid output.
    c=CONSUMERS[0]; lock=json.loads((c/'DEPENDENCIES.lock.json').read_text())
    next(p for p in lock['projects'] if p['id']=='random')['modules']['moxie._random']['file']='src/moxie/missing_for_test.mojo'
    badlock=c/'bad-lock.json'
    before=sha(c/'builds/deps/ASSEMBLY.json')
    try:
        badlock.write_text(json.dumps(lock))
        proc=subprocess.run([sys.executable,str(ASSEMBLY/'tools/consume.py'),'--consumer',str(c),
            '--lock',str(badlock),'--tool',str(ASSEMBLY/'tools/assemble.py')],env=environment(c),capture_output=True,text=True)
        assert proc.returncode and 'missing file' in proc.stderr.lower(),proc.stderr
        assert sha(c/'builds/deps/ASSEMBLY.json')==before
    finally: badlock.unlink(missing_ok=True)
    print('PASS: native contracts, independent consumers, missing sparse helper and missing-source rejection')

def main():
    action=sys.argv[1]
    before=dependency_snapshot()
    try:
        build()
        if action=='test': test()
        elif action=='parity': run(sys.executable,ROOT/'tests/parity.py')
        elif action!='build': raise ValueError(action)
    finally:
        after=dependency_snapshot()
        changed=sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k))
        evidence=ROOT/'docs/private/readiness'; evidence.mkdir(parents=True,exist_ok=True)
        (evidence/f'{action}-isolation.json').write_text(json.dumps({'dependency_files':len(before),'changed':changed},indent=2))
        assert not changed, f'Dependency writes: {changed[:10]}'
    print(f'PASS: {action}; {len(before)} dependency files unchanged')
if __name__=='__main__': main()
