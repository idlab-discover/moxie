"""Family checks against the live local sources of Moxie and other dependencies."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
MOXIE=ROOT.parents[1]
sys.path.insert(0, str(MOXIE / "tools"))
from compiler_limits import compiler_command
CONSUMERS=[ROOT/'tests/consumers'/n for n in ('a','b')]
FLAGS=['-O3','--fp-mode=contract=off']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def environment(root):
    env=os.environ.copy()
    for key,folder in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache',
        'MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        path=root/folder; path.mkdir(parents=True,exist_ok=True); env[key]=str(path)
    env['PYTHONDONTWRITEBYTECODE']='1'
    return env

def run(*args, root=ROOT, check=True):
    return subprocess.run(compiler_command(args),cwd=root,env=environment(root),check=check,text=True,capture_output=True)
def execute(*args,root=ROOT):
    p=run(*args,root=root)
    if p.stdout: print(p.stdout,end='',flush=True)
    return p

def check_owned(lock):
    owner=next(p for p in lock['projects'] if p['id']=='moxie')
    files={'moxie.ensemble.'+p.stem for p in (MOXIE/'src/moxie/ensemble').glob('*.mojo') if p.stem!='__init__'}
    assert {m for m in owner['modules'] if m.startswith('moxie.ensemble.')} == files
    exports=json.loads((ROOT/'EXPORTS.json').read_text())['packages']['moxie.ensemble']
    assert owner['exports']['packages']['moxie.ensemble']==exports
    for project in lock['projects']:
        assert 'moxie-ensemble' not in project['path'], project['path']

def ownership(root):
    deps=root/'builds/deps'
    report=json.loads((deps/'ASSEMBLY.json').read_text())
    for source in (MOXIE/'src/moxie/ensemble').glob('*.mojo'):
        if source.stem == '__init__': continue
        assert report['modules']['moxie.ensemble.'+source.stem]==['moxie']
        assert (deps/'moxie/ensemble'/source.name).read_bytes()==source.read_bytes()
    check_owned(json.loads((root/'builds/resolved.json').read_text()))

def resolve(root):
    """Compose current core owners and the two consumer contributions."""
    lock=json.loads((MOXIE/'DEPENDENCIES.json').read_text())
    for project in lock['projects']:
        project['path']=str((MOXIE/project['path']).resolve())
    if root != ROOT:
        for project in json.loads((root/'DEPENDENCIES.json').read_text())['projects']:
            project['path']=str((root/project['path']).resolve())
            lock['projects'].append(project)
    check_owned(lock)
    path=root/'builds/resolved.json'; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(lock,indent=2)+'\n')
    return path,lock

def assemble(root):
    path,lock=resolve(root)
    owner=Path(next(p for p in lock['projects'] if p['id']=='assembly')['path'])
    execute(sys.executable,owner/'tools/consume_family.py','--consumer',root,'--lock',path,
            '--tool',owner/'tools/assemble_family.py',root=root)

def compile_file(source,output,root=ROOT):
    execute('mojo','build',*FLAGS,'-I',root/'builds/deps','-I',ROOT/'examples',source,'-o',output,root=root)

def build():
    assert run('mojo','--version').stdout.startswith('Mojo 1.0.0 ')
    assemble(ROOT)
    execute('mojo','precompile',ROOT/'builds/deps/moxie','-I',ROOT/'builds/deps','-o',ROOT/'builds/moxie.mojoc')
    for p in sorted((ROOT/'tests').glob('test_*.mojo')):
        compile_file(p,ROOT/'builds'/p.stem)
    for driver in ('sampling_driver','forest_driver','public_forest_driver','public_edge_driver'):
        compile_file(ROOT/'tests'/(driver+'.mojo'),ROOT/'builds'/driver)
    compile_file(ROOT/'examples/estimator.mojo',ROOT/'builds/example')
    compile_file(ROOT/'examples/plumbing.mojo',ROOT/'builds/plumbing_example')
    compile_file(ROOT/'examples/forest_engine.mojo',ROOT/'builds/forest_engine_example')
    for root in CONSUMERS:
        assemble(root)
        execute('mojo','precompile',root/'builds/deps/moxie','-I',root/'builds/deps','-o',root/'builds/moxie.mojoc',root=root)
        compile_file(root/'main.mojo',root/'builds/probe',root)

def negative_compile(file,needle):
    p=run('mojo','build','-I',ROOT/'builds/deps',ROOT/file,'-o',ROOT/'builds/invalid',check=False)
    assert p.returncode and needle in p.stderr,p.stderr

def assembly_errors():
    root=CONSUMERS[0]
    original=resolve(root)[1]
    before=sha(root/'builds/deps/ASSEMBLY.json')
    owner=Path(next(p for p in original['projects'] if p['id']=='assembly')['path'])
    for mode in ['missing','module','export']:
        lock=json.loads(json.dumps(original))
        family=next(p for p in lock['projects'] if p['id']=='moxie')
        a=next(p for p in lock['projects'] if p['id']=='consumer_a')
        b=next(p for p in lock['projects'] if p['id']=='consumer_b')
        if mode=='missing': family['modules']['moxie.ensemble._records']={'file':'src/moxie/ensemble/_absent.mojo'}
        elif mode=='module':
            a['modules']['moxie.ensemble._records']=a['modules']['moxie.ensemble.consumer_a']
        else:
            b['exports']['packages']['moxie.ensemble']['consumer_a']='moxie.ensemble.consumer_b'
        path=root/'builds/bad-lock.json'
        path.write_text(json.dumps(lock))
        p=run(sys.executable,owner/'tools/consume_family.py','--consumer',root,'--lock',path,
              '--tool',owner/'tools/assemble_family.py',root=root,check=False)
        assert p.returncode,(mode,p.stdout)
        assert ('missing file' if mode=='missing' else 'collision') in p.stderr.lower(),p.stderr
        assert sha(root/'builds/deps/ASSEMBLY.json')==before

def test():
    ownership(ROOT)
    for consumer in CONSUMERS: ownership(consumer)
    for p in sorted((ROOT/'tests').glob('test_*.mojo')): execute(ROOT/'builds'/p.stem)
    negative_compile('tests/compile_fail/missing_predict.mojo',"required function 'predict' is not implemented")
    negative_compile('tests/compile_fail/dense_path.mojo','decision_path')
    outputs=[]
    for root in CONSUMERS:
        outputs.append(execute(root/'builds/probe',root=root).stdout)
        report=json.loads((root/'builds/deps/ASSEMBLY.json').read_text())
        for module,owner in [('moxie.ensemble._records','moxie'),('moxie._dense','numerics'),('moxie._random','random'),('moxie.tree._classes','moxie'),('moxie.tree._records','moxie')]:
            assert report['modules'][module]==[owner]
        assert sha(root/'builds/deps/moxie/ensemble/_records.mojo')==sha(MOXIE/'src/moxie/ensemble/_records.mojo')
    assert outputs[0]==outputs[1]
    assert (CONSUMERS[0]/'builds/deps/moxie/ensemble/_records.mojo').stat().st_ino != (CONSUMERS[1]/'builds/deps/moxie/ensemble/_records.mojo').stat().st_ino
    assembly_errors()
    print('PASS: dense contracts, two isolated composed consumers, missing sources and ownership/export conflicts')

def main():
    try:
        if not sys.argv[1].endswith('-only'): build()
        if sys.argv[1] in ('test','test-only'): test()
        elif sys.argv[1] in ('parity','parity-only'):
            # Exact comparisons against the independently built, pinned source reference.
            ownership(ROOT)
            provenance=json.loads((ROOT/'docs/private/forest_engine/oracle-build.json').read_text())
            revision=json.loads((ROOT/'PROVENANCE.json').read_text())['sklearn_revision']
            assert provenance['revision']==revision
            archive=subprocess.check_output(['git','-C',str(MOXIE.parent/'scikit-learn'),'archive',revision])
            assert hashlib.sha256(archive).hexdigest()==provenance['archive_sha256']
            assert Path(provenance['source']).resolve()==ROOT/'builds/reference'
            oracle=ROOT/'builds/oracle-env/bin/python'
            if not oracle.exists(): raise SystemExit('missing builds/oracle-env: run `pixi run ensemble-prepare-oracle` first')
            for script in ('sampling_parity.py','forest_parity.py','public_forest_parity.py','public_edge_parity.py'): execute(oracle,ROOT/'tests'/script)
        elif sys.argv[1]=='start': execute(ROOT/'builds/example')
        elif sys.argv[1]!='build': raise ValueError(sys.argv[1])
    except subprocess.CalledProcessError as exc:
        print(exc.stdout or '',end=''); print(exc.stderr or '',file=sys.stderr,end=''); raise
if __name__=='__main__': main()
