"""Local gates assembling live Moxie HEAD with the current shared assembler."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]/'tools'))
from dependencies import verify_external_dependencies
ASSEMBLY=ROOT.parents[3]/'shared/assembly'
sys.path.insert(0, str(ROOT.parents[3]/'tools'))
from compiler_limits import compiler_command
FLAGS=['-O3','--fp-mode=contract=off']
def env():
    e=os.environ.copy()
    for key,folder in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache','MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        p=ROOT/folder;p.mkdir(parents=True,exist_ok=True);e[key]=str(p)
    e.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return e
def run(*args,check=True):
    r=subprocess.run(compiler_command(args),cwd=ROOT,env=env(),text=True,capture_output=True)
    if check and r.returncode:
        print(r.stdout,r.stderr,file=sys.stderr);raise RuntimeError(f'command failed: {args}')
    return r

def assemble(consumer=ROOT):
    if consumer == ROOT:
        core = ROOT.parents[3]
        return run(sys.executable,core/'shared/assembly/tools/assemble_family.py','--consumer',ROOT,'--lock',core/'DEPENDENCIES.json','--output',ROOT/'builds/deps')
    verify_external_dependencies(consumer/'DEPENDENCIES.json')
    lock=consumer/'DEPENDENCIES.json'
    run(sys.executable,ASSEMBLY/'tools/consume_family.py','--consumer',consumer,'--lock',lock,'--tool',ASSEMBLY/'tools/assemble_family.py')
def compile(source,output,consumer=ROOT):
    run('mojo','build',*FLAGS,'-I',consumer/'builds/deps',source,'-o',output)
def build():
    assert run('mojo','--version').stdout.startswith('Mojo 1.0.0 ')
    metadata=list(Path(sys.prefix,'conda-meta').glob('max-core-*.json'))
    assert len(metadata)==1 and json.loads(metadata[0].read_text())['version']=='26.5.0'
    assemble()
    run('mojo','precompile','-I',ROOT/'builds/deps',ROOT/'builds/deps/moxie','-o',ROOT/'builds/moxie.mojoc')
    for s in [*sorted((ROOT/'tests').glob('test_*.mojo')),*sorted((ROOT/'examples').glob('*.mojo')), ROOT/'tools/benchmark.mojo']:
        compile(s,ROOT/'builds'/s.stem)
    for name in ['a','b']:
        consumer=ROOT/'tests/consumers'/name
        assemble(consumer)
        compile(consumer/'main.mojo',consumer/'builds/probe',consumer)
    print('PASS: build')
def negative():
    cases = {
        'sparse': 'from moxie._sparse import CSR\n',
        'nested': 'from moxie.feature_extraction._dict_vectorizer import FeatureValue\ndef bad():\n    var v = FeatureValue(Dict[String, Int]())\n',
        'iterable': 'from moxie.feature_extraction._dict_vectorizer import FeatureValue\ndef bad():\n    var v = FeatureValue(List[Int]([1,2]))\n',
    }
    for name,body in cases.items():
        path=ROOT/'builds'/f'negative_{name}.mojo'
        path.write_text(body+'\ndef main():\n    pass\n')
        result=run('mojo','build','-I',ROOT/'builds/deps',path,'-o',ROOT/'builds/negative',check=False)
        assert result.returncode,name
    lock=ROOT/'builds/bad-manifest.json'
    original=json.loads((ROOT/'DEPENDENCIES.json').read_text())
    # Keep relative source paths valid when moving this test manifest into builds.
    for project in original['projects']:
        project['path']=str((ROOT/project['path']).resolve())
    next(project for project in original['projects'] if project['id'] == 'moxie')['modules']['moxie.base']['file']='src/moxie/missing.mojo'
    lock.write_text(json.dumps(original))
    before=hashlib.sha256((ROOT/'builds/deps/ASSEMBLY.json').read_bytes()).hexdigest()
    result=run(sys.executable,ASSEMBLY/'tools/consume_family.py','--consumer',ROOT,'--lock',lock,'--tool',ASSEMBLY/'tools/assemble_family.py',check=False)
    assert result.returncode and 'missing' in result.stderr.lower()
    assert hashlib.sha256((ROOT/'builds/deps/ASSEMBLY.json').read_bytes()).hexdigest()==before

def main():
    action=sys.argv[1];build()
    if action=='test':
        for s in sorted((ROOT/'tests').glob('test_*.mojo')):
            print(run(ROOT/'builds'/s.stem).stdout,end='')
        for name in ['a','b']:
            consumer=ROOT/'tests/consumers'/name
            print(run(consumer/'builds/probe').stdout,end='')
            report=json.loads((consumer/'builds/deps/ASSEMBLY.json').read_text())
            assert report['modules']['moxie.feature_extraction._mapping']==['feature_extraction']
            assert report['modules']['moxie.feature_extraction._dict_vectorizer']==['dict_hash']
        negative()
        print('PASS: test')
    elif action=='parity':
        print(run(sys.executable,ROOT/'tests/parity.py').stdout,end='')
    elif action!='build': raise ValueError(action)
if __name__=='__main__':main()
