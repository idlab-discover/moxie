"""Validate the Moxie-owned query engine with its live shared dependencies."""
from pathlib import Path
import json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
ASSEMBLY=ROOT.parents[3]/'shared/assembly'
def run(*args,check=True,capture=False):
    e=os.environ.copy()
    for k,v in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache','MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        p=ROOT/v;p.mkdir(parents=True,exist_ok=True);e[k]=str(p)
    e['PYTHONDONTWRITEBYTECODE']='1'
    return subprocess.run(list(map(str,args)),cwd=ROOT,env=e,check=check,capture_output=capture,text=True)
def build():
    assert run('mojo','--version',capture=True).stdout.startswith('Mojo 1.0.0 ')
    meta=list(Path(sys.prefix,'conda-meta').glob('max-core-*.json'))
    assert len(meta)==1 and json.loads(meta[0].read_text())['version']=='26.5.0'
    lock=json.loads((ROOT/'DEPENDENCIES.json').read_text())
    owner=next(p for p in lock['projects'] if p['id']=='assembly')
    assembly=(ROOT/owner['path']).resolve()
    assert assembly==ASSEMBLY.resolve(),assembly
    run(sys.executable,assembly/'tools/consume_family.py','--consumer',ROOT,'--lock',ROOT/'DEPENDENCIES.json','--tool',assembly/'tools/assemble_family.py')
    run('mojo','precompile',ROOT/'builds/deps/moxie','-I',ROOT/'builds/deps','-o',ROOT/'builds/package.mojoc')
    for folder in ['tests','examples']:
        for source in (ROOT/folder).glob('*.mojo'):
            run('mojo','build','-O3','--fp-mode=contract=off','-I',ROOT/'builds/deps',source,'-o',ROOT/'builds'/source.stem)
if __name__=='__main__':
    if not sys.argv[1].endswith('-only'): build()
    sys.argv[1] = sys.argv[1].removesuffix('-only')
    if sys.argv[1]=='test':
        for p in (ROOT/'tests').glob('test_*.mojo'): run(ROOT/'builds'/p.stem)
        failed=run('mojo','build','-I',ROOT/'builds/deps',ROOT/'tests/compile_fail/sparse.mojo','-o',ROOT/'builds/sparse-invalid',check=False,capture=True)
        assert failed.returncode and 'VALID_METRICS_SPARSE' in failed.stderr
        print('PASS sparse export exclusion')
    elif sys.argv[1]=='parity': run(sys.executable,ROOT/'tests/parity.py')
    elif sys.argv[1]=='example': run(ROOT/'builds/query')
