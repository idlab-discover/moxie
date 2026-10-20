"""Image gates assembling live Moxie HEAD with the current shared assembler."""
from pathlib import Path
import json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]/'tools'))
from dependencies import verify_external_dependencies
sys.path.insert(0, str(ROOT.parents[3]/'tools'))
from compiler_limits import compiler_command
FLAGS=['-O3','--fp-mode=contract=off']
def env():
    e=os.environ.copy()
    for k,v in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache','MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        p=ROOT/v;p.mkdir(parents=True,exist_ok=True);e[k]=str(p)
    e.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return e
def run(*args):
    return subprocess.run(compiler_command(args),cwd=ROOT,env=env(),check=True)
def assemble(consumer=ROOT):
    if consumer == ROOT:
        core = ROOT.parents[3]
        return run(sys.executable,core/'shared/assembly/tools/assemble_family.py','--consumer',ROOT,'--lock',core/'DEPENDENCIES.json','--output',ROOT/'builds/deps')
    verify_external_dependencies(consumer/'DEPENDENCIES.json')
    d=json.loads((consumer/'DEPENDENCIES.json').read_text())
    p=next(p for p in d['projects'] if p['id']=='assembly')
    a=(consumer/p['path']).resolve()
    run(sys.executable,a/'tools/consume_family.py','--consumer',consumer,'--lock',consumer/'DEPENDENCIES.json','--tool',a/'tools/assemble_family.py')
def build():
    version=subprocess.check_output(['mojo','--version'],env=env(),text=True)
    assert version.startswith('Mojo 1.0.0 '),version
    metadata=list(Path(sys.prefix,'conda-meta').glob('max-core-*.json'))
    assert len(metadata)==1 and json.loads(metadata[0].read_text())['version']=='26.5.0'
    assemble()
    run('mojo','precompile',ROOT/'builds/deps/moxie','-I',ROOT/'builds/deps','-o',ROOT/'builds/moxie.mojoc')
    for name in ['a','b']:
        c=ROOT/'tests/consumers'/name
        assemble(c)
        run('mojo','build',*FLAGS,'-I',c/'builds/deps',c/'main.mojo','-o',c/'builds/probe')
    for s in [*ROOT.glob('tests/test_*.mojo'),*ROOT.glob('examples/*.mojo')]:
        run('mojo','build',*FLAGS,'-I',ROOT/'builds/deps',s,'-o',ROOT/'builds'/s.stem)
if __name__=='__main__':
    action=sys.argv[1]
    if action not in ('build','test','parity'):raise ValueError(action)
    build()
    if action=='test':
        for name in ['a','b']:run(ROOT/'tests/consumers'/name/'builds/probe')
        for s in ROOT.glob('tests/test_*.mojo'):run(ROOT/'builds'/s.stem)
        for module in ['moxie/_dense.mojo','moxie/_random.mojo','moxie/feature_extraction/_image_contracts.mojo']:
            paths=[ROOT/'tests/consumers'/name/'builds/deps'/module for name in ['a','b']]
            assert paths[0].read_bytes()==paths[1].read_bytes()==(ROOT/'builds/deps'/module).read_bytes()
            assert paths[0].stat().st_ino!=paths[1].stat().st_ino
        print('PASS isolated shared module identity')
    if action=='parity':run(sys.executable,ROOT/'tests/parity.py')
