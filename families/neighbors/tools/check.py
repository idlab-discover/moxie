"""Assemble live Moxie sources with family-local validation outputs."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT.parents[1]/'shared/assembly'
CONSUMERS = [ROOT/'tests/consumers'/n for n in ['a','b']]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def env(c):
    e=os.environ.copy()
    for k,v in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache','MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        p=c/v; p.mkdir(parents=True,exist_ok=True); e[k]=str(p)
    e['PYTHONDONTWRITEBYTECODE']='1'
    e['OMP_NUM_THREADS']='1'
    e['OPENBLAS_NUM_THREADS']='1'
    return e

def run(*args,c=ROOT,capture=False,check=True,oracle=False):
    e = env(c)
    if oracle and not e.get('PYTHONPATH') and not e.get('SKLEARN_ORACLE_PYTHON'):
        e['PYTHONPATH']=str(ROOT.parents[1]/'builds/neighbors/oracle-deps')
    return subprocess.run(list(map(str,args)),cwd=c,env=e,check=check,capture_output=capture,text=True)

def assemble(c):
    lock=json.loads((c/'DEPENDENCIES.json').read_text())
    owner=next(p for p in lock['projects'] if p['id']=='assembly')
    assembly=(c/owner['path']).resolve()
    assert assembly==ASSEMBLY.resolve(),assembly
    run(sys.executable,assembly/'tools/consume_family.py','--consumer',c,'--lock',c/'DEPENDENCIES.json','--tool',assembly/'tools/assemble_family.py',c=c)

def compile_file(c,source,name):
    run('mojo','build','-O3','--fp-mode=contract=off','-I',c/'builds/deps','-I',ROOT/'examples',source,'-o',c/'builds'/name,c=c)

def build_consumer(c):
    assemble(c)
    run('mojo','precompile',c/'builds/deps/moxie','-I',c/'builds/deps','-o',c/'builds/package.mojoc',c=c)
    compile_file(c,c/'main.mojo','probe')
    compile_file(c,ROOT/'tests/consumers/centroid.mojo','centroid_probe')
    compile_file(c,ROOT/'tests/consumers/nca.mojo','nca_probe')
    compile_file(c,ROOT/'tests/test_kde_sampling.mojo','kde_sampling_probe')
    compile_file(c,ROOT/'tests/test_callable_metrics.mojo','callable_metrics_probe')
    compile_file(c,ROOT/'tests/consumers/kdtree.mojo','kdtree_probe')
    compile_file(c,ROOT/'tests/consumers/balltree.mojo','balltree_probe')

def build():
    assert run('mojo','--version',capture=True).stdout.startswith('Mojo 1.0.0 ')
    meta=list(Path(sys.prefix,'conda-meta').glob('max-core-*.json'))
    assert len(meta)==1 and json.loads(meta[0].read_text())['version']=='26.5.0'
    assemble(ROOT)
    run('mojo','precompile',ROOT/'builds/deps/moxie','-I',ROOT/'builds/deps','-o',ROOT/'builds/package.mojoc')
    for name in ['test_contracts','test_capabilities','test_family','test_plumbing','test_estimators','test_callable_metrics','test_general_labels','test_integration','test_lof','test_kde','test_kde_sampling','test_centroid','test_nca','test_nca_solver','test_kdtree','test_balltree','evaluate_kdtree','evaluate_balltree','evaluate_estimators','evaluate_callable_metrics','evaluate_nca','evaluate_nca_solver']:
        compile_file(ROOT,ROOT/'tests'/f'{name}.mojo',name)
    compile_file(ROOT,ROOT/'examples/estimator.mojo','example')
    compile_file(ROOT,ROOT/'examples/query_buffers.mojo','query_buffers')
    compile_file(ROOT,ROOT/'examples/neighbors.mojo','neighbors_example')
    compile_file(ROOT,ROOT/'examples/centroid.mojo','centroid_example')
    compile_file(ROOT,ROOT/'examples/nca.mojo','nca_example')
    for c in CONSUMERS:
        build_consumer(c)

def test():
    for name in ['test_contracts','test_capabilities','test_family','test_plumbing','test_estimators','test_callable_metrics','test_general_labels','test_integration','test_lof','test_kde','test_kde_sampling','test_centroid','test_nca','test_nca_solver','test_kdtree','test_balltree','centroid_example','nca_example','query_buffers']:
        run(ROOT/'builds'/name)
    for file,diagnostic in [('missing_predict.mojo',"required function 'predict' is not implemented"),('sparse_neighbors.mojo','VALID_METRICS_SPARSE'),('callable_dtype.mojo','metric dtype must match estimator dtype')]:
        result=run('mojo','build','-I',ROOT/'builds/deps','-I',ROOT/'examples',ROOT/'tests/compile_fail'/file,'-o',ROOT/'builds/invalid',capture=True,check=False)
        assert result.returncode and diagnostic in result.stderr,result.stderr
    for c in CONSUMERS:
        run(c/'builds/probe',c=c)
        run(c/'builds/centroid_probe',c=c)
        run(c/'builds/nca_probe',c=c)
        run(c/'builds/kde_sampling_probe',c=c)
        run(c/'builds/callable_metrics_probe',c=c)
        run(c/'builds/kdtree_probe',c=c)
        run(c/'builds/balltree_probe',c=c)
        report=json.loads((c/'builds/deps/ASSEMBLY.json').read_text())
        for module,owner in [('moxie.neighbors._contracts','neighbors'),('moxie._dense','numerics'),('moxie._random','random')]:
            assert report['modules'][module]==[owner]
        assert not (c/'builds/deps/moxie/_sparse.mojo').exists()
    for module in ['neighbors/_contracts','_dense','_random']:
        paths=[c/'builds/deps/moxie'/f'{module}.mojo' for c in CONSUMERS]
        assert sha(paths[0])==sha(paths[1]) and paths[0].stat().st_ino!=paths[1].stat().st_ino
    # Fail closed on missing live sources and ownership/export conflicts, preserving good output.
    c=CONSUMERS[0]
    for mode in ['missing','module','export']:
        lock=json.loads((c/'DEPENDENCIES.json').read_text())
        owner=next(p for p in lock['projects'] if p['id']=='consumer_b')
        if mode=='missing': next(project for project in lock['projects'] if project['id'] == 'moxie')['modules']['moxie.base']['file']='src/moxie/missing.mojo'
        elif mode=='module': owner['modules']['moxie.consumer_a']=owner['modules']['moxie.consumer_b']
        else: owner['exports']['packages']['moxie']['ConsumerA']='moxie.consumer_b'
        bad=c/'bad-lock.json'; before=sha(c/'builds/deps/ASSEMBLY.json')
        try:
            bad.write_text(json.dumps(lock))
            assembly=(c/next(p for p in lock['projects'] if p['id']=='assembly')['path']).resolve()
            p=run(sys.executable,assembly/'tools/consume_family.py','--consumer',c,'--lock',bad,'--tool',assembly/'tools/assemble_family.py',c=c,capture=True,check=False)
            assert p.returncode and mode in p.stderr.lower(),p.stderr
            assert before==sha(c/'builds/deps/ASSEMBLY.json')
        finally: bad.unlink(missing_ok=True)
    print('PASS: dense contracts, isolated composed consumers, sparse exclusion, missing sources and conflicts')

if __name__=='__main__':
    if not sys.argv[1].endswith('-only'): build()
    sys.argv[1] = sys.argv[1].removesuffix('-only')
    if sys.argv[1]=='test': test()
    elif sys.argv[1]=='parity':
        for script in ['parity_callable_metrics.py','parity_general_labels.py','parity_estimators.py','parity_lof.py','parity_kde.py','parity_kde_sampling.py','parity_centroid.py','parity_nca_solver.py','parity_nca.py','parity_kdtree.py','parity_balltree.py']:
            run(os.environ.get('SKLEARN_ORACLE_PYTHON', ROOT.parents[1]/'builds/neighbors/oracle-env/bin/python'), ROOT/'tests'/script, oracle=True)
    elif sys.argv[1]=='example': run(ROOT/'builds/neighbors_example')
    elif sys.argv[1]!='build': raise ValueError(sys.argv[1])
