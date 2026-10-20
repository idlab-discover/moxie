"""Compile current Moxie and family sources; keep all output local."""
from pathlib import Path
import json, os, subprocess, sys
ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT.parents[1]/'shared/assembly'
CONSUMERS = [ROOT/'tests/consumers'/n for n in ('a','b')]
AFFINITY = ROOT/'algorithms/affinity'
KMEANS = ROOT/'algorithms/kmeans'
MINIBATCH = ROOT/'algorithms/minibatch'
FLAGS = ['-O3','--fp-mode=contract=off']
def environment(c):
    env = os.environ.copy()
    for k,v in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache','MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        path=c/v; path.mkdir(parents=True,exist_ok=True); env[k]=str(path)
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['OPENBLAS_NUM_THREADS']='1'
    env['OMP_NUM_THREADS']='1'
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
        if c == ROOT and manifest is None:
            core = ROOT.parents[1]
            lock = json.loads((core/'DEPENDENCIES.json').read_text())
            for project in lock['projects']:
                project['path'] = str((core/project['path']).resolve())
            resolved = c/'builds/resolved.json'
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(json.dumps(lock, indent=2)+'\n')
            compose_sources(c, resolved, c/'builds/deps')
        else:
            compose_sources(c, manifest or c/'DEPENDENCIES.json', c/'builds/deps')
    except (AssemblyError, OSError, ValueError, TypeError) as exc:
        if check:
            raise
        return subprocess.CompletedProcess([], 1, '', str(exc))
    return subprocess.CompletedProcess([], 0, '', '')

def check_integration():
    core = ROOT.parents[1]
    lock = json.loads((core/'DEPENDENCIES.json').read_text())
    report = json.loads((ROOT/'builds/deps/ASSEMBLY.json').read_text())
    exports = {}
    for project in lock['projects']:
        owner = (core/project['path']).resolve()
        assert 'moxie-cluster' not in owner.parts, owner
        if not owner.is_relative_to(ROOT):
            continue
        for module, spec in project['modules'].items():
            assert report['modules'][module] == [project['id']], module
            target = ROOT/'builds/deps'/Path(*module.split('.')).with_suffix('.mojo')
            assert target.read_bytes() == (owner/spec['file']).read_bytes(), module
        for destination, source in project['assets'].items():
            assert (ROOT/'builds/deps'/destination).read_bytes() == (owner/source).read_bytes()
        exports.update(project['exports']['packages'].get('moxie.cluster', {}))
    assert exports == json.loads((ROOT/'EXPORTS.json').read_text())['packages']['moxie.cluster']
    for name in ('HDBSCAN', 'SpectralClustering', 'SpectralBiclustering', 'SpectralCoclustering'):
        assert name not in exports
    print('PASS: core cluster owners, source bytes, licenses and supported exports', flush=True)

def compile_one(c,source,name):
    run('mojo','build',*FLAGS,'-I',c/'builds/deps','-I',ROOT/'examples',source,'-o',c/'builds'/name,c=c)

def build(c=ROOT):
    assert run('mojo','--version',capture=True).stdout.startswith('Mojo 1.0.0 ')
    metadata=list(Path(sys.prefix,'conda-meta').glob('max-core-*.json'))
    assert len(metadata)==1 and json.loads(metadata[0].read_text())['version']=='26.5.0'
    assemble(c)
    if c == ROOT: check_integration()
    run('mojo','precompile',c/'builds/deps/moxie','-I',c/'builds/deps','-o',c/'builds/moxie.mojoc',c=c)
    if c==ROOT:
        for path in sorted((ROOT/'tests').glob('test_*.mojo')): compile_one(c,path,path.stem)
        compile_one(c,AFFINITY/'tests/test_affinity.mojo','test_affinity')
        compile_one(c,KMEANS/'tests/test_kmeans.mojo','test_kmeans')
        compile_one(c,ROOT/'algorithms/birch/tests/test_birch.mojo','test_birch')
        compile_one(c,ROOT/'algorithms/birch/tests/driver.mojo','birch_driver')
        compile_one(c,ROOT/'algorithms/bisecting/tests/test_bisecting.mojo','test_bisecting')
        compile_one(c,ROOT/'algorithms/bisecting/tests/driver.mojo','bisecting_driver')
        compile_one(c,ROOT/'algorithms/mean_shift/tests/test_mean_shift.mojo','test_mean_shift')
        compile_one(c,ROOT/'algorithms/mean_shift/tests/driver.mojo','mean_shift_driver')
        compile_one(c,ROOT/'algorithms/agglomerative/tests/test_agglomerative.mojo','test_agglomerative')
        compile_one(c,ROOT/'algorithms/agglomerative/tests/driver.mojo','agglomerative_driver')
        compile_one(c,ROOT/'algorithms/optics/tests/test_optics.mojo','test_optics')
        compile_one(c,ROOT/'algorithms/optics/tests/driver.mojo','optics_driver')
        compile_one(c,ROOT/'algorithms/dbscan/tests/test_dbscan.mojo','test_dbscan')
        compile_one(c,ROOT/'algorithms/dbscan/tests/driver.mojo','dbscan_driver')
        compile_one(c,MINIBATCH/'tests/test_minibatch.mojo','test_minibatch')
        compile_one(c,MINIBATCH/'tests/driver.mojo','minibatch_driver')
        compile_one(c,KMEANS/'tests/driver.mojo','kmeans_driver')
        compile_one(c,AFFINITY/'tests/driver.mojo','affinity_driver')
        compile_one(c,AFFINITY/'examples/affinity.mojo','example_affinity')
        for path in sorted((ROOT/'examples').glob('*.mojo')): compile_one(c,path,'example_'+path.stem)
        for child in CONSUMERS: build(child)
    else: compile_one(c,c/'main.mojo','probe')

def test(c=ROOT):
    if c != ROOT:
        run(c/'builds/probe',c=c)
        return
    for path in sorted((ROOT/'tests').glob('test_*.mojo')): run(ROOT/'builds'/path.stem)
    run(ROOT/'builds/test_affinity')
    run(ROOT/'builds/test_kmeans')
    run(ROOT/'builds/test_minibatch')
    run(ROOT/'builds/test_birch')
    run(ROOT/'builds/test_bisecting')
    run(ROOT/'builds/test_agglomerative')
    run(ROOT/'builds/test_optics')
    run(ROOT/'builds/test_dbscan')
    run(ROOT/'builds/test_mean_shift')
    for child in CONSUMERS:
        test(child)
        report=json.loads((child/'builds/deps/ASSEMBLY.json').read_text())
        for module,owner in [('moxie._dense','numerics'),('moxie._random','random'),('moxie.cluster._contracts','cluster'),('moxie.cluster._affinity_propagation','affinity'),('moxie.cluster._centroid_engine','centroid_engine'),('moxie.cluster._kmeans','kmeans'),('moxie.cluster._mini_batch_kmeans','minibatch'),('moxie.cluster._birch','birch'),('moxie.cluster._bisect_k_means','bisecting'),('moxie.cluster._agglomerative','agglomerative'),('moxie.cluster._optics','optics'),('moxie.cluster._dbscan','dbscan'),('moxie.cluster._mean_shift','mean_shift'),('moxie.neighbors._unsupervised','neighbors.nearest')]:
            assert report['modules'][module]==[owner]
        assert 'moxie.consumer_a' in report['modules'] and 'moxie.consumer_b' in report['modules']
    for source,expected in [('missing_predict.mojo',"required function 'predict' is not implemented"),('missing_algorithm.mojo','HDBSCAN'),('dbscan_predict.mojo',"'DBSCAN' value has no attribute 'predict'"),('agglomerative_predict.mojo',"'AgglomerativeClustering' value has no attribute 'predict'"),('feature_fit_predict.mojo',"'FeatureAgglomeration' value has no attribute 'fit_predict'"),('optics_predict.mojo',"'OPTICS' value has no attribute 'predict'")]:
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
    run(sys.executable, ROOT/'tests/test_live_sources.py')
    print('PASS: isolated composition, missing sources and ownership/export rejection')

if __name__=='__main__':
    action=sys.argv[1]
    consumer=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT
    assert consumer in [ROOT,*CONSUMERS]
    if not action.endswith('-only'): build(consumer)
    action=action.removesuffix('-only')
    if action=='test': test(consumer)
    elif action=='start': run(ROOT/'builds/example_affinity')
    elif action=='parity':
        assert consumer == ROOT
        run(sys.executable,AFFINITY/'tests/parity.py','--driver',ROOT/'builds/affinity_driver','--output-dir',ROOT/'builds')
        run(sys.executable,ROOT/'tests/feature_names_parity.py')
        run(sys.executable,KMEANS/'tests/parity.py')
        run(sys.executable,MINIBATCH/'tests/parity.py')
        run(sys.executable,ROOT/'algorithms/birch/tests/parity.py')
        run(sys.executable,ROOT/'algorithms/bisecting/tests/parity.py')
        run(sys.executable,ROOT/'algorithms/agglomerative/tests/parity.py')
        run(sys.executable,ROOT/'algorithms/optics/tests/parity.py')
        run(sys.executable,ROOT/'algorithms/dbscan/tests/parity.py')
        run(sys.executable,ROOT/'algorithms/mean_shift/tests/parity.py')
    elif action!='build': raise ValueError(action)
