"""Family checks through Moxie's shared assembler over current local sources."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
MOXIE=ROOT.parents[1]
ASSEMBLY=MOXIE/'shared/assembly'
ALGORITHMS = ('empirical', 'shrinkage', 'robust', 'graphical')
CONSUMERS=[ROOT/'tests/consumers'/n for n in ('a','b')]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def environment(c):
    env=os.environ.copy()
    for key,folder in {'XDG_CACHE_HOME':'.pixi/cache','PIXI_CACHE_DIR':'.pixi/pixi-cache','MODULAR_CACHE_DIR':'.pixi/modular-cache','TMPDIR':'builds/tmp'}.items():
        p=c/folder;p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
    env['PYTHONDONTWRITEBYTECODE']='1'
    return env

def run(*args,c=ROOT,capture=False,check=True):
    return subprocess.run([str(a) for a in args],cwd=c,env=environment(c),check=check,capture_output=capture,text=True)

def assemble(c,lock=None):
    lock=lock or (MOXIE/'DEPENDENCIES.json' if c==ROOT else c/'DEPENDENCIES.json')
    data=json.loads(lock.read_text())
    return run(sys.executable,ASSEMBLY/'tools/consume_family.py','--consumer',c,'--lock',lock,'--tool',ASSEMBLY/'tools/assemble_family.py',c=c)

def compile_file(c,source,name):
    run('mojo','build','-O3','--fp-mode=contract=off','-I',c/'builds/deps','-I',ROOT/'examples','-I',ROOT/'tests/support',source,'-o',c/'builds'/name,c=c)

def build():
    assert run('mojo','--version',capture=True).stdout.startswith('Mojo 1.0.0 ')
    metadata=list(Path(sys.prefix,'conda-meta').glob('max-core-*.json'))
    assert len(metadata)==1 and json.loads(metadata[0].read_text())['version']=='26.5.0'
    for c in [ROOT]+CONSUMERS:
        assemble(c)
        run('mojo','precompile',c/'builds/deps/moxie','-I',c/'builds/deps','-o',c/'builds/moxie.mojoc',c=c)
        if c==ROOT:
            for name in ['test_contracts','test_capabilities','test_family','test_plumbing','test_engines']:
                compile_file(c,ROOT/'tests'/f'{name}.mojo',name)
            compile_file(c,ROOT/'examples/estimator.mojo','example')
            compile_file(c,ROOT/'examples/ownership.mojo','ownership')
            compile_file(c,ROOT/'tests/test_public.mojo','test_public')
            compile_file(c,ROOT/'tests/test_params.mojo','test_params')
            compile_file(c,ROOT/'tests/test_fluent.mojo','test_fluent')
            compile_file(c,ROOT/'tests/audit_driver.mojo','audit_driver')
            compile_file(c,ROOT/'tests/shrinkage_boundary_driver.mojo','shrinkage_boundary_driver')
            compile_file(c,ROOT/'tests/singular_driver.mojo','singular_driver')
            compile_file(c,ROOT/'algorithms/graphical/tests/parity_driver.mojo','audit_graphical_driver')
            compile_file(c,ROOT/'algorithms/graphical/tests/cv_parity_driver.mojo','audit_cv_driver')
            for algorithm in ALGORITHMS:
                for source in sorted((ROOT/'algorithms'/algorithm/'tests').glob('*contracts.mojo')):
                    compile_file(c,source,f'{algorithm}_{source.stem}')
        else: compile_file(c,c/'main.mojo','probe')

def test():
    reference = MOXIE.parent/'scikit-learn/sklearn/covariance/__init__.py'
    tree = ast.parse(reference.read_text())
    public = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets))
    exports = json.loads((ROOT/'EXPORTS.json').read_text())['packages']['moxie.covariance']
    assert set(exports) == set(public) | {'CovarianceEstimator', 'RobustCovarianceDetector'}
    assert all(exports[name] != 'moxie.covariance._api' for name in public)
    manifest = json.loads((MOXIE/'DEPENDENCIES.json').read_text())
    owner = next(p for p in manifest['projects'] if p['id'] == 'moxie')
    assert owner['exports']['packages']['moxie.covariance'] == exports
    receipt = json.loads((ROOT/'builds/deps/ASSEMBLY.json').read_text())
    for module, owners in receipt['modules'].items():
        if module.startswith('moxie.covariance.'):
            assert owners == ['moxie'], (module, owners)
            assert owner['modules'][module]['file'].startswith('src/moxie/covariance/')
    assert all((MOXIE/p['path']).resolve() != MOXIE.parent/'moxie-covariance'
               for p in manifest['projects'])
    run(ROOT/'builds/test_public')
    run(ROOT/'builds/test_fluent')
    run(sys.executable, ROOT/'tests/test_diagnostics.py')
    run(sys.executable, ROOT/'tests/test_likelihood.py')
    run(sys.executable, ROOT/'tests/singular_boundaries.py')
    result = run(ROOT/'builds/test_params', capture=True)
    (ROOT/'builds/params-contracts.txt').write_text(result.stdout)
    print('PASS: keyword parameter contracts (Float64/Float32)')
    for algorithm in ALGORITHMS:
        for source in sorted((ROOT/'algorithms'/algorithm/'tests').glob('*contracts.mojo')):
            run(ROOT/'builds'/f'{algorithm}_{source.stem}')
    for name in ['test_contracts','test_capabilities','test_family','test_plumbing','test_engines','ownership']:
        run(ROOT/'builds'/name)
    for c in CONSUMERS:
        run(c/'builds/probe',c=c)
        report=json.loads((c/'builds/deps/ASSEMBLY.json').read_text())
        assert report['modules']['moxie.covariance._state']==['moxie']
        assert report['modules']['moxie._dense']==['numerics']
        assert sha(c/'builds/deps/moxie/covariance/_state.mojo')==sha(MOXIE/'src/moxie/covariance/_state.mojo')
        assert not (c/'builds/deps/moxie/_sparse.mojo').exists()
    for source,expected in [('tests/compile_fail/missing_predict.mojo',"required function 'predict'")]:
        result=run('mojo','build','-I',ROOT/'builds/deps',ROOT/source,'-o',ROOT/'builds/invalid',capture=True,check=False)
        assert result.returncode and expected in result.stderr,result.stderr
    # Corruption, ownership and export collision checks use only local bad locks.
    c=CONSUMERS[0]
    for mode in ['missing','module','export']:
        data=json.loads(((MOXIE/'DEPENDENCIES.json' if c==ROOT else c/'DEPENDENCIES.json')).read_text())
        if mode=='missing':
            next(p for p in data['projects'] if p['id']=='probe_a')['modules']['moxie.probe_a']={'file':'src/moxie/absent.mojo'}
        else:
            a=next(p for p in data['projects'] if p['id']=='probe_a')
            b=next(p for p in data['projects'] if p['id']=='probe_b')
            if mode=='module':
                b['modules']['moxie.probe_a']={'file':'src/moxie/probe_b.mojo'}
            else: b['exports']['packages']['moxie']['ProbeA']='moxie.probe_b'
        bad=c/'builds/bad.json'
        # Lock-relative paths must still resolve from builds/.
        for p in data['projects']: p['path']=str((c/p['path']).resolve())
        bad.write_text(json.dumps(data))
        before=sha(c/'builds/deps/ASSEMBLY.json')
        result=run(sys.executable,ASSEMBLY/'tools/consume_family.py','--consumer',c,'--lock',bad,'--tool',ASSEMBLY/'tools/assemble_family.py',c=c,capture=True,check=False)
        assert result.returncode and {'missing':'missing file','module':'module collision','export':'export collision'}[mode] in result.stderr.lower(),result.stderr
        assert sha(c/'builds/deps/ASSEMBLY.json')==before
    print('PASS: isolated assembly, missing sources, conflicting ownership/exports, unsupported capability checks')

if __name__=='__main__':
    if not sys.argv[1].endswith('-only'): build()
    if sys.argv[1] in ('test', 'test-only'): test()
    elif sys.argv[1]=='start': run(ROOT/'builds/ownership')
    elif sys.argv[1] in ('parity', 'parity-only'):
        for algorithm in ALGORITHMS:
            run(sys.executable, ROOT/'algorithms'/algorithm/'tools/check.py', 'parity')
        for name in ('params_parity', 'gaps_parity'):
            run(sys.executable, ROOT/'tests'/f'{name}.py')
    elif sys.argv[1]!='build': raise ValueError(sys.argv[1])
