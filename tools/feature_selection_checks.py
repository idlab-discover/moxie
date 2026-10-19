"""Family checks against live local Moxie HEAD and sibling sources via the shared assembler."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
from feature_selection_assembly import assemble as assemble_live
from compiler_limits import compiler_command
TESTS=ROOT/'tests/feature_selection'
EXAMPLES=ROOT/'examples/feature_selection'
UNIVARIATE=TESTS/'univariate'
CONSUMERS=[TESTS/'consumers'/n for n in ['a','b']]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(*args, cwd=ROOT, check=True, capture=False):
    env=os.environ.copy()
    for key,folder in [('XDG_CACHE_HOME','cache'),('PIXI_CACHE_DIR','pixi-cache'),('MODULAR_CACHE_DIR','modular-cache'),('TMPDIR','tmp')]:
        path=cwd/'.pixi'/folder; path.mkdir(parents=True,exist_ok=True); env[key]=str(path)
    env['PYTHONDONTWRITEBYTECODE']='1'
    return subprocess.run(compiler_command(args),cwd=cwd,env=env,check=check,capture_output=capture,text=True)
def assemble(c):
    assemble_live(c, run)
def compile_file(c,source,name):
    run('mojo','build','-O3','-I',c/'builds/deps','-I',EXAMPLES,'-I',ROOT/'examples',source,'-o',c/'builds'/name,cwd=c)
def main(action=None):
    action=action or sys.argv[1]
    assert action in ['build','test','example','parity','test-only','parity-only']
    assert run('mojo','--version',capture=True).stdout.startswith('Mojo 1.0.0 ')
    packages=list(Path(sys.prefix,'conda-meta').glob('max-core-*.json'))
    assert len(packages)==1 and json.loads(packages[0].read_text())['version']=='26.5.0'
    if not action.endswith('-only'):
        for c in [ROOT]+CONSUMERS:
            assemble(c)
            run('mojo','precompile','-I',c/'builds/deps',c/'builds/deps/moxie','-o',c/'builds/package.mojoc',cwd=c)
        compile_file(ROOT, ROOT/'examples/estimator.mojo','example')
        compile_file(ROOT, ROOT/'examples/feature_selection.mojo','selection-example')
        compile_file(ROOT, EXAMPLES/'from_model.mojo','from-model-example')
        compile_file(ROOT, EXAMPLES/'from_model_incremental.mojo','from-model-incremental-example')
        compile_file(ROOT, EXAMPLES/'rfe.mojo','rfe-example')
        compile_file(ROOT, EXAMPLES/'rfe_path.mojo','rfe-path-example')
        compile_file(ROOT, EXAMPLES/'mutual_info.mojo','mutual-info-example')
        compile_file(ROOT, EXAMPLES/'sequential.mojo','sequential-example')
        compile_file(ROOT, EXAMPLES/'rfecv.mojo','rfecv-example')
        for name in ['contracts','capabilities','family','selection','extensions','variance','univariate','from_model','from_model_incremental','rfe','rfe_path','mutual_info','sequential','rfecv','classifier_int','classifier_string','fluent','pipeline']:
            source = ROOT/f'tests/test_{name}.mojo' if name in ('contracts', 'capabilities') else TESTS/f'test_{name}.mojo'
            compile_file(ROOT, source, 'feature-selection-' + name)
        for source in sorted(UNIVARIATE.glob('test_*.mojo')):
            compile_file(ROOT, source, 'univariate-' + source.stem)
        compile_file(ROOT, EXAMPLES/'univariate.mojo', 'univariate-example')
        for c in CONSUMERS: compile_file(c,c/'main.mojo','probe')
    if action=='example': run(ROOT/'builds/selection-example')
    if action in ('test', 'test-only'):
        run(sys.executable, TESTS/'test_compiler_limits.py')
        run(sys.executable, TESTS/'test_assembly.py')
        run(sys.executable, TESTS/'test_ownership.py')
        for name in ['contracts','capabilities','family','selection','extensions','variance','univariate','from_model','from_model_incremental','rfe','rfe_path','mutual_info','sequential','rfecv','classifier_int','classifier_string','fluent','pipeline']: run(ROOT/'builds'/('feature-selection-' + name))
        for source in sorted(UNIVARIATE.glob('test_*.mojo')):
            run(ROOT/'builds'/('univariate-' + source.stem))
        run(ROOT/'builds/univariate-example')
        run(ROOT/'builds/from-model-example')
        run(ROOT/'builds/from-model-incremental-example')
        run(ROOT/'builds/rfe-example')
        run(ROOT/'builds/rfe-path-example')
        run(ROOT/'builds/mutual-info-example')
        run(ROOT/'builds/sequential-example')
        run(ROOT/'builds/rfecv-example')
        for c in CONSUMERS:
            run(c/'builds/probe',cwd=c)
            assert sha(c/'builds/deps/moxie/feature_selection/_selection.mojo')==sha(ROOT/'src/moxie/feature_selection/_selection.mojo')
            report=json.loads((c/'builds/deps/ASSEMBLY.json').read_text())
            assert report['modules']['moxie.feature_selection._selection']==['moxie']
            assert report['modules']['moxie._dense']==['numerics']
            assert report['modules']['moxie.tree._classes']==['moxie']
        for source in ['tests/feature_selection/compile_fail/missing_predict.mojo','tests/feature_selection/compile_fail/sparse.mojo','tests/feature_selection/univariate/compile_fail/mixed_score_dtype.mojo','tests/feature_selection/compile_fail/rfe_probability.mojo','tests/feature_selection/compile_fail/rfecv_probability.mojo','tests/feature_selection/compile_fail/sequential_predict.mojo']:
            result=run('mojo','build','-I',ROOT/'builds/deps',ROOT/source,'-o',ROOT/'builds/invalid',check=False,capture=True)
            expected="required function 'predict' is not implemented" if 'missing_predict' in source else ('score' if 'mixed_score_dtype' in source else ('predict_proba' if 'rfe_probability' in source or 'rfecv_probability' in source else ('predict' if 'sequential_predict' in source else 'SparseSelectionState')))
            assert result.returncode != 0 and expected in result.stderr, result.stderr
        mismatch=run('mojo','build','-I',ROOT/'builds/deps','-I',EXAMPLES,TESTS/'compile_fail/rfe_observer.mojo','-o',ROOT/'builds/invalid-observer',check=False,capture=True)
        assert mismatch.returncode != 0 and 'RFE observer estimator type must match' in mismatch.stderr, mismatch.stderr
        incremental=run('mojo','build','-I',ROOT/'builds/deps',TESTS/'compile_fail/from_model_partial_fit.mojo','-o',ROOT/'builds/invalid-incremental',check=False,capture=True)
        assert incremental.returncode != 0 and 'partial_fit' in incremental.stderr, incremental.stderr
        # Missing live sources and overlapping contributions must fail without replacing output.
        c=CONSUMERS[0]
        original=json.loads((c/'DEPENDENCIES.json').read_text())
        report_hash=sha(c/'builds/deps/ASSEMBLY.json')
        for mode in ['missing','ownership','exports']:
            lock=json.loads(json.dumps(original))
            family=next(p for p in lock['projects'] if p['id']=='moxie')
            contribution=next(p for p in lock['projects'] if p['id']=='consumer_a')
            if mode=='missing':
                next(project for project in lock['projects'] if project['id'] == 'moxie')['modules']['moxie.base']['file']='src/moxie/absent.mojo'
            elif mode=='ownership':
                contribution['modules']['moxie.feature_selection._selection']={'file':'src/moxie/consumer_a.mojo'}
            else:
                contribution['exports']['packages']['moxie.feature_selection']={'SelectionState':'moxie.consumer_a'}
            bad=c/'bad-lock.json'
            try:
                bad.write_text(json.dumps(lock))
                result=assemble_live(c, run, lock_path=bad, check=False, capture=True)
                expected={'missing':'missing file','ownership':'module collision','exports':'export'}[mode]
                assert result.returncode != 0 and expected in result.stderr.lower(), result.stderr
                assert sha(c/'builds/deps/ASSEMBLY.json')==report_hash
            finally:
                bad.unlink(missing_ok=True)
    if action in ('parity', 'parity-only'):
        run(sys.executable, TESTS/'parity_variance.py')
        run(sys.executable, UNIVARIATE/'parity.py', '--assembly-root', ROOT)
        run(sys.executable, TESTS/'parity_from_model.py')
        run(sys.executable, TESTS/'parity_from_model_incremental.py')
        run(sys.executable, TESTS/'parity_rfe.py')
        run(sys.executable, TESTS/'parity_rfe_path.py')
        run(sys.executable, TESTS/'parity_mutual_info.py')
        run(sys.executable, TESTS/'parity_labels.py')
        run(sys.executable, TESTS/'parity_sequential.py')
        run(sys.executable, TESTS/'parity_rfecv.py')
        run(sys.executable, TESTS/'parity_classifier_selection.py')
    print('PASS', action)
if __name__=='__main__': main()
