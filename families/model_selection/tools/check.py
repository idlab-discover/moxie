"""Assemble live Moxie HEAD and family sources with the current shared tool."""
from pathlib import Path
import json, os, subprocess, sys
ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT.parents[1] / 'shared/assembly'
CONSUMERS = [ROOT / 'tests/consumers' / name for name in ('a', 'b')]
def env(root):
    result = os.environ.copy()
    for key, directory in {'XDG_CACHE_HOME':'.pixi/cache', 'PIXI_CACHE_DIR':'.pixi/pixi-cache', 'MODULAR_CACHE_DIR':'.pixi/modular-cache', 'TMPDIR':'builds/tmp'}.items():
        path=root/directory; path.mkdir(parents=True, exist_ok=True); result[key]=str(path)
    result['PYTHONDONTWRITEBYTECODE']='1'
    return result

def run(*args, root=ROOT, check=True):
    return subprocess.run(list(map(str,args)), cwd=root, env=env(root), check=check, text=True, capture_output=not check)

def assemble(root):
    # Read every declared source live; the assembler path is the current Moxie tool.
    lock=json.loads((root/'DEPENDENCIES.json').read_text())
    tool=next(p for p in lock['projects'] if p['id']=='assembly')
    owner=(root/tool['path']).resolve()
    assert owner==ASSEMBLY.resolve(), owner
    run(sys.executable, owner/'tools/consume_family.py', '--consumer', root, '--lock', root/'DEPENDENCIES.json', '--tool', owner/'tools/assemble_family.py', root=root)

def compile_file(source, name, root=ROOT, optimization='-O0'):
    run('mojo','build',optimization,'-I',root/'builds/deps','-I',ROOT/'examples',source,'-o',root/'builds'/name,root=root)

def main():
    action=sys.argv[1]
    version=subprocess.check_output(['mojo','--version'], text=True, env=env(ROOT))
    assert version.startswith('Mojo 1.0.0 ')
    metadata=list(Path(sys.prefix, 'conda-meta').glob('max-core-*.json'))
    assert len(metadata)==1 and json.loads(metadata[0].read_text())['version']=='26.5.0'
    for root in [ROOT,*CONSUMERS]:
        assemble(root)
        run('mojo','precompile','-I',root/'builds/deps',root/'builds/deps/moxie','-o',root/'builds/package.mojoc',root=root)
    for source in sorted((ROOT/'tests').glob('*.mojo')): compile_file(source,source.stem)
    compile_file(ROOT/'algorithms/search/tests/capabilities.mojo', 'test_search_api')
    compile_file(ROOT/'algorithms/validation/tests/prediction_lifecycle.mojo', 'test_prediction_api', optimization='-O3')
    compile_file(ROOT/'algorithms/validation/tests/permutation_lifecycle.mojo', 'test_permutation_api', optimization='-O3')
    compile_file(ROOT/'algorithms/validation/tests/learning_lifecycle.mojo', 'test_learning_api', optimization='-O3')
    # Match the validation child's O3 build: NuMojo reshape triggers a
    # Mojo 1.0 masked-gather lowering error at O0.
    compile_file(ROOT/'algorithms/validation/tests/curve_lifecycle.mojo', 'test_curve_api', optimization='-O3')
    for source in sorted((ROOT/'examples').glob('*.mojo')): compile_file(source, 'example_'+source.stem)
    for root in CONSUMERS: compile_file(root/'main.mojo','probe',root)
    if action=='test':
        run(ROOT/'builds/test_search_api')
        run(ROOT/'builds/test_curve_api')
        run(ROOT/'builds/test_learning_api')
        run(ROOT/'builds/test_permutation_api')
        run(ROOT/'builds/test_prediction_api')
        for source in sorted((ROOT/'tests').glob('*.mojo')): run(ROOT/'builds'/source.stem)
        for root in CONSUMERS:
            run(root/'builds/probe',root=root)
            report=json.loads((root/'builds/deps/ASSEMBLY.json').read_text())
            assert report['modules']['moxie.model_selection._evaluation']==['model_selection']
            assert report['modules']['moxie._dense']==['numerics']
            assert report['modules']['moxie._random']==['random']
            assert not (root/'builds/deps/moxie/_sparse.mojo').exists()
        for name, diagnostic in [('missing_predict',"required function 'predict' is not implemented")]:
            proc=run('mojo','build','-I',ROOT/'builds/deps',ROOT/f'tests/compile_fail/{name}.mojo','-o',ROOT/'builds/invalid',check=False)
            assert proc.returncode and diagnostic in proc.stderr,proc.stderr
        compile_file(ROOT/'tests/compile_fail/future_algorithm.mojo', 'integrated_splitter')
        run(ROOT/'builds/integrated_splitter')
        run(sys.executable,ROOT/'tests/assembly.py')
        run(ROOT/'builds/example_plumbing')
    elif action=='start': run(ROOT/'builds/example_estimator')
    elif action!='build': raise ValueError(action)
    print('PASS', action)
if __name__=='__main__': main()
