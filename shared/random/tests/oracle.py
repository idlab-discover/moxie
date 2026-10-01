"""Execute pinned sklearn function bodies; build unedited Cython sampling source."""
from pathlib import Path
import ast, hashlib, json, numbers, shutil, subprocess, sys, types
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SKLEARN=ROOT.parents[2]/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES=['sklearn/dummy.py','sklearn/utils/validation.py','sklearn/utils/_random.pyx',
       'sklearn/utils/_random.pxd','sklearn/utils/_typedefs.pxd']
def verify():
    assert subprocess.check_output(['git','-C',str(SKLEARN),'rev-parse','HEAD'],text=True).strip()==REV
    assert not subprocess.check_output(['git','-C',str(SKLEARN),'status','--porcelain','--',*FILES],text=True)
    return {f:hashlib.sha256((SKLEARN/f).read_bytes()).hexdigest() for f in FILES}
def load():
    hashes=verify()
    path=SKLEARN/'sklearn/utils/validation.py'
    body=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='check_random_state')
    ns={'np':np,'numbers':numbers}
    exec(compile(ast.Module(body=[body],type_ignores=[]),str(path),'exec'),ns)
    check=ns['check_random_state']
    assert check.__code__.co_filename==str(path)
    # Build exact source in an isolated test-only sklearn namespace.
    target=ROOT/'builds/oracle'
    package=target/'sklearn/utils'; package.mkdir(parents=True,exist_ok=True)
    for f in FILES[2:]: shutil.copyfile(SKLEARN/f,target/f)
    (target/'sklearn/__init__.py').write_text('')
    (package/'__init__.py').write_text('')
    (package/'validation.py').write_text('import numpy as np\nimport numbers\n'+ast.unparse(body)+'\n')
    shutil.copyfile(ROOT/'tests/our_rand_r_oracle.pyx', target/'our_rand_r_oracle.pyx')
    setup="""from setuptools import setup, Extension
from Cython.Build import cythonize
setup(ext_modules=cythonize([Extension('sklearn.utils._random',['sklearn/utils/_random.pyx']), Extension('our_rand_r_oracle',['our_rand_r_oracle.pyx'])], compiler_directives={'language_level':3}))
"""
    (target/'setup.py').write_text(setup)
    subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=target,check=True,stdout=subprocess.DEVNULL)
    sys.path.insert(0,str(target))
    from sklearn.utils._random import sample_without_replacement
    assert str(target) in sys.modules['sklearn.utils._random'].__file__
    # Dummy's original predict methods run unchanged on explicit fitted attributes.
    path=SKLEARN/'sklearn/dummy.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DummyClassifier')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['predict','predict_proba']]
    ns={'np':np,'check_random_state':check,'check_is_fitted':lambda x:None,'_num_samples':len}
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(path),'exec'),ns)
    class Dummy: pass
    for name in ['predict','predict_proba']:
        assert ns[name].__code__.co_filename==str(path)
        setattr(Dummy,name,ns[name])
    (ROOT/'docs/private/random/oracle-run.json').write_text(json.dumps({'revision':REV,'numpy':np.__version__,'files':hashes,'cython':sys.modules['sklearn.utils._random'].__file__},indent=2))
    return sample_without_replacement,Dummy,check
