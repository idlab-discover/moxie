"""Build exact pinned Cython kernels in a test-only local sklearn namespace."""
from pathlib import Path
import ast, hashlib, importlib, json, shutil, subprocess, sys
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parents[4]/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
MODULES=['cluster/_k_means_common','cluster/_k_means_lloyd','cluster/_k_means_elkan','cluster/_k_means_minibatch','utils/_cython_blas']
FILES=['sklearn/'+m+'.pyx' for m in MODULES]+['sklearn/cluster/_k_means_common.pxd','sklearn/utils/_cython_blas.pxd','sklearn/utils/_openmp_helpers.pxd','sklearn/utils/_blas_int.pxi.in','sklearn/utils/extmath.py']
def load():
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==REV
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',*FILES],text=True)
    target=ROOT/'builds/oracle'
    for package in ('sklearn','sklearn/cluster','sklearn/utils'):
        directory=target/package; directory.mkdir(parents=True,exist_ok=True)
        (directory/'__init__.py').write_text('')
    hashes={}
    for f in FILES:
        hashes[f]=hashlib.sha256((SOURCE/f).read_bytes()).hexdigest()
        if not (target/f).exists() or (target/f).read_bytes() != (SOURCE/f).read_bytes():
            shutil.copyfile(SOURCE/f,target/f)
    # Configure the upstream template for the installed SciPy BLAS ABI.
    import scipy
    scipy_pxd=Path(scipy.__file__).parent/'linalg/cython_blas.pxd'
    definition='from scipy.linalg.cython_blas cimport blas_int' if 'ctypedef' in scipy_pxd.read_text() and 'blas_int' in scipy_pxd.read_text() else 'ctypedef int blas_int'
    configured = (SOURCE/'sklearn/utils/_blas_int.pxi.in').read_text().replace('@BLAS_INT_DEF@',definition)
    generated = target/'sklearn/utils/_blas_int.pxi'
    if not generated.exists() or generated.read_text() != configured:
        generated.write_text(configured)
    source=SOURCE/'sklearn/utils/extmath.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='row_norms')
    namespace={'np':np,'sparse':sparse,'get_namespace':lambda x:(np,False),'_is_numpy_namespace':lambda xp:xp is np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    import types
    extmath=types.ModuleType('sklearn.utils.extmath'); extmath.row_norms=namespace['row_norms']
    assert extmath.row_norms.__code__.co_filename==str(source)
    sys.modules['sklearn.utils.extmath']=extmath
    setup="from setuptools import setup, Extension\nfrom Cython.Build import cythonize\n"
    setup+=f"modules={MODULES!r}\n"
    setup+="setup(ext_modules=cythonize([Extension('sklearn.'+m.replace('/','.'),['sklearn/'+m+'.pyx'],extra_compile_args=['-O2','-ffp-contract=off']) for m in modules],compiler_directives={'language_level':3}))\n"
    (target/'setup.py').write_text(setup)
    report=ROOT/'docs/private/centroid'; report.mkdir(parents=True,exist_ok=True)
    with (report/'oracle-build.log').open('w') as log:
        subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=target,check=True,stdout=log,stderr=log)
    sys.path.insert(0,str(target))
    modules=[importlib.import_module('sklearn.'+m.replace('/','.')) for m in MODULES]
    for module in modules: assert Path(module.__file__).is_relative_to(target)
    (report/'oracle.json').write_text(json.dumps({'revision':REV,'files':hashes,'modules':[m.__file__ for m in modules],'numpy':np.__version__,'scipy':scipy.__version__,'blas_int':definition},indent=2))
    return modules[:4]
if __name__=='__main__': load()
