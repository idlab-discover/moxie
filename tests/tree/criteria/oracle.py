"""Compile pinned Cython criterion bodies; flatten declarations/imports only.

No installed sklearn algorithm is used. All generated code and extension output
stay under builds/tree/criteria/oracle. The wrapper only exposes the original private methods.
"""
from pathlib import Path
import hashlib,importlib,json,subprocess,sys,re
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT.parent/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES=['sklearn/tree/_criterion.pyx','sklearn/tree/_criterion.pxd',
       'sklearn/tree/_utils.pyx','sklearn/tree/_utils.pxd',
       'sklearn/utils/_sorting.pyx','sklearn/utils/_typedefs.pxd']
def load():
    (ROOT/'docs/private/tree/criteria').mkdir(parents=True, exist_ok=True)
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==REV
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',*FILES],text=True)
    s={f:(SOURCE/f).read_text() for f in FILES}
    criterion=s[FILES[0]]
    criterion=re.sub(r'^from sklearn\..*\n','',criterion,flags=re.M)
    pxd=s[FILES[1]]
    for cls in ['Criterion','ClassificationCriterion','RegressionCriterion']:
        block=pxd.split('cdef class '+cls,1)[1].split('cdef class ',1)[0]
        fields='\n'.join(line for line in block.splitlines() if line.startswith('    cdef ') and '(' not in line.split('#')[0] and not line.strip().endswith('nogil'))
        criterion=re.sub(r'(cdef class '+cls+r'(?:\([^\n]*\))?:\n)',lambda m:m[0]+fields+'\n',criterion,count=1)
    utils=s[FILES[2]]
    fenwick=utils[utils.index('cdef class WeightedFenwickTree:'):utils.index('cdef class PytestWeightedFenwickTree')]
    fields=s[FILES[3]].split('cdef class WeightedFenwickTree:',1)[1].split('    cdef void reset',1)[0]
    fenwick=fenwick.replace('cdef class WeightedFenwickTree:\n','cdef class WeightedFenwickTree:\n'+fields,1)
    alloc=utils[utils.index('cdef int safe_realloc'):utils.index('def _realloc_test')]
    sorting=re.sub(r'^from sklearn\..*\n','',s[FILES[4]],flags=re.M)
    imports='''from libc.stdlib cimport free, realloc
from libc.string cimport memset
from libc.math cimport log as ln
from libc.math cimport INFINITY
import numpy as np
cimport numpy as cnp
'''
    code=imports+s[FILES[5]]+'\nctypedef float64_t* realloc_ptr\n'+alloc+'\n'+fenwick+'\n'+sorting+'\ncdef inline float64_t log(float64_t x) noexcept nogil:\n    return ln(x) / ln(2.0)\n'+criterion
    code+='''

def evaluate(int kind, const float64_t[:, ::1] y,
             const float64_t[:] weights, const intp_t[:] samples,
             intp_t start, intp_t end, float64_t global_weight, positions,
             const intp_t[::1] classes):
    cdef Criterion criterion
    cdef intp_t width = max(classes)
    cdef cnp.ndarray dest = np.zeros((y.shape[1], width), dtype=np.float64)
    cdef float64_t left, right
    if kind == 0: criterion = Gini(y.shape[1], np.asarray(classes))
    elif kind == 1: criterion = Entropy(y.shape[1], np.asarray(classes))
    elif kind == 2: criterion = MSE(y.shape[1], y.shape[0])
    elif kind == 3: criterion = MAE(y.shape[1], y.shape[0])
    else: criterion = Poisson(y.shape[1], y.shape[0])
    criterion.init(y, weights, global_weight, samples, start, end)
    criterion.node_value(<float64_t*>dest.data)
    node_impurity = criterion.node_impurity()
    if kind == 3:
        # At reset, the right child is the whole node. This original method
        # indexes right_abs_errors[start], avoiding the known stale-slot defect.
        criterion.children_impurity(&left, &right)
        node_impurity = right
    result = []
    for position in positions:
        criterion.update(position)
        criterion.children_impurity(&left, &right)
        result.append([criterion.weighted_n_left, criterion.weighted_n_right,
            left, right, criterion.proxy_impurity_improvement(),
            criterion.impurity_improvement(node_impurity, left, right)])
    return node_impurity, dest, result


def mae_offset_probe():
    cdef const float64_t[:, ::1] y = np.arange(4., dtype=np.float64).reshape(-1,1)
    cdef const float64_t[:] weights = np.ones(4, dtype=np.float64)
    cdef const intp_t[:] samples = np.arange(4, dtype=np.intp)
    cdef Criterion criterion = MAE(1, 4)
    criterion.init(y, weights, 4., samples, 0, 4)
    root = criterion.node_impurity()
    criterion.init(y, weights, 4., samples, 1, 3)
    return root, criterion.node_impurity()
'''
    target=ROOT/'builds/tree/criteria/oracle';target.mkdir(parents=True,exist_ok=True)
    (target/'criteria_oracle.pyx').write_text(code)
    (target/'setup.py').write_text('''from setuptools import setup,Extension
from Cython.Build import cythonize
import numpy
setup(ext_modules=cythonize([Extension('criteria_oracle',['criteria_oracle.pyx'],include_dirs=[numpy.get_include()],extra_compile_args=['-O3','-ffp-contract=off'])],compiler_directives={'language_level':3,'boundscheck':False,'wraparound':False,'cdivision':True}))
''')
    subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=target,check=True)
    sys.path.insert(0,str(target))
    result=importlib.import_module('criteria_oracle')
    assert str(target) in result.__file__
    (ROOT/'docs/private/tree/criteria/oracle.json').write_text(json.dumps({'revision':REV,'files':{f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in FILES},'extracted_sha256':hashlib.sha256(code.encode()).hexdigest(),'module':result.__file__},indent=2)+'\n')
    return result
if __name__=='__main__':load()
