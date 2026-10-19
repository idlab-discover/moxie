"""Compile unchanged pinned sklearn Tree method bodies in a test-only extension.

Builders and partial_dependence are outside this engine. This extraction retains
Tree through _get_node_ndarray, including dense apply/predict/depth/importances,
allocation/state loaders and the exact inline routing and bitset implementations.
Only import/declaration plumbing is flattened; no oracle algorithm is rewritten.
"""
from pathlib import Path
import hashlib,importlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT.parent/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES=['sklearn/tree/_tree.pyx','sklearn/tree/_tree.pxd','sklearn/tree/_utils.pxd',
       'sklearn/tree/_utils.pyx','sklearn/utils/_typedefs.pxd','sklearn/utils/_bitset.pxd']
def between(s,start,end): return s[s.index(start):s.index(end)]
def load():
    (ROOT/'docs/private/tree/storage').mkdir(parents=True, exist_ok=True)
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==REV
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',*FILES],text=True)
    source={f:(SOURCE/f).read_text() for f in FILES}
    tree=source[FILES[0]]; pxd=source[FILES[1]]; utils=source[FILES[2]]
    bitset=source[FILES[5]]
    imports='''from cpython cimport Py_INCREF, PyObject, PyTypeObject
from libc.math cimport INFINITY, isnan
from libc.stdlib cimport free, realloc
from libc.string cimport memcpy, memset
from libc.stdint cimport INTPTR_MAX
import struct
import numpy as np
cimport numpy as cnp
cnp.import_array()
from scipy.sparse import issparse, csr_array
'''
    # Exact source spans, recorded separately below for auditability.
    spans=[source[FILES[4]],
        between(bitset,'ctypedef uint32_t BITSET_INNER_DTYPE_C','cdef inline void init_bitset'),
        between(bitset,'cdef inline uint8_t in_bitset(','cpdef inline uint8_t in_bitset_memoryview'),
        between(utils,'cpdef enum:','cdef enum:\n    # Max value'),
        # Cell belongs to an unrelated neighbor engine; omit only that fused arm.
        between(utils,'ctypedef fused realloc_ptr:','cdef int safe_realloc').replace('    (Cell*)\n',''),
        between(source[FILES[3]],'cdef int safe_realloc','def _realloc_test'),
        between(utils,'cdef inline cnp.ndarray sizet_ptr_to_ndarray','cdef inline intp_t rand_int'),
        between(tree,'cdef extern from "numpy/arrayobject.h":','# =============================================================================\n# Types'),
        between(tree,'TREE_LEAF = -1','cdef inline void _init_parent_record'),
    ]
    fields=between(pxd,'    # Input/Output layout','    # Methods')
    methods=between(tree,'    # Wrap for outside world.','    def compute_partial_dependence')
    methods=methods[:methods.index('    cpdef object decision_path')] + methods[methods.index('    cpdef compute_node_depths'):]
    checks=between(tree,'def _check_n_classes','cdef class _CCPPruneController')
    code=imports+'\n'.join(spans)+'\ncdef class Tree:\n'+fields+methods+'\n'+checks
    target=ROOT/'builds/tree/storage/oracle';target.mkdir(parents=True,exist_ok=True)
    pyx=target/'storage_oracle.pyx';pyx.write_text(code)
    (target/'setup.py').write_text('''from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
setup(ext_modules=cythonize([Extension("storage_oracle", ["storage_oracle.pyx"], include_dirs=[numpy.get_include()], extra_compile_args=["-O3", "-ffp-contract=off"])], compiler_directives={"language_level":3}))
''')
    subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=target,check=True)
    sys.path.insert(0,str(target)); module=importlib.import_module('storage_oracle')
    assert str(target) in module.__file__
    evidence={'revision':REV,'files':{f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in FILES},
              'extension':module.__file__,'extracted_sha256':hashlib.sha256(code.encode()).hexdigest(),
              'adaptation':'unchanged Tree method bodies; flattened Cython declarations; builders, decision_path and partial_dependence excluded'}
    (ROOT/'docs/private/tree/storage/oracle.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return module
if __name__=='__main__': load()
