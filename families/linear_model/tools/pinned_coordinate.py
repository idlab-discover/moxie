"""Compile the exact local sklearn coordinate Cython engine for validation only."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import hashlib
import parity_io as io
from childcheck import environment


def load(root):
    files = ['sklearn/linear_model/_base.py', 'sklearn/linear_model/_coordinate_descent.py',
             'sklearn/linear_model/_cd_fast.pyx', 'sklearn/utils/_random.pxd',
             'sklearn/utils/_cython_blas.pxd', 'sklearn/utils/_typedefs.pxd']
    hashes = io.verify_reference(files)
    folder = root / 'builds/reference'
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / 'pinned_cd_fast.pyx'
    source.write_bytes((io.REFERENCE / files[2]).read_bytes())
    setup = folder / 'setup.py'
    setup.write_text('from setuptools import setup, Extension\nfrom Cython.Build import cythonize\nimport numpy\n'
                     f'setup(ext_modules=cythonize([Extension("pinned_cd_fast", ["pinned_cd_fast.pyx"], include_dirs=[numpy.get_include()])], include_path=[{str(io.REFERENCE)!r}], compiler_directives={{"language_level": 3}}, build_dir="cython"))\n')
    key = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    stamp = folder / 'source.sha256'
    extensions = list(folder.glob('pinned_cd_fast*.so'))
    if not extensions or not stamp.exists() or stamp.read_text() != key:
        with (folder / 'compile.log').open('w') as log:
            subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace', '--build-temp', 'temp'], cwd=folder, env=environment(root), stdout=log, stderr=subprocess.STDOUT, check=True)
        stamp.write_text(key)
    path = next(folder.glob('pinned_cd_fast*.so'))
    spec = importlib.util.spec_from_file_location('pinned_cd_fast', path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    import sklearn.linear_model as package
    package._cd_fast = native
    sys.modules['sklearn.linear_model._cd_fast'] = native
    io.load_pinned(files[0], register=True)
    module = io.load_pinned(files[1], register=True)
    assert module.cd_fast is native
    io.pinned_code(module.ElasticNet.fit, files[1])
    return module, hashes
