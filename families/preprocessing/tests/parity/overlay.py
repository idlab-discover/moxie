"""Build an oracle tree: pinned scikit-learn Python sources over wheel-compiled extensions."""
import shutil
import subprocess
import sys
from pathlib import Path

import importlib.util

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / '../../../scikit-learn').resolve()
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
TARGET = ROOT / 'builds/oracle'


def build():
    head = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == REVISION, head
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain', '--', 'sklearn'], text=True)
    wheel = Path(importlib.util.find_spec('sklearn').origin).parent
    if TARGET.exists():
        shutil.rmtree(TARGET)
    tracked = subprocess.check_output(['git', '-C', str(SOURCE), 'ls-files', 'sklearn'], text=True).split()
    for relative in tracked:
        if not relative.endswith(('.pyx', '.pxd', '.pxi', '.tp', '.c', '.cpp', '.h', 'meson.build')):
            destination = TARGET / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / relative, destination)
    for extension in wheel.rglob('*.so'):
        destination = TARGET / 'sklearn' / extension.relative_to(wheel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(extension)
    for extra in wheel.rglob('*'):
        # Generated metadata modules (e.g. _build_utils, __check_build) the wheel installs.
        if extra.is_file() and extra.suffix == '.py' and not (TARGET / 'sklearn' / extra.relative_to(wheel)).exists():
            destination = TARGET / 'sklearn' / extra.relative_to(wheel)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extra, destination)
    _compile_preprocessing_extensions()
    return TARGET


# Extensions whose pinned signatures differ from the support wheel are rebuilt from
# the pinned .pyx with the locked Cython; all others reuse wheel binaries.
PINNED_EXTENSIONS = ['sklearn/preprocessing/_target_encoder_fast.pyx',
                     'sklearn/preprocessing/_csr_polynomial_expansion.pyx']


def _compile_preprocessing_extensions():
    import numpy
    from Cython.Build import cythonize
    from setuptools import Extension
    from setuptools.dist import Distribution
    from setuptools.command.build_ext import build_ext
    extensions = []
    for relative in PINNED_EXTENSIONS:
        module = relative[:-4].replace('/', '.')
        (TARGET / relative).with_suffix('.cpython-314-x86_64-linux-gnu.so').unlink(missing_ok=True)
        for old in (TARGET / relative).parent.glob(Path(relative).stem + '.*.so'):
            old.unlink()
        extensions.append(Extension(module, [str(SOURCE / relative)], language='c++',
                                    include_dirs=[numpy.get_include()],
                                    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_9_API_VERSION')]))
    modules = cythonize(extensions, include_path=[str(SOURCE)], build_dir=str(TARGET.parent / 'oracle-cython'),
                        compiler_directives={'language_level': 3, 'boundscheck': False, 'wraparound': False,
                                             'initializedcheck': False, 'cdivision': True}, quiet=True)
    distribution = Distribution({'ext_modules': modules})
    command = build_ext(distribution)
    command.build_lib = str(TARGET)
    command.build_temp = str(TARGET.parent / 'oracle-temp')
    command.ensure_finalized()
    command.run()


def activate():
    if not (TARGET / 'sklearn/__init__.py').exists():
        build()
    sys.path.insert(0, str(TARGET))
    import sklearn
    assert Path(sklearn.__file__).resolve() == (TARGET / 'sklearn/__init__.py').resolve(), sklearn.__file__
    return sklearn


if __name__ == '__main__':
    build()
    sk = activate()
    import numpy as np
    from sklearn.preprocessing import StandardScaler, TargetEncoder
    print(sk.__version__, StandardScaler().fit_transform(np.array([[1., 2.], [3., 5.]])))
    print(TargetEncoder(random_state=0).fit_transform(np.array([[0], [1], [0], [1], [0], [1]] * 3), np.arange(18.) + 0.5))
