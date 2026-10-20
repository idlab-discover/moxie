"""Differential-test plumbing shared by the decomposition parity scripts.

The estimator under comparison is always the *pinned local* scikit-learn source
file (`$MOXIE_WORKSPACE/scikit-learn` at REVISION), executed as a module inside
the installed `sklearn.decomposition` package so that its relative imports
(validation, base classes, utilities) resolve to the support wheel. Only that
infrastructure comes from the wheel; the algorithm code is the pinned file,
which `load_pinned` verifies by revision, cleanliness and code filename.

Native results come from compiled Mojo drivers: `run_native` writes each input
matrix to a text file, runs the driver and parses its `name value` lines.
"""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import tempfile

# Pinned reference sources live outside this family's write scope. Direct
# parity-script invocation must not populate their __pycache__ directories.
sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get("MOXIE_WORKSPACE", ROOT.parents[2])).resolve()
SOURCE = WORKSPACE / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"


def verify_source(relative):
    head = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    assert head == REVISION, f"scikit-learn checkout is at {head}, expected {REVISION}"
    dirty = subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain", "--", relative], text=True)
    assert not dirty, f"pinned source {relative} has local modifications"
    return SOURCE / relative


def load_pinned(relative, *, extensions=None):
    """Execute pinned `sklearn/.../<module>.py` as `sklearn...._pinned_<module>`.

    `extensions` optionally maps relative-import names (e.g. "_cdnmf_fast") to
    already-imported modules, for pinned Cython kernels compiled locally.
    """
    path = verify_source(relative)
    package = ".".join(Path(relative).with_suffix("").parts[:-1])
    importlib.import_module(package)
    name = package + "._pinned_" + path.stem
    for alias, module in (extensions or {}).items():
        sys.modules[package + "." + alias] = module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert module.__file__ == str(path)
    return module


def write_matrix(directory, name, array):
    array = np.atleast_2d(np.asarray(array, dtype=np.float64))
    path = Path(directory) / (name + ".txt")
    with open(path, "w") as handle:
        handle.write(f"{array.shape[0]} {array.shape[1]}\n")
        handle.write(" ".join(repr(float(v)) for v in array.ravel()))
        handle.write("\n")
    return path


def run_native(binary, dtype, matrices, *options):
    """Run `binary f32|f64 <matrix files...> <options...>` and parse its output."""
    with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
        paths = [write_matrix(directory, f"m{i}", m) for i, m in enumerate(matrices)]
        args = [str(binary), "f32" if dtype == np.float32 else "f64", *map(str, paths), *map(str, options)]
        process = subprocess.run(args, text=True, capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(f"native driver failed ({process.returncode}): {' '.join(args[:2] + args[2 + len(paths):])}\n"
                           f"{process.stdout[-2000:]}\n{process.stderr[-3000:]}")
    result = {}
    for line in process.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        name, value = parts
        try:
            result.setdefault(name, []).append(float(value))
        except ValueError:
            result[name] = value
    return {k: (np.array(v) if isinstance(v, list) else v) for k, v in result.items()}


def tolerance(dtype, float64=3e-9, float32=6e-4):
    return float32 if dtype == np.float32 else float64


def assert_close(got, expected, tol, what, atol=None):
    expected = np.asarray(expected, dtype=np.float64).ravel()
    got = np.asarray(got if got is not None else [], dtype=np.float64).ravel()
    np.testing.assert_allclose(got, expected, rtol=tol, atol=tol if atol is None else atol,
                               equal_nan=True, err_msg=what)


def align_signs(got, expected, axis):
    """Flip native rows (axis=0) or columns (axis=1) to match the reference signs.

    Used only where the reference itself has no sign convention (e.g. LAPACK
    factor signs in FactorAnalysis), so both are compared up to that symmetry.
    Returns the aligned array and the sign vector.
    """
    got = np.array(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    count = got.shape[axis]
    signs = np.ones(count)
    for i in range(count):
        a = got[i] if axis == 0 else got[:, i]
        b = expected[i] if axis == 0 else expected[:, i]
        if np.dot(a, b) < 0:
            signs[i] = -1.0
    return (got * signs[:, None] if axis == 0 else got * signs[None, :]), signs


def compile_pinned_extension(relative):
    """Compile a pinned sklearn .pyx (validation only) and import it.

    The pinned source file is copied verbatim into ignored builds/parity and
    built with Cython against the local checkout's include path.
    """
    source = verify_source(relative)
    workdir = ROOT / "builds/parity" / source.stem
    workdir.mkdir(parents=True, exist_ok=True)
    name = "_pinned_" + source.stem
    (workdir / (name + ".pyx")).write_bytes(source.read_bytes())
    setup = workdir / "setup.py"
    setup.write_text(
        "from setuptools import setup, Extension\nfrom Cython.Build import cythonize\nimport numpy\n"
        f"setup(ext_modules=cythonize([Extension({name!r}, [{name + '.pyx'!r}], include_dirs=[numpy.get_include()])],"
        f" include_path=[{str(SOURCE)!r}], compiler_directives={{'language_level': 3}}))\n")
    built = list(workdir.glob(name + "*.so"))
    if not built or built[0].stat().st_mtime < source.stat().st_mtime:
        process = subprocess.run([sys.executable, str(setup), "build_ext", "--inplace"], cwd=workdir,
                                 text=True, capture_output=True)
        (workdir / "build.log").write_text(process.stdout + process.stderr)
        process.check_returncode()
        built = list(workdir.glob(name + "*.so"))
    spec = importlib.util.spec_from_file_location(name, built[0])
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    return extension
