"""Build the pinned sklearn SAG engine and its dataset sampler for parity."""

from importlib.util import module_from_spec, spec_from_file_location
import subprocess
import sys
from sysconfig import get_config_var
from unittest.mock import patch

from Cython import Tempita

from parity_io import REFERENCE, load_pinned, pinned_code

BASE = "sklearn/linear_model/_base.py"
SAG = "sklearn/linear_model/_sag.py"
SAG_FAST = "sklearn/linear_model/_sag_fast.pyx.tp"
DATASET = "sklearn/utils/_seq_dataset"
SOURCES = (SAG, SAG_FAST, f"{DATASET}.pyx.tp", f"{DATASET}.pxd.tp")


def load(folder):
    """Compile the pinned stochastic engine and its dataset sampler."""
    templates = {
        SAG_FAST: "pinned_sag_fast.pyx",
        f"{DATASET}.pyx.tp": f"{DATASET}.pyx",
        f"{DATASET}.pxd.tp": f"{DATASET}.pxd",
    }
    for source, destination in templates.items():
        path = folder / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = Tempita.Template((REFERENCE / source).read_text()).substitute()
        if not path.exists() or path.read_text() != rendered:
            path.write_text(rendered)

    setup = folder / "setup.py"
    setup.write_text(
        "from setuptools import setup, Extension\n"
        "from Cython.Build import cythonize\n"
        "from numpy import get_include\n"
        "setup(ext_modules=cythonize([\n"
        f"    Extension('sklearn.utils._seq_dataset', ['{DATASET}.pyx'], "
        "include_dirs=[get_include()], extra_compile_args=['-ffp-contract=off']),\n"
        "    Extension('pinned_sag_fast', ['pinned_sag_fast.pyx'], "
        "include_dirs=[get_include()], extra_compile_args=['-ffp-contract=off']),\n"
        f"], include_path=['.', {str(REFERENCE)!r}], build_dir='cython'))\n"
    )
    with (folder / "compile.log").open("w") as log:
        subprocess.run(
            [sys.executable, setup.name, "build_ext", "--build-lib", "extensions"],
            cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True,
        )

    def load_extension(name):
        library = folder / "extensions" / (name.replace(".", "/") + get_config_var("EXT_SUFFIX"))
        spec = spec_from_file_location(name, library)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    base = load_pinned(BASE)
    dataset = load_extension("sklearn.utils._seq_dataset")
    with patch.dict(sys.modules, {"sklearn.utils._seq_dataset": dataset}):
        native = load_extension("pinned_sag_fast")
    base.ArrayDataset32 = dataset.ArrayDataset32
    base.ArrayDataset64 = dataset.ArrayDataset64
    sag = load_pinned(SAG)
    sag.make_dataset = base.make_dataset
    sag.sag32 = native.sag32
    sag.sag64 = native.sag64
    pinned_code(sag.sag_solver, SAG)
    pinned_code(sag.make_dataset, BASE)
    return sag.sag_solver
