"""Compare the native precomputed hierarchy to the pinned sklearn kernels."""

import ast
import hashlib
import importlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import warnings

from Cython import __version__ as cython_version
from Cython.Build import cythonize
import numpy as np
import scipy
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from setuptools import Distribution, Extension
import sklearn


FAMILY = Path(__file__).resolve().parents[3]
SOURCE = FAMILY.parents[2] / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
REFERENCE = Path("sklearn/cluster/_hdbscan")
BUILD = FAMILY / "builds"
DRIVER = BUILD / "hdbscan_hierarchy_driver"


def build_pinned_extensions(stems, files, *, source_edits=()):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(SOURCE), *args], text=True)

    if git("rev-parse", "HEAD").strip() != REVISION or git("status", "--porcelain", "--", *files):
        raise ValueError("HDBSCAN parity requires the clean pinned sklearn sources")

    target = BUILD / "hdbscan-oracle"
    target.mkdir(parents=True, exist_ok=True)
    extensions = []
    for stem in stems:
        name = f"moxie_oracle{stem}"
        source = SOURCE / REFERENCE / f"{stem}.pyx"
        declarations = source.with_suffix(".pxd")
        if declarations.is_file():
            # Cython resolves same-module declarations by extension name.
            content = source.read_text()
            for old, new in source_edits:
                assert content.count(old) == 1, old
                content = content.replace(old, new)
            source = target / f"{name}.pyx"
            if not source.exists() or source.read_text() != content:
                source.write_text(content)
            shutil.copy2(declarations, source.with_suffix(".pxd"))
        extensions.append(
            Extension(
                name,
                [str(source)],
                include_dirs=[np.get_include()],
                extra_compile_args=["-ffp-contract=off"],
            )
        )
    modules = cythonize(
        extensions,
        include_path=[str(target), str(SOURCE)],
        build_dir=str(target / "generated"),
        compiler_directives={"language_level": 3},
        quiet=True,
    )
    distribution = Distribution({"ext_modules": modules})
    command = distribution.get_command_obj("build_ext")
    command.build_lib = str(target)
    command.build_temp = str(target / "objects")
    distribution.run_command("build_ext")

    sys.path.insert(0, str(target))
    return {stem: importlib.import_module(f"moxie_oracle{stem}") for stem in stems}


def pinned_oracle():
    files = [
        REFERENCE / name for name in ("_linkage.pyx", "_reachability.pyx", "_tree.pxd", "hdbscan.py")
    ]
    files.append(Path("sklearn/utils/validation.py"))
    extensions = build_pinned_extensions(("_reachability", "_linkage"), files)
    scope = {
        "np": np,
        "sp": scipy.sparse,
        "issparse": issparse,
        "warn": warnings.warn,
        "make_single_linkage": extensions["_linkage"].make_single_linkage,
        "mst_from_mutual_reachability": extensions["_linkage"].mst_from_mutual_reachability,
        "mutual_reachability_graph": extensions["_reachability"].mutual_reachability_graph,
    }
    for path, functions in [
        (files[-1], {"_allclose_dense_sparse"}),
        (REFERENCE / "hdbscan.py", {"_hdbscan_brute", "_brute_mst", "_process_mst"}),
    ]:
        nodes = [
            node for node in ast.parse((SOURCE / path).read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in functions
        ]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)

    return scope["_hdbscan_brute"], {
        str(path): hashlib.sha256((SOURCE / path).read_bytes()).hexdigest() for path in files
    }


def run_native(distances, min_samples, alpha, layout):
    command = [
        DRIVER, distances.dtype.itemsize * 8, *distances.shape, min_samples,
        alpha, layout, *distances.ravel(),
    ]
    return subprocess.run(list(map(str, command)), capture_output=True, text=True)


def main():
    oracle, hashes = pinned_oracle()
    rng = np.random.default_rng(26)
    samples = rng.normal(size=(19, 3))
    tied = np.arange(37, dtype=float)[rng.permutation(37)] % 13
    missing = np.array([[0, 1, np.inf, np.inf], [1, 0, np.inf, np.inf],
                        [np.inf, np.inf, 0, 2], [np.inf, np.inf, 2, 0]])
    large = rng.integers(0, 40, size=259)
    cases = {
        "two": np.array([[0, 5], [5, 0]]),
        "euclidean": cdist(samples, samples),
        "tied": abs(tied[:, None] - tied),
        "duplicates": np.zeros((7, 7)),
        "missing_pairs": missing,
        "near_symmetric": np.array([[0, 1, 7], [1 + 5e-8, 0, 2], [7, 2, 0]]),
        "large_ties": abs(large[:, None] - large),
    }
    checked = []
    for dtype in (np.float32, np.float64):
        for case, data in cases.items():
            matrix = data.astype(dtype)
            for min_samples in sorted({1, min(3, len(matrix)), len(matrix)}):
                for alpha in (0.75, 1.0, 2.5):
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="The minimum spanning tree contains")
                        expected = oracle(
                            matrix.astype(np.float64), min_samples, alpha,
                            metric="precomputed", copy=True, metric_params={},
                        )
                    layouts = ("contiguous", "strided") if case != "large_ties" else ("contiguous",)
                    for layout in layouts:
                        result = run_native(matrix, min_samples, alpha, layout)
                        context = (dtype.__name__, case, min_samples, alpha, layout)
                        if result.returncode:
                            raise AssertionError((context, result.stderr))
                        actual = np.loadtxt(result.stdout.splitlines(), ndmin=2)
                        np.testing.assert_array_equal(actual[:, 0], expected["left_node"], err_msg=str(context))
                        np.testing.assert_array_equal(actual[:, 1], expected["right_node"], err_msg=str(context))
                        np.testing.assert_allclose(
                            actual[:, 2], expected["value"], rtol=2e-12, atol=0, err_msg=str(context)
                        )
                        checked.append(context)

    invalid = (
        (np.zeros((2, 3)), 1, 1, "contiguous", "square"),
        (np.zeros((2, 2)), 1, 1, "flat", "rank-2"),
        (np.zeros((0, 0)), 1, 1, "contiguous", "nonempty"),
        (np.zeros((1, 1)), 1, 1, "contiguous", "more than one sample"),
        (np.zeros((2, 2)), 0, 1, "contiguous", "min_samples"),
        (np.zeros((2, 2)), 3, 1, "contiguous", "min_samples"),
        (np.zeros((2, 2)), 1, 0, "contiguous", "alpha"),
        (np.zeros((2, 2)), 1, np.inf, "contiguous", "alpha"),
        (np.array([[0, np.nan], [np.nan, 0]]), 1, 1, "contiguous", "NaN"),
        (np.array([[0, np.inf], [0, 0]]), 1, 1, "contiguous", "symmetric"),
        (np.array([[0, 1], [2, 0]]), 1, 1, "contiguous", "symmetric"),
    )
    for dtype in (np.float32, np.float64):
        for matrix, min_samples, alpha, layout, error in invalid:
            result = run_native(matrix.astype(dtype), min_samples, alpha, layout)
            output = result.stdout + result.stderr
            if not result.returncode or "ValueError:" not in output or error not in output:
                raise AssertionError((dtype, error, result.returncode, output))

    receipt = {
        "revision": REVISION,
        "source_hashes": hashes,
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn_extension_abi": sklearn.__version__,
        "cython": cython_version,
        "platform": platform.platform(),
        "seed": 26,
        "flags": ["-O3", "--fp-mode=contract=off"],
        "cases": checked,
        "invalid_cases": 2 * len(invalid),
    }
    (BUILD / "hdbscan-hierarchy-parity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"PASS: {len(checked)} sklearn hierarchy comparisons and {2 * len(invalid)} rejection cases")


if __name__ == "__main__":
    main()
