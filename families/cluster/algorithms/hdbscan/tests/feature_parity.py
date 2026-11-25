"""Compare finite feature linkage and flat clustering with pinned sklearn."""

import hashlib
import itertools
import json
import platform
import subprocess
import sys

import numpy as np
from sklearn.neighbors import KDTree, BallTree

from hierarchy_parity import (
    BUILD, REFERENCE, REVISION, SOURCE, pinned_linkage_oracles,
    cython_version, scipy, sklearn,
)
from tree_parity import pinned_tree_oracle


DRIVER = BUILD / "hdbscan_feature_driver"


def native(X, min_samples, alpha, algorithm, metric, p, leaf_size, layout, method):
    args = [
        DRIVER, X.dtype.itemsize * 8, *X.shape, min_samples, alpha, algorithm,
        metric, p, leaf_size, layout, method, *X.ravel(),
    ]
    return subprocess.run(list(map(str, args)), capture_output=True, text=True)


def compare(result, expected, labels, probabilities, metric, context):
    assert result.returncode == 0, (context, result.stdout, result.stderr)
    linkage, points = result.stdout.split("points\n")
    linkage = np.loadtxt(linkage.splitlines(), ndmin=2)
    points = np.loadtxt(points.splitlines(), ndmin=2)
    detail = str(context)
    np.testing.assert_array_equal(linkage[:, 0], expected["left_node"], err_msg=detail)
    np.testing.assert_array_equal(linkage[:, 1], expected["right_node"], err_msg=detail)
    # Cosine forms 1 - similarity, so ulps of similarity can dominate small distances.
    height_atol = 2e-15 if metric == "cosine" else 0
    probability_rtol = 1e-10 if metric == "cosine" else 2e-12
    np.testing.assert_allclose(
        linkage[:, 2], expected["value"], rtol=2e-12, atol=height_atol, err_msg=detail
    )
    np.testing.assert_array_equal(points[:, 0], labels, err_msg=detail)
    np.testing.assert_allclose(
        points[:, 1], probabilities, rtol=probability_rtol, atol=0, err_msg=detail
    )


def main():
    brute, prims, hashes = pinned_linkage_oracles()
    oracle_tree = pinned_tree_oracle()
    rng = np.random.default_rng(26)
    groups = rng.permutation(np.vstack([
        rng.normal(loc=center, scale=0.15, size=(11, 3))
        for center in (0, 2, 4)
    ]))
    tied = np.array([[0], [2], [1], [10], [11], [12], [0], [10], [1], [11]])
    angles = rng.uniform(-0.5, 0.5, size=(19, 2))
    cases = (
        ("clustered", groups, "euclidean", ("auto", "brute", "kd_tree", "ball_tree")),
        ("manhattan", groups, "manhattan", ("auto", "brute", "kd_tree", "ball_tree")),
        ("minkowski", groups, "minkowski", ("auto", "brute", "kd_tree", "ball_tree")),
        ("tied", tied, "manhattan", ("auto", "brute", "kd_tree", "ball_tree")),
        ("duplicate", np.zeros((7, 3)), "euclidean", ("auto", "brute", "kd_tree", "ball_tree")),
        ("cosine", groups, "cosine", ("auto", "brute")),
        ("cosine_zero", np.array([[0, 0], [1, 0], [-1, 0], [0, 1]]), "cosine", ("auto", "brute")),
        ("haversine", angles, "haversine", ("auto", "ball_tree")),
        ("braycurtis", angles, "braycurtis", ("auto", "brute", "ball_tree")),
        ("two", np.array([[0, 0], [2, 3]]), "euclidean", ("auto", "brute", "kd_tree", "ball_tree")),
    )

    checked = []
    for name, data, metric, algorithms in cases:
        p = 3 if metric == "minkowski" else 2
        params = {"p": p} if metric == "minkowski" else {}
        for dtype, algorithm in itertools.product((np.float32, np.float64), algorithms):
            X = data.astype(dtype)
            promoted = X.astype(np.float64)
            for k, alpha, (layout, leaf_size, method) in itertools.product(
                sorted({1, min(len(X), 3), len(X)}),
                (0.5, 1.0, 2.5),
                (("contiguous", 40, "eom"), ("strided", 2, "leaf"), ("offset", 4, "eom")),
            ):
                backend = algorithm
                if algorithm == "auto":
                    if metric in KDTree.valid_metrics:
                        backend = "kd_tree"
                    elif metric in BallTree.valid_metrics:
                        backend = "ball_tree"
                    else:
                        backend = "brute"
                if backend == "brute":
                    expected = brute(promoted, k, alpha, metric=metric, **params)
                else:
                    expected = prims(
                        promoted, backend, k, alpha, metric=metric, leaf_size=leaf_size, **params
                    )
                labels, probabilities = oracle_tree.tree_to_labels(expected, 3, method)
                context = (name, dtype.__name__, algorithm, k, alpha, leaf_size, layout, method)
                result = native(X, k, alpha, algorithm, metric, p, leaf_size, layout, method)
                compare(result, expected, labels, probabilities, metric, context)
                checked.append(context)

    x = np.array([[0., 0.], [1., 1.], [2., 2.]])
    invalid = (
        (x, 1, 1, "unknown", "euclidean", 2, 1, "contiguous", "invalid HDBSCAN algorithm"),
        (x, 1, 1, "brute", "precomputed", 2, 1, "contiguous", "feature metric"),
        (x, 1, 1, "kd_tree", "cosine", 2, 1, "contiguous", "KDTree"),
        (x, 1, 1, "ball_tree", "cosine", 2, 1, "contiguous", "BallTree"),
        (x, 1, 1, "auto", "haversine", 2, 0, "contiguous", "leaf_size"),
        (x, 1, 1, "auto", "euclidean", 2, 1, "flat", "rank-2"),
        (x[:0], 1, 1, "auto", "euclidean", 2, 1, "contiguous", "nonempty"),
        (x[:1], 1, 1, "auto", "euclidean", 2, 1, "contiguous", "more than one sample"),
        (x, 0, 1, "auto", "euclidean", 2, 1, "contiguous", "min_samples"),
        (x, 4, 1, "auto", "euclidean", 2, 1, "contiguous", "min_samples"),
        (x, 1, 0, "auto", "euclidean", 2, 1, "contiguous", "alpha"),
        (x, 1, np.inf, "auto", "euclidean", 2, 1, "contiguous", "alpha"),
        (np.array([[0., 0.], [np.inf, 0.]]), 1, 1, "auto", "euclidean", 2, 1, "strided", "finite features"),
        (np.array([[0., 0.], [np.nan, 0.]]), 1, 1, "brute", "euclidean", 2, 1, "offset", "finite features"),
    )
    for dtype in (np.float32, np.float64):
        for data, *options, error in invalid:
            result = native(data.astype(dtype), *options, "eom")
            output = result.stdout + result.stderr
            assert result.returncode and "ValueError:" in output and error in output, (
                dtype.__name__, error, result.returncode, output,
            )

    for path in (REFERENCE / "_tree.pyx", REFERENCE / "_tree.pxd"):
        hashes[str(path)] = hashlib.sha256((SOURCE / path).read_bytes()).hexdigest()
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
    (BUILD / "hdbscan-feature-parity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"PASS: {len(checked)} sklearn feature linkage and extraction comparisons "
          f"and {2 * len(invalid)} rejection cases")


if __name__ == "__main__":
    main()
