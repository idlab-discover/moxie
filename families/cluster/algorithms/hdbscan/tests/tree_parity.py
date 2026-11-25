"""Compare native HDBSCAN tree extraction with the pinned Cython oracle."""

import hashlib
from itertools import product
import json
import subprocess

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist

from hierarchy_parity import BUILD, REFERENCE, REVISION, SOURCE, build_pinned_extensions, pinned_oracle


DRIVER = BUILD / "hdbscan_tree_driver"
FIELDS = ("left_node", "right_node", "value")


def native(hierarchy, min_size, method, allow_root, epsilon, max_size, layout="strided"):
    rows = [row[name].item() for row in hierarchy for name in FIELDS]
    args = [DRIVER, len(hierarchy) + 1, min_size, method, int(allow_root), epsilon, max_size, layout, *rows]
    return subprocess.run(list(map(str, args)), capture_output=True, text=True)


def check_case(tree, hierarchy, context, options, *, matrix=None):
    if matrix is None:
        result = native(hierarchy, *options)
    else:
        min_size, method, allow_root, epsilon, max_size = options
        layout = f"precomputed{matrix.dtype.itemsize * 8}"
        args = [
            DRIVER, len(matrix), min_size, method, int(allow_root), epsilon,
            max_size, layout, 3, 1.25, *matrix.ravel(),
        ]
        result = subprocess.run(list(map(str, args)), capture_output=True, text=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        try:
            labels, probabilities = tree.tree_to_labels(hierarchy, *options)
        except KeyError as error:
            assert error.args == (len(hierarchy) + 1,), (context, options, error)
            output = result.stdout + result.stderr
            assert result.returncode and "KeyError: HDBSCAN root is absent" in output, (
                context, options, result,
            )
            return False
        expected_edges = tree._condense_tree(hierarchy, options[0])
    assert result.returncode == 0, (context, options, result.stderr)
    points, edges = result.stdout.split("edges\n")
    points = np.loadtxt(points.splitlines(), ndmin=2)
    edges = np.loadtxt(edges.splitlines(), ndmin=2)
    detail = str((context, options))
    np.testing.assert_array_equal(points[:, 0], labels, err_msg=detail)
    np.testing.assert_allclose(points[:, 1], probabilities, rtol=2e-12, atol=0, err_msg=detail)
    for column, field in ((0, "parent"), (1, "child"), (3, "cluster_size")):
        np.testing.assert_array_equal(edges[:, column], expected_edges[field], err_msg=detail)
    np.testing.assert_allclose(edges[:, 2], expected_edges["value"], rtol=2e-12, atol=0, err_msg=detail)
    return True


def main():
    tree_files = [REFERENCE / name for name in ("_tree.pyx", "_tree.pxd")]
    # NumPy 2.5 rejects implicit scalar conversion of singleton vectors.
    # Pull out the same element explicitly in the epsilon traversal.
    parent = "cluster_tree[cluster_tree['child'] == leaf]['parent']"
    inverse = "1 / cluster_tree[cluster_tree['child'] == parent]['value']"
    tree = build_pinned_extensions(
        ("_tree",),
        tree_files,
        source_edits=(
            (f"parent = {parent}", f"parent = {parent}.item()"),
            (f"parent_eps = {inverse}", f"parent_eps = ({inverse}).item()"),
        ),
    )["_tree"]
    oracle, hashes = pinned_oracle()
    rng = np.random.default_rng(26)
    cases = {}
    for i in range(16):
        groups = (
            rng.normal(scale=0.07, size=(7, 2)),
            rng.normal(loc=1.0, scale=0.15, size=(8, 2)),
            rng.normal(loc=2.0, scale=0.34, size=(11, 2)),
        )
        x = rng.permutation(np.vstack(groups))
        cases[f"random_{i}"] = np.array(
            list(map(tuple, linkage(x))), dtype=tree.HIERARCHY_dtype
        )
    for name, rows in {
        "two": [(0, 1, 4, 2)],
        "zero_ties": [(0, 1, 0, 2), (2, 3, 0, 2), (4, 5, 0, 4)],
        "infinite_split": [(0, 1, 1, 2), (2, 3, 2, 2), (4, 5, np.inf, 4)],
        "root_pruning": [(0, 1, 1, 2), (2, 5, 2, 3), (3, 6, 3, 4), (4, 7, 4, 5)],
        "epsilon_ties": [
            (0, 1, 1, 2), (2, 3, 1, 2), (4, 5, 1, 2), (6, 7, 1, 2),
            (8, 9, 2, 4), (10, 11, 3, 4), (12, 13, 4, 8),
        ],
        "infinite_ancestor": [
            (0, 1, 1, 2), (2, 3, 1, 2), (4, 5, 1, 2), (6, 7, 1, 2),
            (8, 9, 2, 4), (10, 11, np.inf, 4), (12, 13, np.inf, 8),
        ],
        "interleaved_pruning": [
            (0, 1, 1, 2), (2, 3, 2, 2), (4, 8, 3, 3), (5, 9, 4, 3),
            (6, 10, 5, 4), (7, 11, 6, 4), (12, 13, 7, 8),
        ],
    }.items():
        cases[name] = np.array(rows, dtype=tree.HIERARCHY_dtype)

    compositions = 0
    root_errors = 0
    for dtype in (np.float32, np.float64):
        values = rng.integers(0, 15, size=(27, 2))
        matrix = cdist(values, values).astype(dtype)
        expected = oracle(matrix.astype(np.float64), 3, 1.25, metric="precomputed", copy=True, metric_params={})
        for options in product((2, 5, 12), ("eom", "leaf"), (False, True), (0, 2, 6), (None, 4)):
            if check_case(tree, expected, f"precomputed_{dtype.__name__}", options, matrix=matrix):
                compositions += 1
            else:
                root_errors += 1
        cases[f"precomputed_{dtype.__name__}"] = expected

    checked = 0
    for name, hierarchy in cases.items():
        sizes = sorted({2, min(4, len(hierarchy) + 1), len(hierarchy) + 2})
        for options in product(sizes, ("eom", "leaf"), (False, True), (0, 0.25, 1.0, 2.0, 4.0, 8.0), (None, 3)):
            if check_case(tree, hierarchy, name, options):
                checked += 1
            else:
                root_errors += 1

    valid = cases["infinite_split"]
    nan = valid.copy()
    nan["value"][0] = np.nan
    unordered = valid.copy()
    unordered["value"][0] = 3
    reused = valid.copy()
    reused["right_node"][0] = reused["left_node"][0]
    invalid = [
        (valid, (1, "eom", False, 0, None), "strided", "min_cluster_size"),
        (valid, (2, "other", False, 0, None), "strided", "cluster_selection_method"),
        (valid, (2, "eom", False, -1, None), "strided", "cluster_selection_epsilon"),
        (valid, (2, "eom", False, np.inf, None), "strided", "cluster_selection_epsilon"),
        (valid, (2, "eom", False, 0, 0), "strided", "max_cluster_size"),
        (valid, (2, "eom", False, 0, None), "forest", "complete"),
        (valid, (2, "eom", False, 0, None), "missing", "requires merge distances"),
        (reused, (2, "eom", False, 0, None), "strided", "used twice"),
        (unordered, (2, "eom", False, 0, None), "strided", "nondecreasing"),
        (nan, (2, "eom", False, 0, None), "strided", "must not contain NaN"),
    ]
    for hierarchy, options, layout, error in invalid:
        result = native(hierarchy, *options, layout=layout)
        output = result.stdout + result.stderr
        assert result.returncode and "ValueError:" in output and error in output, result

    assert root_errors > 0
    receipt = {
        "revision": REVISION,
        "source_hashes": hashes | {
            str(path): hashlib.sha256((SOURCE / path).read_bytes()).hexdigest()
            for path in tree_files
        },
        "numpy": np.__version__,
        "oracle_compatibility": "Explicit singleton extraction on two epsilon traversal reads.",
        "seed": 26,
        "tree_cases": list(cases),
        "comparisons": checked,
        "precomputed_compositions": compositions,
        "reference_root_errors": root_errors,
        "rejections": len(invalid),
    }
    (BUILD / "hdbscan-tree-parity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        f"PASS: {checked} pinned tree comparisons, {compositions} precomputed compositions, "
        f"{root_errors} matching reference errors, {len(invalid)} rejection cases"
    )


if __name__ == "__main__":
    main()
