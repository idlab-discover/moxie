"""Compare complete binned trees and predictions against the pinned grower."""
from itertools import product
from pathlib import Path
import subprocess

import numpy as np
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower

from hgb_splitting_parity import bits64


BUILDS = Path(__file__).resolve().parents[1] / "builds"


def make_case(
    X, gradients, counts, *, n_bins=8, hessians=None, categorical=None,
    monotonic=None, groups=None, l2=0.0, min_hessian=1e-3, min_leaf=1,
    min_gain=0.0, max_leaves=None, max_depth=None, shrinkage=1.0, query=None,
):
    X = np.asfortranarray(X, dtype=np.uint8)
    n_rows, n_features = X.shape
    gradients = np.asarray(gradients, dtype=np.float32)
    categorical = np.asarray(
        categorical if categorical is not None else np.zeros(n_features), dtype=np.uint8
    )
    monotonic = np.asarray(
        monotonic if monotonic is not None else np.zeros(n_features), dtype=np.int8
    )
    missing = np.any(X == n_bins - 1, axis=0).astype(np.uint8)
    unit_hessian = hessians is None
    hessians = np.asarray([1.0] if unit_hessian else hessians, dtype=np.float32)
    if query is None:
        query = np.concatenate([X, np.full((1, n_features), n_bins - 1, dtype=np.uint8)])
    query = np.ascontiguousarray(query, dtype=np.uint8)
    grower = TreeGrower(
        X, gradients, hessians, n_bins=n_bins, n_bins_non_missing=counts,
        has_missing_values=missing, is_categorical=categorical, monotonic_cst=monotonic,
        interaction_cst=None if groups is None else list(map(set, groups)),
        l2_regularization=l2, min_hessian_to_split=min_hessian, min_samples_leaf=min_leaf,
        min_gain_to_split=min_gain, max_leaf_nodes=max_leaves, max_depth=max_depth,
        shrinkage=shrinkage, n_threads=1,
    )
    grower.grow()
    thresholds = [
        np.arange(count if cat else max(0, count - 1), dtype=np.float64)
        for count, cat in zip(counts, categorical)
    ]
    predictor = grower.make_predictor(thresholds)
    predictions = predictor.predict_binned(query, n_bins - 1, n_threads=1)

    expected = [[grower.n_nodes, len(grower.finalized_leaves)]]
    remaining = [grower.root]
    categorical_splits = 0
    while remaining:
        node = remaining.pop()
        common = [node.depth, node.partition_start, node.partition_stop]
        if node.is_leaf:
            expected.append([*common, -1, *bits64([node.value])])
        else:
            split = node.split_info
            expected.append([
                *common, split.feature_idx, split.bin_idx, int(split.missing_go_to_left),
                int(split.is_categorical), *bits64([node.value, split.gain]),
            ])
            if split.is_categorical:
                expected.append(list(map(int, split.left_cat_bitset)))
                categorical_splits += 1
            remaining.extend((node.right_child, node.left_child))
    expected.extend([int(row)] for row in grower.splitter.partition)
    expected.extend([value] for value in bits64(predictions))

    fields = [
        n_rows, n_features, n_bins, int(unit_hessian), max_leaves or 0, max_depth or 0,
        min_leaf, *bits64([l2, min_hessian, min_gain, shrinkage]), len(groups or ()),
        len(query), *np.column_stack([counts, categorical, monotonic]).ravel(),
    ]
    for group in groups or ():
        fields.extend([len(group), *group])
    fields.extend([*X.ravel(order="F"), *gradients.view(np.int32)])
    if not unit_hessian:
        fields.extend(hessians.view(np.int32))
    fields.extend(query.ravel())
    return " ".join(map(str, fields)), expected, (grower.n_nodes, categorical_splits, n_rows, len(query))


def targeted_cases():
    yield "singleton", make_case([[0]], [5], [1])
    yield "no split at minimum gain", make_case([[0], [1]], [-8, 8], [2], min_gain=256)
    yield "root Hessian floor", make_case([[0], [1]], [-8, 8], [2], hessians=[0.1, 0.1], min_hessian=0.5)
    yield "missing values only", make_case([[0], [7], [0], [7]], [-1, 1, -1, 1], [1])
    yield "missing prediction goes left", make_case([[0], [0], [1]], [-1, -1, 2], [2])
    yield "tied missing prediction goes right", make_case([[0], [1]], [-1, 1], [2])
    yield "zero gradient", make_case([[0], [1], [2]], [0, 0, 0], [3])
    yield "overlapping groups and tied features", make_case(
        [[0, 0, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1]],
        [-1, 1, 1, -1], [2] * 3, groups=[[0, 1], [1, 2]],
    )

    X = np.array(list(product(range(2), repeat=5)), dtype=np.uint8)
    gradients = -(X @ np.array([16, 8, 4, 2, 1]) - 15.5)
    for limit in range(2, 18):
        yield f"equal gain frontier capped at {limit}", make_case(
            X, gradients, [2] * 5, max_leaves=limit, shrinkage=0.25,
        )

    categories = np.repeat(np.array([0, 1, 2, 3], dtype=np.uint8), [32, 24, 60, 52])
    gradients = np.repeat([2.0, -6.0, 5.0, -7.0], [32, 24, 60, 52])
    for with_missing in (False, True):
        X = categories.copy()
        if with_missing:
            X[X == 3] = 7
        yield f"categorical missing={with_missing}", make_case(
            X[:, None], gradients, [3 if with_missing else 4], categorical=[True],
        )

    wide_codes = np.repeat([0, 31, 32, 63, 127, 254, 255], 24)
    yield "categorical high bitset words", make_case(
        wide_codes[:, None], np.repeat([2, -1, 4, -5, -8, 7, 6], 24), [255],
        n_bins=256, categorical=[True], hessians=np.linspace(0.8, 1.25, len(wide_codes)),
    )

    # Strongly disparate magnitudes exercise the pinned root reduction order.
    rng = np.random.default_rng(109)
    X = np.arange(255, dtype=np.uint8).reshape(-1, 1)
    for _ in range(5):
        gradients = rng.choice([-1.0, 1.0], len(X)) * 2.0 ** rng.integers(-22, 30, len(X))
        yield "large dynamic range", make_case(
            X, gradients, [255], n_bins=256, max_leaves=15,
        )


def generated_cases():
    rng = np.random.default_rng(7711)
    for seed in range(120):
        n_rows = int(rng.integers(19, 211))
        X = rng.integers(0, 7, size=(n_rows, 6), dtype=np.uint8)
        n_bins = 256 if seed % 5 == 0 else 8
        if seed % 3 != 0:
            X[rng.random(X.shape) < 0.13] = n_bins - 1
        categorical = [False, False, seed % 3 == 1, False, seed % 2 == 0, False]
        monotonic = [1, -1, 0, 0, 0, 0] if seed % 4 < 2 else None
        signal = np.minimum(X.astype(np.int32), 7) @ np.array([4, -2, 0, 1, 0, 2])
        if seed % 6 == 4:
            signal = np.where(X[:, 4] % 2 == 0, -50, 50) + signal
        gradient = -(signal - signal.mean() + rng.normal(0, 4, n_rows))
        hessians = rng.uniform(0.2, 2.0, n_rows) if seed % 2 else None
        query = rng.integers(0, 8, size=(17, 6), dtype=np.uint8)
        query[query == 7] = n_bins - 1
        yield f"generated {seed}", make_case(
            X, gradient, [7] * 6, n_bins=n_bins, hessians=hessians,
            categorical=categorical, monotonic=monotonic,
            groups=[[0, 1, 2], [1, 3], [2, 4, 5]] if seed % 3 == 0 else None,
            max_leaves=[None, 2, 5, 11][seed % 4],
            max_depth=[None, 1, 2, 4, 8][seed % 5],
            min_leaf=1 + seed % 8, l2=[0.0, 0.125, 2.0][seed % 3],
            min_hessian=[1e-3, 0.8, 3.5][seed % 3],
            min_gain=[0.0, 0.25, 5.0][seed % 3],
            shrinkage=[1.0, 0.5, 0.125][seed % 3], query=query,
        )


def main():
    cases = [*targeted_cases(), *generated_cases()]
    fixtures = BUILDS / "hgb_grower_fixtures.txt"
    fixtures.write_text("".join(f"{case[0]}\n" for _, case in cases))
    run = subprocess.run(
        [BUILDS / "hgb_grower_driver", fixtures], check=True, text=True, capture_output=True
    )
    actual = [list(map(int, line.split())) for line in run.stdout.splitlines()]
    offset = 0
    for label, (_, expected, _) in cases:
        for index, want in enumerate(expected):
            got = actual[offset] if offset < len(actual) else None
            if got != want:
                raise AssertionError(f"{label}, record {index}: got {got}, expected {want}")
            offset += 1
    assert offset == len(actual), "extra records from grower"
    nodes, cat, positions, predictions = map(sum, zip(*(case[2] for _, case in cases)))
    print(
        f"hgb grower parity: {len(cases)} cases, {nodes} nodes ({cat} categorical splits), "
        f"{positions} partition positions and {predictions} binned predictions matched bit for bit"
    )


if __name__ == "__main__":
    main()
