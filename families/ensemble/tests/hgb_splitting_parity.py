"""Compare split decisions, Float64 fields and stable row order to pinned sklearn."""
from pathlib import Path
import subprocess

import numpy as np
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import Splitter, compute_node_value


BUILDS = Path(__file__).resolve().parents[1] / "builds"


def bits64(values):
    return np.asarray(values, dtype=np.float64).view(np.int64).tolist()


def make_case(
    X, gradients, counts, *, n_bins=8, hessians=None, categorical=None,
    monotonic=None, rows=None, selected=None, l2=0.0, min_hessian=1e-3,
    min_leaf=1, min_gain=0.0, bounds=(-np.inf, np.inf), threads=1,
):
    X = np.asfortranarray(X, dtype=np.uint8)
    n_rows, n_features = X.shape
    gradients = np.asarray(gradients, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.uint32)
    missing = np.any(X == n_bins - 1, axis=0).astype(np.uint8)
    categorical = np.zeros(n_features, dtype=np.uint8) if categorical is None else (
        np.asarray(categorical, dtype=np.uint8)
    )
    monotonic = np.zeros(n_features, dtype=np.int8) if monotonic is None else (
        np.asarray(monotonic, dtype=np.int8)
    )
    rows = np.arange(n_rows, dtype=np.uint32) if rows is None else (
        np.asarray(rows, dtype=np.uint32)
    )
    selected = np.arange(n_features, dtype=np.uint32) if selected is None else (
        np.asarray(selected, dtype=np.uint32)
    )
    constant = hessians is None
    hessians = np.ones(1, dtype=np.float32) if constant else (
        np.asarray(hessians, dtype=np.float32)
    )
    builder = HistogramBuilder(X, n_bins, gradients, hessians, constant, threads)
    histograms = builder.compute_histograms_brute(rows)
    total_gradient = gradients[rows].sum(dtype=np.float64)
    total_hessian = float(len(rows)) if constant else hessians[rows].sum(dtype=np.float64)
    value = compute_node_value(total_gradient, total_hessian, *bounds, l2)
    expected = [bits64([value])]
    if len(selected):
        splitter = Splitter(
            X, counts, n_bins - 1, missing, categorical, monotonic, l2,
            min_hessian, min_leaf, min_gain, constant, n_threads=threads,
        )
        split = splitter.find_node_split(
            len(rows), histograms, total_gradient, total_hessian, value,
            *bounds, allowed_features=selected,
        )
        if split.gain != -1:
            missing_left = split.missing_go_to_left
            if split.is_categorical:
                # The C struct only initializes this field if training saw missing
                # values. The bitset is the source of truth for categorical routing.
                missing_left = (
                    int(split.left_cat_bitset[(n_bins - 1) // 32])
                    >> ((n_bins - 1) % 32)
                ) & 1
            expected.append([
                split.feature_idx, split.bin_idx, int(missing_left),
                split.n_samples_left, split.n_samples_right, int(split.is_categorical),
                *bits64([
                    split.gain, split.sum_gradient_left, split.sum_hessian_left,
                    split.sum_gradient_right, split.sum_hessian_right,
                    split.value_left, split.value_right,
                ]),
            ])
            if split.is_categorical:
                expected.extend([int(word)] for word in split.left_cat_bitset)
            left, right, boundary = splitter.split_indices(split, rows.copy())
            assert boundary == split.n_samples_left
            expected.extend([int(row)] for row in np.concatenate([left, right]))
    if len(expected) == 1:
        expected.append("no split")

    fields = [
        n_rows, n_features, n_bins, len(selected), len(rows), int(constant),
        bits64(l2), bits64(min_hessian), min_leaf, bits64(min_gain),
        *bits64(bounds), *bits64([total_gradient, total_hessian]),
        *selected, *rows, *np.column_stack([counts, missing, categorical, monotonic]).ravel(),
        *X.ravel(order="F"), *gradients.view(np.int32),
    ]
    if not constant:
        fields.extend(hessians.view(np.int32))
    return " ".join(map(str, fields)), expected


def targeted_cases():
    yield "missing left", make_case([[0], [7], [1], [0], [7], [1]], [2, 2, -2, 2, 2, -2], [2])
    yield "missing right", make_case([[0], [7], [1], [0], [7], [1]], [2, -2, -2, 2, -2, -2], [2])
    yield "missing only split", make_case([[7], [0], [7], [0]], [2, -2, 2, -2], [1])
    yield "all missing", make_case([[7], [7]], [-3, 3], [1])
    yield "empty selection", make_case([[0], [1]], [-1, 1], [2], selected=[])
    X = [[0, 0], [1, 1], [0, 0], [1, 1]]
    gradients = [4, -4, 4, -4]
    yield "strict feature tie", make_case(X, gradients, [2, 2], selected=[1, 0])
    yield "negative monotonic rejects", make_case(X, gradients, [2, 2], selected=[1], monotonic=[0, -1])
    yield "clipped monotonic gain", make_case(X, gradients, [2, 2], monotonic=[0, 1], bounds=(-1, 1))
    yield "strict min gain", make_case(X, gradients, [2, 2], bounds=(-1, 1), min_gain=16)
    yield "min hessian", make_case(X, gradients, [2, 2], hessians=[0.01, 1, 0.02, 1], min_hessian=0.04)
    yield "min samples", make_case(X, gradients, [2, 2], min_leaf=3)
    yield "L2 shrinkage", make_case(X, gradients, [2, 2], l2=8.0)
    yield "no Hessian mass", make_case(X, gradients, [2, 2], hessians=[0, 0, 0, 0])
    yield "global missing away from node", make_case(
        [[7], [0], [7], [1], [1], [0]], [0, 2, 0, -2, -2, 2],
        [2], rows=[3, 5, 4, 1], threads=3,
    )
    yield "category support", make_case(
        np.repeat([0, 1, 2], [12, 9, 6])[:, None],
        np.repeat([4, -8, 20], [12, 9, 6]), [3], categorical=[1],
    )
    yield "high bitset words and missing", make_case(
        np.repeat([0, 31, 32, 63, 127, 254, 255], [12, 12, 12, 12, 7, 12, 12])[:, None],
        np.repeat([2, -1, 4, -5, -80, 7, 6], [12, 12, 12, 12, 7, 12, 12]),
        [255], n_bins=256, categorical=[1], threads=3,
    )
    sizes = [10, 10, 10, 10, 10, 7]
    yield "tied category scores", make_case(
        np.repeat([0, 1, 2, 3, 4, 255], sizes)[:, None],
        np.repeat([3, 3, 3, 3, -3, -50], sizes),
        [5], n_bins=256, categorical=[1],
    )


def randomized_cases():
    rng = np.random.default_rng(72)
    for i in range(160):
        n_rows = 127 if i % 2 else 257
        X = rng.integers(0, 7, size=(n_rows, 5), dtype=np.uint8)
        is_categorical = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        X[::7, 0] = 7
        X[::5, 1] = 7
        X[:, 2] = rng.integers(0, 3, size=n_rows)
        X[:, 4] = X[:, 2]
        signal = i % 4
        gradients = rng.normal(size=n_rows) + 3 * np.cos(X[:, signal])
        hessians = rng.random(n_rows, dtype=np.float32) if i % 3 else None
        if hessians is not None:
            hessians[::9] = 0
        rows = rng.permutation(n_rows)[:n_rows // 2] if i % 2 else None
        selected = rng.permutation(5)[: 1 + i % 5]
        yield f"randomized {i}", make_case(
            X, gradients, [7, 7, 3, 7, 3], hessians=hessians,
            categorical=is_categorical, monotonic=[i % 3 - 1, 0, 0, 0, 1],
            rows=rows, selected=selected, l2=i % 7 / 3, min_leaf=1 + i % 13,
            min_gain=i % 4 / 3, bounds=(-1.3, 0.7) if i % 4 else (-np.inf, np.inf),
        )


def main():
    cases = [*targeted_cases(), *randomized_cases()]
    fixture = BUILDS / "hgb_splitting_fixtures.txt"
    fixture.write_text("".join(record + "\n" for _, (record, _) in cases))
    lines = subprocess.check_output([BUILDS / "hgb_splitting_driver", fixture], text=True).splitlines()
    offset = 0
    splits = 0
    category_splits = 0
    rows = 0
    for name, (_, expected) in cases:
        actual = [
            line if line == "no split" else list(map(int, line.split()))
            for line in lines[offset : offset + len(expected)]
        ]
        if actual != expected:
            for i, (got, want) in enumerate(zip(actual, expected)):
                if got != want:
                    raise AssertionError(f"{name}, record {i}: {got} != {want}")
            raise AssertionError(f"{name}: expected {len(expected)} records, got {len(actual)}")
        offset += len(expected)
        if expected[1] != "no split":
            splits += 1
            category_splits += expected[1][5]
            rows += sum(expected[1][3:5])
    assert offset == len(lines)
    print(f"PASS: {len(cases)} cases, {splits} exact splits ({category_splits} categorical)"
          f" and {rows:,} row positions")


if __name__ == "__main__":
    main()
