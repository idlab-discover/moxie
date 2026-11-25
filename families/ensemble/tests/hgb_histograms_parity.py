"""Compare node histograms against pinned scikit-learn, using raw float bits."""
from itertools import product
from pathlib import Path
import subprocess

import numpy as np
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder


BUILDS = Path(__file__).resolve().parents[1] / "builds"


def make_case(rng, n_rows, n_features, n_bins, constant, partitioned, selector):
    X = np.asfortranarray(
        rng.integers(0, n_bins, (n_rows, n_features), dtype=np.uint8)
    )
    X[:, 0] = n_bins - 1  # Missing-only feature with many collisions.
    gradients = (
        rng.standard_normal(n_rows) * np.exp2(rng.integers(-45, 45, n_rows))
    ).astype(np.float32)
    hessians = rng.random(n_rows, dtype=np.float32)
    hessians[::3] = 0
    if constant:
        hessians = np.ones(1, dtype=np.float32)

    root = np.arange(n_rows, dtype=np.uint32)
    parent = rng.permutation(root)[: n_rows // 2] if partitioned else root
    sibling = rng.permutation(parent)[: len(parent) // 2]
    all_features = np.arange(n_features, dtype=np.uint32)
    selected = (all_features, all_features[::-2].copy(), all_features[:0])[selector]

    oracle = HistogramBuilder(X, n_bins, gradients, hessians, constant, 1)
    parent_hist = oracle.compute_histograms_brute(parent)
    sibling_hist = oracle.compute_histograms_brute(sibling, selected)
    parent_expected = parent_hist.copy()
    other_hist = np.asarray(
        oracle.compute_histograms_subtraction(parent_hist, sibling_hist, selected)
    )

    expected = []
    for histogram, features in (
        (parent_expected, all_features),
        (sibling_hist, np.sort(selected)),
        (other_hist, np.sort(selected)),
    ):
        for feature in features:
            column = np.asarray(histogram)[feature]
            expected.extend(zip(
                column["count"].tolist(),
                column["sum_gradients"].view(np.int64).tolist(),
                column["sum_hessians"].view(np.int64).tolist(),
            ))

    fields = [
        n_rows, n_features, n_bins, len(selected), len(sibling), len(parent), int(constant),
        *selected, *sibling, *parent, *X.ravel(order="F"), *gradients.view(np.int32),
    ]
    if not constant:
        fields.extend(hessians.view(np.int32))
    return " ".join(map(str, fields)), expected


def main():
    rng = np.random.default_rng(42)
    cases = [
        make_case(rng, rows, features, bins, constant, partitioned, selector)
        for (rows, features, bins), constant, partitioned, selector in product(
            [(1, 1, 3), (13, 5, 8), (34, 7, 31), (301, 5, 256)],
            (False, True),
            (False, True),
            range(3),
        )
    ]
    fixture = BUILDS / "hgb_histograms_fixtures.txt"
    fixture.write_text("".join(line + "\n" for line, _ in cases))
    output = subprocess.check_output(
        [BUILDS / "hgb_histograms_driver", fixture], text=True
    ).splitlines()
    offset = 0
    for index, (_, expected) in enumerate(cases):
        actual = [
            tuple(map(int, line.split()))
            for line in output[offset : offset + len(expected)]
        ]
        np.testing.assert_array_equal(actual, expected, err_msg=f"case {index}")
        offset += len(expected)
    assert offset == len(output)
    print(f"PASS: {len(cases)} histogram cases, {offset:,} exact bin records")


if __name__ == "__main__":
    main()
