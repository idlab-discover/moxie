"""Compare the native bin mapper with pinned scikit-learn thresholds and bytes."""

from pathlib import Path
import subprocess

import numpy as np
import sklearn
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper


ROOT = Path(__file__).resolve().parents[1]


def cases():
    rng = np.random.RandomState(11)
    regular = np.arange(1200, dtype=np.float64).reshape(400, 3) / 7
    repeated = rng.randint(-5, 8, size=(793, 4)).astype(np.float64)
    repeated[::7, 0] = np.nan
    repeated[::13, 2] = np.nan
    weights = rng.randint(0, 9, size=len(repeated)).astype(np.float64)
    categories = [[0, 1, 31, 32, 127, 254], None, [], [2, 7, 9]]
    mixed = np.array(
        [[0, -1, np.nan, 9], [31, np.nan, np.nan, -1], [254, 2, -1, 7], [32, 3, -2, 2]]
    )
    query = np.array([[1, 0, np.nan, 8], [np.nan, 2, -1, 2], [127, 9, 0, 10]])
    yield "categorical", mixed, query, 16, None, None, categories

    edges = np.array(
        [
            [-np.inf, -np.inf, 0, 1e300],
            [0, np.inf, 1, 1.1e300],
            [1, -np.inf, 2, 1.2e300],
            [np.inf, np.inf, 3, np.inf],
            [np.nan, np.nan, 4, np.nan],
        ]
    )
    for n_bins in (3, 4, 256):
        yield "nonfinite", edges, edges, n_bins, None, None, [None] * edges.shape[1]
    for n_bins in (3, 4, 7, 19, 128, 256):
        yield "uniform", regular, regular[::-5], n_bins, None, None, [None] * 3
        yield "uniform_sampled", regular, regular[::-5], n_bins, 113, None, [None] * 3
        yield "repeated", repeated, repeated[::11], n_bins, None, None, [None] * 4
        yield "weighted", repeated, repeated[::11], n_bins, None, weights, [None] * 4
        yield "weighted_sampled", repeated, repeated[::11], n_bins, 349, weights, [None] * 4
        uneven = rng.lognormal(0, 4, len(regular))
        uneven[::9] = 0
        yield "continuous_weighted", regular, regular[::9], n_bins, None, uneven, [None] * 3
        signals = rng.normal(size=(n_bins * 3 + 1, 3))
        yield "asymmetric_quantiles", signals, signals[::7], n_bins, None, None, [None] * 3
    # Integer ranks, near-integer ranks, singletons, and more bins than observations.
    for n_rows in range(1, 40):
        signals = rng.normal(size=(n_rows, 2))
        n_bins = max(3, 1 + n_rows // 2)
        yield "short_quantiles", signals, signals, n_bins, None, None, [None, None]


def fixture(data, query, n_bins, limit, weights, domains):
    fields = [*data.shape, len(query), n_bins, limit or 0, int(weights is not None)]
    for array in (data, query, weights):
        if array is not None:
            fields.extend(np.asarray(array, dtype=np.float64).ravel().view(np.int64))
    for domain in domains:
        fields.append(-1 if domain is None else len(domain))
        if domain is not None:
            fields.extend(domain)
    return " ".join(map(str, fields))


def ints(group):
    return [int(token) for token in group.split()]


def main():
    assert Path(sklearn.__file__).resolve().is_relative_to(ROOT / "builds/oracle-env")
    fixtures = []
    expected = []
    for name, data, query, n_bins, limit, weights, domains in cases():
        fixtures.append(fixture(data, query, n_bins, limit, weights, domains))
        with np.errstate(invalid="ignore", over="ignore"):
            mapper = _BinMapper(
                n_bins=n_bins,
                subsample=limit,
                is_categorical=[d is not None for d in domains],
                known_categories=[None if d is None else np.asarray(d, dtype=np.float64) for d in domains],
                random_state=42,
                n_threads=1,
            ).fit(data, sample_weight=weights)
        expected.append((f"{name}, bins={n_bins}, rows={len(data)}", mapper, mapper.transform(query)))

    path = ROOT / "builds/hgb_binning_fixtures.txt"
    path.write_text("\n".join(fixtures) + "\n")
    result = subprocess.check_output([ROOT / "builds/hgb_binning_driver", path], text=True)
    thresholds_checked = 0
    codes_checked = 0
    for line, (context, mapper, binned) in zip(result.splitlines(), expected, strict=True):
        missing, *columns, codes, feature_map, bitsets = line.split("|")
        assert int(missing) == mapper.missing_values_bin_idx_, context
        for column, count, thresholds in zip(
            columns, mapper.n_bins_non_missing_, mapper.bin_thresholds_, strict=True
        ):
            actual_count, *raw = ints(column)
            assert actual_count == count, context
            np.testing.assert_array_equal(np.asarray(raw, dtype=np.int64).view(np.float64), thresholds, err_msg=context)
            thresholds_checked += len(raw)
        np.testing.assert_array_equal(ints(codes), binned.ravel(), err_msg=context)
        codes_checked += binned.size
        expected_bits, mapping = mapper.make_known_categories_bitsets()
        np.testing.assert_array_equal(ints(feature_map), mapping, err_msg=context)
        np.testing.assert_array_equal(ints(bitsets), expected_bits.ravel(), err_msg=context)
    print(f"PASS: {len(expected)} HGB binning cases, {thresholds_checked} exact thresholds, {codes_checked} exact bin codes")


if __name__ == "__main__":
    main()
