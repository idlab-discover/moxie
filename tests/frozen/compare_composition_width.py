"""Compare frozen composition widths, indexed steps and slices with pinned sklearn."""

from pathlib import Path
import subprocess

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"


def main():
    revision = subprocess.check_output(["git", "-C", SOURCE, "rev-parse", "HEAD"], text=True)
    assert revision.strip() == REVISION, revision
    for module in (
        "pipeline.py", "frozen/_frozen.py", "discriminant_analysis.py",
        "preprocessing/_data.py", "compose/_column_transformer.py",
    ):
        pinned = subprocess.check_output(["git", "-C", SOURCE, "show", f"{REVISION}:sklearn/{module}"])
        assert (Path(sklearn.__file__).parent / module).read_bytes() == pinned, module

    i = np.arange(26)
    y = i % 2
    X = np.column_stack((y * 3 + i % 7 / 4, i % 11 / 3))
    wider = np.zeros((len(i), 4))
    pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    union = FeatureUnion([("scale", StandardScaler())])
    assert not hasattr(FrozenEstimator(pipe), "n_features_in_")
    assert not hasattr(FrozenEstimator(union), "n_features_in_")
    frozen_pipe = FrozenEstimator(pipe.fit(X, y))
    frozen_union = FrozenEstimator(union.fit(wider).fit(X))
    frozen_pipe.fit(wider, y)
    frozen_union.fit(wider, None)
    expected = [clone(frozen_pipe).n_features_in_, clone(frozen_union).n_features_in_]

    with np.testing.assert_raises(IndexError):
        frozen_pipe[2]
    assert frozen_pipe[0] is clone(frozen_pipe)[-2]
    assert frozen_pipe[1] is FrozenEstimator(frozen_pipe)[-1]
    with np.testing.assert_raises(KeyError):
        frozen_pipe["missing"]
    named_prefix = clone(frozen_pipe)["standardscaler"]
    named_final = FrozenEstimator(frozen_pipe)["lineardiscriminantanalysis"]
    assert named_prefix is frozen_pipe[0]
    assert named_final is frozen_pipe[1]
    expected_scaled = named_prefix.transform(X)
    expected_predictions = named_final.predict(expected_scaled)

    native = subprocess.check_output([ROOT / "builds/tests/frozen/test_frozen_composition"], text=True)
    [widths] = [line.split()[1:] for line in native.splitlines() if line.startswith("composition_width ")]
    actual = list(map(int, widths))
    assert actual == expected, (actual, expected)
    print(f"Frozen Pipeline and FeatureUnion input widths match sklearn exactly: {actual}")

    def values(key, dtype):
        return np.array(
            [line.split()[1:] for line in native.splitlines() if line.startswith(key + " ")],
            dtype=dtype,
        )

    [shape] = values("composition_step_shape", np.int64)
    np.testing.assert_array_equal(shape, expected_scaled.shape)
    scaled = values("composition_step_scaled", np.float64).reshape(shape)
    predictions = values("composition_step_prediction", np.int64).ravel()
    np.testing.assert_allclose(scaled, expected_scaled, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(predictions, expected_predictions)
    print(f"Frozen pipeline steps matched sklearn with max transform error "
          f"{abs(scaled - expected_scaled).max():.3g} and {predictions.size} exact predictions")

    prefix_slice, final_slice = FrozenEstimator(frozen_pipe)[:-1], clone(frozen_pipe)[-1:99]
    full_slice, empty_slice = frozen_pipe[-99:99], frozen_pipe[2:0]
    assert prefix_slice[-1] is named_prefix
    assert final_slice["lineardiscriminantanalysis"] is named_final
    assert full_slice[-1] is named_final
    assert empty_slice.__sklearn_is_fitted__()
    with np.testing.assert_raises(KeyError):
        prefix_slice["lineardiscriminantanalysis"]
    with np.testing.assert_raises(ValueError):
        frozen_pipe[::2]
    [slice_lengths] = values("composition_slice_lengths", np.int64)
    np.testing.assert_array_equal(slice_lengths, list(map(len, (prefix_slice, final_slice, full_slice, empty_slice))))
    sliced_scaled = values("composition_slice_scaled", np.float64).reshape(shape)
    np.testing.assert_allclose(sliced_scaled, prefix_slice.transform(X), rtol=0, atol=1e-14)
    sliced_predictions = values("composition_slice_prediction", np.int64).ravel()
    np.testing.assert_array_equal(sliced_predictions, final_slice.predict(prefix_slice.transform(X)))
    np.testing.assert_array_equal(sliced_predictions, full_slice.predict(X))
    expected_probabilities = full_slice.predict_proba(X)
    sliced_probabilities = values("composition_slice_probability", np.float64).reshape(expected_probabilities.shape)
    np.testing.assert_allclose(sliced_probabilities, expected_probabilities, rtol=0, atol=1e-12)
    print(f"Frozen pipeline slices matched {sliced_scaled.size} transformed values "
          f"within 1e-14, {sliced_predictions.size} exact predictions and "
          f"{sliced_probabilities.size} probabilities within 1e-12")

    columns = ColumnTransformer(
        [("scale", StandardScaler(), [0]), ("empty", StandardScaler(), [])],
        remainder=MinMaxScaler(),
    )
    with np.testing.assert_raises(TypeError):
        FrozenEstimator(columns)["scale"]
    frozen_columns = FrozenEstimator(columns.fit(X))
    nested_columns = FrozenEstimator(clone(frozen_columns))
    with np.testing.assert_raises(KeyError):
        frozen_columns["missing"]
    assert not hasattr(frozen_columns["empty"], "n_features_in_")
    expected_columns = np.column_stack((
        nested_columns["scale"].transform(X[:, :1]),
        frozen_columns["remainder"].transform(X[:, 1:]),
    ))
    indexed_columns = values("column_lookup", np.float64)
    np.testing.assert_allclose(indexed_columns, expected_columns, rtol=0, atol=1e-14)
    print(f"Frozen column lookups matched {indexed_columns.size} sklearn values with max error "
          f"{abs(indexed_columns - expected_columns).max():.3g}")


if __name__ == "__main__":
    main()
