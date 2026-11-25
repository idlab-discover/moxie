import numojo as nm
from std.math import nan
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from moxie import clone
from moxie.ensemble import RandomForestRegressor
from moxie.ensemble._sampling import MaxSamples
from moxie.impute import IterativeImputer

comptime Forest = RandomForestRegressor[DType.float64]
comptime Array = nm.NDArray[DType.float64]


def fixture(start: Int, rows: Int, shift: Int = 0) raises -> Array:
    var data = nm.zeros[DType.float64](nm.Shape(rows, 3))
    for row in range(rows):
        var i = start + row
        var x = Float64(i + shift)
        data.unsafe_set(row * 3, x)
        data.unsafe_set(row * 3 + 1, x * x / 32)
        data.unsafe_set(row * 3 + 2, x * x * x / 1024)
        for j in range(3):
            if i % 7 == j and (start != 0 or j < 2):
                data.unsafe_set(row * 3 + j, nan[DType.float64]())
    return data^


def emit(X: Array) raises:
    print(X.shape[0], X.shape[1])
    for i in range(X.size):
        print(X.unsafe_get(i))


def main() raises:
    var regressor = Forest(
        n_estimators=5, max_depth=10, bootstrap=True,
        max_samples=MaxSamples(0.5), n_jobs=2, random_state=0,
    )
    var model = IterativeImputer[DType.float64, Forest](
        regressor^, random_state=0, max_iter=40, tol=1e-1,
    )
    var X = fixture(0, 32)
    var query = fixture(32, 7)
    with assert_raises():
        _ = model.transform(query)
    var fitted = model.fit_transform(X)
    var replay = model.transform(query)
    print(model.n_iter_, model.n_features_with_missing_, len(model.imputation_sequence_))
    for step in model.imputation_sequence_:
        print(step.feat_idx, step.estimator.n_features_in_, len(step.estimator))
        for feature in step.neighbor_feat_idx:
            print(feature)
        for tree in step.estimator.estimators_():
            print(tree.get_params().random_state.value())
    emit(fitted)
    emit(replay)

    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    with assert_raises():
        _ = fresh.transform(query)
    var snapshot = model.copy()
    _ = model.fit(fixture(0, 32, shift=512))
    var changed = model.transform(query)
    var saved = snapshot.transform(query)
    assert_true(changed.unsafe_get(9) != replay.unsafe_get(9))
    for i in range(replay.size):
        assert_equal(saved.unsafe_get(i), replay.unsafe_get(i))
