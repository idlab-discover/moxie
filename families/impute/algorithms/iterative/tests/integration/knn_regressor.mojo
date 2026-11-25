import numojo as nm
from std.math import nan
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from moxie import clone
from moxie.impute import IterativeImputer
from moxie.neighbors import KNeighborsRegressor


def fixture[dtype: DType](start: Int, rows: Int, shift: Int = 0) raises -> nm.NDArray[dtype]:
    comptime features = 4
    var result = nm.zeros[dtype](nm.Shape(rows, features))
    for row in range(rows):
        var i = start + row
        for j in range(features):
            var residue = ((i + 3) * (i + 17 * j + 8) * (j + 9)) % 251
            var value = Scalar[dtype](residue) / 8 + Scalar[dtype](shift)
            if i % 4 == (j + 1) % 4 and (start != 0 or j < 3):
                value = nan[dtype]()
            result.unsafe_set(row * features + j, value)
    return result^


def emit[dtype: DType](X: nm.NDArray[dtype]) raises:
    print(X.shape[0], X.shape[1])
    for i in range(X.size):
        print(Float64(X.unsafe_get(i)))


def check[dtype: DType]() raises:
    comptime Regressor = KNeighborsRegressor[dtype]
    var regressor = Regressor(n_neighbors=10)
    var X = fixture[dtype](0, 32)
    var query = fixture[dtype](32, 7)
    var model = IterativeImputer[dtype, Regressor](
        regressor^, random_state=0, max_iter=40, tol=1e-2,
    )
    with assert_raises():
        _ = model.transform(query)
    var train = model.fit_transform(X)
    var held_out = model.transform(query)
    print(model.n_iter_, model.n_features_with_missing_, len(model.imputation_sequence_))
    for step in model.imputation_sequence_:
        print(step.feat_idx, step.estimator.n_features_in_, step.estimator.n_samples_fit_)
    emit(train)
    emit(held_out)

    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    with assert_raises():
        _ = fresh.transform(query)
    var snapshot = model.copy()
    _ = model.fit(fixture[dtype](0, 32, shift=512))
    var changed = model.transform(query)
    var replay = snapshot.transform(query)
    assert_true(changed.unsafe_get(3) > 512)
    for i in range(replay.size):
        assert_equal(replay.unsafe_get(i), held_out.unsafe_get(i))


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
