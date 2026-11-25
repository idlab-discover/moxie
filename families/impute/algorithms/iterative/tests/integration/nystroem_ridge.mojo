import numojo as nm
from std.math import isnan, nan
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from moxie import clone
from moxie.impute import IterativeImputer
from moxie.kernel_approximation import Nystroem
from moxie.linear_model import Ridge
from moxie.pipeline import make_pipeline


def fixture[dtype: DType](start: Int, rows: Int) raises -> nm.NDArray[dtype]:
    comptime width = 3
    var result = nm.zeros[dtype](nm.Shape(rows, width))
    for row in range(rows):
        var i = start + row
        for j in range(width):
            var value = Scalar[dtype](((i + 3) * (i + 17 * j + 8)) % 67) / 4
            if i % 5 == j and (start != 0 or j < 2):
                value = nan[dtype]()
            result.unsafe_set(row * width + j, value)
    return result^


def emit[dtype: DType](X: nm.NDArray[dtype]) raises:
    print(X.shape[0], X.shape[1])
    for i in range(X.size):
        print(Float64(X.unsafe_get(i)))


def check[dtype: DType]() raises:
    var regressor = make_pipeline(
        Nystroem[dtype](kernel="polynomial", degree=2.0, random_state=0),
        Ridge[dtype](alpha=1e4),
    )
    var model = IterativeImputer[dtype, type_of(regressor)](
        regressor^, random_state=0, max_iter=40, tol=1e-1,
    )
    comptime rows = 6 if dtype == DType.float32 else 17
    var X = fixture[dtype](0, rows)
    var query = fixture[dtype](17, 7)
    with assert_raises():
        _ = model.transform(query)
    var train = model.fit_transform(X)
    var held_out = model.transform(query)
    print(model.n_iter_, model.n_features_with_missing_, len(model.imputation_sequence_))
    for step in model.imputation_sequence_:
        ref basis_indices = step.estimator.transformer.component_indices_
        print(step.feat_idx, len(step.neighbor_feat_idx), basis_indices.size)
        for neighbor in step.neighbor_feat_idx:
            print(neighbor)
        for i in range(basis_indices.size):
            print(basis_indices.unsafe_get(i))
    emit(train)
    emit(held_out)

    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_false(fresh.get_params().estimator.value().__sklearn_is_fitted__())
    with assert_raises():
        _ = fresh.transform(query)

    var snapshot = model.copy()
    # Mutating a nested fitted array must affect only its owning copy.
    var basis = model.imputation_sequence_[0].estimator.transformer.components_.view()
    var original = basis.unsafe_get(0)
    basis.unsafe_set(0, original + 1)
    assert_equal(
        snapshot.imputation_sequence_[0].estimator.transformer.components_.unsafe_get(0),
        original,
    )
    for i in range(X.size):
        if not isnan(X.unsafe_get(i)):
            X.unsafe_set(i, X.unsafe_get(i) + 16)
    _ = model.fit(X)
    var changed = model.transform(query)
    var replay = snapshot.transform(query)
    assert_true(changed.unsafe_get(9) > 16)
    for i in range(replay.size):
        assert_equal(replay.unsafe_get(i), held_out.unsafe_get(i))


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
