"""FrozenEstimator around production estimators and inside native pipelines.

Mirrors sklearn usage: a pre-fitted step keeps its learned state when the
surrounding Pipeline is fitted or cloned, and attribute forwarding exposes the
inner estimator's feature metadata.
"""
import numojo as nm
from std.testing import assert_equal, assert_false, assert_true, assert_raises
from moxie.base import clone, BaseEstimator
from moxie._feature_metadata import InputFeatureCount
from moxie.frozen import FrozenEstimator
from moxie.discriminant_analysis import LinearDiscriminantAnalysis
from moxie.preprocessing import StandardScaler
from moxie.pipeline import Pipeline

comptime LDA = LinearDiscriminantAnalysis[DType.float64, Int]
comptime Dense = nm.NDArray[DType.float64]


struct WidthProbe(BaseEstimator, InputFeatureCount):
    """Synthetic dispatch probe: ColumnTransformer is the production conformer."""
    comptime Params = Int
    var width: Int

    def __init__(out self, width: Int):
        self.width = width

    def get_params(self, deep: Bool = True) raises -> Int:
        return self.width

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Int
    ) raises -> ref[self] Self:
        self.width = params
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.width)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.width >= 0

    def get_n_features_in(self) raises -> Int:
        if self.width < 0:
            raise Error("NotFittedError: probe is not fitted")
        return self.width


def dataset(n: Int, shift: Float64) raises -> Dense:
    var X = Dense(nm.Shape(n, 2))
    for i in range(n):
        X[nm.Item(i, 0)] = Float64(i % 2) * 3 + Float64(i % 7) / 4 + shift
        X[nm.Item(i, 1)] = Float64(i % 11) / 3 - 2 * shift
    return X^


def labels(n: Int) -> List[Int]:
    var y = List[Int]()
    for i in range(n):
        y.append(i % 2)
    return y^


def assert_same(a: Dense, b: Dense) raises:
    assert_equal(a.ndim, b.ndim)
    for d in range(a.ndim):
        assert_equal(a.shape[d], b.shape[d])
    for i in range(a.size):
        assert_equal(a.item(i), b.item(i))


def frozen_classifier() raises:
    var X = dataset(24, 0.0)
    var y = labels(24)
    var model = LDA()
    _ = model.fit(X, y)
    var reference = LDA()  # LDA is not Copyable; its fit is deterministic.
    _ = reference.fit(X, y)
    var frozen = FrozenEstimator(model^)
    # fit is a no-op even with different data and labels.
    var other_y = List[Int]()
    for i in range(24):
        other_y.append(i % 3)
    _ = frozen.fit(dataset(24, 5.0), other_y)
    assert_equal(frozen.estimator().classes_, reference.classes_)
    assert_same(frozen.predict_proba(X), reference.predict_proba(X))
    assert_same(frozen.predict_log_proba(X), reference.predict_log_proba(X))
    assert_same(frozen.decision_function(X), reference.decision_function(X))
    assert_same(frozen.transform(X), reference.transform(X))
    assert_equal(frozen.predict(X), reference.predict(X))
    assert_equal(frozen.score(X, y), reference.score(X, y))
    var cloned = clone(frozen)
    assert_true(cloned.__sklearn_is_fitted__())
    assert_same(cloned.estimator().coef_, reference.coef_)
    assert_same(cloned.predict_proba(X), reference.predict_proba(X))
    var unfitted = FrozenEstimator(LDA())
    with assert_raises(contains="NotFittedError"):
        _ = unfitted.fit(X, y)


def frozen_pipeline_prefix() raises:
    var X_train = dataset(30, 0.0)
    var X_new = dataset(30, 4.0)
    var y = labels(30)
    var scaler = StandardScaler()
    _ = scaler.fit(X_train)
    var reference = scaler.copy()
    var pipe = Pipeline(FrozenEstimator(scaler^), LDA())
    _ = pipe.fit(X_new, y)
    # The frozen prefix kept its training-set statistics.
    assert_same(pipe.transformer.estimator().mean_.value(), reference.mean_.value())
    var expected = LDA()
    _ = expected.fit(reference.transform(X_new), y)
    assert_same(pipe.predict_proba(X_new), expected.predict_proba(reference.transform(X_new)))
    # Pipeline clone keeps the frozen step fitted and unfits the final estimator.
    var fresh = clone(pipe)
    assert_true(fresh.transformer.__sklearn_is_fitted__())
    assert_false(fresh.__sklearn_is_fitted__())
    _ = fresh.fit(X_new, y)
    assert_same(fresh.predict_proba(X_new), pipe.predict_proba(X_new))
    # Feature metadata is forwarded like sklearn's attribute delegation.
    var names: List[String] = ["height", "width"]
    assert_equal(pipe.transformer.get_feature_names_out(names.copy()), names)
    assert_equal(pipe.transformer.get_feature_names_out(),
                 reference.get_feature_names_out())
    var unfitted = FrozenEstimator(StandardScaler())
    with assert_raises():
        _ = unfitted.get_feature_names_out()
    assert_equal(FrozenEstimator(WidthProbe(3)).get_n_features_in(), 3)
    with assert_raises(contains="NotFittedError"):
        _ = FrozenEstimator(WidthProbe(-1)).get_n_features_in()


def frozen_pipeline_final() raises:
    var X = dataset(26, 0.0)
    var y = labels(26)
    var model = LDA()
    _ = model.fit(X, y)
    var reference = LDA()
    _ = reference.fit(X, y)
    var pipe = Pipeline(StandardScaler(), FrozenEstimator(model^))
    _ = pipe.fit(dataset(26, 3.0), y)
    # Only the scaler is refitted; the frozen classifier still uses raw-space coefficients.
    assert_same(pipe.estimator.estimator().coef_, reference.coef_)
    var scaled = pipe.transformer.transform(X)
    assert_same(pipe.predict_proba(X), reference.predict_proba(scaled))


def frozen_float32() raises:
    var X = nm.zeros[DType.float32](nm.Shape(6, 2))
    for i in range(6):
        X[nm.Item(i, 0)] = Float32(i)
        X[nm.Item(i, 1)] = Float32(i * i) / 4
    var scaler = StandardScaler[DType.float32]()
    _ = scaler.fit(X)
    var reference = scaler.copy()
    var frozen = FrozenEstimator(scaler^)
    _ = frozen.fit(nm.ones[DType.float32](nm.Shape(3, 2)))
    var out = frozen.transform(X)
    comptime assert out.dtype == DType.float32
    var expected = reference.transform(X)
    for i in range(out.size):
        assert_equal(out.item(i), expected.item(i))
    var back = frozen.inverse_transform(out)
    var expected_back = reference.inverse_transform(expected)
    for i in range(back.size):
        assert_equal(back.item(i), expected_back.item(i))


def main() raises:
    frozen_classifier()
    frozen_pipeline_prefix()
    frozen_pipeline_final()
    frozen_float32()
    print("Frozen composition contracts passed")
