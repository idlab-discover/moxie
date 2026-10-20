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
from moxie.pipeline import Pipeline, FeatureUnion, union_step
from moxie.compose import ColumnTransformer, ColumnEntry
from moxie.model_selection import (
    CVAdapter, TypedScorer, IndexArray, IndexSplit, KFold, cross_val_score,
)

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


comptime Frozen = FrozenEstimator[LinearDiscriminantAnalysis[DType.float64, Int]]


def take_rows(X: nm.NDArray[DType.float64], rows: IndexArray) raises -> nm.NDArray[DType.float64]:
    var out = nm.NDArray[DType.float64](nm.Shape(rows.size, X.shape[1]))
    for i in range(rows.size):
        for j in range(X.shape[1]):
            out[nm.Item(i, j)] = X[nm.Item(Int(rows[nm.Item(i)]), j)]
    return out^


struct FrozenAccuracy(TypedScorer):
    comptime Estimator = Frozen
    comptime Input = nm.NDArray[DType.float64]
    comptime Target = List[Int]
    comptime Metadata = NoneType

    def __init__(out self):
        pass

    def score(self, estimator: Frozen, X: Self.Input, y: List[Int], metadata: NoneType) raises -> Float64:
        return estimator.score(X, y)


@fieldwise_init
struct FrozenCV(CVAdapter):
    comptime Scorer = FrozenAccuracy

    def n_samples(self, X: nm.NDArray[DType.float64], y: List[Int]) raises -> Int:
        return len(y)

    def input_rows(self, X: nm.NDArray[DType.float64], rows: IndexArray, train: IndexArray) raises -> nm.NDArray[DType.float64]:
        return take_rows(X, rows)

    def target_rows(self, y: List[Int], rows: IndexArray) raises -> List[Int]:
        var out = List[Int]()
        for i in range(rows.size):
            out.append(y[Int(rows[nm.Item(i)])])
        return out^

    def metadata_rows(self, metadata: NoneType, rows: IndexArray) raises -> NoneType:
        return None

    def fit(self, mut estimator: Frozen, X: nm.NDArray[DType.float64], y: List[Int], metadata: NoneType) raises:
        _ = estimator.fit(X, y)


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


def kfold_splits(n: Int) raises -> List[IndexSplit]:
    var cursor = KFold(n_splits=3).split(n)
    var splits = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.value().copy())
    return splits^


def frozen_cross_validation() raises:
    # sklearn: cross_val_score(FrozenEstimator(clf), X, y) clones the frozen
    # wrapper (keeping clf) and its fit is a no-op, so each fold scores clf.
    var X_train = dataset(20, 0.0)
    var y_train = labels(20)
    var X = dataset(24, 1.5)
    var y = List[Int]()
    for i in range(24):
        y.append((i // 3) % 2)
    var model = LDA()
    _ = model.fit(X_train, y_train)
    var frozen = FrozenEstimator(model^)
    var splits = kfold_splits(24)
    var scores = cross_val_score[FrozenCV](frozen, X, y, FrozenCV(), FrozenAccuracy(), kfold_splits(24), None, None)
    assert_equal(scores.size, 3)
    var adapter = FrozenCV()
    for k in range(3):
        var test = splits[k].test.copy()
        var expected = frozen.score(adapter.input_rows(X, test, splits[k].train), adapter.target_rows(y, test))
        assert_equal(scores.item(k), expected)
    var original = frozen.predict_proba(X_train)
    assert_same(frozen.predict_proba(X_train), original)


def frozen_union_and_columns() raises:
    # FeatureUnion.fit_transform and ColumnTransformer.fit_transform call the
    # frozen branch's no-op fit, then transform with the training statistics.
    var X_train = dataset(12, 0.0)
    var X_new = dataset(12, 6.0)
    var scaler = StandardScaler()
    _ = scaler.fit(X_train)
    var expected = scaler.transform(X_new)
    var union = FeatureUnion(union_step[DType.float64]("frozen", FrozenEstimator(scaler.copy())))
    assert_same(union.fit_transform(X_new, None), expected)
    var columns = ColumnTransformer(ColumnEntry("frozen", FrozenEstimator(scaler.copy()), [0, 1]))
    assert_same(columns.fit_transform(X_new), expected)
    var names: List[String] = ["a", "b"]
    var expected_names: List[String] = ["frozen__a", "frozen__b"]
    assert_equal(columns.get_feature_names_out(names.copy()), expected_names)
    assert_equal(union.get_feature_names_out(names^), expected_names)


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
    frozen_cross_validation()
    frozen_union_and_columns()
    frozen_float32()
    print("Frozen composition contracts passed")
