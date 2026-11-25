"""FrozenEstimator around production estimators and inside native pipelines.

Mirrors sklearn usage: a pre-fitted step keeps its learned state when the
surrounding Pipeline is fitted or cloned, and attribute forwarding exposes the
inner estimator's feature metadata.
"""
import numojo as nm
from std.testing import assert_equal, assert_false, assert_true, assert_raises
from moxie.base import clone, BaseEstimator, ClassifierMixin
from moxie._feature_metadata import InputFeatureCount
from moxie.frozen import FrozenEstimator
from moxie.discriminant_analysis import LinearDiscriminantAnalysis
from moxie.preprocessing import MinMaxScaler, StandardScaler
from moxie.pipeline import Pipeline, FeatureUnion, make_pipeline, union_step
from moxie.compose import ColumnTransformer, ColumnEntry
from moxie.model_selection import (
    DenseSupervisedAdapter, ErrorPolicy, GridSearchCV, IndexSplit, KFold,
    ParameterGrid, RandomizedSearchCV, cross_val_score, cross_validate,
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


def input_width[E: InputFeatureCount](estimator: E) raises -> Int:
    return estimator.get_n_features_in()


def frozen_composition_width() raises:
    var X = dataset(26, 0.0)
    var y = labels(26)
    var wider = nm.zeros[DType.float64](nm.Shape(26, 4))
    var pipe = make_pipeline(StandardScaler(), LDA())
    var unfitted_pipe = FrozenEstimator(clone(pipe))
    with assert_raises():
        _ = input_width(unfitted_pipe)
    _ = pipe.fit(X, y)
    var frozen_pipe = FrozenEstimator(pipe^)
    _ = frozen_pipe.fit(wider, y)
    var pipeline_width = input_width(clone(frozen_pipe))
    assert_equal(pipeline_width, 2)

    var union = FeatureUnion(union_step[DType.float64]("scale", StandardScaler()))
    var unfitted_union = FrozenEstimator(clone(union))
    with assert_raises():
        _ = input_width(unfitted_union)
    _ = union.fit(wider)
    _ = union.fit(X)
    var frozen_union = FrozenEstimator(union^)
    _ = frozen_union.fit(wider)
    var union_width = input_width(clone(frozen_union))
    assert_equal(union_width, 2)
    print("composition_width", pipeline_width, union_width)


def frozen_pipeline_steps() raises:
    var X = dataset(26, 0.0)
    var y = labels(26)
    var pipe = make_pipeline(StandardScaler(), LDA())
    var unfitted = FrozenEstimator(clone(pipe))
    assert_false(unfitted.step[0]().__sklearn_is_fitted__())
    assert_false(
        unfitted.named_step[StandardScaler[]]("standardscaler").__sklearn_is_fitted__()
    )
    assert_false(unfitted.step[1]().__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = unfitted.step[1]().predict(X)

    _ = pipe.fit(X, y)
    var frozen = FrozenEstimator(pipe^)
    _ = frozen.fit(dataset(26, 5.0), y)
    var cloned = clone(frozen)
    var nested = FrozenEstimator(cloned.copy())
    assert_equal(len(frozen), 2)
    assert_equal(len(nested), 2)
    assert_true(
        Pointer(to=frozen.step[0]()) == Pointer(to=frozen.estimator().transformer)
    )
    assert_true(Pointer(to=frozen.step[0]()) == Pointer(to=cloned.step[-2]()))
    assert_true(Pointer(to=frozen.step[1]()) == Pointer(to=nested.step[-1]()))

    var prefix_name = String("standardscaler")
    ref prefix = cloned.named_step[StandardScaler[]](prefix_name)
    ref final_step = nested.named_step[LDA]("lineardiscriminantanalysis")
    assert_true(Pointer(to=prefix) == Pointer(to=frozen.step[0]()))
    assert_true(Pointer(to=final_step) == Pointer(to=frozen.step[1]()))
    var named = nested.named_steps()
    var keys = named.keys()
    assert_equal(len(named), 2)
    assert_equal(keys[0], prefix_name)
    assert_equal(keys[1], "lineardiscriminantanalysis")
    assert_true(Pointer(to=named.get[StandardScaler[]](keys[0])) == Pointer(to=prefix))
    assert_true(Pointer(to=named.get[LDA](keys[1])) == Pointer(to=final_step))
    with assert_raises(contains="KeyError"):
        _ = frozen.named_step[StandardScaler[]]("missing")
    with assert_raises(contains="TypeError"):
        _ = frozen.named_step[StandardScaler[]]("lineardiscriminantanalysis")

    var scaled = prefix.transform(X)
    var predictions = final_step.predict(scaled)
    assert_equal(predictions, frozen.predict(X))
    print("composition_step_shape", scaled.shape[0], scaled.shape[1])
    for i in range(scaled.size):
        print("composition_step_scaled", scaled.item(i))
    for value in predictions:
        print("composition_step_prediction", value)

    var prefix_slice = nested.slice[stop=-1]()
    var final_slice = cloned.slice[-1, 99]()
    var full_slice = clone(frozen).slice[-99, 99]()
    var empty_slice = frozen.slice[2, 0]()
    assert_equal(len(prefix_slice), 1)
    assert_equal(len(final_slice), 1)
    assert_equal(len(full_slice), 2)
    assert_equal(len(empty_slice), 0)
    assert_true(empty_slice.__sklearn_is_fitted__())
    assert_false(unfitted.slice[stop=1]().__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = unfitted.slice[stop=1]().transform(X)
    assert_true(Pointer(to=prefix_slice.step[-1]()) == Pointer(to=prefix))
    assert_true(Pointer(to=final_slice.named_step[LDA]("lineardiscriminantanalysis")) == Pointer(to=final_step))
    assert_true(Pointer(to=full_slice.step[-1]()) == Pointer(to=final_step))
    with assert_raises(contains="KeyError"):
        _ = prefix_slice.named_step[LDA]("lineardiscriminantanalysis")
    with assert_raises(contains="KeyError"):
        _ = empty_slice.named_step[StandardScaler[]](prefix_name)
    var sliced_scaled = prefix_slice.transform(X)
    var sliced_predictions = final_slice.predict(sliced_scaled)
    var sliced_probabilities = final_slice.predict_proba(sliced_scaled)
    assert_same(sliced_scaled, scaled)
    assert_equal(sliced_predictions, full_slice.predict(X))
    assert_same(sliced_probabilities, full_slice.predict_proba(X))
    _ = frozen.set_params(estimator=make_pipeline(StandardScaler(), LDA()))
    assert_false(frozen.__sklearn_is_fitted__())
    assert_equal(full_slice.predict(X), sliced_predictions)
    print("composition_slice_lengths", len(prefix_slice), len(final_slice), len(full_slice), len(empty_slice))
    for i in range(sliced_scaled.size):
        print("composition_slice_scaled", sliced_scaled.item(i))
    for value in sliced_predictions:
        print("composition_slice_prediction", value)
    for i in range(sliced_probabilities.size):
        print("composition_slice_probability", sliced_probabilities.item(i))

    var transformers = make_pipeline(StandardScaler(), MinMaxScaler())
    _ = transformers.fit(X)
    var frozen_transformers = FrozenEstimator(transformers^)
    var first = frozen_transformers.step[-2]().transform(X)
    assert_same(
        frozen_transformers.step[1]().transform(first),
        frozen_transformers.slice[]().transform(X),
    )

    var two_scalers = make_pipeline(StandardScaler(), StandardScaler())
    _ = two_scalers.fit(X)
    var frozen_scalers = FrozenEstimator(two_scalers^)
    ref first_scaler = frozen_scalers.named_step[StandardScaler[]]("standardscaler-1")
    ref last_scaler = frozen_scalers.named_step[StandardScaler[]]("standardscaler-2")
    assert_true(Pointer(to=first_scaler) == Pointer(to=frozen_scalers.step[0]()))
    assert_true(Pointer(to=last_scaler) == Pointer(to=frozen_scalers.step[1]()))


def kfold_splits(n: Int) raises -> List[IndexSplit]:
    var cursor = KFold(n_splits=3).split(n)
    var splits = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            return splits^
        splits.append(pair.take())


def check_fitted_cv_and_search[E: ClassifierMixin & Deinitable](
    frozen: FrozenEstimator[E], X: FrozenEstimator[E].Input, y: List[Int], name: String,
) raises:
    comptime assert E.Prediction == List[Int]
    comptime Adapter = DenseSupervisedAdapter[FrozenEstimator[E], List[Int]]
    var adapter = Adapter()
    var original = rebind_var[List[Int]](frozen.predict(X))
    var scores = cross_val_score[Adapter](frozen, X, y, adapter)
    var evaluation = cross_validate[Adapter](
        frozen, X, y, adapter, return_estimator=True,
    )
    assert_equal(scores.ndim, 1)
    assert_equal(scores.size, 5)
    assert_equal(len(evaluation.estimators.value()), 5)
    for fold in range(scores.size):
        assert_equal(scores.item(fold), evaluation.scores.test_scores.item(0, fold, 0))
        ref fitted = evaluation.estimators.value()[fold]
        assert_true(fitted.__sklearn_is_fitted__())
        assert_equal(rebind_var[List[Int]](fitted.predict(X)), original)
        print("frozen_cv", name, "default", fold, scores.item(fold))

    var explicit = cross_val_score[Adapter](
        frozen, X, y, adapter, cv=kfold_splits(len(y)),
    )
    assert_equal(explicit.size, 3)
    for fold in range(explicit.size):
        print("frozen_cv", name, "kfold", fold, explicit.item(fold))

    var grid = Dict[String, List[Float64]]()
    var search = GridSearchCV[Adapter](frozen, grid, adapter)
    var random = RandomizedSearchCV[Adapter](
        frozen, ParameterGrid[Float64](grid), adapter, n_iter=1, random_state=42,
    )
    _ = search.fit(X, y)
    _ = random.fit(X, y)
    assert_equal(search.n_splits_(), 5)
    assert_equal(random.n_splits_(), 5)
    assert_equal(len(search.best_params_()), 0)
    assert_equal(len(random.best_params_()), 0)
    var grid_results = search.cv_results_()
    var random_results = random.cv_results_()
    for fold in range(scores.size):
        assert_equal(grid_results.scores.test_scores.item(0, fold, 0), scores.item(fold))
        assert_equal(random_results.scores.test_scores.item(0, fold, 0), scores.item(fold))
    var grid_predictions = rebind_var[List[Int]](search.predict(X))
    var random_predictions = rebind_var[List[Int]](random.predict(X))
    assert_equal(grid_predictions, original)
    assert_equal(random_predictions, original)
    assert_equal(rebind_var[List[Int]](frozen.predict(X)), original)
    assert_equal(search.best_score_(), random.best_score_())
    print("frozen_search", name, "grid", search.best_score_())
    print("frozen_search", name, "random", random.best_score_())
    print("frozen_refit", name, grid_predictions)


def frozen_cv_and_search() raises:
    var X_train = dataset(20, 0.0)
    var y_train = labels(20)
    var X = dataset(24, 1.5)
    var y = List[Int]()
    for i in range(24):
        y.append((i // 3) % 2)
    comptime Adapter = DenseSupervisedAdapter[FrozenEstimator[LDA], List[Int]]
    with assert_raises(contains="NotFittedError"):
        _ = cross_val_score[Adapter](
            FrozenEstimator(LDA()), X, y, Adapter(), error_score=ErrorPolicy(True, 0),
        )

    var model = LDA()
    _ = model.fit(X_train, y_train)
    check_fitted_cv_and_search(FrozenEstimator(model^), X, y, "lda")

    var pipe = make_pipeline(StandardScaler(), LDA())
    _ = pipe.fit(X_train, y_train)
    check_fitted_cv_and_search(FrozenEstimator(pipe^), X, y, "pipeline")


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


def frozen_column_transformer_lookup() raises:
    var X = dataset(26, 0.0)
    var columns = ColumnTransformer(
        ColumnEntry("scale", StandardScaler(), [0]),
        ColumnEntry("empty", StandardScaler(), List[Int]()),
        remainder=MinMaxScaler(),
    )
    var unfitted = FrozenEstimator(columns.copy())
    with assert_raises(contains="NotFittedError"):
        _ = unfitted.named_transformers_[StandardScaler[]]("scale")
    _ = columns.fit(X)
    var frozen = FrozenEstimator(columns^)
    var cloned = clone(frozen)
    var nested = FrozenEstimator(cloned^)
    var scaler = nested.named_transformers_[StandardScaler[]]("scale")
    var remainder = frozen.named_transformers_[MinMaxScaler[]]("remainder")
    assert_false(frozen.named_transformers_[StandardScaler[]]("empty").__sklearn_is_fitted__())
    with assert_raises(contains="KeyError"):
        _ = frozen.named_transformers_[StandardScaler[]]("missing")
    with assert_raises(contains="KeyError"):
        _ = frozen.named_transformers_[MinMaxScaler[]]("scale")

    var first = nm.zeros[DType.float64](nm.Shape(26, 1))
    var last = nm.zeros[DType.float64](nm.Shape(26, 1))
    for row in range(26):
        first[nm.Item(row, 0)] = X[nm.Item(row, 0)]
        last[nm.Item(row, 0)] = X[nm.Item(row, 1)]
    var scaled = scaler.transform(first)
    var ranged = remainder.transform(last)
    var combined = frozen.transform(X)
    assert_equal(scaled.shape[1], 1)
    assert_equal(ranged.shape[1], 1)
    for row in range(26):
        var scale_value = scaled[nm.Item(row, 0)]
        var range_value = ranged[nm.Item(row, 0)]
        assert_equal(scale_value, combined[nm.Item(row, 0)])
        assert_equal(range_value, combined[nm.Item(row, 1)])
        print("column_lookup", scale_value, range_value)
    _ = scaler.fit(nm.zeros[DType.float64](nm.Shape(26, 1)))
    assert_true(scaler.transform(first)[nm.Item(0, 0)] != scaled[nm.Item(0, 0)])
    assert_same(frozen.transform(X), combined)


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
    frozen_composition_width()
    frozen_pipeline_steps()
    frozen_cv_and_search()
    frozen_union_and_columns()
    frozen_column_transformer_lookup()
    frozen_float32()
    print("Frozen composition contracts passed")
