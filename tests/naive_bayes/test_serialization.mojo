from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.math import isnan
from std.memory import bitcast
from numojo import NDArrayStrides, Shape, Item, array
from moxie._labels import StringTarget
from moxie._naive_bayes_dense import ClassDomain, dense_from_lists
from moxie.naive_bayes import IntListDomain, NumericDomain, StringDomain
from moxie.naive_bayes_serialization import (
    BernoulliNBSchema, CategoricalNBSchema, ComplementNBSchema, GaussianNBSchema, MultinomialNBSchema,
)
from moxie.serialization import BinaryWriter, dumps, dtype_code, load_model, loads, save_model


def fitted_round_trip[D: ClassDomain, dtype: DType](y: D.Target) raises:
    comptime Schema = GaussianNBSchema[D, dtype]
    var X = dense_from_lists([[1.0, 2.0], [10.0, 3.0], [2.0, 4.0], [7.0, 1.0]])
    var model = Schema.Model(priors=List[Float64]([0.25, 0.75]))
    _ = model.fit(X, y)
    _ = model.set_params(
        priors=Optional[List[Float64]](List[Float64]([0.75, 0.25])),
        var_smoothing=0.125,
    )

    var before = dumps(Schema.encode(model))
    var restored = Schema.decode(loads(before))
    assert_equal(restored.n_features_in_, 2)
    var learned_priors: List[Scalar[dtype]] = [0.25, 0.75]
    assert_equal(restored.class_prior_.tolist(), learned_priors)
    assert_equal(restored.priors.value(), List[Float64]([0.75, 0.25]))
    assert_equal(restored.var_smoothing, 0.125)
    assert_equal(dumps(Schema.encode(restored)), before)
    assert_equal(restored.predict_proba(X).tolist(), model.predict_proba(X).tolist())
    var predicted = restored.predict(X)
    var expected = model.predict(X)
    for i in range(4):
        assert_true(D.equal_at(predicted, expected, i))
    assert_false(restored.__sklearn_clone__().__sklearn_is_fitted__())

    # The next weighted update can use the rebuilt class domain without classes=.
    var weights: List[Float64] = [0.5, 0.25, 0.75, 0.5]
    _ = restored.partial_fit(X, y, sample_weight=weights.copy())
    assert_equal(dumps(Schema.encode(model)), before)
    _ = model.partial_fit(X, y, sample_weight=weights^)
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))


def file_round_trips() raises:
    comptime Schema = GaussianNBSchema[]
    var path = "builds/naive-bayes-gaussian-model.moxie"
    var X = dense_from_lists([[2.0], [10.0], [3.0], [9.0]])
    var y: List[Int] = [1, 2, 1, 2]
    var model = Schema.Model(priors=List[Float64]([0.25, 0.75]), var_smoothing=0.0)
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_false(restored.__sklearn_is_fitted__())
    assert_equal(restored.priors.value(), model.priors.value())
    assert_equal(restored.var_smoothing, 0.0)
    with assert_raises(contains="NotFittedError"):
        _ = restored.predict(X)

    _ = restored.fit(X, y)
    save_model[Schema](restored, path)
    var fitted = load_model[Schema](path)
    assert_true(fitted.__sklearn_is_fitted__())
    assert_equal(fitted.predict(X), y)
    assert_equal(dumps(Schema.encode(fitted)), dumps(Schema.encode(restored)))


def logical_state_and_overflow() raises:
    comptime Schema = GaussianNBSchema[]
    var X = dense_from_lists([[2.0], [10.0], [3.0], [9.0]])
    var y: List[Int] = [1, 2, 1, 2]
    var model = Schema.Model(var_smoothing=0.0)
    _ = model.fit(X, y)
    var buffer = dense_from_lists([[-0.0, 100.0], [42.0, 100.0]])
    model.theta_ = buffer.view_with_layout(Shape(2, 1), NDArrayStrides(-2, 1), 2)
    var restored = Schema.decode(Schema.encode(model))
    assert_equal(restored.theta_.item(0, 0), 42.0)
    assert_equal(bitcast[DType.uint64, 1](restored.theta_.item(1, 0)), UInt64(1) << 63)
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))

    # Finite inputs can overflow moments. Preserve the resulting NaN payloads.
    var extreme = dense_from_lists([[1e200], [10.0], [-1e200], [9.0]])
    _ = model.fit(extreme, y)
    assert_true(isnan(model.epsilon_))
    var bytes = dumps(Schema.encode(model))
    restored = Schema.decode(loads(bytes))
    assert_true(isnan(restored.var_.item(0, 0)))
    assert_equal(dumps(Schema.encode(restored)), bytes)


def malformed_state() raises:
    comptime Schema = GaussianNBSchema[]
    var X = dense_from_lists([[2.0], [10.0], [3.0], [9.0]])
    var y: List[Int] = [-1, 2, -1, 2]
    var model = Schema.Model()
    var unfitted = Schema.encode(model)
    model.n_features_in_ = -1
    with assert_raises(contains="unfitted GaussianNB has learned state"):
        _ = Schema.encode(model)
    _ = model.fit(X, y)
    var archive = Schema.encode(model)
    with assert_raises(contains="dtype mismatch"):
        _ = GaussianNBSchema[dtype=DType.float32].decode(archive)
    with assert_raises(contains="label domain mismatch"):
        _ = GaussianNBSchema[NumericDomain[DType.int64]].decode(archive)

    var bad = archive.copy()
    bad.schema_version = 4
    with assert_raises(contains="unsupported estimator"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.manifest.append(0)
    with assert_raises(contains="trailing data"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.payloads[0].shape = [2, 1]
    with assert_raises(contains="invalid shape for class_count_"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.payloads[0].dtype = dtype_code[DType.int64]()
    with assert_raises(contains="array dtype or shape mismatch"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.payloads[0].name = "unknown"
    with assert_raises(contains="missing array class_count_"):
        _ = Schema.decode(bad)
    unfitted.payloads.append(archive.payloads[0].copy())
    with assert_raises(contains="unexpected GaussianNB arrays"):
        _ = Schema.decode(unfitted)

    bad = archive.copy()
    var negative_count = BinaryWriter()
    negative_count.scalar(Float64(-1))
    negative_count.scalar(Float64(2))
    bad.payloads[0].data = negative_count^.finish()
    with assert_raises(contains="invalid GaussianNB class count"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    var swapped_classes = BinaryWriter()
    swapped_classes.uint(UInt64(dtype_code[DType.float64]()))
    swapped_classes.uint(1)  # IntListDomain
    swapped_classes.scalar(model.var_smoothing)
    swapped_classes.boolean(False)  # no configured priors
    swapped_classes.boolean(True)  # fitted
    swapped_classes.integer(model.n_features_in_)
    swapped_classes.scalar(model.epsilon_)
    swapped_classes.uint(2)
    swapped_classes.integer(2)
    swapped_classes.integer(-1)
    bad.manifest = swapped_classes^.finish()
    with assert_raises(contains="sorted label domain"):
        _ = Schema.decode(bad)

    model.class_prior_[Item(0)] = -0.25
    with assert_raises(contains="invalid GaussianNB learned prior"):
        _ = Schema.encode(model)
    model.class_prior_[Item(0)] = 0.5
    model.classes_ = [2, -1]
    with assert_raises(contains="sorted label domain"):
        _ = Schema.encode(model)


def multinomial_round_trip[D: ClassDomain, dtype: DType](y: D.Target) raises:
    comptime Schema = MultinomialNBSchema[D, dtype]
    var X = dense_from_lists([[2.0, 1.0], [1.0, 9.0], [3.0, 1.0], [1.0, 8.0]])
    var backing = array[DType.float64]([0.25, 9.0, 0.0], [3])
    var alpha = backing.view_with_layout(Shape(2), NDArrayStrides(-2), 2)
    var model = Schema.Model(alpha_values=alpha^, force_alpha=False,
                             class_prior=List[Float64]([0.25, 2.0]))
    var weights: List[Float64] = [0.5, 1.0, 0.25, 0.75]
    _ = model.fit(X, y, sample_weight=weights.copy())
    _ = model.set_params(force_alpha=True, fit_prior=False,
                         class_prior=Optional[List[Float64]](List[Float64]([0.75, 0.25])))
    var before = dumps(Schema.encode(model))
    var restored = Schema.decode(loads(before))
    assert_equal(dumps(Schema.encode(restored)), before)
    assert_equal(restored.alpha_values.value().tolist(), List[Float64]([0.0, 0.25]))
    assert_equal(restored.fit_warnings[0].message, model.fit_warnings[0].message)
    assert_equal(restored.fit_warnings[0].category, model.fit_warnings[0].category)
    assert_equal(restored.predict_proba(X).tolist(), model.predict_proba(X).tolist())
    var expected = model.predict(X)
    var predicted = restored.predict(X)
    for i in range(4):
        assert_true(D.equal_at(expected, predicted, i))
    assert_false(restored.__sklearn_clone__().__sklearn_is_fitted__())

    _ = restored.partial_fit(X, y, sample_weight=weights.copy())
    assert_equal(len(restored.fit_warnings), 0)
    assert_equal(dumps(Schema.encode(model)), before)
    _ = model.partial_fit(X, y, sample_weight=weights^)
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))


def complement_round_trip[D: ClassDomain, dtype: DType](y: D.Target) raises:
    comptime Schema = ComplementNBSchema[D, dtype]
    var X = dense_from_lists([[2.0, 1.0], [1.0, 9.0], [3.0, 1.0], [1.0, 8.0]])
    var backing = array[DType.float64]([0.25, 9.0, 0.0], [3])
    var alpha = backing.view_with_layout(Shape(2), NDArrayStrides(-2), 2)
    var model = Schema.Model(alpha_values=alpha^, force_alpha=False,
                             class_prior=List[Float64]([0.25, 2.0]))
    var weights: List[Float64] = [0.5, 1.0, 0.25, 0.75]
    _ = model.set_params(norm=True)
    _ = model.fit(X, y, sample_weight=weights.copy())
    _ = model.set_params(norm=False)
    _ = model.set_params(force_alpha=True, fit_prior=False,
                         class_prior=Optional[List[Float64]](List[Float64]([0.75, 0.25])))
    var before = dumps(Schema.encode(model))
    var restored = Schema.decode(loads(before))
    assert_equal(dumps(Schema.encode(restored)), before)
    assert_equal(restored.alpha_values.value().tolist(), List[Float64]([0.0, 0.25]))
    assert_equal(restored.fit_warnings[0].message, model.fit_warnings[0].message)
    assert_equal(restored.fit_warnings[0].category, model.fit_warnings[0].category)
    assert_false(restored.norm)
    assert_equal(restored.feature_all_.tolist(), model.feature_all_.tolist())
    assert_equal(restored.predict_proba(X).tolist(), model.predict_proba(X).tolist())
    var expected = model.predict(X)
    var predicted = restored.predict(X)
    for i in range(4):
        assert_true(D.equal_at(expected, predicted, i))
    assert_false(restored.__sklearn_clone__().__sklearn_is_fitted__())

    _ = restored.partial_fit(X, y, sample_weight=weights.copy())
    assert_equal(len(restored.fit_warnings), 0)
    assert_equal(dumps(Schema.encode(model)), before)
    _ = model.partial_fit(X, y, sample_weight=weights^)
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))


def multinomial_file_and_validation() raises:
    comptime Schema = MultinomialNBSchema[]
    var path = "builds/naive-bayes-multinomial-model.moxie"
    var X = dense_from_lists([[2.0, 0.0], [0.0, 0.0]])
    var y: List[Int] = [-1, 2]
    var model = Schema.Model(alpha=0.0)
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_false(restored.__sklearn_is_fitted__())
    assert_equal(restored.alpha, 0.0)
    with assert_raises(contains="NotFittedError"):
        _ = restored.predict(X)

    _ = model.fit(X, y)
    save_model[Schema](model, path)
    restored = load_model[Schema](path)
    assert_true(isnan(restored.feature_log_prob_.item(1, 0)))
    assert_equal(restored.feature_log_prob_.item(0, 1), -Float64("inf"))
    var archive = Schema.encode(restored)
    assert_equal(dumps(archive), dumps(Schema.encode(model)))

    with assert_raises(contains="unsupported estimator"):
        _ = GaussianNBSchema[].decode(archive)
    with assert_raises(contains="dtype mismatch"):
        _ = MultinomialNBSchema[dtype=DType.float32].decode(archive)
    with assert_raises(contains="label domain mismatch"):
        _ = MultinomialNBSchema[StringDomain].decode(archive)
    var bad = archive.copy()
    bad.manifest.append(0)
    with assert_raises(contains="trailing data"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.payloads[2].shape = [4]
    with assert_raises(contains="invalid shape for feature_count_"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    _ = bad.payloads.pop()
    var shape: List[Int] = [2, 2]
    var values: List[Float64] = [1.0, 0.0, 0.0, 0.0]
    bad.add_array[DType.float64]("feature_log_prob_", shape, values)
    with assert_raises(contains="invalid MultinomialNB feature statistics"):
        _ = Schema.decode(bad)
    model.class_count_[Item(0)] = -1.0
    with assert_raises(contains="invalid MultinomialNB class count"):
        _ = Schema.encode(model)


def complement_file_and_validation() raises:
    comptime Schema = ComplementNBSchema[]
    var path = "builds/naive-bayes-complement-model.moxie"
    var X = dense_from_lists([[2.0, 0.0], [0.0, 0.0]])
    var y: List[Int] = [-1, 2]
    var model = Schema.Model(alpha=0.0, norm=True)
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_false(restored.__sklearn_is_fitted__())
    assert_true(restored.norm)
    with assert_raises(contains="NotFittedError"):
        _ = restored.predict(X)
    model.feature_all_ = array[DType.float64]([2.0], [1])
    with assert_raises(contains="unfitted ComplementNB"):
        _ = Schema.encode(model)

    _ = model.set_params(norm=False)
    _ = model.fit(X, y)
    save_model[Schema](model, path)
    restored = load_model[Schema](path)
    assert_equal(restored.feature_all_.tolist(), List[Float64]([2.0, 0.0]))
    assert_true(isnan(restored.feature_log_prob_.item(0, 0)))
    assert_equal(restored.feature_log_prob_.item(1, 1), Float64("inf"))
    var archive = Schema.encode(restored)
    assert_equal(dumps(archive), dumps(Schema.encode(model)))

    with assert_raises(contains="unsupported estimator"):
        _ = MultinomialNBSchema[].decode(archive)
    var bad = archive.copy()
    _ = bad.payloads.pop()
    with assert_raises(contains="unexpected ComplementNB arrays"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.payloads[4].shape = [1, 2]
    with assert_raises(contains="invalid shape for feature_all_"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    _ = bad.payloads.pop()
    var shape: List[Int] = [2]
    var values: List[Float64] = [2.0, 1.0]
    bad.add_array[DType.float64]("feature_all_", shape, values)
    with assert_raises(contains="feature_all_ disagrees"):
        _ = Schema.decode(bad)
    model.feature_log_prob_[Item(1, 0)] = -1.0
    with assert_raises(contains="invalid ComplementNB feature statistics"):
        _ = Schema.encode(model)


def bernoulli_file_round_trip() raises:
    comptime Schema = BernoulliNBSchema[StringDomain, DType.float32]
    var path = "builds/naive-bayes-bernoulli-model.moxie"
    var X = dense_from_lists([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]).astype[DType.float32]()
    var y = StringTarget(["β", "a\0z", "β", "a\0z"])
    var alpha_storage = array[DType.float64]([0.25, 99.0, -0.0], [3])
    var alpha = alpha_storage.view_with_layout(Shape(2), NDArrayStrides(-2), 2)
    var model = Schema.Model(alpha=2.0, alpha_values=alpha^, force_alpha=False, binarize=None)
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_false(restored.__sklearn_is_fitted__())
    assert_false(Bool(restored.binarize))
    assert_equal(restored.alpha, 2.0)
    assert_equal(restored.alpha_values.value().tolist(), List[Float64]([-0.0, 0.25]))
    assert_equal(bitcast[DType.uint64, 1](restored.alpha_values.value().item(0)), UInt64(1) << 63)
    with assert_raises(contains="NotFittedError"):
        _ = restored.predict(X)

    _ = model.fit(X, y)
    _ = restored.fit(X, y)
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
    _ = model.set_params(binarize=Optional[Float64](0.5), force_alpha=True, fit_prior=False,
                         class_prior=Optional[List[Float64]](List[Float64]([0.75, 0.25])))
    save_model[Schema](model, path)
    var before = dumps(Schema.encode(model))
    restored = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(restored)), before)
    assert_equal(restored.binarize.value(), 0.5)
    assert_equal(restored.fit_warnings[0].message, model.fit_warnings[0].message)
    assert_equal(restored.classes_.get(0), "a\0z")
    assert_equal(restored.classes_.get(1), "β")
    assert_equal(restored.predict_proba(X).tolist(), model.predict_proba(X).tolist())
    assert_false(restored.__sklearn_clone__().__sklearn_is_fitted__())
    var weights: List[Float64] = [0.25, 2.0, 0.5, 1.0]
    _ = restored.partial_fit(X, y, sample_weight=weights.copy())
    assert_equal(dumps(Schema.encode(model)), before)
    _ = model.partial_fit(X, y, sample_weight=weights^)
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
    assert_equal(len(restored.fit_warnings), 0)


def bernoulli_raw_state_and_validation() raises:
    comptime Schema = BernoulliNBSchema[]
    var X = dense_from_lists([[-4.0, 7.0], [0.0, 2.0]])
    var y: List[Int] = [1, 2]
    var model = Schema.Model(binarize=None)
    _ = model.fit(X, y)
    # The learned state came from raw inputs. The new threshold only affects future calls.
    _ = model.set_params(binarize=Optional[Float64](0.0))
    var archive = Schema.encode(model)
    var before = dumps(archive)
    var restored = Schema.decode(loads(before))
    assert_equal(restored.feature_count_.item(0, 0), -4.0)
    assert_true(isnan(restored.feature_log_prob_.item(0, 0)))
    assert_true(restored.feature_log_prob_.item(0, 1) > 0.0)
    assert_equal(dumps(Schema.encode(restored)), before)

    with assert_raises(contains="dtype mismatch"):
        _ = BernoulliNBSchema[dtype=DType.float32].decode(archive)
    with assert_raises(contains="label domain mismatch"):
        _ = BernoulliNBSchema[NumericDomain[DType.int64]].decode(archive)
    with assert_raises(contains="unsupported estimator"):
        _ = MultinomialNBSchema[].decode(archive)
    var bad = archive.copy()
    bad.schema_version = 4
    with assert_raises(contains="unsupported estimator"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.manifest.append(0)
    with assert_raises(contains="trailing data"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.payloads[2].shape = [4]
    with assert_raises(contains="invalid shape for feature_count_"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    _ = bad.payloads.pop()
    with assert_raises(contains="unexpected BernoulliNB arrays"):
        _ = Schema.decode(bad)
    model.class_count_[Item(0)] = -1.0
    with assert_raises(contains="invalid BernoulliNB class count"):
        _ = Schema.encode(model)


def categorical_file_round_trip() raises:
    comptime Schema = CategoricalNBSchema[StringDomain, DType.uint8]
    var path = "builds/naive-bayes-categorical-model.moxie"
    var X = array[DType.uint8]([5, 0, 0, 3], [2, 2])
    var small = array[DType.uint8]([0, 1, 1, 0], [2, 2])
    var grown = array[DType.uint8]([7, 2, 1, 0], [2, 2])
    var y = StringTarget(["β", "a\0z"])
    var model = Schema.Model(alpha=0.0, force_alpha=False, min_categories=3)
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_false(restored.__sklearn_is_fitted__())
    assert_equal(restored.min_categories.value(), 3)
    assert_false(Bool(restored.min_categories_values))
    with assert_raises(contains="NotFittedError"):
        _ = restored.predict(X)

    _ = model.fit(X, y)
    _ = restored.fit(X, y)
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
    _ = model.partial_fit(small, y)
    var expected_categories: List[Int64] = [3, 3]
    assert_equal(model.n_categories_.tolist(), expected_categories)
    assert_equal(model.category_count_[0].shape.tolist(), [2, 6])
    assert_equal(model.category_count_[1].shape.tolist(), [2, 4])
    _ = model.set_params(force_alpha=True, fit_prior=False,
                         min_categories_values=Optional[List[Int]](List[Int]([2, 4])))
    var before = dumps(Schema.encode(model))
    save_model[Schema](model, path)
    restored = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(restored)), before)
    assert_equal(restored.min_categories.value(), 3)
    assert_equal(restored.min_categories_values.value(), [2, 4])
    assert_equal(restored.classes_.get(0), "a\0z")
    assert_equal(restored.classes_.get(1), "β")
    assert_equal(restored.fit_warnings[0].message, model.fit_warnings[0].message)
    assert_equal(restored.predict_proba(X).tolist(), model.predict_proba(X).tolist())
    assert_false(restored.__sklearn_clone__().__sklearn_is_fitted__())
    var weights: List[Float64] = [0.25, 2.0]
    _ = restored.partial_fit(grown, y, sample_weight=weights.copy())
    assert_equal(dumps(Schema.encode(model)), before)
    _ = model.partial_fit(grown, y, sample_weight=weights^)
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
    assert_equal(restored.category_count_[0].shape.tolist(), [2, 8])
    assert_equal(len(restored.fit_warnings), 0)


def categorical_archive_validation() raises:
    comptime Schema = CategoricalNBSchema[]
    var model = Schema.Model()
    _ = model.fit(dense_from_lists([[0.0, 2.0], [0.0, 0.0]]), List[Int]([2, 1]))
    var archive = Schema.encode(model)
    with assert_raises(contains="dtype mismatch"):
        _ = CategoricalNBSchema[dtype=DType.float32].decode(archive)
    with assert_raises(contains="label domain mismatch"):
        _ = CategoricalNBSchema[NumericDomain[DType.int64]].decode(archive)
    var bad = archive.copy()
    bad.schema_version = 4
    with assert_raises(contains="unsupported estimator"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    _ = bad.payloads.pop()
    with assert_raises(contains="unexpected CategoricalNB arrays"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.payloads[3].shape = [2, 4]
    with assert_raises(contains="invalid packed CategoricalNB feature tables"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    var width_start = len(bad.manifest) - 8
    bad.manifest[width_start] = 4
    with assert_raises(contains="invalid CategoricalNB feature table width"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.payloads[2].data[0] = 2
    with assert_raises(contains="invalid CategoricalNB category count"):
        _ = Schema.decode(bad)
    model.category_count_[1][Item(0, 0)] = -1.0
    with assert_raises(contains="invalid CategoricalNB feature statistics"):
        _ = Schema.encode(model)


def categorical_wide_archive() raises:
    comptime Schema = CategoricalNBSchema[NumericDomain[DType.int64]]
    var row = List[Float64](length=512, fill=0.0)
    var X = dense_from_lists([row.copy(), row^])
    var model = Schema.Model()
    _ = model.fit(X, array[DType.int64]([2, 1], [2]))
    var before = dumps(Schema.encode(model))
    var restored = Schema.decode(loads(before))
    assert_equal(dumps(Schema.encode(restored)), before)
    assert_equal(len(restored.category_count_), 512)


def main() raises:
    fitted_round_trip[IntListDomain, DType.float64](List[Int]([2, -1, 2, -1]))
    fitted_round_trip[StringDomain, DType.float32](StringTarget(["β", "a\0z", "β", "a\0z"]))
    fitted_round_trip[NumericDomain[DType.int64], DType.float64](
        array[DType.int64]([9223372036854775807, -1, 9223372036854775807, -1], [4])
    )
    fitted_round_trip[NumericDomain[DType.float32], DType.float32](
        array[DType.float32]([2, -1, 2, -1], [4])
    )
    fitted_round_trip[NumericDomain[DType.float64], DType.float64](
        array[DType.float64]([2, -1, 2, -1], [4])
    )
    file_round_trips()
    logical_state_and_overflow()
    malformed_state()
    print("PASS: GaussianNB serialization contracts")
    multinomial_round_trip[IntListDomain, DType.float64](List[Int]([2, -1, 2, -1]))
    multinomial_round_trip[StringDomain, DType.float32](StringTarget(["β", "a\0z", "β", "a\0z"]))
    multinomial_round_trip[NumericDomain[DType.int64], DType.float64](
        array[DType.int64]([9223372036854775807, -1, 9223372036854775807, -1], [4])
    )
    multinomial_round_trip[NumericDomain[DType.float32], DType.float32](
        array[DType.float32]([2, -1, 2, -1], [4])
    )
    multinomial_round_trip[NumericDomain[DType.float64], DType.uint8](
        array[DType.float64]([2, -1, 2, -1], [4])
    )
    multinomial_file_and_validation()
    print("PASS: MultinomialNB serialization contracts")
    complement_round_trip[IntListDomain, DType.float64](List[Int]([2, -1, 2, -1]))
    complement_round_trip[StringDomain, DType.float32](StringTarget(["β", "a\0z", "β", "a\0z"]))
    complement_round_trip[NumericDomain[DType.int64], DType.float64](
        array[DType.int64]([9223372036854775807, -1, 9223372036854775807, -1], [4])
    )
    complement_round_trip[NumericDomain[DType.float32], DType.float32](
        array[DType.float32]([2, -1, 2, -1], [4])
    )
    complement_round_trip[NumericDomain[DType.float64], DType.uint8](
        array[DType.float64]([2, -1, 2, -1], [4])
    )
    complement_file_and_validation()
    print("PASS: ComplementNB serialization contracts")
    bernoulli_file_round_trip()
    bernoulli_raw_state_and_validation()
    print("PASS: BernoulliNB serialization contracts")
    categorical_file_round_trip()
    categorical_archive_validation()
    categorical_wide_archive()
    print("PASS: CategoricalNB serialization contracts")
