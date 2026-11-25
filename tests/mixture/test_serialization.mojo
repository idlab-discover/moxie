from numojo import NDArray, NDArrayStrides, Shape, full, zeros
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.math import inf, nan, isnan
from moxie.base import clone
from moxie.mixture import GaussianMixture
from moxie.mixture_serialization import GaussianMixtureSchema
from moxie.serialization import dumps, loads, save_model, load_model


def data[dtype: DType]() raises -> NDArray[dtype]:
    var values: List[Float64] = [-3, -1, -2, -2, -4, -1, -3, 0, 6, 2, 4, 1, 5, 2, 4, 3]
    var x = zeros[dtype](Shape(8, 2))
    for i in range(len(values)):
        x.unsafe_set(i, values[i].cast[dtype]())
    return x^


def same_array[dtype: DType](actual: NDArray[dtype], expected: NDArray[dtype]) raises:
    assert_equal(actual.ndim, expected.ndim)
    for axis in range(actual.ndim):
        assert_equal(actual.shape[axis], expected.shape[axis])
    for i in range(actual.size):
        assert_equal(actual.unsafe_get(i), expected.unsafe_get(i))


def unfitted_configuration() raises:
    comptime Schema = GaussianMixtureSchema[]
    var backing = zeros[DType.float64](Shape(4))
    backing.unsafe_set(3, 0.25)
    backing.unsafe_set(1, 0.75)
    var reversed = backing.view_with_layout(Shape(2), NDArrayStrides(-2), 3)
    var one = full[DType.float64](Shape(1), nan[DType.float64]())
    var scalar = one[0]
    var empty = zeros[DType.float64](Shape(1, 0, 2))
    var model = GaussianMixture(
        -5, covariance_type="pending", tol=-0.5, reg_covar=0.25, max_iter=3,
        n_init=2, init_params="random", weights_init=reversed^, means_init=scalar^,
        precisions_init=empty^, warm_start=True, verbose=2, verbose_interval=3,
    )
    var path = "builds/mixture-gaussian-unfitted.moxie"
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_false(restored.__sklearn_is_fitted__())
    assert_false(Bool(restored.random_state))
    assert_equal(restored.n_components, -5)
    assert_equal(restored.covariance_type, "pending")
    assert_equal(restored.weights_init.value().unsafe_get(0), 0.25)
    assert_equal(restored.weights_init.value().unsafe_get(1), 0.75)
    assert_equal(restored.means_init.value().ndim, 0)
    assert_true(isnan(restored.means_init.value().unsafe_get(0)))
    assert_equal(restored.precisions_init.value().size, 0)
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
    backing.unsafe_set(3, 99)
    assert_equal(restored.weights_init.value().unsafe_get(0), 0.25)
    var x = data[DType.float64]()
    with assert_raises(contains="NotFittedError"):
        _ = restored.predict(x)
    with assert_raises(contains="invalid GaussianMixture parameters"):
        _ = restored.fit(x)


def fitted_round_trips[dtype: DType]() raises:
    comptime Schema = GaussianMixtureSchema[dtype]
    var x = data[dtype]()
    var path = "builds/mixture-gaussian-fitted-" + String(dtype) + ".moxie"
    var kinds: List[String] = ["full", "tied", "diag", "spherical"]
    for kind in kinds:
        var model = GaussianMixture[dtype](
            2, covariance_type=kind, random_state=7, max_iter=1, warm_start=True,
        )
        _ = model.fit(x)
        assert_true(model.convergence_warning_ != "")
        var labels = model.predict(x)
        var probabilities = model.predict_proba(x)
        var scores = model.score_samples(x)
        var sample = model.sample(8)
        # These pending parameters cannot reproduce the fitted state.
        _ = model.set_params(n_components=-1, covariance_type="pending", max_iter=5)
        save_model[Schema](model, path)
        var restored = load_model[Schema](path)
        assert_equal(restored.n_components, -1)
        assert_equal(restored.covariance_type, "pending")
        assert_equal(restored.convergence_warning_, model.convergence_warning_)
        assert_equal(restored.n_features_in_, 2)
        assert_true(restored.__sklearn_is_fitted__())
        assert_false(clone(restored).__sklearn_is_fitted__())
        assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
        _ = restored.set_params(n_components=2, covariance_type=kind)
        _ = model.set_params(n_components=2, covariance_type=kind)
        same_array(restored.predict(x), labels)
        same_array(restored.predict_proba(x), probabilities)
        same_array(restored.score_samples(x), scores)
        assert_equal(restored.score(x), model.score(x))
        assert_equal(restored.bic(x), model.bic(x))
        assert_equal(restored.aic(x), model.aic(x))
        var restored_sample = restored.sample(8)
        same_array(restored_sample[0], sample[0])
        same_array(restored_sample[1], sample[1])
        same_array(restored.fit_predict(x), model.fit_predict(x))
        assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
        var initial = model.means_.unsafe_get(0)
        restored.means_.unsafe_set(0, 10)
        assert_equal(model.means_.unsafe_get(0), initial)

    var initializer = GaussianMixture[dtype](
        2, covariance_type="spherical", max_iter=0, init_params="random_from_data",
        random_state=7,
    )
    _ = initializer.fit(x)
    save_model[Schema](initializer, path)
    var restored = load_model[Schema](path)
    assert_equal(restored.n_iter_, 0)
    assert_equal(restored.lower_bound_, -inf[DType.float64]())
    assert_equal(len(restored.lower_bounds_), 0)
    assert_equal(restored.convergence_warning_, "")
    same_array(restored.predict(x), initializer.predict(x))

    # Every sample seeds a component. Gaussian initialization adds ten dtype
    # eps to each count, so the Float32 weight sum is slightly above one.
    var tiny = zeros[dtype](Shape(2, 1))
    tiny.unsafe_set(0, -1)
    tiny.unsafe_set(1, 1)
    _ = initializer.fit(tiny)
    save_model[Schema](initializer, path)
    restored = load_model[Schema](path)
    same_array(restored.weights_, initializer.weights_)


def invalid_archives() raises:
    comptime Schema = GaussianMixtureSchema[]
    var x = data[DType.float64]()
    var model = GaussianMixture(2, covariance_type="spherical", max_iter=1, random_state=7)
    _ = model.fit(x)
    var archive = Schema.encode(model)
    with assert_raises(contains="dtype mismatch"):
        _ = GaussianMixtureSchema[DType.float32].decode(archive)
    var wrong = archive.copy()
    wrong.type_id = "moxie.mixture.BayesianGaussianMixture"
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    wrong.schema_version += 1
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    wrong.manifest.append(0)
    with assert_raises():
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    wrong.payloads.append(wrong.payloads[0].copy())
    with assert_raises(contains="unexpected GaussianMixture arrays"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    wrong.payloads[1].name = "unknown"
    with assert_raises(contains="missing array means_"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    wrong.payloads[1].shape = [4]
    with assert_raises(contains="rank-two"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    _ = wrong.payloads.pop(0)
    var shape: List[Int] = [2]
    var weights: List[Float64] = [-0.25, 1.25]
    wrong.add_array[DType.float64]("weights_", shape, weights)
    with assert_raises(contains="invalid GaussianMixture weight"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    _ = wrong.payloads.pop(2)
    var covariances: List[Float64] = [-1.0, 1.0]
    wrong.add_array[DType.float64]("covariances_", shape, covariances)
    with assert_raises():
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    _ = wrong.payloads.pop(3)
    var precisions: List[Float64] = [1.0, 1.0]
    wrong.add_array[DType.float64]("precisions_", shape, precisions)
    with assert_raises(contains="precisions differ"):
        _ = Schema.decode(loads(dumps(wrong)))
    model.lower_bounds_.append(0.0)
    with assert_raises(contains="history length"):
        _ = Schema.encode(model)


def main() raises:
    unfitted_configuration()
    fitted_round_trips[DType.float32]()
    fitted_round_trips[DType.float64]()
    invalid_archives()
    print("PASS: GaussianMixture archive round trips and invalid state")
