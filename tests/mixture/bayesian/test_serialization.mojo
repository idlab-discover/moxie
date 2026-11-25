from numojo import NDArray, NDArrayStrides, Shape, zeros
from std.math import inf, nan
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from moxie.base import clone
from moxie.mixture import BayesianGaussianMixture
from moxie.mixture_serialization import BayesianGaussianMixtureSchema, GaussianMixtureSchema
from moxie.serialization import ModelArchive, dumps, load_model, save_model


def data[dtype: DType]() raises -> NDArray[dtype]:
    # The empirical covariance prior has rank one. The default ridge makes
    # the posterior covariances positive definite in both tested dtypes.
    var x = zeros[dtype](Shape(8, 2))
    for row in range(8):
        x.unsafe_set(row * 2, Scalar[dtype](row))
    return x^


def same_array[dtype: DType](actual: NDArray[dtype], expected: NDArray[dtype]) raises:
    assert_equal(actual.shape, expected.shape)
    for index in range(actual.size):
        assert_equal(actual.unsafe_get(index), expected.unsafe_get(index))


def unfitted_configuration() raises:
    comptime Schema = BayesianGaussianMixtureSchema[]
    var backing = zeros[DType.float64](Shape(4))
    backing.unsafe_set(3, 1.25)
    backing.unsafe_set(1, -0.75)
    var reversed = backing.view_with_layout(Shape(2), NDArrayStrides(-2), 3)
    var model = BayesianGaussianMixture(
        n_components=-2, covariance_type="pending", tol=-0.25, reg_covar=0.1,
        max_iter=12, n_init=4, init_params="random_from_data",
        weight_concentration_prior_type="pending", weight_concentration_prior=0.03125,
        mean_precision_prior=1.5, mean_prior=reversed^, degrees_of_freedom_prior=3.5,
        covariance_prior=zeros[DType.float64](Shape(1, 0, 2)), warm_start=True,
        verbose=2, verbose_interval=3,
    )
    var path = "builds/mixture-bayesian-unfitted.moxie"
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_false(restored.__sklearn_is_fitted__())
    assert_false(Bool(restored.random_state))
    assert_equal(restored.n_components, -2)
    assert_equal(restored.tol, -0.25)
    assert_equal(restored.reg_covar, 0.1)
    assert_equal(restored.max_iter, 12)
    assert_equal(restored.n_init, 4)
    assert_equal(restored.init_params, "random_from_data")
    assert_equal(restored.covariance_type, "pending")
    assert_equal(restored.weight_concentration_prior_type, "pending")
    assert_equal(restored.weight_concentration_prior.or_else(0.0), 0.03125)
    assert_equal(restored.mean_precision_prior.or_else(0.0), 1.5)
    assert_equal(restored.degrees_of_freedom_prior.or_else(0.0), 3.5)
    assert_true(restored.warm_start)
    assert_equal(restored.verbose, 2)
    assert_equal(restored.verbose_interval, 3)
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
    backing.unsafe_set(3, 99)
    assert_true(Bool(restored.mean_prior))
    ref restored_mean = restored.mean_prior.value()
    assert_equal(restored_mean.unsafe_get(0), 1.25)
    assert_equal(restored_mean.unsafe_get(1), -0.75)
    assert_true(Bool(restored.covariance_prior))
    ref restored_covariance = restored.covariance_prior.value()
    assert_equal(restored_covariance.shape, Shape(1, 0, 2))
    var x = data[DType.float64]()
    with assert_raises(contains="NotFittedError"):
        _ = restored.predict(x)
    with assert_raises(contains="invalid BayesianGaussianMixture parameters"):
        _ = restored.fit(x)


def fitted_round_trips[dtype: DType]() raises:
    comptime Schema = BayesianGaussianMixtureSchema[dtype]
    var x = data[dtype]()
    var path = "builds/mixture-bayesian-fitted-" + String(dtype) + ".moxie"
    var kinds: List[String] = ["full", "tied", "diag", "spherical"]
    var prior_types: List[String] = ["dirichlet_process", "dirichlet_distribution"]
    for kind in kinds:
        for prior_type in prior_types:
            var model = BayesianGaussianMixture[dtype](
                n_components=2, covariance_type=kind, weight_concentration_prior_type=prior_type,
                warm_start=True, max_iter=1, random_state=7,
            )
            _ = model.fit(x)
            var labels = model.predict(x)
            var probabilities = model.predict_proba(x)
            var scores = model.score_samples(x)
            var sample = model.sample(8)
            _ = model.set_params(
                n_components=3, covariance_type="pending", weight_concentration_prior_type="pending",
                weight_concentration_prior=9.0, max_iter=5,
            )
            save_model[Schema](model, path)
            var restored = load_model[Schema](path)
            assert_true(restored.__sklearn_is_fitted__())
            assert_false(clone(restored).__sklearn_is_fitted__())
            assert_equal(restored.random_state.or_else(-1), 7)
            assert_equal(restored.n_components, 3)
            assert_equal(restored.covariance_type, "pending")
            assert_equal(restored.weight_concentration_prior_type, "pending")
            assert_equal(restored.weight_concentration_prior.or_else(0.0), 9.0)
            assert_equal(restored.weight_concentration_prior_, 0.5)
            assert_equal(restored.n_features_in_, 2)
            assert_equal(restored.n_iter_, 1)
            assert_equal(restored.convergence_warning_, model.convergence_warning_)
            assert_true(restored.convergence_warning_ != "")
            assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))

            _ = restored.set_params(n_components=2, covariance_type=kind, weight_concentration_prior_type=prior_type)
            _ = model.set_params(n_components=2, covariance_type=kind, weight_concentration_prior_type=prior_type)
            same_array(restored.predict(x), labels)
            same_array(restored.predict_proba(x), probabilities)
            same_array(restored.score_samples(x), scores)
            assert_equal(restored.score(x), model.score(x))
            var restored_sample = restored.sample(8)
            same_array(restored_sample[0], sample[0])
            same_array(restored_sample[1], sample[1])
            same_array(restored.fit_predict(x), model.fit_predict(x))
            assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
            var original = model.weight_concentration_[0].unsafe_get(0)
            restored.weight_concentration_[0].unsafe_set(0, 999)
            assert_equal(model.weight_concentration_[0].unsafe_get(0), original)

    var zero_iteration = BayesianGaussianMixture[dtype](
        covariance_type="spherical", max_iter=0, random_state=7,
    )
    _ = zero_iteration.fit(x)
    save_model[Schema](zero_iteration, path)
    var restored = load_model[Schema](path)
    assert_equal(restored.n_iter_, 0)
    assert_equal(restored.lower_bound_, -inf[DType.float64]())
    assert_equal(len(restored.lower_bounds_), 0)
    assert_equal(restored.convergence_warning_, "")
    same_array(restored.predict(x), zero_iteration.predict(x))


def replace_array(mut archive: ModelArchive, name: String, shape: List[Int], values: List[Float64]) raises:
    for index in range(len(archive.payloads)):
        if archive.payloads[index].name == name:
            _ = archive.payloads.pop(index)
            archive.add_array[DType.float64](name, shape, values)
            return
    raise Error("Missing test payload")


def invalid_archives() raises:
    comptime Schema = BayesianGaussianMixtureSchema[]
    var x = data[DType.float64]()
    var model = BayesianGaussianMixture(n_components=2, covariance_type="spherical", max_iter=1, random_state=7)
    _ = model.fit(x)
    var archive = Schema.encode(model)
    with assert_raises(contains="dtype mismatch"):
        _ = BayesianGaussianMixtureSchema[DType.float32].decode(archive)
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = GaussianMixtureSchema[].decode(archive)
    var wrong = archive.copy()
    wrong.schema_version += 1
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    _ = wrong.manifest.pop()
    with assert_raises(contains="history length"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    wrong.manifest.append(0)
    with assert_raises(contains="trailing data"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    wrong.payloads.append(wrong.payloads[0].copy())
    with assert_raises(contains="unexpected BayesianGaussianMixture arrays"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    replace_array(wrong, "weight_concentration_a", [2], [0.0, 1.0])
    with assert_raises(contains="weight concentration"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    replace_array(wrong, "weight_concentration_b", [1, 2], [1.0, 1.0])
    with assert_raises(contains="array rank"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    replace_array(wrong, "degrees_of_freedom_", [2], [0.5, 2.5])
    with assert_raises(contains="degrees of freedom"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    replace_array(wrong, "covariance_prior_", [1], [-1.0])
    with assert_raises(contains="prior variance"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    replace_array(wrong, "mean_prior_", [2], [nan[DType.float64](), 0.0])
    with assert_raises(contains="non-finite mixture mean prior"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    replace_array(wrong, "weights_", [2], [0.0, 1.0])
    with assert_raises(contains="weights differ from their concentration"):
        _ = Schema.decode(wrong)
    wrong = archive.copy()
    replace_array(wrong, "precisions_cholesky_", [2], [0.0, 1.0])
    with assert_raises(contains="nonpositive mixture precision diagonal"):
        _ = Schema.decode(wrong)
    model.convergence_warning_ = "unknown"
    with assert_raises(contains="invalid BayesianGaussianMixture convergence warning"):
        _ = Schema.encode(model)


def main() raises:
    unfitted_configuration()
    fitted_round_trips[DType.float32]()
    fitted_round_trips[DType.float64]()
    invalid_archives()
    print("PASS: BayesianGaussianMixture archive round trips and invalid state")
