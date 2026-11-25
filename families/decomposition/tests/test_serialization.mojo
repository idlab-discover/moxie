"""Dense decomposition persistence contracts."""

from std.math import inf, isinf, isnan
from std.sys import argv
from std.testing import assert_equal, assert_raises, assert_true
from numojo import NDArray, NDArrayStrides, Shape, zeros
from moxie.base import clone
from moxie.decomposition import (
    FactorAnalysis, FastICA, KernelPCA, PCA, IncrementalPCA, TruncatedSVD, NMF, MiniBatchNMF,
)
from moxie.decomposition._fastica import WhitenOption
from moxie.decomposition._kernel_pca import KernelPower
from moxie.decomposition._pca import ComponentCount
from moxie.decomposition_serialization import (
    FactorAnalysisSchema, FastICASchema, KernelPCASchema, PCASchema,
    IncrementalPCASchema, TruncatedSVDSchema, NMFSchema, MiniBatchNMFSchema,
)
from moxie.serialization import dumps, loads, save_model, load_model


def same_array[dtype: DType](left: NDArray[dtype], right: NDArray[dtype]) raises:
    assert_equal(left.ndim, right.ndim)
    for axis in range(left.ndim):
        assert_equal(left.shape[axis], right.shape[axis])
    for i in range(left.size):
        assert_equal(left.unsafe_get(i), right.unsafe_get(i))


def training_data[dtype: DType]() raises -> NDArray[dtype]:
    var X = zeros[dtype](Shape(4, 3))
    var values = [4, 1, 0, -3, 2, 6, 9, 2, 3, 4, 7, 2]
    for i in range(len(values)):
        X.unsafe_set(i, Scalar[dtype](values[i]))
    return X^


def test_pca[dtype: DType](path: String) raises:
    comptime Schema = PCASchema[dtype]
    var X = training_data[dtype]()
    var fresh = PCA[dtype](0.8, copy=False, whiten=True, svd_solver="full",
        tol=0.25, iterated_power=3, n_oversamples=4, power_iteration_normalizer="QR", random_state=42)
    save_model[Schema](fresh, path)
    var unfitted = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(fresh)), dumps(Schema.encode(unfitted)))
    with assert_raises(contains="NotFittedError: PCA"):
        _ = unfitted.transform(X)

    var model = PCA[dtype](2, whiten=True, svd_solver="auto")
    _ = model.fit(X)
    _ = model.set_params(svd_solver="randomized", n_components=ComponentCount(String("mle")))
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(restored)))
    assert_equal(restored.svd_solver, String("randomized"))
    assert_equal(restored._fit_svd_solver, String("full"))
    assert_equal(restored.n_components_, 2)
    assert_true(not clone(restored).__sklearn_is_fitted__())
    var projected = restored.transform(X)
    same_array(model.transform(X), projected)
    same_array(model.inverse_transform(projected), restored.inverse_transform(projected))
    same_array(model.get_covariance(), restored.get_covariance())
    same_array(model.get_precision(), restored.get_precision())
    same_array(model.score_samples(X), restored.score_samples(X))
    var owned = Schema.decode(Schema.encode(model))
    model.mean_.value().unsafe_set(0, 100)
    same_array(owned.transform(X), projected)

    var rank_zero = PCA[dtype](0)
    _ = rank_zero.fit(X)
    var empty = Schema.decode(loads(dumps(Schema.encode(rank_zero))))
    assert_true(empty.__sklearn_is_fitted__())
    assert_equal(empty.n_components_, 0)
    assert_equal(empty.components_.value().shape[0], 0)

    var constant = PCA[dtype](1)
    _ = constant.fit(zeros[dtype](Shape(3, 2)))
    var degenerate = Schema.decode(Schema.encode(constant))
    assert_true(isnan(degenerate.explained_variance_ratio_.value().unsafe_get(0)))

    var mean = restored.mean_.value().copy()
    var strided = mean.view_with_layout(Shape(3), NDArrayStrides(-1), 2)
    restored.mean_ = strided^
    var linear = Schema.decode(Schema.encode(restored))
    for i in range(3):
        assert_equal(linear.mean_.value().unsafe_get(i), mean.unsafe_get(2 - i))

    var archive = Schema.encode(restored)
    archive.schema_version = 2
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = Schema.decode(archive)
    archive.schema_version = 1
    archive.payloads[0].shape = [3, 2]
    with assert_raises(contains="invalid shape for components_"):
        _ = Schema.decode(archive)
    restored.singular_values_.value().unsafe_set(0, -1)
    with assert_raises(contains="invalid value in singular_values_"):
        _ = Schema.encode(restored)


def test_incremental_pca[dtype: DType](path: String) raises:
    comptime Schema = IncrementalPCASchema[dtype]
    var X = training_data[dtype]()
    var shifted = X.copy()
    for i in range(shifted.size):
        shifted.unsafe_set(i, X.unsafe_get(i) + 11)

    var model = IncrementalPCA[dtype](2, whiten=True, batch_size=3)
    save_model[Schema](model, path)
    var unfitted = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(unfitted)))
    with assert_raises(contains="NotFittedError: IncrementalPCA"):
        _ = unfitted.transform(X)

    _ = model.partial_fit(X)
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(restored.n_samples_seen_, 4)
    assert_equal(restored.batch_size_, 0)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(restored)))
    same_array(model.transform(shifted), restored.transform(shifted))
    _ = model.partial_fit(shifted)
    _ = restored.partial_fit(shifted)
    assert_equal(restored.n_samples_seen_, 8)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(restored)))
    same_array(model.transform(X), restored.transform(X))

    _ = model.fit(X)
    _ = model.set_params(batch_size=Optional[Int](7))
    var fitted = Schema.decode(Schema.encode(model))
    assert_equal(fitted.batch_size_, 3)
    assert_equal(fitted.batch_size.value(), 7)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(fitted)))

    var archive = Schema.encode(model)
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = PCASchema[dtype].decode(archive)
    archive.payloads[2].name = "foreign_array"
    with assert_raises(contains="missing array var_"):
        _ = Schema.decode(archive)

    var single = IncrementalPCA[dtype](1)
    _ = single.partial_fit(zeros[dtype](Shape(1, 2)))
    var single_restored = Schema.decode(Schema.encode(single))
    assert_true(isnan(single_restored.explained_variance_.value().unsafe_get(0)))
    _ = single_restored.partial_fit(X.view_with_layout(Shape(4, 2), NDArrayStrides(3, 1), 0))
    assert_equal(single_restored.n_samples_seen_, 5)


def test_truncated_svd[dtype: DType](path: String) raises:
    comptime Schema = TruncatedSVDSchema[dtype]
    var X = training_data[dtype]()
    var model = TruncatedSVD[dtype](2, algorithm="arpack", n_iter=4,
        n_oversamples=3, power_iteration_normalizer="QR", random_state=7, tol=0.01)
    save_model[Schema](model, path)
    var unfitted = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(unfitted)))
    with assert_raises(contains="NotFittedError: This TruncatedSVD"):
        _ = unfitted.transform(X)
    assert_equal(unfitted.algorithm, String("arpack"))
    assert_equal(unfitted.n_iter, 4)
    assert_equal(unfitted.n_oversamples, 3)
    assert_equal(unfitted.power_iteration_normalizer, String("QR"))
    assert_equal(unfitted.random_state.value(), 7)
    assert_equal(unfitted.tol, 0.01)

    _ = model.fit(X)
    _ = model.set_params(n_components=1, algorithm="randomized")
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(restored)))
    assert_equal(restored.n_components, 1)
    assert_equal(restored.components_.value().shape[0], 2)
    assert_equal(restored.algorithm, String("randomized"))
    assert_equal(restored.get_feature_names_out(), model.get_feature_names_out())
    assert_true(not clone(restored).__sklearn_is_fitted__())
    var projected = model.transform(X)
    same_array(projected, restored.transform(X))
    same_array(model.inverse_transform(projected), restored.inverse_transform(projected))
    model.components_.value().unsafe_set(0, 100)
    same_array(projected, restored.transform(X))

    var archive = Schema.encode(restored)
    archive.schema_version = 2
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = Schema.decode(archive)
    archive.schema_version = 1
    archive.payloads[1].name = "foreign_array"
    with assert_raises(contains="missing array singular_values_"):
        _ = Schema.decode(archive)

    var constant = TruncatedSVD[dtype](2, random_state=7)
    _ = constant.fit(zeros[dtype](Shape(1, 3)))
    var degenerate = Schema.decode(Schema.encode(constant))
    assert_equal(degenerate.components_.value().shape[0], 1)
    assert_true(isnan(degenerate.explained_variance_ratio_.value().unsafe_get(0)))


def test_factor_analysis[dtype: DType](path: String) raises:
    comptime Schema = FactorAnalysisSchema[dtype]
    var X = training_data[dtype]()
    var init = List[Float64](length=3, fill=1.25)
    var model = FactorAnalysis[dtype](1, tol=0.001, max_iter=8, noise_variance_init=init^,
        svd_method="lapack", iterated_power=2, rotation=String("varimax"), random_state=None)
    save_model[Schema](model, path)
    var unfitted = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(unfitted)))
    with assert_raises(contains="NotFittedError: This FactorAnalysis"):
        _ = unfitted.transform(X)
    assert_equal(unfitted.noise_variance_init.value(), List[Float64](length=3, fill=1.25))
    assert_equal(unfitted.svd_method, String("lapack"))
    assert_equal(unfitted.iterated_power, 2)
    assert_equal(unfitted.rotation.value(), String("varimax"))
    assert_true(not unfitted.random_state)
    assert_equal(unfitted.tol, 0.001)

    _ = model.fit(X)
    _ = model.set_params(copy=False, max_iter=1, n_components=Optional[Optional[Int]](Optional[Int](0)))
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(restored)))
    assert_equal(restored.n_components.value(), 0)
    assert_equal(restored.components_.value().shape[0], 1)
    assert_equal(restored.max_iter, 1)
    assert_true(not restored.copy)
    assert_true(restored.n_iter_ > restored.max_iter)
    assert_equal(restored.get_feature_names_out(), model.get_feature_names_out())
    assert_true(not clone(restored).__sklearn_is_fitted__())
    same_array(model.transform(X), restored.transform(X))
    same_array(model.get_covariance(), restored.get_covariance())
    same_array(model.get_precision(), restored.get_precision())
    same_array(model.score_samples(X), restored.score_samples(X))
    assert_equal(model.score(X), restored.score(X))

    var archive = Schema.encode(restored)
    archive.schema_version = 2
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = Schema.decode(archive)
    archive.schema_version = 1
    archive.payloads[0].shape = [3, 1]
    with assert_raises(contains="invalid shape for components_"):
        _ = Schema.decode(archive)
    restored.noise_variance_.value().unsafe_set(0, 0)
    with assert_raises(contains="invalid value in noise_variance_"):
        _ = Schema.encode(restored)
    restored.noise_variance_.value().unsafe_set(0, 1)
    restored.loglike_.append(0)
    with assert_raises(contains="invalid FactorAnalysis likelihood history length"):
        _ = Schema.encode(restored)

    var no_factors = FactorAnalysis[dtype](0)
    _ = no_factors.fit(X)
    var zero_rank = Schema.decode(Schema.encode(no_factors))
    assert_equal(zero_rank.components_.value().shape[0], 0)
    same_array(no_factors.score_samples(X), zero_rank.score_samples(X))
    var singular = FactorAnalysis[dtype](1, svd_method="lapack", max_iter=1)
    _ = singular.fit(zeros[dtype](Shape(3, 2)))
    var degenerate = Schema.decode(Schema.encode(singular))
    assert_true(isinf(degenerate.loglike_[0]) and degenerate.loglike_[0] > 0)


def test_fastica[dtype: DType](path: String) raises:
    comptime Schema = FastICASchema[dtype]
    var X = training_data[dtype]()
    var weights = zeros[DType.float64](Shape(2, 2))
    weights.unsafe_set(0, 1)
    weights.unsafe_set(1, 0.25)
    weights.unsafe_set(3, 1)
    var args = Dict[String, Float64]()
    args["alpha"] = 1.5
    args["extra"] = -2
    var model = FastICA[dtype](2, fun_args=args^, w_init=weights.T(),
        whiten_solver="eigh", random_state=37)
    save_model[Schema](model, path)
    var unfitted = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(unfitted)))
    assert_equal(unfitted.w_init.value().unsafe_get(1), 0)
    assert_equal(unfitted.w_init.value().unsafe_get(2), 0.25)
    assert_equal(unfitted.fun_args.value()["alpha"], 1.5)
    assert_equal(unfitted.fun_args.value()["extra"], -2)
    with assert_raises(contains="NotFittedError: This FastICA"):
        _ = unfitted.transform(X)

    _ = model.fit(X)
    _ = model.set_params(whiten=WhitenOption(False), max_iter=1,
        n_components=Optional[Optional[Int]](Optional[Int](3)))
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(restored)))
    assert_equal(restored.whiten[Bool], False)
    assert_equal(restored.max_iter, 1)
    assert_equal(restored.n_components.value(), 3)
    assert_equal(restored.components_.value().shape[0], 2)
    same_array(restored._unmixing.value(), model._unmixing.value())
    var projected = restored.transform(X)
    same_array(projected, model.transform(X))
    same_array(restored.inverse_transform(projected), model.inverse_transform(projected))
    assert_true(not clone(restored).__sklearn_is_fitted__())
    model.components_.value().unsafe_set(0, 100)
    same_array(restored.transform(X), projected)

    var new_weights = zeros[DType.float64](Shape(2))
    new_weights.unsafe_set(1, inf[DType.float64]())
    restored.w_init = new_weights^
    var changed_config = Schema.decode(Schema.encode(restored))
    assert_equal(changed_config.w_init.value().ndim, 1)
    assert_true(isinf(changed_config.w_init.value().unsafe_get(1)))
    same_array(changed_config.transform(X), projected)

    var scalar = zeros[DType.float64](Shape(1))
    scalar.unsafe_set(0, 2.5)
    var with_scalar = FastICA[dtype](w_init=scalar[0])
    var restored_scalar = Schema.decode(Schema.encode(with_scalar))
    assert_equal(restored_scalar.w_init.value().ndim, 0)
    assert_equal(restored_scalar.w_init.value().unsafe_get(0), 2.5)

    var uncentered = FastICA[dtype](whiten=False, random_state=37)
    _ = uncentered.fit(X)
    var roundtrip = Schema.decode(Schema.encode(uncentered))
    assert_true(not roundtrip.mean_)
    assert_true(not roundtrip.whitening_)
    same_array(roundtrip.transform(X), uncentered.transform(X))

    var archive = Schema.encode(restored)
    archive.schema_version = 2
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = Schema.decode(archive)
    archive.schema_version = 1
    archive.payloads[3].shape = [1, 4]
    with assert_raises(contains="invalid shape for _unmixing"):
        _ = Schema.decode(archive)
    restored.whitening_ = None
    with assert_raises(contains="missing whitening_"):
        _ = Schema.encode(restored)


def test_kernel_pca[dtype: DType](path: String) raises:
    comptime Schema = KernelPCASchema[dtype]
    var X = training_data[dtype]()
    var params = Dict[String, Float64]()
    params["unused"] = 3.5
    var model = KernelPCA[dtype](2, kernel="rbf", gamma=0.1, kernel_params=params^,
        alpha=0.5, fit_inverse_transform=True, eigen_solver="dense", max_iter=11,
        iterated_power=3, random_state=7, n_jobs=-1)
    save_model[Schema](model, path)
    var unfitted = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(unfitted)))
    assert_equal(unfitted.kernel_params.value()["unused"], 3.5)
    assert_equal(unfitted.iterated_power[Int], 3)
    assert_equal(unfitted.n_jobs.value(), -1)
    with assert_raises(contains="NotFittedError: KernelPCA"):
        _ = unfitted.transform(X)

    _ = model.fit(X)
    _ = model.set_params(fit_inverse_transform=False,
        gamma=Optional[Optional[Float64]](Optional[Float64](0.25)),
        iterated_power=KernelPower(String("auto")))
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(restored)))
    assert_equal(restored.gamma.value(), 0.25)
    assert_equal(restored.gamma_, 0.1)
    assert_equal(restored.iterated_power[String], String("auto"))
    assert_true(restored.dual_coef_)
    assert_true(not restored.fit_inverse_transform)
    var projected = restored.transform(X)
    same_array(projected, model.transform(X))
    with assert_raises(contains="inverse transform is not available"):
        _ = restored.inverse_transform(projected)
    _ = model.set_params(fit_inverse_transform=True)
    _ = restored.set_params(fit_inverse_transform=True)
    same_array(restored.inverse_transform(projected), model.inverse_transform(projected))
    assert_true(not clone(restored).__sklearn_is_fitted__())
    model.X_fit_.value().unsafe_set(0, 100)
    same_array(restored.transform(X), projected)

    var zero = KernelPCA[dtype]()
    var constant = zeros[dtype](Shape(4, 3))
    _ = zero.fit(constant)
    var roundtrip = Schema.decode(loads(dumps(Schema.encode(zero))))
    assert_equal(roundtrip.transform(constant).shape[1], 0)
    assert_true(not roundtrip.dual_coef_)

    var archive = Schema.encode(restored)
    archive.schema_version = 2
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = Schema.decode(archive)
    archive.schema_version = 1
    archive.payloads[3].shape = [2, 2]
    with assert_raises(contains="invalid shape for _kernel_mean"):
        _ = Schema.decode(archive)
    restored.eigenvalues_.value().unsafe_set(0, -1)
    with assert_raises(contains="invalid value in eigenvalues_"):
        _ = Schema.encode(restored)


def positive_training_data[dtype: DType]() raises -> NDArray[dtype]:
    var X = training_data[dtype]()
    X.unsafe_set(2, 1)
    X.unsafe_set(3, 3)
    return X^


def test_nmf[dtype: DType](path: String) raises:
    comptime Schema = NMFSchema[dtype]
    var X = positive_training_data[dtype]()
    # Overcomplete fits can have more components than features.
    var model = NMF[dtype](4, init=String("random"), max_iter=5, random_state=9,
        alpha_W=0.03, alpha_H=0.05, l1_ratio=0.2, shuffle=True)
    save_model[Schema](model, path)
    var fresh = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(fresh)), dumps(Schema.encode(model)))
    with assert_raises(contains="NotFittedError: This NMF"):
        _ = fresh.transform(X)
    _ = model.fit(X)
    var W = model.transform(X)
    var params = model.get_params()
    params.config.n_components = NoneType()
    params.solver = "mu"
    params.config.beta_loss = 1.5
    _ = model.set_params(params)
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
    assert_true(restored.n_components.isa[NoneType]())
    assert_equal(restored.n_components_, 4)
    assert_equal(restored.beta_loss[Float64], 1.5)
    params.config.n_components = 4
    _ = model.set_params(params)
    _ = restored.set_params(params)
    same_array(restored.transform(X), model.transform(X))
    var inverse = model.inverse_transform(W)
    same_array(restored.inverse_transform(W), inverse)
    assert_true(not clone(restored).__sklearn_is_fitted__())
    model.components_.value().unsafe_set(0, 100)
    same_array(restored.inverse_transform(W), inverse)

    var archive = Schema.encode(restored)
    archive.schema_version = 2
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = Schema.decode(archive)
    archive.schema_version = 1
    archive.payloads[0].shape = [3, 4]
    with assert_raises(contains="invalid shape for components_"):
        _ = Schema.decode(archive)
    restored.components_.value().unsafe_set(0, -1)
    with assert_raises(contains="invalid value in components_"):
        _ = Schema.encode(restored)


def test_minibatch_nmf[dtype: DType](path: String) raises:
    comptime Schema = MiniBatchNMFSchema[dtype]
    var X = positive_training_data[dtype]()
    var next = X.copy()
    next.unsafe_set(0, 8)
    var model = MiniBatchNMF[dtype](2, init=String("random"), beta_loss=1.5,
        batch_size=2, max_iter=7, max_no_improvement=None, forget_factor=0.4,
        fresh_restarts=True, fresh_restarts_max_iter=5, transform_max_iter=4, random_state=6)
    save_model[Schema](model, path)
    var fresh = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(fresh)), dumps(Schema.encode(model)))
    with assert_raises(contains="NotFittedError: This MiniBatchNMF"):
        _ = fresh.transform(X)
    _ = model.partial_fit(X)
    var params = model.get_params()
    params.config.n_components = String("auto")
    params.config.beta_loss = String("frobenius")
    params.forget_factor = 0.8
    params.transform_max_iter = 8
    _ = model.set_params(params)
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
    assert_equal(restored.n_steps_, 1)
    assert_equal(restored.n_iter_, 0)
    assert_equal(restored._beta, 1.5)
    assert_equal(restored._transform_max_iter, 4)
    assert_true(not clone(restored).__sklearn_is_fitted__())
    same_array(restored.transform(X), model.transform(X))
    _ = restored.partial_fit(next)
    _ = model.partial_fit(next)
    assert_equal(restored.n_steps_, 2)
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
    same_array(restored.transform(X), model.transform(X))

    var archive = Schema.encode(restored)
    archive.schema_version = 2
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = Schema.decode(archive)
    archive.schema_version = 1
    archive.payloads[2].shape = [3, 2]
    with assert_raises(contains="invalid shape for _components_denominator"):
        _ = Schema.decode(archive)
    restored._components_numerator = None
    with assert_raises(contains="missing _components_numerator"):
        _ = Schema.encode(restored)


def main() raises:
    var path = String(argv()[1])
    test_pca[DType.float32](path)
    test_pca[DType.float64](path)
    test_incremental_pca[DType.float32](path)
    test_incremental_pca[DType.float64](path)
    test_truncated_svd[DType.float32](path)
    test_truncated_svd[DType.float64](path)
    test_factor_analysis[DType.float32](path)
    test_factor_analysis[DType.float64](path)
    test_fastica[DType.float32](path)
    test_fastica[DType.float64](path)
    test_kernel_pca[DType.float32](path)
    test_kernel_pca[DType.float64](path)
    test_nmf[DType.float32](path)
    test_nmf[DType.float64](path)
    test_minibatch_nmf[DType.float32](path)
    test_minibatch_nmf[DType.float64](path)
    var model = PCA[DType.float32]()
    save_model[PCASchema[DType.float32]](model, path)
    with assert_raises(contains="projection dtype mismatch"):
        _ = load_model[PCASchema[DType.float64]](path)
    var factor = FactorAnalysis[DType.float32]()
    var archive = FactorAnalysisSchema[DType.float32].encode(factor)
    with assert_raises(contains="projection dtype mismatch"):
        _ = FactorAnalysisSchema[DType.float64].decode(archive)
    print("PASS: PCA, IncrementalPCA, TruncatedSVD, FactorAnalysis, FastICA, KernelPCA, NMF and MiniBatchNMF serialization contracts in Float32 and Float64")
