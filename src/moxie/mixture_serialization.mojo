"""Native persistence schemas owned by the dense mixture family."""
from std.math import abs, isfinite, sqrt
from std.utils.numerics import inf
from numojo import NDArray, Shape, full, zeros
from moxie.mixture import BayesianGaussianMixture, GaussianMixture
from moxie.mixture._bayesian_mixture import mixture_weights
from moxie.mixture._contracts import check_shape, covariance_shape
from moxie.mixture._em import CONVERGENCE_WARNING
from moxie.mixture._precision import (
    compute_precision_cholesky,
    precisions_from_cholesky,
)
from moxie.serialization import BinaryReader, BinaryWriter, ModelArchive, ModelSchema, dtype_code


def _logical_value[dtype: DType](
    value: NDArray[dtype], index: Int
) raises -> Scalar[dtype]:
    # NuMojo's single-index item overload skips strides on rank-one views
    # and rejects scalars.
    if value.ndim == 0:
        return value.unsafe_get(0)
    if value.ndim == 1:
        return value.unsafe_get(index * Int(value.strides[0]))
    return value.item(index)


def _write_array[dtype: DType](
    mut archive: ModelArchive, name: String, value: NDArray[dtype]
) raises:
    var shape = List[Int]()
    for axis in range(value.ndim):
        shape.append(value.shape[axis])
    var values = List[Scalar[dtype]](capacity=value.size)
    for index in range(value.size):
        values.append(_logical_value(value, index))
    archive.add_array[dtype](name, shape, values)


def _read_array[dtype: DType](
    archive: ModelArchive, name: String
) raises -> NDArray[dtype]:
    var payload = archive.array(name)
    var values = payload.values[dtype]()
    if len(payload.shape) == 0:
        # NuMojo's scalar indexing constructs a populated rank-zero array.
        var one = full[dtype](Shape(1), values[0])
        return one[0]
    # Even an empty array needs representable intermediate shape and stride
    # products when NuMojo constructs its metadata.
    var nonzero_product = 1
    for dimension in payload.shape:
        if dimension != 0:
            if nonzero_product > 9223372036854775807 // dimension:
                raise Error("ModelSchemaError: mixture array dimensions overflow")
            nonzero_product *= dimension
    var result = zeros[dtype](Shape(payload.shape))
    for index in range(len(values)):
        result.unsafe_set(index, values[index])
    return result^


def _write_optional_array[dtype: DType](
    mut writer: BinaryWriter, mut archive: ModelArchive, name: String,
    value: Optional[NDArray[dtype]],
) raises:
    writer.boolean(Bool(value))
    if value:
        _write_array(archive, name, value.value())


def _read_optional_array[dtype: DType](
    mut reader: BinaryReader[_], archive: ModelArchive, name: String,
) raises -> Optional[NDArray[dtype]]:
    if reader.boolean():
        return _read_array[dtype](archive, name)
    return None


def _check_finite[dtype: DType](array: NDArray[dtype], name: String) raises:
    for index in range(array.size):
        if not isfinite(_logical_value(array, index)):
            raise Error("ModelSchemaError: non-finite mixture " + name)


def _check_gaussian_state[dtype: DType](model: GaussianMixture[dtype]) raises:
    if not model.__sklearn_is_fitted__():
        if (
            model.n_features_in_ != 0 or model._fitted_covariance_type != ""
            or model.weights_.size != 0 or model.means_.size != 0
            or model.covariances_.size != 0 or model.precisions_.size != 0
            or model.precisions_cholesky_.size != 0 or model.n_iter_ != 0
            or model.converged_ or model.lower_bound_ != 0
            or len(model.lower_bounds_) != 0 or model.convergence_warning_ != ""
        ):
            raise Error("ModelSchemaError: unfitted GaussianMixture has learned state")
        return

    var kind = model._fitted_covariance_type
    var features = model.n_features_in_
    if model.means_.ndim != 2:
        raise Error("ModelSchemaError: expected rank-two GaussianMixture means")
    var components = model.means_.shape[0]
    var covariance = covariance_shape(kind, components, features)
    check_shape(model.weights_, Shape(components))
    check_shape(model.means_, Shape(components, features))
    check_shape(model.covariances_, covariance)
    check_shape(model.precisions_, covariance)
    check_shape(model.precisions_cholesky_, covariance)
    _check_finite(model.means_, "means")
    _check_finite(model.covariances_, "covariances")
    _check_finite(model.precisions_, "precisions")
    _check_finite(model.precisions_cholesky_, "precision factors")

    var epsilon = 2e-6 if dtype == DType.float32 else 1e-8
    var total = 0.0
    for i in range(components):
        var weight = Float64(_logical_value(model.weights_, i))
        if not isfinite(weight) or weight < 0:
            raise Error("ModelSchemaError: invalid GaussianMixture weight")
        total += weight
    # Seed-only initialization can underfill the weight sum at max_iter=0.
    # A fully occupied initialization can exceed 1 by 10 dtype eps from nk.
    if total <= 0 or total > 1.0 + epsilon or (model.n_iter_ > 0 and abs(total - 1.0) > epsilon):
        raise Error("ModelSchemaError: invalid GaussianMixture weight sum")

    if model.n_iter_ != len(model.lower_bounds_):
        raise Error("ModelSchemaError: GaussianMixture history length differs from n_iter_")
    if model.n_iter_ == 0:
        if model.converged_ or model.lower_bound_ != -inf[DType.float64]():
            raise Error("ModelSchemaError: invalid zero-iteration GaussianMixture state")
    elif model.lower_bound_ != model.lower_bounds_[model.n_iter_ - 1]:
        raise Error("ModelSchemaError: GaussianMixture history differs from lower_bound_")
    for lower_bound in model.lower_bounds_:
        if not isfinite(lower_bound):
            raise Error("ModelSchemaError: non-finite GaussianMixture history")
    var warning = String(CONVERGENCE_WARNING) if model.n_iter_ > 0 and not model.converged_ else String("")
    if model.convergence_warning_ != warning:
        raise Error("ModelSchemaError: invalid GaussianMixture convergence warning")


def _check_precision_matrices[dtype: DType](
    kind: String, features: Int, covariances: NDArray[dtype],
    precisions: NDArray[dtype], factor: NDArray[dtype],
) raises:
    # Use the same covariance triangle as fitting. In the full/tied paths
    # the two triangles can differ because of intermediate rounding.
    var epsilon = 1e-6 if dtype == DType.float32 else 1e-12
    var width = features if kind == "full" or kind == "tied" else 1
    var block = width * width
    for start in range(0, factor.size, block):
        for row in range(width):
            var diag = start + row * width + row
            if factor.unsafe_get(diag) <= 0 or precisions.unsafe_get(diag) <= 0:
                raise Error("ModelSchemaError: nonpositive mixture precision diagonal")
            for col in range(row):
                if factor.unsafe_get(start + row * width + col) != 0:
                    raise Error("ModelSchemaError: invalid mixture factor triangle")
    var covariance_factor = compute_precision_cholesky(covariances, kind)
    var reconstructed = precisions_from_cholesky(factor, kind)
    for start in range(0, factor.size, block):
        for row in range(width):
            var row_norm = sqrt(Float64(precisions.unsafe_get(start + row * width + row)))
            for col in range(width):
                var col_norm = sqrt(Float64(precisions.unsafe_get(start + col * width + col)))
                var index = start + row * width + col
                var expected = Float64(reconstructed.unsafe_get(index))
                var stored = Float64(precisions.unsafe_get(index))
                if not isfinite(expected) or abs(expected - stored) / row_norm / col_norm > epsilon:
                    raise Error("ModelSchemaError: mixture precisions differ from their factors")
                var expected_factor = Float64(covariance_factor.unsafe_get(index))
                var stored_factor = Float64(factor.unsafe_get(index))
                if not isfinite(expected_factor) or abs(expected_factor - stored_factor) / row_norm > epsilon:
                    raise Error("ModelSchemaError: mixture covariance and precision factor disagree")


struct GaussianMixtureSchema[dtype: DType = DType.float64](ModelSchema):
    """Version 1 of the experimental GaussianMixture model archive."""

    comptime Model = GaussianMixture[Self.dtype]
    comptime type_id = "moxie.mixture.GaussianMixture"
    comptime version = 1

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        _check_gaussian_state(model)
        var archive = ModelArchive(Self.type_id, Self.version)
        var writer = BinaryWriter()
        writer.integer(dtype_code[Self.dtype]())
        writer.integer(model.n_components)
        writer.string(model.covariance_type)
        writer.scalar[DType.float64](model.tol)
        writer.scalar[DType.float64](model.reg_covar)
        writer.integer(model.max_iter)
        writer.integer(model.n_init)
        writer.string(model.init_params)
        writer.boolean(Bool(model.random_state))
        if model.random_state:
            writer.integer(model.random_state.value())
        writer.boolean(model.warm_start)
        writer.integer(model.verbose)
        writer.integer(model.verbose_interval)
        _write_optional_array(writer, archive, "weights_init", model.weights_init)
        _write_optional_array(writer, archive, "means_init", model.means_init)
        _write_optional_array(writer, archive, "precisions_init", model.precisions_init)
        var fitted = model.__sklearn_is_fitted__()
        writer.boolean(fitted)
        if fitted:
            writer.string(model._fitted_covariance_type)
            writer.integer(model.n_features_in_)
            writer.boolean(model.converged_)
            writer.integer(model.n_iter_)
            writer.scalar[DType.float64](model.lower_bound_)
            writer.string(model.convergence_warning_)
            for lower_bound in model.lower_bounds_:
                writer.scalar[DType.float64](lower_bound)
            _write_array(archive, "weights_", model.weights_)
            _write_array(archive, "means_", model.means_)
            _write_array(archive, "covariances_", model.covariances_)
            _write_array(archive, "precisions_", model.precisions_)
            _write_array(archive, "precisions_cholesky_", model.precisions_cholesky_)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.type_id, Self.version)
        var reader = BinaryReader(Span(archive.manifest))
        if reader.integer() != dtype_code[Self.dtype]():
            raise Error("ModelSchemaError: GaussianMixture archive dtype mismatch")
        var components = reader.integer()
        var kind = reader.string()
        var tol = reader.scalar[DType.float64]()
        var reg_covar = reader.scalar[DType.float64]()
        var max_iter = reader.integer()
        var n_init = reader.integer()
        var init_params = reader.string()
        var seed: Optional[Int] = None
        if reader.boolean():
            seed = reader.integer()
        var warm_start = reader.boolean()
        var verbose = reader.integer()
        var verbose_interval = reader.integer()
        var weights = _read_optional_array[Self.dtype](reader, archive, "weights_init")
        var means = _read_optional_array[Self.dtype](reader, archive, "means_init")
        var precisions = _read_optional_array[Self.dtype](reader, archive, "precisions_init")
        var fitted = reader.boolean()
        var expected_arrays = Int(Bool(weights)) + Int(Bool(means)) + Int(Bool(precisions))
        if len(archive.payloads) != expected_arrays + (5 if fitted else 0):
            raise Error("ModelSchemaError: unexpected GaussianMixture arrays")
        var model = Self.Model(
            components, covariance_type=kind, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            weights_init=weights^, means_init=means^, precisions_init=precisions^,
            random_state=seed, warm_start=warm_start, verbose=verbose,
            verbose_interval=verbose_interval,
        )
        if fitted:
            model._fitted_covariance_type = reader.string()
            model.n_features_in_ = reader.integer()
            model.converged_ = reader.boolean()
            model.n_iter_ = reader.integer()
            model.lower_bound_ = reader.scalar[DType.float64]()
            model.convergence_warning_ = reader.string()
            if model.n_iter_ < 0 or model.n_iter_ != reader.remaining() // 8:
                raise Error("ModelSchemaError: invalid GaussianMixture history length")
            for _ in range(model.n_iter_):
                model.lower_bounds_.append(reader.scalar[DType.float64]())
            model.weights_ = _read_array[Self.dtype](archive, "weights_")
            model.means_ = _read_array[Self.dtype](archive, "means_")
            model.covariances_ = _read_array[Self.dtype](archive, "covariances_")
            model.precisions_ = _read_array[Self.dtype](archive, "precisions_")
            model.precisions_cholesky_ = _read_array[Self.dtype](archive, "precisions_cholesky_")
            model._fitted = True
            _check_gaussian_state(model)
            _check_precision_matrices(
                model._fitted_covariance_type, model.n_features_in_,
                model.covariances_, model.precisions_, model.precisions_cholesky_,
            )
        reader.finish()
        return model^


def _write_optional_scalar(mut writer: BinaryWriter, value: Optional[Float64]):
    writer.boolean(Bool(value))
    if value:
        writer.scalar[DType.float64](value.value())


def _read_optional_scalar(mut reader: BinaryReader[_]) raises -> Optional[Float64]:
    if reader.boolean():
        return reader.scalar[DType.float64]()
    return None


def _check_bayesian_values[dtype: DType](
    array: NDArray[dtype], name: String, minimum: Float64,
) raises:
    for index in range(array.size):
        var value = Float64(_logical_value(array, index))
        if not isfinite(value) or value <= minimum:
            raise Error("ModelSchemaError: invalid BayesianGaussianMixture " + name)


def _check_bayesian_state[dtype: DType](model: BayesianGaussianMixture[dtype]) raises:
    if not model.__sklearn_is_fitted__():
        if (
            model.n_features_in_ != 0 or model._fitted_covariance_type != ""
            or model.weights_.size != 0 or model.means_.size != 0
            or model.covariances_.size != 0 or model.precisions_.size != 0
            or model.precisions_cholesky_.size != 0 or model.n_iter_ != 0
            or model.converged_ or model.lower_bound_ != 0
            or len(model.lower_bounds_) != 0 or model.convergence_warning_ != ""
            or model.weight_concentration_prior_ != 0 or len(model.weight_concentration_) != 0
            or model.mean_precision_prior_ != 0 or model.mean_precision_.size != 0
            or model.mean_prior_.size != 0 or model.degrees_of_freedom_prior_ != 0
            or model.degrees_of_freedom_.size != 0 or model.covariance_prior_.size != 0
        ):
            raise Error("ModelSchemaError: unfitted BayesianGaussianMixture has learned state")
        return

    var kind = model._fitted_covariance_type
    var features = model.n_features_in_
    if model.means_.ndim != 2:
        raise Error("ModelSchemaError: expected rank-two BayesianGaussianMixture means")
    var components = model.means_.shape[0]
    var covariance = covariance_shape(kind, components, features)
    check_shape(model.weights_, Shape(components))
    check_shape(model.means_, Shape(components, features))
    check_shape(model.covariances_, covariance)
    check_shape(model.precisions_, covariance)
    check_shape(model.precisions_cholesky_, covariance)
    check_shape(model.mean_prior_, Shape(features))
    check_shape(model.mean_precision_, Shape(components))
    check_shape(model.degrees_of_freedom_, Shape(1 if kind == "tied" else components))
    var concentration_count = len(model.weight_concentration_)
    if concentration_count != 1 and concentration_count != 2:
        raise Error("ModelSchemaError: invalid BayesianGaussianMixture concentration count")
    for concentration in model.weight_concentration_:
        check_shape(concentration, Shape(components))
        _check_bayesian_values(concentration, "weight concentration", 0.0)

    var width = 1 if kind == "spherical" else features
    var matrix_prior = kind == "full" or kind == "tied"
    check_shape(model.covariance_prior_, Shape(width, width) if matrix_prior else Shape(width))
    _check_finite(model.covariance_prior_, "covariance prior")
    # Empirical priors can be singular, including zero variance on a feature.
    for row in range(width):
        var diagonal = row * width + row if matrix_prior else row
        if _logical_value(model.covariance_prior_, diagonal) < 0:
            raise Error("ModelSchemaError: negative BayesianGaussianMixture prior variance")
        if matrix_prior:
            for col in range(row):
                var a = Float64(_logical_value(model.covariance_prior_, row * width + col))
                var b = Float64(_logical_value(model.covariance_prior_, col * width + row))
                if abs(a - b) > 1e-8 + 1e-5 * min(abs(a), abs(b)):
                    raise Error("ModelSchemaError: asymmetric BayesianGaussianMixture covariance prior")

    if (
        not isfinite(model.weight_concentration_prior_) or model.weight_concentration_prior_ <= 0
        or not isfinite(model.mean_precision_prior_) or model.mean_precision_prior_ <= 0
        or not isfinite(model.degrees_of_freedom_prior_)
        or model.degrees_of_freedom_prior_ <= Float64(features - 1)
    ):
        raise Error("ModelSchemaError: invalid BayesianGaussianMixture resolved scalar priors")
    _check_finite(model.mean_prior_, "mean prior")
    _check_finite(model.means_, "means")
    _check_finite(model.covariances_, "covariances")
    _check_finite(model.precisions_, "precisions")
    _check_finite(model.precisions_cholesky_, "precision factors")
    _check_bayesian_values(model.mean_precision_, "mean precision", 0.0)
    _check_bayesian_values(model.degrees_of_freedom_, "degrees of freedom", Float64(features - 1))

    if model.n_iter_ != len(model.lower_bounds_):
        raise Error("ModelSchemaError: BayesianGaussianMixture history length differs from n_iter_")
    if model.n_iter_ == 0:
        if model.converged_ or model.lower_bound_ != -inf[DType.float64]():
            raise Error("ModelSchemaError: invalid zero-iteration BayesianGaussianMixture state")
    elif model.lower_bound_ != model.lower_bounds_[model.n_iter_ - 1]:
        raise Error("ModelSchemaError: BayesianGaussianMixture history differs from lower_bound_")
    for lower_bound in model.lower_bounds_:
        if not isfinite(lower_bound):
            raise Error("ModelSchemaError: non-finite BayesianGaussianMixture history")
    var warning = String(CONVERGENCE_WARNING) if model.n_iter_ > 0 and not model.converged_ else String("")
    if model.convergence_warning_ != warning:
        raise Error("ModelSchemaError: invalid BayesianGaussianMixture convergence warning")


def _check_bayesian_weights[dtype: DType](model: BayesianGaussianMixture[dtype]) raises:
    var concentration_count = len(model.weight_concentration_)
    var expected = mixture_weights(
        model.weight_concentration_[0],
        model.weight_concentration_[concentration_count - 1],
        concentration_count == 2,
    )
    var epsilon = 1e-6 if dtype == DType.float32 else 1e-12
    var total = 0.0
    for index in range(expected.size):
        var stored = Float64(model.weights_.unsafe_get(index))
        var weight = Float64(expected.unsafe_get(index))
        if not isfinite(stored) or not isfinite(weight) or stored < 0 or abs(weight - stored) > epsilon:
            raise Error("ModelSchemaError: BayesianGaussianMixture weights differ from their concentration")
        total += stored
    if abs(total - 1.0) > epsilon:
        raise Error("ModelSchemaError: invalid BayesianGaussianMixture weight sum")


struct BayesianGaussianMixtureSchema[dtype: DType = DType.float64](ModelSchema):
    """Version 1 of the experimental BayesianGaussianMixture model archive."""

    comptime Model = BayesianGaussianMixture[Self.dtype]
    comptime type_id = "moxie.mixture.BayesianGaussianMixture"
    comptime version = 1

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        _check_bayesian_state(model)
        var archive = ModelArchive(Self.type_id, Self.version)
        var writer = BinaryWriter()
        writer.integer(dtype_code[Self.dtype]())
        writer.integer(model.n_components)
        writer.string(model.covariance_type)
        writer.scalar[DType.float64](model.tol)
        writer.scalar[DType.float64](model.reg_covar)
        writer.integer(model.max_iter)
        writer.integer(model.n_init)
        writer.string(model.init_params)
        writer.string(model.weight_concentration_prior_type)
        _write_optional_scalar(writer, model.weight_concentration_prior)
        _write_optional_scalar(writer, model.mean_precision_prior)
        _write_optional_scalar(writer, model.degrees_of_freedom_prior)
        _write_optional_array(writer, archive, "mean_prior", model.mean_prior)
        _write_optional_array(writer, archive, "covariance_prior", model.covariance_prior)
        writer.boolean(Bool(model.random_state))
        if model.random_state:
            writer.integer(model.random_state.value())
        writer.boolean(model.warm_start)
        writer.integer(model.verbose)
        writer.integer(model.verbose_interval)
        var fitted = model.__sklearn_is_fitted__()
        writer.boolean(fitted)
        if fitted:
            writer.string(model._fitted_covariance_type)
            writer.integer(model.n_features_in_)
            writer.boolean(len(model.weight_concentration_) == 2)
            writer.scalar[DType.float64](model.weight_concentration_prior_)
            writer.scalar[DType.float64](model.mean_precision_prior_)
            writer.scalar[DType.float64](model.degrees_of_freedom_prior_)
            writer.boolean(model.converged_)
            writer.integer(model.n_iter_)
            writer.scalar[DType.float64](model.lower_bound_)
            writer.string(model.convergence_warning_)
            for lower_bound in model.lower_bounds_:
                writer.scalar[DType.float64](lower_bound)
            _write_array(archive, "weights_", model.weights_)
            _write_array(archive, "means_", model.means_)
            _write_array(archive, "covariances_", model.covariances_)
            _write_array(archive, "precisions_", model.precisions_)
            _write_array(archive, "precisions_cholesky_", model.precisions_cholesky_)
            _write_array(archive, "mean_prior_", model.mean_prior_)
            _write_array(archive, "mean_precision_", model.mean_precision_)
            _write_array(archive, "degrees_of_freedom_", model.degrees_of_freedom_)
            _write_array(archive, "covariance_prior_", model.covariance_prior_)
            _write_array(archive, "weight_concentration_a", model.weight_concentration_[0])
            if len(model.weight_concentration_) == 2:
                _write_array(archive, "weight_concentration_b", model.weight_concentration_[1])
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.type_id, Self.version)
        var reader = BinaryReader(Span(archive.manifest))
        if reader.integer() != dtype_code[Self.dtype]():
            raise Error("ModelSchemaError: BayesianGaussianMixture archive dtype mismatch")
        var components = reader.integer()
        var kind = reader.string()
        var tol = reader.scalar[DType.float64]()
        var reg_covar = reader.scalar[DType.float64]()
        var max_iter = reader.integer()
        var n_init = reader.integer()
        var init_params = reader.string()
        var prior_type = reader.string()
        var weight_prior = _read_optional_scalar(reader)
        var mean_precision_prior = _read_optional_scalar(reader)
        var dof_prior = _read_optional_scalar(reader)
        var mean_prior = _read_optional_array[Self.dtype](reader, archive, "mean_prior")
        var covariance_prior = _read_optional_array[Self.dtype](reader, archive, "covariance_prior")
        var seed: Optional[Int] = None
        if reader.boolean():
            seed = reader.integer()
        var warm_start = reader.boolean()
        var verbose = reader.integer()
        var verbose_interval = reader.integer()
        var fitted = reader.boolean()
        var expected_arrays = Int(Bool(mean_prior)) + Int(Bool(covariance_prior))
        var model = Self.Model(
            n_components=components, covariance_type=kind, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            weight_concentration_prior_type=prior_type, weight_concentration_prior=weight_prior,
            mean_precision_prior=mean_precision_prior, mean_prior=mean_prior^,
            degrees_of_freedom_prior=dof_prior, covariance_prior=covariance_prior^,
            random_state=seed, warm_start=warm_start, verbose=verbose,
            verbose_interval=verbose_interval,
        )
        if fitted:
            model._fitted_covariance_type = reader.string()
            model.n_features_in_ = reader.integer()
            var dirichlet_process = reader.boolean()
            expected_arrays += 10 + Int(dirichlet_process)
            if len(archive.payloads) != expected_arrays:
                raise Error("ModelSchemaError: unexpected BayesianGaussianMixture arrays")
            model.weight_concentration_prior_ = reader.scalar[DType.float64]()
            model.mean_precision_prior_ = reader.scalar[DType.float64]()
            model.degrees_of_freedom_prior_ = reader.scalar[DType.float64]()
            model.converged_ = reader.boolean()
            model.n_iter_ = reader.integer()
            model.lower_bound_ = reader.scalar[DType.float64]()
            model.convergence_warning_ = reader.string()
            if model.n_iter_ < 0 or model.n_iter_ != reader.remaining() // 8:
                raise Error("ModelSchemaError: invalid BayesianGaussianMixture history length")
            for _ in range(model.n_iter_):
                model.lower_bounds_.append(reader.scalar[DType.float64]())
            model.weights_ = _read_array[Self.dtype](archive, "weights_")
            model.means_ = _read_array[Self.dtype](archive, "means_")
            model.covariances_ = _read_array[Self.dtype](archive, "covariances_")
            model.precisions_ = _read_array[Self.dtype](archive, "precisions_")
            model.precisions_cholesky_ = _read_array[Self.dtype](archive, "precisions_cholesky_")
            model.mean_prior_ = _read_array[Self.dtype](archive, "mean_prior_")
            model.mean_precision_ = _read_array[Self.dtype](archive, "mean_precision_")
            model.degrees_of_freedom_ = _read_array[Self.dtype](archive, "degrees_of_freedom_")
            model.covariance_prior_ = _read_array[Self.dtype](archive, "covariance_prior_")
            model.weight_concentration_.append(_read_array[Self.dtype](archive, "weight_concentration_a"))
            if dirichlet_process:
                model.weight_concentration_.append(_read_array[Self.dtype](archive, "weight_concentration_b"))
            model._fitted = True
            _check_bayesian_state(model)
            _check_bayesian_weights(model)
            _check_precision_matrices(
                model._fitted_covariance_type, model.n_features_in_,
                model.covariances_, model.precisions_, model.precisions_cholesky_,
            )
        elif len(archive.payloads) != expected_arrays:
            raise Error("ModelSchemaError: unexpected BayesianGaussianMixture arrays")
        reader.finish()
        return model^
