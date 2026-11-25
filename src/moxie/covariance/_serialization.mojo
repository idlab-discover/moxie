"""Experimental schemas for dense covariance estimators."""
import numojo as nm
from numojo import NDArray, Shape
from std.math import isfinite, isnan

from moxie.covariance._elliptic_envelope import EllipticEnvelope
from moxie.covariance._empirical_covariance import CovarianceModel, EmpiricalCovariance
from moxie.covariance._engines import RobustCovarianceState
from moxie.covariance._graph_lasso import AlphaGrid, GraphicalLasso, GraphicalLassoCV, GraphicalLassoCVResults
from moxie.covariance._robust_covariance import MinCovDet, RobustFit
from moxie.covariance._shrunk_covariance_adaptive import LedoitWolf, OAS
from moxie.covariance._shrunk_covariance_fixed import ShrunkCovariance
from moxie.covariance._state import validate_location, validate_square
from moxie.serialization import ArrayPayload, BinaryReader, BinaryWriter, ModelArchive, ModelSchema, dtype_code


def _write_array[dtype: DType, wire_dtype: DType = dtype](
    mut archive: ModelArchive, name: String, array: NDArray[dtype],
) raises:
    var dimensions = List[Int]()
    var values = List[Scalar[wire_dtype]](capacity=array.size)
    dimensions.append(array.shape[0])
    var stride = Int(array.strides[0])
    if array.ndim == 1:
        for i in range(array.shape[0]):
            # NuMojo's flat item() skips strides for rank-one arrays.
            values.append(array.unsafe_get(i * stride).cast[wire_dtype]())
    else:
        dimensions.append(array.shape[1])
        var column_stride = Int(array.strides[1])
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                values.append(array.unsafe_get(i * stride + j * column_stride).cast[wire_dtype]())
    archive.add_array[wire_dtype](name, dimensions, values)


def _decode_array[dtype: DType](payload: ArrayPayload, shape: Shape) raises -> NDArray[dtype]:
    if len(payload.shape) != len(shape):
        raise Error(String("ModelSchemaError: invalid rank for ", payload.name))
    for axis in range(len(shape)):
        if payload.shape[axis] != shape[axis]:
            raise Error(String("ModelSchemaError: invalid shape for ", payload.name))
    var values = payload.values[dtype]()
    var array = nm.zeros[dtype](shape)
    for i in range(len(values)):
        array.unsafe_set(i, values[i])
    return array^


def _read_array[dtype: DType](
    archive: ModelArchive, name: String, shape: Shape,
) raises -> NDArray[dtype]:
    return _decode_array[dtype](archive.array(name), shape)


def _validate_widened[dtype: DType](
    public: NDArray[dtype], widened: NDArray[DType.float64], features: Int,
) raises:
    validate_square(widened, features)
    for i in range(features):
        for j in range(features):
            if public.item(i, j) != widened.item(i, j).cast[dtype]():
                raise Error("ModelSchemaError: widened matrix differs from its public snapshot")


def _validate_state[dtype: DType](model: CovarianceModel[dtype]) raises:
    if not model.is_fitted():
        if (model.state.fitted.is_fitted() or model.state.location_
            or model.state.precision_ or model._covariance64 or model._precision64):
            raise Error("ModelSchemaError: unfitted covariance has learned state")
        return
    if not model.state.fitted.is_fitted() or not model.state.location_:
        raise Error("ModelSchemaError: missing fitted covariance metadata")
    var features = model.state.fitted.n_features_in_.value()
    validate_location(model.state.location_.value(), features)
    validate_square(model.state.covariance_.value(), features)
    if model.state.precision_:
        validate_square(model.state.precision_.value(), features)
    if model._covariance64:
        _validate_widened(model.state.covariance_.value(), model._covariance64.value(), features)
        if Bool(model.state.precision_) != Bool(model._precision64):
            raise Error("ModelSchemaError: inconsistent widened precision")
        if model._precision64:
            _validate_widened(model.state.precision_.value(), model._precision64.value(), features)
    elif model._precision64:
        raise Error("ModelSchemaError: widened precision requires widened covariance")


def _write_state[dtype: DType](
    mut archive: ModelArchive, mut writer: BinaryWriter, model: CovarianceModel[dtype],
) raises:
    _validate_state(model)
    writer.boolean(model.is_fitted())
    if not model.is_fitted():
        return
    writer.integer(model.state.fitted.n_features_in_.value())
    writer.boolean(Bool(model.state.precision_))
    writer.boolean(Bool(model._covariance64))
    _write_array(archive, "location_", model.state.location_.value())
    _write_array(archive, "covariance_", model.state.covariance_.value())
    if model.state.precision_:
        _write_array(archive, "precision_", model.state.precision_.value())
    if model._covariance64:
        _write_array(archive, "covariance64", model._covariance64.value())
        if model._precision64:
            _write_array(archive, "precision64", model._precision64.value())


def _read_state[dtype: DType](
    archive: ModelArchive, mut reader: BinaryReader[_], allow_widened: Bool = True,
    extra_fitted_arrays: Int = 0,
) raises -> CovarianceModel[dtype]:
    var model = CovarianceModel[dtype]()

    if reader.boolean():
        var features = reader.integer()
        if features < 1:
            raise Error("ModelSchemaError: invalid feature count")
        var stored = reader.boolean()
        var widened = reader.boolean()
        if widened and not allow_widened:
            raise Error("ModelSchemaError: estimator has no widened covariance")
        var expected = 2 + Int(stored) + Int(widened) + Int(widened and stored) + extra_fitted_arrays
        if len(archive.payloads) != expected:
            raise Error("ModelSchemaError: unexpected covariance arrays")
        var matrix_shape = Shape(features, features)
        model.state.location_ = _read_array[dtype](archive, "location_", Shape(features))
        model.state.covariance_ = _read_array[dtype](archive, "covariance_", matrix_shape)
        if stored:
            model.state.precision_ = _read_array[dtype](archive, "precision_", matrix_shape)
        if widened:
            model._covariance64 = _read_array[DType.float64](archive, "covariance64", matrix_shape)
            if stored:
                model._precision64 = _read_array[DType.float64](archive, "precision64", matrix_shape)
        model.state.fitted.record_fit(1, features)
    elif len(archive.payloads) != 0:
        raise Error("ModelSchemaError: unfitted covariance has arrays")
    _validate_state(model)
    return model^


def _write_config[dtype: DType](store_precision: Bool, assume_centered: Bool) -> BinaryWriter:
    var writer = BinaryWriter()
    writer.uint(UInt64(dtype_code[dtype]()))
    writer.boolean(store_precision)
    writer.boolean(assume_centered)
    return writer^


def _check_dtype[dtype: DType](mut reader: BinaryReader[_]) raises:
    if reader.uint() != UInt64(dtype_code[dtype]()):
        raise Error("ModelSchemaError: covariance dtype mismatch")


def _write_shrinkage[dtype: DType](
    mut writer: BinaryWriter, model: CovarianceModel[dtype], shrinkage: Optional[Float64],
) raises:
    if model._covariance64 or Bool(shrinkage) != model.is_fitted():
        raise Error("ModelSchemaError: inconsistent adaptive shrinkage state")
    if shrinkage:
        if not isfinite(shrinkage.value()):
            raise Error("ModelSchemaError: nonfinite learned shrinkage")
        writer.scalar[DType.float64](shrinkage.value())


def _read_shrinkage(mut reader: BinaryReader[_], fitted: Bool) raises -> Optional[Float64]:
    if not fitted:
        return None
    var shrinkage = reader.scalar[DType.float64]()
    if not isfinite(shrinkage):
        raise Error("ModelSchemaError: nonfinite learned shrinkage")
    return shrinkage


struct EmpiricalCovarianceSchema[dtype: DType = DType.float64](ModelSchema):
    comptime Model = EmpiricalCovariance[Self.dtype]
    comptime TYPE_ID = "moxie.covariance.EmpiricalCovariance"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TYPE_ID, 1)
        var writer = _write_config[Self.dtype](model.store_precision, model.assume_centered)
        _write_state(archive, writer, model._model)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.TYPE_ID, 1)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(
            store_precision=reader.boolean(),
            assume_centered=reader.boolean(),
        )
        model._model = _read_state[Self.dtype](archive, reader)
        reader.finish()
        return model^


struct ShrunkCovarianceSchema[dtype: DType = DType.float64](ModelSchema):
    comptime Model = ShrunkCovariance[Self.dtype]
    comptime TYPE_ID = "moxie.covariance.ShrunkCovariance"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TYPE_ID, 1)
        var writer = _write_config[Self.dtype](model.store_precision, model.assume_centered)
        writer.scalar[DType.float64](model.shrinkage)
        _write_state(archive, writer, model._model)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.TYPE_ID, 1)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(
            store_precision=reader.boolean(),
            assume_centered=reader.boolean(),
            shrinkage=reader.scalar[DType.float64](),
        )
        model._model = _read_state[Self.dtype](archive, reader)
        reader.finish()
        return model^


struct LedoitWolfSchema[dtype: DType = DType.float64](ModelSchema):
    comptime Model = LedoitWolf[Self.dtype]
    comptime TYPE_ID = "moxie.covariance.LedoitWolf"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TYPE_ID, 1)
        var writer = _write_config[Self.dtype](model.store_precision, model.assume_centered)
        writer.integer(model.block_size)
        _write_state(archive, writer, model._model)
        _write_shrinkage(writer, model._model, model._shrinkage)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.TYPE_ID, 1)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(
            store_precision=reader.boolean(),
            assume_centered=reader.boolean(),
            block_size=reader.integer(),
        )
        model._model = _read_state[Self.dtype](archive, reader, allow_widened=False)
        model._shrinkage = _read_shrinkage(reader, model.__sklearn_is_fitted__())
        reader.finish()
        return model^


struct OASSchema[dtype: DType = DType.float64](ModelSchema):
    comptime Model = OAS[Self.dtype]
    comptime TYPE_ID = "moxie.covariance.OAS"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TYPE_ID, 1)
        var writer = _write_config[Self.dtype](model.store_precision, model.assume_centered)
        _write_state(archive, writer, model._model)
        _write_shrinkage(writer, model._model, model._shrinkage)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.TYPE_ID, 1)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(
            store_precision=reader.boolean(),
            assume_centered=reader.boolean(),
        )
        model._model = _read_state[Self.dtype](archive, reader, allow_widened=False)
        model._shrinkage = _read_shrinkage(reader, model.__sklearn_is_fitted__())
        reader.finish()
        return model^


def _write_robust_config[dtype: DType](
    store_precision: Bool, assume_centered: Bool, support_fraction: Optional[Float64],
    random_state: Optional[Int],
) -> BinaryWriter:
    var writer = _write_config[dtype](store_precision, assume_centered)
    writer.boolean(Bool(support_fraction))
    if support_fraction:
        writer.scalar[DType.float64](support_fraction.value())
    writer.boolean(Bool(random_state))
    if random_state:
        writer.integer(random_state.value())
    return writer^


def _read_support_fraction(mut reader: BinaryReader[_]) raises -> Optional[Float64]:
    if reader.boolean():
        return reader.scalar[DType.float64]()
    return None


def _read_random_state(mut reader: BinaryReader[_]) raises -> Optional[Int]:
    if reader.boolean():
        return reader.integer()
    return None


def _validate_mask(mask: NDArray[DType.bool], samples: Int) raises:
    if mask.ndim != 1 or mask.shape[0] != samples:
        raise Error("ModelSchemaError: invalid robust support shape")
    for i in range(samples):
        if mask.unsafe_get(i * Int(mask.strides[0])):
            return
    raise Error("ModelSchemaError: empty robust support")


def _read_mask(archive: ModelArchive, name: String, samples: Int) raises -> NDArray[DType.bool]:
    var payload = archive.array(name)
    if len(payload.shape) != 1 or payload.shape[0] != samples:
        raise Error(String("ModelSchemaError: invalid shape for ", name))
    var values = payload.values[DType.uint32]()
    var mask = nm.zeros[DType.bool](Shape(samples))
    for i in range(samples):
        if values[i] > 1:
            raise Error(String("ModelSchemaError: invalid boolean value in ", name))
        mask.unsafe_set(i, values[i] != 0)
    return mask^


def _validate_robust[dtype: DType](fit: RobustFit[dtype]) raises:
    if Bool(fit.robust) != fit.model.is_fitted():
        raise Error("ModelSchemaError: inconsistent robust fitted state")
    if fit.robust:
        ref robust = fit.robust.value()
        if not fit.model._covariance64:
            raise Error("ModelSchemaError: missing widened robust covariance")
        var features = fit.model.state.fitted.n_features_in_.value()
        var samples = robust.dist_.size
        if robust.dist_.ndim != 1 or samples < 2:
            raise Error("ModelSchemaError: invalid robust distance shape")
        validate_location(robust.raw_location_, features)
        validate_square(robust.raw_covariance_, features)
        _validate_mask(robust.raw_support_, samples)
        _validate_mask(robust.support_, samples)


def _write_robust[dtype: DType](
    mut archive: ModelArchive, mut writer: BinaryWriter, fit: RobustFit[dtype],
) raises:
    _write_state(archive, writer, fit.model)
    _validate_robust(fit)
    if fit.robust:
        ref robust = fit.robust.value()
        writer.integer(robust.dist_.size)
        _write_array(archive, "raw_location_", robust.raw_location_)
        _write_array(archive, "raw_covariance_", robust.raw_covariance_)
        _write_array[DType.bool, DType.uint32](archive, "raw_support_", robust.raw_support_)
        _write_array[DType.bool, DType.uint32](archive, "support_", robust.support_)
        _write_array(archive, "dist_", robust.dist_)


def _read_robust[dtype: DType](
    archive: ModelArchive, mut reader: BinaryReader[_],
) raises -> RobustFit[dtype]:
    var fit = RobustFit[dtype]()
    fit.model = _read_state[dtype](archive, reader, extra_fitted_arrays=5)
    if fit.model.is_fitted():
        var features = fit.model.state.fitted.n_features_in_.value()
        var samples = reader.integer()
        if samples < 2:
            raise Error("ModelSchemaError: invalid robust distance shape")
        fit.robust = RobustCovarianceState[dtype](
            fit.model.state.copy(),
            _read_array[dtype](archive, "raw_location_", Shape(features)),
            _read_array[dtype](archive, "raw_covariance_", Shape(features, features)),
            _read_mask(archive, "raw_support_", samples),
            _read_mask(archive, "support_", samples),
            _read_array[dtype](archive, "dist_", Shape(samples)),
        )
    _validate_robust(fit)
    return fit^


def _validate_offset(fitted: Bool, offset: Float64) raises:
    if not isfinite(offset) or (not fitted and offset != 0.0):
        raise Error("ModelSchemaError: invalid elliptic envelope offset")


struct MinCovDetSchema[dtype: DType = DType.float64](ModelSchema):
    comptime Model = MinCovDet[Self.dtype]
    comptime TYPE_ID = "moxie.covariance.MinCovDet"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TYPE_ID, 1)
        var writer = _write_robust_config[Self.dtype](
            model.store_precision, model.assume_centered, model.support_fraction, model.random_state,
        )
        _write_robust(archive, writer, model._fit)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.TYPE_ID, 1)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(
            store_precision=reader.boolean(),
            assume_centered=reader.boolean(),
            support_fraction=_read_support_fraction(reader),
            random_state=_read_random_state(reader),
        )
        model._fit = _read_robust[Self.dtype](archive, reader)
        reader.finish()
        return model^


struct EllipticEnvelopeSchema[dtype: DType = DType.float64](ModelSchema):
    comptime Model = EllipticEnvelope[Self.dtype]
    comptime TYPE_ID = "moxie.covariance.EllipticEnvelope"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TYPE_ID, 1)
        var writer = _write_robust_config[Self.dtype](
            model.store_precision, model.assume_centered, model.support_fraction, model.random_state,
        )
        writer.scalar[DType.float64](model.contamination)
        _write_robust(archive, writer, model._fit)
        var fitted = model.__sklearn_is_fitted__()
        _validate_offset(fitted, model._offset)
        if fitted:
            writer.scalar[DType.float64](model._offset)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.TYPE_ID, 1)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(
            store_precision=reader.boolean(),
            assume_centered=reader.boolean(),
            support_fraction=_read_support_fraction(reader),
            random_state=_read_random_state(reader),
            contamination=reader.scalar[DType.float64](),
        )
        model._fit = _read_robust[Self.dtype](archive, reader)
        if model.__sklearn_is_fitted__():
            model._offset = reader.scalar[DType.float64]()
            _validate_offset(True, model._offset)
        reader.finish()
        return model^


def _validate_graphical_state[dtype: DType](
    model: CovarianceModel[dtype], n_iter: Int, n_costs: Int,
) raises:
    if not model.is_fitted():
        if n_iter != 0 or n_costs != 0:
            raise Error("ModelSchemaError: unfitted graphical model has solver diagnostics")
        return
    if not model._covariance64 or not model._precision64 or not model.state.precision_:
        raise Error("ModelSchemaError: graphical state requires stored, widened estimates")
    if model.state.fitted.n_features_in_.value() < 2:
        raise Error("ModelSchemaError: graphical state requires at least two features")
    # The alpha-zero solve records one cost and zero iterations. A zero
    # iteration budget on the iterative path records no costs and n_iter=1.
    if not ((n_iter > 0 and n_costs == n_iter)
            or (n_iter == 0 and n_costs == 1)
            or (n_iter == 1 and n_costs == 0)):
        raise Error("ModelSchemaError: inconsistent graphical iteration count")


def _write_graphical_state[dtype: DType](
    mut archive: ModelArchive, mut writer: BinaryWriter, model: CovarianceModel[dtype],
    n_iter: Int, costs: List[Tuple[Float64, Float64]],
) raises:
    _write_state(archive, writer, model)
    _validate_graphical_state(model, n_iter, len(costs))
    if model.is_fitted():
        writer.integer(n_iter)
        var values = List[Float64](capacity=2 * len(costs))
        for pair in costs:
            values.append(pair[0])
            values.append(pair[1])
        var shape: List[Int] = [len(costs), 2]
        archive.add_array[DType.float64]("costs_", shape, values)


def _read_graphical_costs[dtype: DType](
    archive: ModelArchive, model: CovarianceModel[dtype], n_iter: Int,
) raises -> List[Tuple[Float64, Float64]]:
    var costs = List[Tuple[Float64, Float64]]()
    if model.is_fitted():
        var payload = archive.array("costs_")
        if len(payload.shape) != 2 or payload.shape[1] != 2:
            raise Error("ModelSchemaError: invalid shape for costs_")
        _validate_graphical_state(model, n_iter, payload.shape[0])
        var values = payload.values[DType.float64]()
        for i in range(0, len(values), 2):
            costs.append((values[i], values[i + 1]))
    return costs^


def _write_optional_int(mut writer: BinaryWriter, value: Optional[Int]):
    writer.boolean(Bool(value))
    if value:
        writer.integer(value.value())


def _read_optional_int(mut reader: BinaryReader[_]) raises -> Optional[Int]:
    if reader.boolean():
        return reader.integer()
    return None


def _write_alpha_grid(mut writer: BinaryWriter, grid: AlphaGrid):
    writer.integer(grid.count)
    writer.boolean(Bool(grid.values))
    if grid.values:
        writer.uint(UInt64(len(grid.values.value())))
        for alpha in grid.values.value():
            writer.scalar[DType.float64](alpha)


def _read_alpha_grid(mut reader: BinaryReader[_]) raises -> AlphaGrid:
    var grid = AlphaGrid(reader.integer())
    if reader.boolean():
        var count = reader.count(reader.remaining() // 8)
        var values = List[Float64](capacity=count)
        for _ in range(count):
            values.append(reader.scalar[DType.float64]())
        grid.values = values^
    return grid^


def _validate_cv_results(alpha: Float64, results: GraphicalLassoCVResults) raises:
    if results.alphas.ndim != 1 or results.alphas.size < 2:
        raise Error("ModelSchemaError: CV results require a path and a baseline")
    var count = results.alphas.size
    if (results.split_test_score.ndim != 2 or results.split_test_score.shape[0] != count
        or results.split_test_score.shape[1] < 1
        or results.mean_test_score.ndim != 1 or results.mean_test_score.size != count
        or results.std_test_score.ndim != 1 or results.std_test_score.size != count):
        raise Error("ModelSchemaError: inconsistent CV score shapes")
    var found = False
    var alpha_stride = Int(results.alphas.strides[0])
    for i in range(count - 1):
        var current = results.alphas.unsafe_get(i * alpha_stride)
        if not (current > 0) or current < results.alphas.unsafe_get((i + 1) * alpha_stride):
            raise Error("ModelSchemaError: expected positive CV alphas in descending order")
        found = found or current == alpha
    if not found or results.alphas.unsafe_get((count - 1) * alpha_stride) != 0:
        raise Error("ModelSchemaError: missing selected alpha or zero-alpha baseline")
    # Failed folds can produce -inf scores and NaN standard deviations.
    var deviation_stride = Int(results.std_test_score.strides[0])
    for i in range(count):
        var deviation = results.std_test_score.unsafe_get(i * deviation_stride)
        if not isnan(deviation) and deviation < 0:
            raise Error("ModelSchemaError: negative CV score deviation")


def _read_cv_results(archive: ModelArchive, alpha: Float64) raises -> GraphicalLassoCVResults:
    var split = archive.array("split_test_score")
    if len(split.shape) != 2 or split.shape[0] < 2 or split.shape[1] < 1:
        raise Error("ModelSchemaError: invalid shape for split_test_score")
    var rows = Shape(split.shape[0])
    var results = GraphicalLassoCVResults(
        _read_array[DType.float64](archive, "alphas", rows),
        _decode_array[DType.float64](split, Shape(split.shape[0], split.shape[1])),
        _read_array[DType.float64](archive, "mean_test_score", rows),
        _read_array[DType.float64](archive, "std_test_score", rows),
    )
    _validate_cv_results(alpha, results)
    return results^


struct GraphicalLassoSchema[dtype: DType = DType.float64](ModelSchema):
    comptime Model = GraphicalLasso[Self.dtype]
    comptime TYPE_ID = "moxie.covariance.GraphicalLasso"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TYPE_ID, 1)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        writer.scalar[DType.float64](model.alpha)
        writer.string(model.mode)
        writer.boolean(Bool(model.covariance))
        if model.covariance:
            writer.string(model.covariance.value())
        writer.scalar[DType.float64](model.tol)
        writer.scalar[DType.float64](model.enet_tol)
        writer.integer(model.max_iter)
        writer.boolean(model.verbose)
        writer.scalar[DType.float64](model.eps)
        writer.boolean(model.assume_centered)
        _write_graphical_state(archive, writer, model._model, model._n_iter, model._costs)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.TYPE_ID, 1)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        _check_dtype[Self.dtype](reader)
        var alpha = reader.scalar[DType.float64]()
        var mode = reader.string(reader.remaining())
        var covariance: Optional[String] = None
        if reader.boolean():
            covariance = reader.string(reader.remaining())
        var model = Self.Model(
            alpha, mode=mode^, covariance=covariance^,
            tol=reader.scalar[DType.float64](), enet_tol=reader.scalar[DType.float64](),
            max_iter=reader.integer(), verbose=reader.boolean(),
            eps=reader.scalar[DType.float64](), assume_centered=reader.boolean(),
        )
        model._model = _read_state[Self.dtype](archive, reader, extra_fitted_arrays=1)
        if model.__sklearn_is_fitted__():
            model._n_iter = reader.integer()
            model._costs = _read_graphical_costs(archive, model._model, model._n_iter)
        reader.finish()
        return model^


struct GraphicalLassoCVSchema[dtype: DType = DType.float64](ModelSchema):
    comptime Model = GraphicalLassoCV[Self.dtype]
    comptime TYPE_ID = "moxie.covariance.GraphicalLassoCV"

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(Self.TYPE_ID, 1)
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        _write_alpha_grid(writer, model.alphas)
        writer.integer(model.n_refinements)
        _write_optional_int(writer, model.cv)
        writer.scalar[DType.float64](model.tol)
        writer.scalar[DType.float64](model.enet_tol)
        writer.integer(model.max_iter)
        writer.string(model.mode)
        _write_optional_int(writer, model.n_jobs)
        writer.boolean(model.verbose)
        writer.scalar[DType.float64](model.eps)
        writer.boolean(model.assume_centered)
        _write_graphical_state(archive, writer, model._model, model._n_iter, model._costs)
        if Bool(model._results) != model.__sklearn_is_fitted__():
            raise Error("ModelSchemaError: inconsistent CV fitted state")
        if model._results:
            ref results = model._results.value()
            _validate_cv_results(model._alpha, results)
            writer.scalar[DType.float64](model._alpha)
            _write_array(archive, "alphas", results.alphas)
            _write_array(archive, "split_test_score", results.split_test_score)
            _write_array(archive, "mean_test_score", results.mean_test_score)
            _write_array(archive, "std_test_score", results.std_test_score)
        elif model._alpha != 0:
            raise Error("ModelSchemaError: unfitted CV model has a selected alpha")
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.TYPE_ID, 1)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        _check_dtype[Self.dtype](reader)
        var model = Self.Model(
            alphas=_read_alpha_grid(reader), n_refinements=reader.integer(),
            cv=_read_optional_int(reader),
            tol=reader.scalar[DType.float64](), enet_tol=reader.scalar[DType.float64](),
            max_iter=reader.integer(), mode=reader.string(reader.remaining()),
            n_jobs=_read_optional_int(reader), verbose=reader.boolean(),
            eps=reader.scalar[DType.float64](), assume_centered=reader.boolean(),
        )
        model._model = _read_state[Self.dtype](archive, reader, extra_fitted_arrays=5)
        if model.__sklearn_is_fitted__():
            model._n_iter = reader.integer()
            model._costs = _read_graphical_costs(archive, model._model, model._n_iter)
            model._alpha = reader.scalar[DType.float64]()
            model._results = _read_cv_results(archive, model._alpha)
        reader.finish()
        return model^
