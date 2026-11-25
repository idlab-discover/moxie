"""Native state schemas for the dense cross-decomposition estimators."""
from moxie.serialization import (
    BinaryReader,
    BinaryWriter,
    ModelArchive,
    ModelSchema,
    dtype_code,
)
from moxie.metadata_routing import SampleWeightRequest
from moxie.warnings import FitWarning
from ._contracts import Matrix
from ._pls import (
    _CANONICAL,
    _CCA,
    _CONVERGENCE_WARNING,
    _PLS,
    _REGRESSION,
    _USER_WARNING,
    PLSParams,
    _at,
    _matrix,
    _vat,
    _vector,
)
from ._plssvd import PLSSVD, PLSSVDParams


comptime _VERSION = 1
comptime _PLS_REQUEST_VERSION = 2
comptime _PREFIX = "moxie.cross_decomposition."


def _write_array(
    mut archive: ModelArchive, name: String, array: Matrix,
    rows: Int, cols: Int = -1, *, positive: Bool = False,
) raises:
    var shape: List[Int] = [rows]
    if cols >= 0:
        shape.append(cols)
    if array.ndim != len(shape):
        raise Error("ModelSchemaError: invalid rank for " + name)
    for axis in range(len(shape)):
        if array.shape[axis] != shape[axis]:
            raise Error("ModelSchemaError: invalid shape for " + name)
    var values = List[Float64](capacity=array.shape.size())
    for i in range(rows):
        if cols < 0:
            var value = _vat(array, i)
            if positive and not (value > 0):
                raise Error("ModelSchemaError: nonpositive scale in " + name)
            values.append(value)
        else:
            for j in range(cols):
                values.append(_at(array, i, j))
    archive.add_array[DType.float64](name, shape, values)


def _read_array(
    archive: ModelArchive, name: String,
    rows: Int, cols: Int = -1, *, positive: Bool = False,
) raises -> Matrix:
    for payload in archive.payloads:
        if payload.name != name:
            continue
        if cols < 0:
            if len(payload.shape) != 1 or payload.shape[0] != rows:
                raise Error("ModelSchemaError: invalid vector shape for " + name)
        else:
            if (
                len(payload.shape) != 2
                or payload.shape[0] != rows
                or payload.shape[1] != cols
            ):
                raise Error("ModelSchemaError: invalid matrix shape for " + name)
        if payload.dtype != dtype_code[DType.float64]():
            raise Error("ModelSchemaError: invalid dtype for " + name)
        # Decode from the validated payload directly into owned storage.
        var result = _vector(rows) if cols < 0 else _matrix(rows, cols)
        var reader = BinaryReader(Span(payload.data))
        for i in range(result.size):
            var value = reader.scalar[DType.float64]()
            if positive and not (value > 0):
                raise Error("ModelSchemaError: nonpositive scale in " + name)
            result.unsafe_set(i, value)
        return result^
    raise Error("ModelSchemaError: missing array " + name)


def _check_input_dtype[dtype: DType](mut reader: BinaryReader) raises:
    if reader.uint() != UInt64(dtype_code[dtype]()):
        raise Error("ModelSchemaError: input dtype mismatch")


def _expect_arrays(archive: ModelArchive, count: Int) raises:
    if len(archive.payloads) != count:
        raise Error("ModelSchemaError: unexpected number of learned arrays")


def _check_dimensions(features: Int, targets: Int, components: Int, bound: Int) raises:
    if features < 1 or targets < 1 or components < 1 or components > bound:
        raise Error("ModelSchemaError: invalid learned dimensions")


def _check_weight_ranks(x_weights: Matrix, y_weights: Matrix) raises:
    if x_weights.ndim != 2 or y_weights.ndim != 2:
        raise Error("ModelSchemaError: invalid weight ranks")


def _read_score_request(mut reader: BinaryReader) raises -> SampleWeightRequest:
    var request = SampleWeightRequest()
    if reader.boolean():
        var alias_name = reader.string(reader.remaining())
        try:
            request.set(alias_name.copy())
        except:
            raise Error("ModelSchemaError: invalid PLS sample-weight alias")
        # The setter also accepts the unchanged sentinel and the canonical name.
        if not request.value().isa[String]():
            raise Error("ModelSchemaError: invalid PLS sample-weight alias")
    else:
        request.set(reader.boolean())
    return request^


struct _PLSSchema[flavor: Int, dtype: DType = DType.float64](ModelSchema):
    comptime Model = _PLS[Self.flavor, Self.dtype]

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(_PREFIX + Self.Model.name(), _VERSION)
        var writer = BinaryWriter()
        var params = model.get_params()
        model._check_algorithm_parameter(params.algorithm)
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        writer.integer(params.n_components)
        writer.boolean(params.scale)
        writer.string(params.algorithm)
        writer.integer(params.max_iter)
        writer.scalar(params.tol)
        writer.boolean(params.copy_input)
        var fitted = model.__sklearn_is_fitted__()
        writer.boolean(fitted)
        if fitted:
            # Settings may have changed since fit, including n_components.
            _check_weight_ranks(model.x_weights_, model.y_weights_)
            if model.x_scores_.ndim != 2:
                raise Error("ModelSchemaError: invalid training score rank")
            var p = model.n_features_in_
            var q = model.y_weights_.shape[0]
            var k = model.x_weights_.shape[1]
            var n = model.x_scores_.shape[0]
            var bound = min(n, p) if Self.flavor == _REGRESSION else min(n, min(p, q))
            _check_dimensions(p, q, k, bound)
            if n < 2 or (model._predict_1d and q != 1):
                raise Error("ModelSchemaError: invalid training or prediction shape")
            writer.integer(p)
            writer.integer(q)
            writer.integer(k)
            writer.integer(n)
            writer.boolean(model._predict_1d)
            if len(model.n_iter_) > k or len(model.diagnostics_) > k:
                raise Error("ModelSchemaError: too many iterations or diagnostics")
            writer.uint(UInt64(len(model.n_iter_)))
            for iterations in model.n_iter_:
                if iterations < 1:
                    raise Error("ModelSchemaError: invalid iteration count")
                writer.integer(iterations)
            writer.uint(UInt64(len(model.diagnostics_)))
            for diagnostic in model.diagnostics_:
                writer.string(diagnostic)

            _write_array(archive, "x_weights_", model.x_weights_, p, k)
            _write_array(archive, "y_weights_", model.y_weights_, q, k)
            _write_array(archive, "x_loadings_", model.x_loadings_, p, k)
            _write_array(archive, "y_loadings_", model.y_loadings_, q, k)
            _write_array(archive, "x_scores_", model.x_scores_, n, k)
            _write_array(archive, "y_scores_", model.y_scores_, n, k)
            _write_array(archive, "x_rotations_", model.x_rotations_, p, k)
            _write_array(archive, "y_rotations_", model.y_rotations_, q, k)
            _write_array(archive, "coef_", model.coef_, q, p)
            _write_array(archive, "intercept_", model.intercept_, q)
            _write_array(archive, "_x_mean", model._x_mean, p)
            _write_array(archive, "_y_mean", model._y_mean, q)
            _write_array(archive, "_x_std", model._x_std, p, positive=True)
            _write_array(archive, "_y_std", model._y_std, q, positive=True)
        var request = model._score_request.value()
        if not request.isa[NoneType]():
            archive.schema_version = _PLS_REQUEST_VERSION
            var is_alias = request.isa[String]()
            writer.boolean(is_alias)
            if is_alias:
                writer.string(request[String])
            else:
                writer.boolean(request[Bool])
        archive.manifest = writer^.finish()
        archive.validate()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        var has_request = archive.schema_version == _PLS_REQUEST_VERSION
        archive.expect(
            _PREFIX + Self.Model.name(),
            _PLS_REQUEST_VERSION if has_request else _VERSION,
        )
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        _check_input_dtype[Self.dtype](reader)
        var params = PLSParams(
            reader.integer(), reader.boolean(), reader.string(reader.remaining()),
            reader.integer(), reader.scalar[DType.float64](), reader.boolean(),
        )
        var model = Self.Model(params=params)
        if not reader.boolean():
            if has_request:
                model._score_request = _read_score_request(reader)
            reader.finish()
            _expect_arrays(archive, 0)
            return model^

        _expect_arrays(archive, 14)
        var p = reader.integer()
        var q = reader.integer()
        var k = reader.integer()
        var n = reader.integer()
        var bound = min(n, p) if Self.flavor == _REGRESSION else min(n, min(p, q))
        _check_dimensions(p, q, k, bound)
        model._predict_1d = reader.boolean()
        if n < 2 or (model._predict_1d and q != 1):
            raise Error("ModelSchemaError: invalid training or prediction shape")
        var iterations_count = reader.count(min(k, reader.remaining() // 8))
        for _ in range(iterations_count):
            var iterations = reader.integer()
            if iterations < 1:
                raise Error("ModelSchemaError: invalid iteration count")
            model.n_iter_.append(iterations)
        var diagnostics_count = reader.count(min(k, reader.remaining() // 8))
        for _ in range(diagnostics_count):
            var diagnostic = reader.string(reader.remaining())
            # Archives store warning categories and messages together.
            for category in [_CONVERGENCE_WARNING, _USER_WARNING]:
                var prefix = String(category) + ": "
                if diagnostic.startswith(prefix):
                    model.fit_warnings.append(
                        FitWarning(String(category), String(diagnostic.removeprefix(prefix)))
                    )
                    break
            model.diagnostics_.append(diagnostic^)
        if has_request:
            model._score_request = _read_score_request(reader)
        reader.finish()

        model.x_weights_ = _read_array(archive, "x_weights_", p, k)
        model.y_weights_ = _read_array(archive, "y_weights_", q, k)
        model.x_loadings_ = _read_array(archive, "x_loadings_", p, k)
        model.y_loadings_ = _read_array(archive, "y_loadings_", q, k)
        model.x_scores_ = _read_array(archive, "x_scores_", n, k)
        model.y_scores_ = _read_array(archive, "y_scores_", n, k)
        model.x_rotations_ = _read_array(archive, "x_rotations_", p, k)
        model.y_rotations_ = _read_array(archive, "y_rotations_", q, k)
        model.coef_ = _read_array(archive, "coef_", q, p)
        model.intercept_ = _read_array(archive, "intercept_", q)
        model._x_mean = _read_array(archive, "_x_mean", p)
        model._y_mean = _read_array(archive, "_y_mean", q)
        model._x_std = _read_array(archive, "_x_std", p, positive=True)
        model._y_std = _read_array(archive, "_y_std", q, positive=True)
        model.n_features_in_ = p
        model._fitted = True
        return model^


comptime PLSRegressionSchema[dtype: DType = DType.float64] = _PLSSchema[
    _REGRESSION, dtype
]
"""Native model archive schema for PLSRegression of the selected input dtype."""

comptime PLSCanonicalSchema[dtype: DType = DType.float64] = _PLSSchema[
    _CANONICAL, dtype
]
"""Native model archive schema for PLSCanonical of the selected input dtype."""

comptime CCASchema[dtype: DType = DType.float64] = _PLSSchema[_CCA, dtype]
"""Native model archive schema for CCA of the selected input dtype."""


struct PLSSVDSchema[dtype: DType = DType.float64](ModelSchema):
    """Native model archive schema for PLSSVD of the selected input dtype."""

    comptime Model = PLSSVD[Self.dtype]

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        var archive = ModelArchive(_PREFIX + Self.Model.name(), _VERSION)
        var writer = BinaryWriter()
        var params = model.get_params()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        writer.integer(params.n_components)
        writer.boolean(params.scale)
        writer.boolean(params.copy_input)
        var fitted = model.__sklearn_is_fitted__()
        writer.boolean(fitted)
        if fitted:
            _check_weight_ranks(model.x_weights_, model.y_weights_)
            var p = model.n_features_in_
            var q = model.y_weights_.shape[0]
            var k = model.x_weights_.shape[1]
            _check_dimensions(p, q, k, min(p, q))
            writer.integer(p)
            writer.integer(q)
            writer.integer(k)
            _write_array(archive, "x_weights_", model.x_weights_, p, k)
            _write_array(archive, "y_weights_", model.y_weights_, q, k)
            _write_array(archive, "_x_mean", model._x_mean, p)
            _write_array(archive, "_y_mean", model._y_mean, q)
            _write_array(archive, "_x_std", model._x_std, p, positive=True)
            _write_array(archive, "_y_std", model._y_std, q, positive=True)
        archive.manifest = writer^.finish()
        archive.validate()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(_PREFIX + Self.Model.name(), _VERSION)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        _check_input_dtype[Self.dtype](reader)
        var params = PLSSVDParams(reader.integer(), reader.boolean(), reader.boolean())
        var model = Self.Model(params=params)
        if not reader.boolean():
            reader.finish()
            _expect_arrays(archive, 0)
            return model^

        _expect_arrays(archive, 6)
        var p = reader.integer()
        var q = reader.integer()
        var k = reader.integer()
        reader.finish()
        _check_dimensions(p, q, k, min(p, q))
        model.x_weights_ = _read_array(archive, "x_weights_", p, k)
        model.y_weights_ = _read_array(archive, "y_weights_", q, k)
        model._x_mean = _read_array(archive, "_x_mean", p)
        model._y_mean = _read_array(archive, "_y_mean", q)
        model._x_std = _read_array(archive, "_x_std", p, positive=True)
        model._y_std = _read_array(archive, "_y_std", q, positive=True)
        model.n_features_in_ = p
        model._fitted = True
        return model^
