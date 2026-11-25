"""Experimental native schemas for built-in kernels and dense GP regression."""

from numojo import NDArray, Shape, zeros
from std.math import isfinite
from moxie._optimize import LBFGSBConfig

from moxie.serialization import (
    BinaryReader,
    BinaryWriter,
    ModelArchive,
    ModelSchema,
    dtype_code,
)

from ._gpr import FminLBFGSB, GenericGaussianProcessRegressor, ScalarOrVector
from .kernels import Bounds, Kernel, _Kind, _Node, _check_pairwise_metric

# Kernel evaluation and parameter inspection recurse over the loaded tree.
comptime _MAX_KERNEL_DEPTH = 256


def _write_optional_int(mut writer: BinaryWriter, value: Optional[Int]):
    writer.boolean(Bool(value))
    if value:
        writer.integer(value.value())


def _read_optional_int(mut reader: BinaryReader[_]) raises -> Optional[Int]:
    if reader.boolean():
        return reader.integer()
    return None


def _write_floats(mut writer: BinaryWriter, values: List[Float64]):
    writer.uint(UInt64(len(values)))
    for value in values:
        writer.scalar[DType.float64](value)


def _read_floats(mut reader: BinaryReader[_]) raises -> List[Float64]:
    var count = reader.count(reader.remaining() // 8)
    var values = List[Float64](capacity=count)
    for _ in range(count):
        values.append(reader.scalar[DType.float64]())
    return values^


def _write_bounds(mut writer: BinaryWriter, bounds: Bounds, elements: Int) raises:
    writer.boolean(bounds.is_fixed)
    if bounds.is_fixed:
        return
    if len(bounds.lower) != len(bounds.upper) or (
        len(bounds.lower) != 1 and len(bounds.lower) != elements
    ):
        raise Error("ModelSchemaError: invalid kernel bounds shape")
    _write_floats(writer, bounds.lower)
    _write_floats(writer, bounds.upper)


def _read_bounds(mut reader: BinaryReader[_], elements: Int) raises -> Bounds:
    if reader.boolean():
        return Bounds.fixed()
    var lower = _read_floats(reader)
    var upper = _read_floats(reader)
    if len(lower) != 1 and len(lower) != elements:
        raise Error("ModelSchemaError: invalid kernel bounds shape")
    return Bounds(lower, upper)


def _check_arity(kind: Int, count: Int) raises:
    if kind < _Kind.CONSTANT or kind > _Kind.COMPOUND:
        raise Error("ModelSchemaError: unknown kernel kind")
    if (
        (kind <= _Kind.PAIRWISE and count != 0)
        or ((kind == _Kind.SUM or kind == _Kind.PRODUCT) and count != 2)
        or (kind == _Kind.EXPONENTIATION and count != 1)
        or (kind == _Kind.COMPOUND and count == 0)
    ):
        raise Error("ModelSchemaError: invalid kernel node arity")


def _check_leaf_size(kind: Int, count: Int) raises:
    if count == 0 or (count != 1 and kind != _Kind.RBF and kind != _Kind.MATERN):
        raise Error("ModelSchemaError: invalid kernel parameter count")


def _write_node(mut writer: BinaryWriter, node: _Node) raises:
    var kind = node.kind
    writer.integer(kind)
    writer.uint(UInt64(len(node.children)))
    for child in node.children:
        writer.integer(child)
    if kind <= _Kind.PAIRWISE:
        _check_leaf_size(kind, len(node.values))
        _write_floats(writer, node.values)
        _write_bounds(writer, node.bounds, len(node.values))
    if kind == _Kind.RATIONAL_QUADRATIC or kind == _Kind.EXP_SINE_SQUARED:
        writer.scalar[DType.float64](node.value2)
        _write_bounds(writer, node.bounds2, 1)
    if kind == _Kind.MATERN or kind == _Kind.EXPONENTIATION:
        writer.scalar[DType.float64](node.extra)
    if kind == _Kind.EXPONENTIATION:
        if node.integer_exponent and Float64(node.integer_exponent.value()) != node.extra:
            raise Error("ModelSchemaError: inconsistent integer kernel exponent")
        _write_optional_int(writer, node.integer_exponent)
    if kind == _Kind.PAIRWISE:
        _check_pairwise_metric(node.metric)
        writer.string(node.metric)
        writer.boolean(node.has_kwargs)
        if node.has_kwargs:
            writer.uint(UInt64(len(node.kwargs)))
            for entry in node.kwargs.items():
                writer.string(entry.key)
                writer.scalar[DType.float64](entry.value)


def _read_node(mut reader: BinaryReader[_]) raises -> _Node:
    var kind = reader.integer()
    var count = reader.count(reader.remaining() // 8)
    _check_arity(kind, count)
    var node = _Node(kind)
    for _ in range(count):
        node.children.append(reader.integer())
    if kind <= _Kind.PAIRWISE:
        node.values = _read_floats(reader)
        _check_leaf_size(kind, len(node.values))
        node.bounds = _read_bounds(reader, len(node.values))
    if kind == _Kind.RATIONAL_QUADRATIC or kind == _Kind.EXP_SINE_SQUARED:
        node.value2 = reader.scalar[DType.float64]()
        node.bounds2 = _read_bounds(reader, 1)
    if kind == _Kind.MATERN or kind == _Kind.EXPONENTIATION:
        node.extra = reader.scalar[DType.float64]()
    if kind == _Kind.EXPONENTIATION:
        node.integer_exponent = _read_optional_int(reader)
        if node.integer_exponent and Float64(node.integer_exponent.value()) != node.extra:
            raise Error("ModelSchemaError: inconsistent integer kernel exponent")
    if kind == _Kind.PAIRWISE:
        node.metric = reader.string(reader.remaining())
        _check_pairwise_metric(node.metric)
        node.has_kwargs = reader.boolean()
        if node.has_kwargs:
            var entries = reader.count(reader.remaining() // 16)
            for _ in range(entries):
                var key = reader.string(reader.remaining())
                if key in node.kwargs:
                    raise Error("ModelSchemaError: duplicate pairwise kernel keyword " + key)
                node.kwargs[key] = reader.scalar[DType.float64]()
    return node^


def _check_tree(nodes: List[_Node]) raises:
    # Native tree storage places every child before its parent and the root last.
    var count = len(nodes)
    if count == 0:
        raise Error("ModelSchemaError: empty kernel tree")
    var has_parent = List[Bool](length=count, fill=False)
    var depths = List[Int](capacity=count)
    for i in range(count):
        ref node = nodes[i]
        _check_arity(node.kind, len(node.children))
        var depth = 1
        for child in node.children:
            if child < 0 or child >= i or has_parent[child]:
                raise Error("ModelSchemaError: invalid kernel child index")
            has_parent[child] = True
            depth = max(depth, depths[child] + 1)
        if depth > _MAX_KERNEL_DEPTH:
            raise Error("ModelSchemaError: kernel nesting exceeds 256 levels")
        depths.append(depth)
    for i in range(count - 1):
        if not has_parent[i]:
            raise Error("ModelSchemaError: disconnected kernel node")


def _write_kernel(mut writer: BinaryWriter, kernel: Kernel) raises:
    if kernel._root != len(kernel._nodes) - 1:
        raise Error("ModelSchemaError: invalid kernel root")
    _check_tree(kernel._nodes)
    writer.uint(UInt64(len(kernel._nodes)))
    for node in kernel._nodes:
        _write_node(writer, node)


def _read_kernel(mut reader: BinaryReader[_]) raises -> Kernel:
    var count = reader.count(reader.remaining() // 16)
    var nodes = List[_Node]()
    for _ in range(count):
        nodes.append(_read_node(reader))
    _check_tree(nodes)
    return Kernel(nodes=nodes^, root=count - 1)


struct KernelSchema(ModelSchema):
    """Version 1 state for a built-in, possibly composite Kernel value."""

    comptime Model = Kernel
    comptime TypeId = "moxie.gaussian_process.kernels.Kernel"
    comptime StateVersion = 1

    @staticmethod
    def encode(model: Kernel) raises -> ModelArchive:
        var writer = BinaryWriter()
        _write_kernel(writer, model)
        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Kernel:
        archive.expect(Self.TypeId, Self.StateVersion)
        archive.validate()
        if len(archive.payloads) != 0:
            raise Error("ModelSchemaError: unexpected kernel arrays")
        var reader = BinaryReader(Span(archive.manifest))
        var kernel = _read_kernel(reader)
        reader.finish()
        return kernel^


def _write_optimizer(mut writer: BinaryWriter, optimizer: Optional[FminLBFGSB]) raises:
    writer.boolean(Bool(optimizer))
    if optimizer:
        ref engine = optimizer.value().engine
        LBFGSBConfig(
            history_size=engine.history_size,
            max_iterations=engine.max_iterations,
            max_evaluations=engine.max_evaluations,
            function_tolerance=engine.function_tolerance,
            projected_gradient_tolerance=engine.projected_gradient_tolerance,
            max_line_search_steps=engine.max_line_search_steps,
        ).validate()
        writer.integer(engine.history_size)
        writer.integer(engine.max_iterations)
        writer.integer(engine.max_evaluations)
        writer.scalar[DType.float64](engine.function_tolerance)
        writer.scalar[DType.float64](engine.projected_gradient_tolerance)
        writer.integer(engine.max_line_search_steps)


def _read_optimizer(mut reader: BinaryReader[_]) raises -> Optional[FminLBFGSB]:
    if not reader.boolean():
        return None
    var config = LBFGSBConfig(
        history_size=reader.integer(),
        max_iterations=reader.integer(),
        max_evaluations=reader.integer(),
        function_tolerance=reader.scalar[DType.float64](),
        projected_gradient_tolerance=reader.scalar[DType.float64](),
        max_line_search_steps=reader.integer(),
    )
    config.validate()
    var optimizer = FminLBFGSB()
    optimizer.engine.history_size = config.history_size
    optimizer.engine.max_iterations = config.max_iterations
    optimizer.engine.max_evaluations = config.max_evaluations
    optimizer.engine.function_tolerance = config.function_tolerance
    optimizer.engine.projected_gradient_tolerance = config.projected_gradient_tolerance
    optimizer.engine.max_line_search_steps = config.max_line_search_steps
    return optimizer


def _write_array[dtype: DType](
    mut archive: ModelArchive, name: String, array: NDArray[dtype]
) raises:
    var shape = List[Int]()
    for axis in range(array.ndim):
        shape.append(array.shape[axis])
    var values = List[Scalar[dtype]](capacity=array.size)
    var cols = array.shape[1] if array.ndim == 2 else 1
    var row_stride = Int(array.strides[0])
    var col_stride = Int(array.strides[1]) if array.ndim == 2 else 0
    for i in range(array.shape[0]):
        for j in range(cols):
            var value = array.unsafe_get(i * row_stride + j * col_stride)
            if not isfinite(value):
                raise Error("ModelSchemaError: nonfinite GP array " + name)
            values.append(value)
    archive.add_array[dtype](name, shape, values)


def _read_array[dtype: DType](
    archive: ModelArchive, name: String, expected: List[Int]
) raises -> NDArray[dtype]:
    var payload = archive.array(name)
    if payload.shape != expected:
        raise Error("ModelSchemaError: invalid shape of GP array " + name)
    var values = payload.values[dtype]()
    var array = zeros[dtype](Shape(expected))
    for i in range(len(values)):
        if not isfinite(values[i]):
            raise Error("ModelSchemaError: nonfinite GP array " + name)
        array.unsafe_set(i, values[i])
    return array^


def _learned_target_shape(samples: Int, targets: Int, is_vector: Bool) -> List[Int]:
    var shape: List[Int] = [samples]
    if not is_vector:
        shape.append(targets)
    return shape^


def _check_fitted[dtype: DType](model: GenericGaussianProcessRegressor[dtype]) raises:
    if model.X_train_.ndim != 2 or model._target_matrix.ndim != 2:
        raise Error("ModelSchemaError: invalid GP training array rank")
    var n = model.X_train_.shape[0]
    var d = model.X_train_.shape[1]
    var t = model._target_matrix.shape[1]
    var y_shape = Shape(_learned_target_shape(n, t, model._y_is_vector))
    if (
        n < 1 or d < 1 or t < 1
        or d != model.n_features_in_
        or model._target_matrix.shape[0] != n
        or (model._y_is_vector and t != 1)
        or model.y_train_.shape != y_shape or model.alpha_.shape != y_shape
        or model.L_.shape != Shape(n, n)
        or len(model._y_train_mean) != t or len(model._y_train_std) != t
    ):
        raise Error("ModelSchemaError: inconsistent fitted GP dimensions")
    if not isfinite(model.log_marginal_likelihood_value_):
        raise Error("ModelSchemaError: nonfinite GP log marginal likelihood")
    for k in range(t):
        if not isfinite(model._y_train_mean[k]) or not (
            isfinite(model._y_train_std[k]) and model._y_train_std[k] > 0
        ):
            raise Error("ModelSchemaError: invalid GP target normalization")
    var row_stride = Int(model.L_.strides[0])
    var col_stride = Int(model.L_.strides[1])
    for i in range(n):
        if model.L_.unsafe_get(i * (row_stride + col_stride)) <= 0:
            raise Error("ModelSchemaError: nonpositive GP Cholesky diagonal")
        for j in range(i + 1, n):
            if model.L_.unsafe_get(i * row_stride + j * col_stride) != 0:
                raise Error("ModelSchemaError: GP Cholesky factor must be lower triangular")


struct GaussianProcessRegressorSchema[dtype: DType = DType.float64](ModelSchema):
    """Version 1 state for a dense GPR using FminLBFGSB or optimizer=None."""

    comptime Model = GenericGaussianProcessRegressor[Self.dtype]
    comptime TypeId = "moxie.gaussian_process.GaussianProcessRegressor"
    comptime StateVersion = 1

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive:
        model._validate_params()
        var writer = BinaryWriter()
        writer.uint(UInt64(dtype_code[Self.dtype]()))
        writer.boolean(Bool(model.kernel))
        if model.kernel:
            _write_kernel(writer, model.kernel.value())
        writer.boolean(model.alpha.isa[List[Float64]]())
        if model.alpha.isa[List[Float64]]():
            _write_floats(writer, model.alpha[List[Float64]])
        else:
            writer.scalar[DType.float64](model.alpha[Float64])
        _write_optimizer(writer, model.optimizer)
        writer.integer(model.n_restarts_optimizer)
        writer.boolean(model.normalize_y)
        writer.boolean(model.copy_X_train)
        _write_optional_int(writer, model.n_targets)
        _write_optional_int(writer, model.random_state)

        var archive = ModelArchive(Self.TypeId, Self.StateVersion)
        var fitted = model.__sklearn_is_fitted__()
        writer.boolean(fitted)
        if fitted:
            _check_fitted(model)
            _write_kernel(writer, model.kernel_)
            writer.integer(model.X_train_.shape[0])
            writer.integer(model.n_features_in_)
            writer.integer(model._target_matrix.shape[1])
            writer.boolean(model._y_is_vector)
            writer.scalar[DType.float64](model.log_marginal_likelihood_value_)
            _write_floats(writer, model._y_train_mean)
            _write_floats(writer, model._y_train_std)
            writer.uint(UInt64(len(model.convergence_warnings_)))
            for warning in model.convergence_warnings_:
                writer.string(warning)
            _write_array(archive, "X_train_", model.X_train_)
            _write_array(archive, "y_train_", model.y_train_)
            _write_array(archive, "L_", model.L_)
            _write_array(archive, "alpha_", model.alpha_)
            _write_array(archive, "target_matrix", model._target_matrix)
        archive.manifest = writer^.finish()
        return archive^

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model:
        archive.expect(Self.TypeId, Self.StateVersion)
        archive.validate()
        var reader = BinaryReader(Span(archive.manifest))
        if reader.uint() != UInt64(dtype_code[Self.dtype]()):
            raise Error("ModelSchemaError: GP input dtype mismatch")
        var kernel: Optional[Kernel] = None
        if reader.boolean():
            kernel = _read_kernel(reader)
        var alpha: ScalarOrVector
        if reader.boolean():
            alpha = _read_floats(reader)
        else:
            alpha = reader.scalar[DType.float64]()
        var model = Self.Model(
            kernel,
            alpha=alpha^,
            optimizer=_read_optimizer(reader),
            n_restarts_optimizer=reader.integer(),
            normalize_y=reader.boolean(),
            copy_X_train=reader.boolean(),
            n_targets=_read_optional_int(reader),
            random_state=_read_optional_int(reader),
        )
        var fitted = reader.boolean()
        if len(archive.payloads) != (5 if fitted else 0):
            raise Error("ModelSchemaError: unexpected GP array count")
        if fitted:
            model.kernel_ = _read_kernel(reader)
            var n = reader.integer()
            model.n_features_in_ = reader.integer()
            var t = reader.integer()
            model._y_is_vector = reader.boolean()
            if n < 1 or model.n_features_in_ < 1 or t < 1 or (
                model._y_is_vector and t != 1
            ):
                raise Error("ModelSchemaError: invalid fitted GP dimensions")
            model.log_marginal_likelihood_value_ = reader.scalar[DType.float64]()
            model._y_train_mean = _read_floats(reader)
            model._y_train_std = _read_floats(reader)
            var warnings_count = reader.count(reader.remaining() // 8)
            for _ in range(warnings_count):
                model.convergence_warnings_.append(reader.string(reader.remaining()))
            var y_shape = _learned_target_shape(n, t, model._y_is_vector)
            model.X_train_ = _read_array[Self.dtype](
                archive, "X_train_", [n, model.n_features_in_]
            )
            model.y_train_ = _read_array[DType.float64](archive, "y_train_", y_shape)
            model.L_ = _read_array[DType.float64](archive, "L_", [n, n])
            model.alpha_ = _read_array[DType.float64](archive, "alpha_", y_shape)
            model._target_matrix = _read_array[DType.float64](archive, "target_matrix", [n, t])
            _check_fitted(model)
            model._fitted = True
        reader.finish()
        return model^
