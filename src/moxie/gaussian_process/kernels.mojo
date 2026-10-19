# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Composable Gaussian-process kernels (`sklearn.gaussian_process.kernels`).

Every kernel is a value of the type-erased `Kernel`. The sklearn-named leaf
structs (`RBF`, `Matern`, ...) carry typed constructor parameters, convert
implicitly to `Kernel`, and build composite kernels with `+`, `*` and `**`.
`Sum`, `Product`, `Exponentiation` and `CompoundKernel` construct composite
`Kernel` values directly.

Conventions follow sklearn: `theta()` holds the natural logarithm of the
non-fixed hyperparameters in sorted hyperparameter-name order (operators
recurse k1 before k2), `bounds()` holds their log bounds `(n_dims, 2)`, and
gradients `(n, n, n_dims)` are taken with respect to log hyperparameters.
Kernel evaluation always computes in Float64 (see docs/api.md).
"""
from numojo import NDArray, Shape, zeros
from std.collections import Dict
from std.math import abs, cos, isfinite, isinf, pi, sin, sqrt, tanh

from moxie.gaussian_process._plumbing import (
    Dense64,
    HyperparameterSpec,
    KernelContract,
    KernelEvaluation,
    validate_gradient,
)
from moxie.gaussian_process._kernel_math import (
    additive_chi2,
    as_dense64,
    concat_gradients,
    multiply_gradient,
    empty_gradient,
    euclidean_from_squared,
    fexp,
    flog,
    fpow,
    fill,
    format_g3,
    inner_products,
    manhattan_distances,
    matern_general_values,
    row_normalized,
    scale_columns,
    squared_distances,
    vector_fill,
)

comptime Hyperparameter = HyperparameterSpec
"""sklearn `Hyperparameter`: name, n_elements, fixed flag, raw-unit bounds.

Only numeric hyperparameters exist, so `value_type` is always "numeric" and is
not stored. Fixed hyperparameters carry an empty `(0, 2)` bounds matrix.
"""

comptime _FD_EPSILON: Float64 = 1e-10
comptime _FLOAT_EPSILON: Float64 = 2.220446049250313e-16


struct _Kind:
    comptime CONSTANT = 0
    comptime WHITE = 1
    comptime RBF = 2
    comptime MATERN = 3
    comptime RATIONAL_QUADRATIC = 4
    comptime EXP_SINE_SQUARED = 5
    comptime DOT_PRODUCT = 6
    comptime PAIRWISE = 7
    comptime SUM = 8
    comptime PRODUCT = 9
    comptime EXPONENTIATION = 10
    comptime COMPOUND = 11


@always_inline
def _is_inf(x: Float64) -> Bool:
    return isinf(x) and x > 0


def _kind_name(kind: Int) -> String:
    if kind == _Kind.CONSTANT:
        return "ConstantKernel"
    if kind == _Kind.WHITE:
        return "WhiteKernel"
    if kind == _Kind.RBF:
        return "RBF"
    if kind == _Kind.MATERN:
        return "Matern"
    if kind == _Kind.RATIONAL_QUADRATIC:
        return "RationalQuadratic"
    if kind == _Kind.EXP_SINE_SQUARED:
        return "ExpSineSquared"
    if kind == _Kind.DOT_PRODUCT:
        return "DotProduct"
    if kind == _Kind.PAIRWISE:
        return "PairwiseKernel"
    if kind == _Kind.SUM:
        return "Sum"
    if kind == _Kind.PRODUCT:
        return "Product"
    if kind == _Kind.EXPONENTIATION:
        return "Exponentiation"
    return "CompoundKernel"


def _py_float(x: Float64) -> String:
    """Python `str(float)` for the values that appear in kernel reprs."""
    if x == Float64(Int(x)) and abs(x) < 1e16:
        return String(Int(x)) + ".0"
    return String(x)


# ===----------------------------------------------------------------------=== #
# Bounds
# ===----------------------------------------------------------------------=== #


struct Bounds(Copyable, Equatable, Movable, Writable):
    """Raw-unit hyperparameter bounds: `"fixed"`, one pair, or one pair per element.

    `Bounds(1e-5, 1e5)` is sklearn's `(1e-5, 1e5)`; `Bounds("fixed")` (or the
    implicit conversion from the literal `"fixed"`) marks the hyperparameter
    fixed. `Bounds(lower_list, upper_list)` gives per-element bounds for
    anisotropic length scales. Values are not validated, as in sklearn.
    """

    var is_fixed: Bool
    var lower: List[Float64]
    var upper: List[Float64]

    def __init__(out self, lower: Float64, upper: Float64):
        """Scalar `(lower, upper)` pair, broadcast over vector elements."""
        self.is_fixed = False
        self.lower = [lower]
        self.upper = [upper]

    def __init__(out self, lower: List[Float64], upper: List[Float64]) raises:
        """Per-element bounds; both lists must have the same nonzero length."""
        if len(lower) == 0 or len(lower) != len(upper):
            raise Error(
                "ValueError: per-element bounds need equally long, nonempty"
                " lower and upper lists (got "
                + String(len(lower))
                + " and "
                + String(len(upper))
                + ")"
            )
        self.is_fixed = False
        self.lower = lower.copy()
        self.upper = upper.copy()

    @implicit
    def __init__(out self, text: StringLiteral):
        """`"fixed"`; any other text yields an invalid marker that raises on use.
        """
        self.is_fixed = String(text) == "fixed"
        self.lower = List[Float64]()
        self.upper = List[Float64]()

    @staticmethod
    def fixed() -> Bounds:
        """The sklearn `"fixed"` marker."""
        return Bounds("fixed")

    def __eq__(self, other: Bounds) -> Bool:
        if self.is_fixed != other.is_fixed:
            return False
        if self.is_fixed:
            return True
        if len(self.lower) != len(other.lower):
            return False
        for i in range(len(self.lower)):
            if self.lower[i] != other.lower[i] or self.upper[i] != other.upper[i]:
                return False
        return True

    def __ne__(self, other: Bounds) -> Bool:
        return not self == other

    def write_to(self, mut writer: Some[Writer]):
        if self.is_fixed:
            writer.write("fixed")
            return
        if len(self.lower) == 0:
            writer.write("<invalid bounds>")
            return
        if len(self.lower) == 1:
            writer.write(
                "(", _py_float(self.lower[0]), ", ", _py_float(self.upper[0]), ")"
            )
            return
        writer.write("[")
        for i in range(len(self.lower)):
            if i > 0:
                writer.write(", ")
            writer.write(
                "[", _py_float(self.lower[i]), ", ", _py_float(self.upper[i]), "]"
            )
        writer.write("]")

    def expanded(self, n_elements: Int, name: String) raises -> Dense64:
        """Raw `(n_elements, 2)` bounds; `(0, 2)` when fixed."""
        if self.is_fixed:
            return zeros[DType.float64](Shape(0, 2))
        if len(self.lower) == 0:
            raise Error(
                "ValueError: bounds of "
                + name
                + " must be a (lower, upper) pair or \"fixed\""
            )
        if len(self.lower) != 1 and len(self.lower) != n_elements:
            raise Error(
                "ValueError: Bounds on "
                + name
                + " should have either 1 or "
                + String(n_elements)
                + " dimensions. Given are "
                + String(len(self.lower))
            )
        var out = zeros[DType.float64](Shape(n_elements, 2))
        for i in range(n_elements):
            var k = 0 if len(self.lower) == 1 else i
            out.unsafe_set(2 * i, self.lower[k])
            out.unsafe_set(2 * i + 1, self.upper[k])
        return out^


# ===----------------------------------------------------------------------=== #
# Parameter values (typed replacement for sklearn's get_params dict values)
# ===----------------------------------------------------------------------=== #


struct _ParamKind:
    comptime NUMBER = 0
    comptime NUMBERS = 1
    comptime BOUNDS = 2
    comptime TEXT = 3
    comptime KERNEL = 4
    comptime KERNELS = 5
    comptime MAPPING = 6
    comptime NONE = 7


struct ParamValue(Copyable, Movable, Writable):
    """One kernel parameter value: number, number list, Bounds, text, kernel(s),
    keyword mapping, or None."""

    var kind: Int
    var numbers: List[Float64]
    var bounds: Bounds
    var text: String
    var kernels: List[Kernel]
    var mapping: Dict[String, Float64]

    def __init__(out self, *, kind: Int):
        self.kind = kind
        self.numbers = List[Float64]()
        self.bounds = Bounds.fixed()
        self.text = String()
        self.kernels = List[Kernel]()
        self.mapping = Dict[String, Float64]()

    @staticmethod
    def none() -> ParamValue:
        """Python `None` (e.g. `pairwise_kernels_kwargs=None`)."""
        return ParamValue(kind=_ParamKind.NONE)

    @implicit
    def __init__(out self, value: Float64):
        self = ParamValue(kind=_ParamKind.NUMBER)
        self.numbers.append(value)

    @implicit
    def __init__(out self, value: List[Float64]):
        self = ParamValue(kind=_ParamKind.NUMBERS)
        self.numbers = value.copy()

    @implicit
    def __init__(out self, value: Bounds):
        self = ParamValue(kind=_ParamKind.BOUNDS)
        self.bounds = value.copy()

    @implicit
    def __init__(out self, value: String):
        self = ParamValue(kind=_ParamKind.TEXT)
        self.text = value

    @implicit
    def __init__(out self, value: Kernel):
        self = ParamValue(kind=_ParamKind.KERNEL)
        self.kernels.append(value.copy())

    @implicit
    def __init__(out self, value: List[Kernel]):
        self = ParamValue(kind=_ParamKind.KERNELS)
        self.kernels = value.copy()

    @implicit
    def __init__(out self, value: Dict[String, Float64]):
        self = ParamValue(kind=_ParamKind.MAPPING)
        self.mapping = value.copy()

    def is_none(self) -> Bool:
        return self.kind == _ParamKind.NONE

    def as_float(self, name: String) raises -> Float64:
        """The scalar value; a one-element number list also qualifies."""
        if (
            self.kind == _ParamKind.NUMBER or self.kind == _ParamKind.NUMBERS
        ) and len(self.numbers) == 1:
            return self.numbers[0]
        raise Error("TypeError: parameter " + name + " expects a scalar number")

    def as_floats(self, name: String) raises -> List[Float64]:
        if self.kind == _ParamKind.NUMBER or self.kind == _ParamKind.NUMBERS:
            if len(self.numbers) == 0:
                raise Error("ValueError: parameter " + name + " is empty")
            return self.numbers.copy()
        raise Error("TypeError: parameter " + name + " expects numbers")

    def as_bounds(self, name: String) raises -> Bounds:
        if self.kind == _ParamKind.BOUNDS:
            return self.bounds.copy()
        if self.kind == _ParamKind.TEXT and self.text == "fixed":
            return Bounds.fixed()
        raise Error("TypeError: parameter " + name + " expects Bounds or \"fixed\"")

    def as_text(self, name: String) raises -> String:
        if self.kind == _ParamKind.TEXT:
            return self.text
        raise Error("TypeError: parameter " + name + " expects a string")

    def as_kernel(self, name: String) raises -> Kernel:
        if self.kind == _ParamKind.KERNEL:
            return self.kernels[0].copy()
        raise Error("TypeError: parameter " + name + " expects a Kernel")

    def as_kernels(self, name: String) raises -> List[Kernel]:
        if self.kind == _ParamKind.KERNELS:
            return self.kernels.copy()
        raise Error("TypeError: parameter " + name + " expects a list of Kernels")

    def as_mapping(self, name: String) raises -> Optional[Dict[String, Float64]]:
        if self.kind == _ParamKind.NONE:
            return None
        if self.kind == _ParamKind.MAPPING:
            return self.mapping.copy()
        raise Error("TypeError: parameter " + name + " expects a mapping or None")

    def write_to(self, mut writer: Some[Writer]):
        if self.kind == _ParamKind.NUMBER:
            writer.write(_py_float(self.numbers[0]))
        elif self.kind == _ParamKind.NUMBERS:
            writer.write("[")
            for i in range(len(self.numbers)):
                if i > 0:
                    writer.write(", ")
                writer.write(_py_float(self.numbers[i]))
            writer.write("]")
        elif self.kind == _ParamKind.BOUNDS:
            writer.write(self.bounds)
        elif self.kind == _ParamKind.TEXT:
            writer.write(self.text)
        elif self.kind == _ParamKind.KERNEL:
            writer.write(self.kernels[0])
        elif self.kind == _ParamKind.KERNELS:
            writer.write("[")
            for i in range(len(self.kernels)):
                if i > 0:
                    writer.write(", ")
                writer.write(self.kernels[i])
            writer.write("]")
        elif self.kind == _ParamKind.MAPPING:
            writer.write("{")
            var first = True
            for entry in self.mapping.items():
                if not first:
                    writer.write(", ")
                first = False
                writer.write("'", entry.key, "': ", _py_float(entry.value))
            writer.write("}")
        else:
            writer.write("None")


struct KernelParams(Copyable, Movable, Sized, Writable):
    """Ordered `get_params` result: parameter names and typed values.

    Names follow sklearn, including nested `k1__length_scale` entries for deep
    parameters. `set_params` accepts any subset of these names.
    """

    var names: List[String]
    var values: List[ParamValue]

    def __init__(out self):
        self.names = List[String]()
        self.values = List[ParamValue]()

    def __len__(self) -> Int:
        return len(self.names)

    def set(mut self, name: String, var value: ParamValue):
        """Insert or replace one entry."""
        for i in range(len(self.names)):
            if self.names[i] == name:
                self.values[i] = value^
                return
        self.names.append(name)
        self.values.append(value^)

    def __contains__(self, name: String) -> Bool:
        for n in self.names:
            if n == name:
                return True
        return False

    def __getitem__(self, name: String) raises -> ParamValue:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.values[i].copy()
        raise Error("KeyError: " + name)

    def keys(self) -> List[String]:
        return self.names.copy()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("{")
        for i in range(len(self.names)):
            if i > 0:
                writer.write(", ")
            writer.write("'", self.names[i], "': ", self.values[i])
        writer.write("}")


# ===----------------------------------------------------------------------=== #
# Kernel tree storage
# ===----------------------------------------------------------------------=== #


struct _Node(Copyable, Movable):
    var kind: Int
    var values: List[Float64]
    var bounds: Bounds
    var value2: Float64
    var bounds2: Bounds
    var extra: Float64
    var metric: String
    var kwargs: Dict[String, Float64]
    var has_kwargs: Bool
    var children: List[Int]

    def __init__(out self, kind: Int):
        self.kind = kind
        self.values = List[Float64]()
        self.bounds = Bounds.fixed()
        self.value2 = 0.0
        self.bounds2 = Bounds.fixed()
        self.extra = 0.0
        self.metric = String()
        self.kwargs = Dict[String, Float64]()
        self.has_kwargs = False
        self.children = List[Int]()


@fieldwise_init
struct _Eval(Movable):
    var K: Dense64
    var G: Dense64

    def take_covariance(deinit self) -> Dense64:
        return self.K^

    def into_evaluation(deinit self, with_gradient: Bool) -> KernelEvaluation:
        if with_gradient:
            return KernelEvaluation(covariance=self.K^, gradient=self.G^)
        return KernelEvaluation(covariance=self.K^, gradient=None)


def _is_leaf(kind: Int) -> Bool:
    return kind <= _Kind.PAIRWISE


def _check_pairwise_metric(metric: String) raises:
    if (
        metric == "linear"
        or metric == "rbf"
        or metric == "laplacian"
        or metric == "poly"
        or metric == "polynomial"
        or metric == "sigmoid"
        or metric == "cosine"
        or metric == "chi2"
        or metric == "additive_chi2"
        or metric == "precomputed"
    ):
        return
    raise Error(
        "ValueError: PairwiseKernel metric must be one of 'additive_chi2',"
        " 'chi2', 'cosine', 'laplacian', 'linear', 'poly', 'polynomial',"
        " 'precomputed', 'rbf', 'sigmoid'; got '"
        + metric
        + "' (callable metrics are not supported)"
    )


def _kwarg(node: _Node, name: String, default: Float64) -> Float64:
    if node.has_kwargs:
        var found = node.kwargs.get(name)
        if found:
            return found.value()
    return default


def _validate_pairwise_input(A: Dense64) raises:
    if A.shape[0] == 0 or A.shape[1] == 0:
        raise Error("ValueError: PairwiseKernel requires nonempty arrays")
    for i in range(A.size):
        if not isfinite(A.unsafe_get(i)):
            raise Error("ValueError: PairwiseKernel input contains NaN or infinity")

def _pairwise_matrix(
    node: _Node, gamma_value: Float64, X: Dense64, Y: Dense64, same: Bool
) raises -> Dense64:
    """sklearn `pairwise_kernels(X, Y, metric, gamma, filter_params=True, **kw)`.
    """
    var metric = node.metric
    _check_pairwise_metric(metric)
    _validate_pairwise_input(X)
    if not same:
        _validate_pairwise_input(Y)
    if metric == "precomputed":
        if X.shape[1] != Y.shape[0]:
            raise Error("ValueError: Precomputed metric requires shape (n_queries, n_indexed)")
        return X.copy()
    if X.shape[1] != Y.shape[1]:
        raise Error(
            "ValueError: Incompatible dimension for X and Y matrices: X.shape[1]"
            " == "
            + String(X.shape[1])
            + " while Y.shape[1] == "
            + String(Y.shape[1])
        )
    if metric == "linear":
        return inner_products(X, Y)
    if metric == "cosine":
        var xn = row_normalized(X)
        if same:
            return inner_products(xn, xn)
        return inner_products(xn, row_normalized(Y))
    if metric == "additive_chi2":
        return additive_chi2(X, Y, same)
    var K: Dense64
    if metric == "rbf":
        K = squared_distances(X, Y, same)
        var p = K.unsafe_ptr()
        for i in range(K.size):
            p[unsafe_offset=i] = fexp(-gamma_value * p[unsafe_offset=i])
        return K^
    if metric == "laplacian":
        K = manhattan_distances(X, Y)
        var p = K.unsafe_ptr()
        for i in range(K.size):
            p[unsafe_offset=i] = fexp(-gamma_value * p[unsafe_offset=i])
        return K^
    if metric == "chi2":
        K = additive_chi2(X, Y, same)
        var p = K.unsafe_ptr()
        for i in range(K.size):
            p[unsafe_offset=i] = fexp(gamma_value * p[unsafe_offset=i])
        return K^
    K = inner_products(X, Y)
    var p = K.unsafe_ptr()
    var coef0 = _kwarg(node, "coef0", 1.0)
    if metric == "sigmoid":
        for i in range(K.size):
            p[unsafe_offset=i] = tanh(gamma_value * p[unsafe_offset=i] + coef0)
        return K^
    var degree = _kwarg(node, "degree", 3.0)
    for i in range(K.size):
        p[unsafe_offset=i] = fpow(gamma_value * p[unsafe_offset=i] + coef0, degree)
    return K^


def _check_length_scale(node: _Node, n_features: Int) raises:
    if len(node.values) > 1 and len(node.values) != n_features:
        raise Error(
            "ValueError: Anisotropic kernel must have the same number of"
            " dimensions as data ("
            + String(len(node.values))
            + "!="
            + String(n_features)
            + ")"
        )


def _matern_from_distances(nu: Float64, dists: Dense64, same: Bool) raises -> Dense64:
    """Matern covariance from scaled Euclidean distances (sklearn branches)."""
    var n = dists.shape[0]
    var m = dists.shape[1]
    var K: Dense64
    if nu == 0.5 or nu == 1.5 or nu == 2.5 or _is_inf(nu):
        K = zeros[DType.float64](Shape(n, m))
        var src = dists.unsafe_ptr()
        var dst = K.unsafe_ptr()
        for i in range(n * m):
            var d = src[unsafe_offset=i]
            var v: Float64
            if nu == 0.5:
                v = fexp(-d)
            elif nu == 1.5:
                var t = d * sqrt(3.0)
                v = (1.0 + t) * fexp(-t)
            elif nu == 2.5:
                var t = d * sqrt(5.0)
                v = (1.0 + t + t * t / 3.0) * fexp(-t)
            else:
                v = fexp(-d * d / 2.0)
            dst[unsafe_offset=i] = v
    else:
        K = matern_general_values(nu, dists, same, _FLOAT_EPSILON)
    if same:
        var dst = K.unsafe_ptr()
        for i in range(n):
            dst[unsafe_offset = i * m + i] = 1.0
    return K^


def _elementwise_pow(K: Dense64, exponent: Float64) raises -> Dense64:
    var out = K.copy()
    var p = out.unsafe_ptr()
    for i in range(out.size):
        p[unsafe_offset=i] = fpow(p[unsafe_offset=i], exponent)
    return out^


# ===----------------------------------------------------------------------=== #
# Kernel
# ===----------------------------------------------------------------------=== #


struct Kernel(KernelContract, Equatable, Writable):
    """A composable Gaussian-process covariance function.

    Leaf kernels are built with the sklearn-named constructors (`RBF(...)`,
    `ConstantKernel(...)`, ...) and combined with `+`, `*`, `**`, `Sum`,
    `Product`, `Exponentiation` and `CompoundKernel`. Values are independent:
    `copy()` is sklearn's `clone` for kernels.
    """

    comptime Input = Dense64
    comptime Params = KernelParams

    var _nodes: List[_Node]
    var _root: Int

    def __init__(out self, *, var nodes: List[_Node], root: Int):
        self._nodes = nodes^
        self._root = root

    @implicit
    def __init__[K: KernelLike](out self, component: K):
        """Convert any leaf struct (RBF, Matern, ...) into a Kernel."""
        self = component.as_kernel()

    @staticmethod
    def _leaf(var node: _Node) -> Kernel:
        var nodes = List[_Node]()
        nodes.append(node^)
        return Kernel(nodes=nodes^, root=0)

    # --- tree plumbing ------------------------------------------------------

    def _append_subtree(self, mut out: List[_Node], idx: Int) -> Int:
        var node = self._nodes[idx].copy()
        var mapped = List[Int]()
        for c in self._nodes[idx].children:
            mapped.append(self._append_subtree(out, c))
        node.children = mapped^
        out.append(node^)
        return len(out) - 1

    def _subtree(self, idx: Int) -> Kernel:
        var nodes = List[_Node]()
        var root = self._append_subtree(nodes, idx)
        return Kernel(nodes=nodes^, root=root)

    @staticmethod
    def _combine(kind: Int, parts: List[Kernel], extra: Float64 = 0.0) -> Kernel:
        var nodes = List[_Node]()
        var node = _Node(kind)
        node.extra = extra
        for part in parts:
            node.children.append(part._append_subtree(nodes, part._root))
        nodes.append(node^)
        return Kernel(nodes=nodes^, root=len(nodes) - 1)

    def _rebuild_replacing(
        self, mut out: List[_Node], idx: Int, target: Int, replacement: Kernel
    ) -> Int:
        if idx == target:
            return replacement._append_subtree(out, replacement._root)
        var node = self._nodes[idx].copy()
        var mapped = List[Int]()
        for c in self._nodes[idx].children:
            mapped.append(self._rebuild_replacing(out, c, target, replacement))
        node.children = mapped^
        out.append(node^)
        return len(out) - 1

    def _replace(mut self, target: Int, replacement: Kernel):
        var nodes = List[_Node]()
        var root = self._rebuild_replacing(nodes, self._root, target, replacement)
        self._nodes = nodes^
        self._root = root

    # --- structure access ---------------------------------------------------

    def name(self) -> String:
        """sklearn class name of the outermost kernel (e.g. "Product")."""
        return _kind_name(self._nodes[self._root].kind)

    def _child(self, position: Int, accessor: String) raises -> Kernel:
        var node = self._nodes[self._root].copy()
        if (
            node.kind == _Kind.SUM or node.kind == _Kind.PRODUCT
        ) and position < 2 and accessor != "kernel":
            return self._subtree(node.children[position])
        if node.kind == _Kind.EXPONENTIATION and accessor == "kernel":
            return self._subtree(node.children[0])
        raise Error(
            "AttributeError: '" + self.name() + "' object has no attribute '" + accessor + "'"
        )

    def k1(self) raises -> Kernel:
        """First operand of a Sum or Product."""
        return self._child(0, "k1")

    def k2(self) raises -> Kernel:
        """Second operand of a Sum or Product."""
        return self._child(1, "k2")

    def kernel(self) raises -> Kernel:
        """Base kernel of an Exponentiation."""
        return self._child(0, "kernel")

    def exponent(self) raises -> Float64:
        """Exponent of an Exponentiation."""
        if self._nodes[self._root].kind != _Kind.EXPONENTIATION:
            raise Error("AttributeError: '" + self.name() + "' object has no attribute 'exponent'")
        return self._nodes[self._root].extra

    def kernels(self) raises -> List[Kernel]:
        """Children of a CompoundKernel."""
        if self._nodes[self._root].kind != _Kind.COMPOUND:
            raise Error("AttributeError: '" + self.name() + "' object has no attribute 'kernels'")
        var out = List[Kernel]()
        for c in self._nodes[self._root].children:
            out.append(self._subtree(c))
        return out^

    # --- parameters ---------------------------------------------------------

    def _leaf_params(self, idx: Int) -> KernelParams:
        ref node = self._nodes[idx]
        var params = KernelParams()
        var k = node.kind
        if k == _Kind.CONSTANT:
            params.set("constant_value", node.values[0])
            params.set("constant_value_bounds", node.bounds.copy())
        elif k == _Kind.WHITE:
            params.set("noise_level", node.values[0])
            params.set("noise_level_bounds", node.bounds.copy())
        elif k == _Kind.RBF or k == _Kind.MATERN:
            if len(node.values) == 1:
                params.set("length_scale", node.values[0])
            else:
                params.set("length_scale", node.values.copy())
            params.set("length_scale_bounds", node.bounds.copy())
            if k == _Kind.MATERN:
                params.set("nu", node.extra)
        elif k == _Kind.RATIONAL_QUADRATIC:
            params.set("length_scale", node.values[0])
            params.set("alpha", node.value2)
            params.set("length_scale_bounds", node.bounds.copy())
            params.set("alpha_bounds", node.bounds2.copy())
        elif k == _Kind.EXP_SINE_SQUARED:
            params.set("length_scale", node.values[0])
            params.set("periodicity", node.value2)
            params.set("length_scale_bounds", node.bounds.copy())
            params.set("periodicity_bounds", node.bounds2.copy())
        elif k == _Kind.DOT_PRODUCT:
            params.set("sigma_0", node.values[0])
            params.set("sigma_0_bounds", node.bounds.copy())
        elif k == _Kind.PAIRWISE:
            params.set("gamma", node.values[0])
            params.set("gamma_bounds", node.bounds.copy())
            params.set("metric", node.metric)
            if node.has_kwargs:
                params.set("pairwise_kernels_kwargs", node.kwargs.copy())
            else:
                params.set("pairwise_kernels_kwargs", ParamValue.none())
        return params^

    def _params_at(self, idx: Int, deep: Bool) -> KernelParams:
        ref node = self._nodes[idx]
        if _is_leaf(node.kind):
            return self._leaf_params(idx)
        var params = KernelParams()
        if node.kind == _Kind.SUM or node.kind == _Kind.PRODUCT:
            params.set("k1", self._subtree(node.children[0]))
            params.set("k2", self._subtree(node.children[1]))
            if deep:
                var p1 = self._params_at(node.children[0], True)
                for i in range(len(p1.names)):
                    params.set("k1__" + p1.names[i], p1.values[i].copy())
                var p2 = self._params_at(node.children[1], True)
                for i in range(len(p2.names)):
                    params.set("k2__" + p2.names[i], p2.values[i].copy())
        elif node.kind == _Kind.EXPONENTIATION:
            params.set("kernel", self._subtree(node.children[0]))
            params.set("exponent", node.extra)
            if deep:
                var p = self._params_at(node.children[0], True)
                for i in range(len(p.names)):
                    params.set("kernel__" + p.names[i], p.values[i].copy())
        else:
            var children = List[Kernel]()
            for c in node.children:
                children.append(self._subtree(c))
            params.set("kernels", children^)
        return params^

    def get_params(self, deep: Bool = True) raises -> KernelParams:
        """Parameter names mapped to typed values, in sklearn order."""
        return self._params_at(self._root, deep)

    def _invalid_parameter(self, idx: Int, key: String) -> Error:
        return Error(
            "ValueError: Invalid parameter "
            + key
            + " for kernel "
            + _kind_name(self._nodes[idx].kind)
            + ". Check the list of available parameters with"
            " `kernel.get_params().keys()`."
        )

    def _find_child(self, idx: Int, name: String) raises -> Int:
        ref node = self._nodes[idx]
        if (node.kind == _Kind.SUM or node.kind == _Kind.PRODUCT) and (
            name == "k1" or name == "k2"
        ):
            return node.children[0 if name == "k1" else 1]
        if node.kind == _Kind.EXPONENTIATION and name == "kernel":
            return node.children[0]
        raise self._invalid_parameter(idx, name)

    def _set_leaf_param(mut self, idx: Int, key: String, value: ParamValue) raises:
        var k = self._nodes[idx].kind
        ref node = self._nodes[idx]
        if k == _Kind.CONSTANT and key == "constant_value":
            node.values = [value.as_float(key)]
        elif k == _Kind.CONSTANT and key == "constant_value_bounds":
            node.bounds = value.as_bounds(key)
        elif k == _Kind.WHITE and key == "noise_level":
            node.values = [value.as_float(key)]
        elif k == _Kind.WHITE and key == "noise_level_bounds":
            node.bounds = value.as_bounds(key)
        elif (k == _Kind.RBF or k == _Kind.MATERN) and key == "length_scale":
            node.values = value.as_floats(key)
        elif (k == _Kind.RBF or k == _Kind.MATERN) and key == "length_scale_bounds":
            node.bounds = value.as_bounds(key)
        elif k == _Kind.MATERN and key == "nu":
            node.extra = value.as_float(key)
        elif (
            k == _Kind.RATIONAL_QUADRATIC or k == _Kind.EXP_SINE_SQUARED
        ) and key == "length_scale":
            node.values = value.as_floats(key)
        elif (
            k == _Kind.RATIONAL_QUADRATIC or k == _Kind.EXP_SINE_SQUARED
        ) and key == "length_scale_bounds":
            node.bounds = value.as_bounds(key)
        elif (k == _Kind.RATIONAL_QUADRATIC and key == "alpha") or (
            k == _Kind.EXP_SINE_SQUARED and key == "periodicity"
        ):
            node.value2 = value.as_float(key)
        elif (k == _Kind.RATIONAL_QUADRATIC and key == "alpha_bounds") or (
            k == _Kind.EXP_SINE_SQUARED and key == "periodicity_bounds"
        ):
            node.bounds2 = value.as_bounds(key)
        elif k == _Kind.DOT_PRODUCT and key == "sigma_0":
            node.values = [value.as_float(key)]
        elif k == _Kind.DOT_PRODUCT and key == "sigma_0_bounds":
            node.bounds = value.as_bounds(key)
        elif k == _Kind.PAIRWISE and key == "gamma":
            node.values = [value.as_float(key)]
        elif k == _Kind.PAIRWISE and key == "gamma_bounds":
            node.bounds = value.as_bounds(key)
        elif k == _Kind.PAIRWISE and key == "metric":
            node.metric = value.as_text(key)
        elif k == _Kind.PAIRWISE and key == "pairwise_kernels_kwargs":
            var mapping = value.as_mapping(key)
            node.has_kwargs = Bool(mapping)
            node.kwargs = mapping.value().copy() if mapping else Dict[String, Float64]()
        else:
            raise self._invalid_parameter(idx, key)

    def _set_param_at(mut self, idx: Int, key: String, value: ParamValue) raises:
        var split = key.find("__")
        if split >= 0:
            var head = String(key[byte=0:split])
            var rest = String(key[byte = split + 2 : key.byte_length()])
            var child = self._find_child(idx, head)
            self._set_param_at(child, rest, value)
            return
        var kind = self._nodes[idx].kind
        if _is_leaf(kind):
            self._set_leaf_param(idx, key, value)
        elif (kind == _Kind.SUM or kind == _Kind.PRODUCT) and (key == "k1" or key == "k2"):
            self._replace(self._find_child(idx, key), value.as_kernel(key))
        elif kind == _Kind.EXPONENTIATION and key == "kernel":
            self._replace(self._find_child(idx, key), value.as_kernel(key))
        elif kind == _Kind.EXPONENTIATION and key == "exponent":
            self._nodes[idx].extra = value.as_float(key)
        elif kind == _Kind.COMPOUND and key == "kernels":
            var replacement = Kernel._combine(_Kind.COMPOUND, value.as_kernels(key))
            self._replace(idx, replacement)
        else:
            raise self._invalid_parameter(idx, key)

    def set_params(mut self, params: KernelParams) raises:
        """Apply parameters in order; nested names use `<component>__<name>`."""
        for i in range(len(params.names)):
            self._set_param_at(self._root, params.names[i], params.values[i])

    def set_param(mut self, name: String, value: ParamValue) raises:
        """Set one (possibly nested) parameter, e.g. `"k2__length_scale"`."""
        self._set_param_at(self._root, name, value)

    # --- hyperparameters, theta and bounds ----------------------------------

    def _hyperparameters_at(
        self, idx: Int, prefix: String, mut out: List[HyperparameterSpec]
    ) raises:
        ref node = self._nodes[idx]
        var k = node.kind
        if k == _Kind.SUM or k == _Kind.PRODUCT:
            self._hyperparameters_at(node.children[0], prefix + "k1__", out)
            self._hyperparameters_at(node.children[1], prefix + "k2__", out)
            return
        if k == _Kind.EXPONENTIATION:
            self._hyperparameters_at(node.children[0], prefix + "kernel__", out)
            return
        if k == _Kind.COMPOUND:
            return  # sklearn exposes no hyperparameter_* attributes here.
        if k == _Kind.RATIONAL_QUADRATIC:
            out.append(_spec(prefix + "alpha", 1, node.bounds2))
            out.append(_spec(prefix + "length_scale", 1, node.bounds))
            return
        if k == _Kind.EXP_SINE_SQUARED:
            out.append(_spec(prefix + "length_scale", 1, node.bounds))
            out.append(_spec(prefix + "periodicity", 1, node.bounds2))
            return
        var n_elements = len(node.values) if len(node.values) > 1 else 1
        out.append(_spec(prefix + _primary_name(k), n_elements, node.bounds))

    def hyperparameters(self) raises -> List[HyperparameterSpec]:
        """Hyperparameter specifications in theta order (fixed ones included)."""
        var out = List[HyperparameterSpec]()
        self._hyperparameters_at(self._root, "", out)
        return out^

    def _theta_at(self, idx: Int, mut out: List[Float64]):
        ref node = self._nodes[idx]
        var k = node.kind
        if not _is_leaf(k):
            for c in node.children:
                self._theta_at(c, out)
            return
        if k == _Kind.RATIONAL_QUADRATIC:
            if not node.bounds2.is_fixed:
                out.append(flog(node.value2))
            if not node.bounds.is_fixed:
                out.append(flog(node.values[0]))
            return
        if k == _Kind.EXP_SINE_SQUARED:
            if not node.bounds.is_fixed:
                out.append(flog(node.values[0]))
            if not node.bounds2.is_fixed:
                out.append(flog(node.value2))
            return
        if not node.bounds.is_fixed:
            for v in node.values:
                out.append(flog(v))

    def _theta_list(self) -> List[Float64]:
        var out = List[Float64]()
        self._theta_at(self._root, out)
        return out^

    def theta(self) raises -> Dense64:
        """Log-transformed non-fixed hyperparameters, shape `(n_dims,)`."""
        var values = self._theta_list()
        var out = zeros[DType.float64](Shape(len(values)))
        for i in range(len(values)):
            out.unsafe_set(i, values[i])
        return out^

    def n_dims(self) -> Int:
        """Number of non-fixed hyperparameters (length of theta)."""
        return len(self._theta_list())

    def _bounds_at(self, idx: Int, mut rows: List[Float64]) raises:
        ref node = self._nodes[idx]
        var k = node.kind
        if not _is_leaf(k):
            for c in node.children:
                self._bounds_at(c, rows)
            return
        var specs = List[HyperparameterSpec]()
        self._hyperparameters_at(idx, "", specs)
        for spec in specs:
            if spec.fixed:
                continue
            for i in range(spec.n_elements):
                rows.append(flog(spec.bounds.unsafe_get(2 * i)))
                rows.append(flog(spec.bounds.unsafe_get(2 * i + 1)))

    def bounds(self) raises -> Dense64:
        """Log-transformed bounds `(n_dims, 2)`; `(0, 2)` when nothing is free."""
        var rows = List[Float64]()
        self._bounds_at(self._root, rows)
        var n = len(rows) // 2
        var out = zeros[DType.float64](Shape(n, 2))
        for i in range(len(rows)):
            out.unsafe_set(i, rows[i])
        return out^

    def _set_theta_at(mut self, idx: Int, theta: List[Float64], mut pos: Int):
        var k = self._nodes[idx].kind
        if not _is_leaf(k):
            var children = self._nodes[idx].children.copy()
            for c in children:
                self._set_theta_at(c, theta, pos)
            return
        ref node = self._nodes[idx]
        if k == _Kind.RATIONAL_QUADRATIC:
            if not node.bounds2.is_fixed:
                node.value2 = fexp(theta[pos])
                pos += 1
            if not node.bounds.is_fixed:
                node.values[0] = fexp(theta[pos])
                pos += 1
            return
        if k == _Kind.EXP_SINE_SQUARED:
            if not node.bounds.is_fixed:
                node.values[0] = fexp(theta[pos])
                pos += 1
            if not node.bounds2.is_fixed:
                node.value2 = fexp(theta[pos])
                pos += 1
            return
        if not node.bounds.is_fixed:
            for i in range(len(node.values)):
                node.values[i] = fexp(theta[pos])
                pos += 1

    def set_theta(mut self, theta: Dense64) raises:
        """sklearn `kernel.theta = theta`: exponentiate into the hyperparameters."""
        var expected = self.n_dims()
        if theta.ndim != 1 or theta.size != expected:
            raise Error(
                "ValueError: theta has not the correct number of entries."
                " Should be "
                + String(expected)
                + "; given are "
                + String(theta.size)
            )
        var values = List[Float64]()
        var stride = Int(theta.strides[0])
        for i in range(theta.size):
            values.append(theta.unsafe_get(i * stride))
        var pos = 0
        self._set_theta_at(self._root, values, pos)

    def clone_with_theta(self, theta: Dense64) raises -> Kernel:
        """An independent copy with the given log hyperparameters."""
        var cloned = self.copy()
        cloned.set_theta(theta)
        return cloned^

    # --- evaluation ---------------------------------------------------------

    def _gradient_width(self, idx: Int) -> Int:
        var out = List[Float64]()
        self._theta_at(idx, out)
        return len(out)

    def _eval_leaf(
        self, idx: Int, X: Dense64, Y: Dense64, same: Bool, want_grad: Bool
    ) raises -> _Eval:
        ref node = self._nodes[idx]
        var k = node.kind
        if k != _Kind.PAIRWISE:
            validate_gradient(not same, want_grad)
        var n = X.shape[0]
        var m = Y.shape[0]
        var d = X.shape[1]
        if k == _Kind.CONSTANT:
            var c = node.values[0]
            var G = empty_gradient(n)
            if want_grad and not node.bounds.is_fixed:
                G = zeros[DType.float64](Shape(n, n, 1))
                var g = G.unsafe_ptr()
                for i in range(n * n):
                    g[unsafe_offset=i] = c
            return _Eval(K=fill(n, m, c), G=G^)
        if k == _Kind.WHITE:
            var c = node.values[0]
            var K = zeros[DType.float64](Shape(n, m))
            var G = empty_gradient(n)
            if same:
                for i in range(n):
                    K.unsafe_set(i * n + i, c)
                if want_grad and not node.bounds.is_fixed:
                    G = zeros[DType.float64](Shape(n, n, 1))
                    for i in range(n):
                        G.unsafe_set(i * n + i, c)
            return _Eval(K=K^, G=G^)
        if k == _Kind.DOT_PRODUCT:
            var s2 = node.values[0] * node.values[0]
            var K = inner_products(X, Y)
            var p = K.unsafe_ptr()
            for i in range(K.size):
                p[unsafe_offset=i] = p[unsafe_offset=i] + s2
            var G = empty_gradient(n)
            if want_grad and not node.bounds.is_fixed:
                G = zeros[DType.float64](Shape(n, n, 1))
                var g = G.unsafe_ptr()
                for i in range(n * n):
                    g[unsafe_offset=i] = 2.0 * s2
            return _Eval(K=K^, G=G^)
        if k == _Kind.PAIRWISE:
            var gamma_value = node.values[0]
            var K = _pairwise_matrix(node, gamma_value, X, Y, same)
            var G = empty_gradient(n)
            if want_grad and not node.bounds.is_fixed:
                var theta0 = flog(gamma_value)
                var f0 = _pairwise_matrix(node, fexp(theta0), X, Y, same)
                var f1 = _pairwise_matrix(node, fexp(theta0 + _FD_EPSILON), X, Y, same)
                G = zeros[DType.float64](Shape(n, m, 1))
                var g = G.unsafe_ptr()
                var a = f0.unsafe_ptr()
                var b = f1.unsafe_ptr()
                for i in range(n * m):
                    g[unsafe_offset=i] = (b[unsafe_offset=i] - a[unsafe_offset=i]) / _FD_EPSILON
            return _Eval(K=K^, G=G^)
        if k == _Kind.RATIONAL_QUADRATIC:
            if len(node.values) > 1:
                raise Error(
                    "AttributeError: RationalQuadratic kernel only supports isotropic"
                    " version, please use a single scalar for length_scale"
                )
            var ls = node.values[0]
            var alpha = node.value2
            var D2 = squared_distances(X, Y, same)
            var K = zeros[DType.float64](Shape(n, m))
            var base = zeros[DType.float64](Shape(n, m))
            var dp = D2.unsafe_ptr()
            var kp = K.unsafe_ptr()
            var bp = base.unsafe_ptr()
            for i in range(n * m):
                var b = 1.0 + dp[unsafe_offset=i] / (2.0 * alpha * (ls * ls))
                bp[unsafe_offset=i] = b
                kp[unsafe_offset=i] = fpow(b, -alpha)
            if same:
                for i in range(n):
                    kp[unsafe_offset = i * m + i] = 1.0
            var width = (0 if node.bounds.is_fixed else 1) + (0 if node.bounds2.is_fixed else 1)
            var G = zeros[DType.float64](Shape(n, n, width)) if want_grad else empty_gradient(n)
            if want_grad and width > 0:
                var g = G.unsafe_ptr()
                for i in range(n * n):
                    var col = 0
                    var kv = kp[unsafe_offset=i]
                    var b = bp[unsafe_offset=i]
                    var dist = dp[unsafe_offset=i]
                    if not node.bounds2.is_fixed:
                        g[unsafe_offset = i * width + col] = kv * (
                            -alpha * flog(b) + dist / (2.0 * (ls * ls) * b)
                        )
                        col += 1
                    if not node.bounds.is_fixed:
                        g[unsafe_offset = i * width + col] = dist * kv / ((ls * ls) * b)
            return _Eval(K=K^, G=G^)
        if k == _Kind.EXP_SINE_SQUARED:
            var ls = node.values[0]
            var period = node.value2
            var D = euclidean_from_squared(squared_distances(X, Y, same))
            var K = zeros[DType.float64](Shape(n, m))
            var dp = D.unsafe_ptr()
            var kp = K.unsafe_ptr()
            for i in range(n * m):
                var s: Float64
                if same:
                    s = sin(pi * dp[unsafe_offset=i] / period)
                else:
                    s = sin(pi / period * dp[unsafe_offset=i])
                kp[unsafe_offset=i] = fexp(-2.0 * (s / ls) * (s / ls))
            var width = (0 if node.bounds.is_fixed else 1) + (0 if node.bounds2.is_fixed else 1)
            var G = zeros[DType.float64](Shape(n, n, width)) if want_grad else empty_gradient(n)
            if want_grad and width > 0:
                var g = G.unsafe_ptr()
                for i in range(n * n):
                    var arg = pi * dp[unsafe_offset=i] / period
                    var s = sin(arg)
                    var c = cos(arg)
                    var kv = kp[unsafe_offset=i]
                    var col = 0
                    if not node.bounds.is_fixed:
                        g[unsafe_offset = i * width + col] = 4.0 / (ls * ls) * (s * s) * kv
                        col += 1
                    if not node.bounds2.is_fixed:
                        g[unsafe_offset = i * width + col] = 4.0 * arg / (ls * ls) * c * s * kv
            return _Eval(K=K^, G=G^)
        # RBF and Matern share scaled distances.
        _check_length_scale(node, d)
        var ls = node.values.copy()
        var anisotropic = len(ls) > 1
        var Xs = scale_columns(X, ls)
        var D2: Dense64
        if same:
            D2 = squared_distances(Xs, Xs, True)
        else:
            D2 = squared_distances(Xs, scale_columns(Y, ls), False)
        var K: Dense64
        if k == _Kind.RBF:
            K = D2.copy()
            var kp = K.unsafe_ptr()
            for i in range(K.size):
                kp[unsafe_offset=i] = fexp(-0.5 * kp[unsafe_offset=i])
            if same:
                for i in range(n):
                    kp[unsafe_offset = i * m + i] = 1.0
        else:
            K = _matern_from_distances(node.extra, euclidean_from_squared(D2), same)
        if not want_grad or node.bounds.is_fixed:
            return _Eval(K=K^, G=empty_gradient(n))
        var nu = node.extra
        var general = k == _Kind.MATERN and not (
            nu == 0.5 or nu == 1.5 or nu == 2.5 or _is_inf(nu)
        )
        if general:
            # sklearn: forward differences of clone_with_theta(theta)(X).
            var theta0 = List[Float64]()
            for v in ls:
                theta0.append(flog(v))
            var width = len(theta0)
            var G = zeros[DType.float64](Shape(n, n, width))
            var base_scale = List[Float64]()
            for t in theta0:
                base_scale.append(fexp(t))
            var f0 = _matern_from_distances(
                nu, euclidean_from_squared(squared_distances(scale_columns(X, base_scale), scale_columns(X, base_scale), True)), True
            )
            var g = G.unsafe_ptr()
            var a = f0.unsafe_ptr()
            for c in range(width):
                var shifted = List[Float64]()
                for j in range(width):
                    shifted.append(fexp(theta0[j] + (_FD_EPSILON if j == c else 0.0)))
                var Xc = scale_columns(X, shifted)
                var f1 = _matern_from_distances(
                    nu, euclidean_from_squared(squared_distances(Xc, Xc, True)), True
                )
                var b = f1.unsafe_ptr()
                for e in range(n * n):
                    g[unsafe_offset = e * width + c] = (
                        b[unsafe_offset=e] - a[unsafe_offset=e]
                    ) / _FD_EPSILON
            return _Eval(K=K^, G=G^)
        var width = len(ls) if anisotropic else 1
        var G = zeros[DType.float64](Shape(n, n, width))
        var g = G.unsafe_ptr()
        var kp = K.unsafe_ptr()
        var d2p = D2.unsafe_ptr()
        var xp = X.unsafe_ptr()
        var parts = List[Float64](length=width, fill=0.0)
        for i in range(n):
            for j in range(n):
                var e = i * n + j
                var total: Float64 = 0.0
                if anisotropic:
                    for c in range(width):
                        var diff = xp[unsafe_offset = i * d + c] - xp[unsafe_offset = j * d + c]
                        parts[c] = diff * diff / (ls[c] * ls[c])
                        total += parts[c]
                else:
                    parts[0] = d2p[unsafe_offset=e]
                    total = parts[0]
                var kv = kp[unsafe_offset=e]
                for c in range(width):
                    var Dc = parts[c]
                    var value: Float64
                    if k == _Kind.RBF or _is_inf(nu):
                        value = Dc * kv
                    elif nu == 0.5:
                        var denom = sqrt(total)
                        value = kv * (Dc / denom) if denom != 0.0 else 0.0
                    elif nu == 1.5:
                        value = 3.0 * Dc * fexp(-sqrt(3.0 * total))
                    else:
                        var tmp = sqrt(5.0 * total)
                        value = 5.0 / 3.0 * Dc * (tmp + 1.0) * fexp(-tmp)
                    g[unsafe_offset = e * width + c] = value
        return _Eval(K=K^, G=G^)

    def _eval(
        self, idx: Int, X: Dense64, Y: Dense64, same: Bool, want_grad: Bool
    ) raises -> _Eval:
        var k = self._nodes[idx].kind
        if _is_leaf(k):
            return self._eval_leaf(idx, X, Y, same, want_grad)
        if k == _Kind.COMPOUND:
            raise Error(
                "ValueError: CompoundKernel cannot be nested inside another kernel"
            )
        var c0 = self._nodes[idx].children[0]
        if self._nodes[c0].kind == _Kind.COMPOUND or (
            len(self._nodes[idx].children) > 1
            and self._nodes[self._nodes[idx].children[1]].kind == _Kind.COMPOUND
        ):
            raise Error(
                "ValueError: CompoundKernel cannot be nested inside another kernel"
            )
        if k == _Kind.EXPONENTIATION:
            var e = self._eval(c0, X, Y, same, want_grad)
            var power = self._nodes[idx].extra
            if want_grad:
                # sklearn scales this gradient in place: broadcasting may not expand it.
                if e.G.shape[1] != e.K.shape[1] and e.K.shape[1] != 1:
                    raise Error("ValueError: exponentiation cannot expand the gradient shape")
                var factor = _elementwise_pow(e.K, power - 1.0)
                for i in range(factor.size):
                    factor.unsafe_set(i, power * factor.unsafe_get(i))
                multiply_gradient(e.G, factor)
            e.K = _elementwise_pow(e.K, power)
            return e^
        var c1 = self._nodes[idx].children[1]
        var e1 = self._eval(c0, X, Y, same, want_grad)
        var e2 = self._eval(c1, X, Y, same, want_grad)
        var k1p = e1.K.unsafe_ptr()
        var k2p = e2.K.unsafe_ptr()
        if k == _Kind.SUM:
            for i in range(e1.K.size):
                k1p[unsafe_offset=i] = k1p[unsafe_offset=i] + k2p[unsafe_offset=i]
            if want_grad:
                e1.G = concat_gradients(e1.G, e2.G)
            return e1^
        if want_grad:
            multiply_gradient(e1.G, e2.K)
            multiply_gradient(e2.G, e1.K)
            e1.G = concat_gradients(e1.G, e2.G)
        for i in range(e1.K.size):
            k1p[unsafe_offset=i] = k1p[unsafe_offset=i] * k2p[unsafe_offset=i]
        return e1^

    def _evaluate_compound(
        self, X: Dense64, Y: Dense64, same: Bool, want_grad: Bool
    ) raises -> KernelEvaluation:
        var children = self._nodes[self._root].children.copy()
        var count = len(children)
        var n = X.shape[0]
        var m = Y.shape[0]
        var results = List[_Eval]()
        var width = -1
        for c in children:
            var e = self._eval(c, X, Y, same, want_grad)
            if want_grad:
                if width >= 0 and e.G.shape[2] != width:
                    raise Error(
                        "ValueError: CompoundKernel gradients require children with"
                        " equal numbers of free hyperparameters (got "
                        + String(width)
                        + " and "
                        + String(e.G.shape[2])
                        + ")"
                    )
                width = e.G.shape[2]
            results.append(e^)
        var K = zeros[DType.float64](Shape(n, m, count))
        var kp = K.unsafe_ptr()
        for t in range(count):
            var src = results[t].K.unsafe_ptr()
            for e in range(n * m):
                kp[unsafe_offset = e * count + t] = src[unsafe_offset=e]
        if not want_grad:
            return KernelEvaluation(covariance=K^, gradient=None)
        var p = width if width > 0 else 0
        var gradient_cols = results[0].G.shape[1] if count else n
        for t in range(count):
            if results[t].G.shape[1] != gradient_cols:
                raise Error("ValueError: CompoundKernel gradient shapes must match")
        var G = zeros[DType.float64](Shape(n, gradient_cols, p, count))
        var gp = G.unsafe_ptr()
        for t in range(count):
            var src = results[t].G.unsafe_ptr()
            for e in range(n * gradient_cols):
                for c in range(p):
                    gp[unsafe_offset = (e * p + c) * count + t] = src[unsafe_offset = e * p + c]
        return KernelEvaluation(covariance=K^, gradient=G^)

    def _check_features_at(self, idx: Int, X: Dense64, Y: Dense64) raises:
        ref node = self._nodes[idx]
        if not _is_leaf(node.kind):
            for child in node.children:
                self._check_features_at(child, X, Y)
            return
        if node.kind == _Kind.PAIRWISE and node.metric == "precomputed":
            return
        if Y.shape[1] != X.shape[1] and self._vector_at(idx):
            raise Error(
                "ValueError: X has "
                + String(X.shape[1])
                + " features but Y has "
                + String(Y.shape[1])
                + "; kernel "
                + self.name()
                + " needs equal feature counts"
            )

    def evaluate(
        self, X: Dense64, Y: Optional[Dense64] = None, eval_gradient: Bool = False
    ) raises -> KernelEvaluation:
        """sklearn `kernel(X, Y, eval_gradient)`.

        Returns covariance `(n, m)` and, when requested, the log-theta gradient
        `(n, n, n_dims)`. PairwiseKernel also allows explicit-Y gradients.
        Other leaves require `Y=None`. CompoundKernel returns
        `(n, m, k)` covariance and `(n, n, n_dims_child, k)` gradients.
        """
        var Xd = as_dense64(X, "kernel(X)")
        var same = not Bool(Y)
        var Yd = Xd.copy() if same else as_dense64(Y.value(), "kernel(X, Y)")
        if not same:
            self._check_features_at(self._root, Xd, Yd)
        if self._nodes[self._root].kind == _Kind.COMPOUND:
            return self._evaluate_compound(Xd, Yd, same, eval_gradient)
        return self._eval(self._root, Xd, Yd, same, eval_gradient).into_evaluation(
            eval_gradient
        )

    def __call__[
        dtype: DType
    ](self, X: NDArray[dtype], Y: Optional[NDArray[dtype]] = None) raises -> Dense64:
        """Covariance matrix k(X, Y) (k(X, X) when Y is None), in Float64."""
        var Xd = as_dense64(X, "kernel(X)")
        if self._nodes[self._root].kind == _Kind.COMPOUND:
            raise Error(
                "ValueError: CompoundKernel returns a 3-D covariance; use evaluate()"
            )
        if Y:
            var Yd = as_dense64(Y.value(), "kernel(X, Y)")
            self._check_features_at(self._root, Xd, Yd)
            return self._eval(self._root, Xd, Yd, False, False).take_covariance()
        return self._eval(self._root, Xd, Xd, True, False).take_covariance()

    def _diag_at(self, idx: Int, X: Dense64) raises -> Dense64:
        ref node = self._nodes[idx]
        var k = node.kind
        var n = X.shape[0]
        if k == _Kind.CONSTANT or k == _Kind.WHITE:
            return vector_fill(n, node.values[0])
        if (
            k == _Kind.RBF
            or k == _Kind.MATERN
            or k == _Kind.RATIONAL_QUADRATIC
            or k == _Kind.EXP_SINE_SQUARED
        ):
            return vector_fill(n, 1.0)
        if k == _Kind.DOT_PRODUCT:
            var out = zeros[DType.float64](Shape(n))
            var d = X.shape[1]
            var xp = X.unsafe_ptr()
            for i in range(n):
                var acc: Float64 = 0.0
                for c in range(d):
                    var v = xp[unsafe_offset = i * d + c]
                    acc += v * v
                out.unsafe_set(i, acc + node.values[0] * node.values[0])
            return out^
        if k == _Kind.PAIRWISE:
            _validate_pairwise_input(X)
            var out = zeros[DType.float64](Shape(n))
            var d = X.shape[1]
            for i in range(n):
                var row = zeros[DType.float64](Shape(1, d))
                for c in range(d):
                    row.unsafe_set(c, X.unsafe_get(i * d + c))
                var K = _pairwise_matrix(node, node.values[0], row, row, True)
                out.unsafe_set(i, K.unsafe_get(0))
            return out^
        if k == _Kind.COMPOUND:
            raise Error("ValueError: CompoundKernel cannot be nested inside another kernel")
        var d0 = self._diag_at(node.children[0], X)
        if k == _Kind.EXPONENTIATION:
            return _elementwise_pow(d0, node.extra)
        var d1 = self._diag_at(node.children[1], X)
        for i in range(n):
            if k == _Kind.SUM:
                d0.unsafe_set(i, d0.unsafe_get(i) + d1.unsafe_get(i))
            else:
                d0.unsafe_set(i, d0.unsafe_get(i) * d1.unsafe_get(i))
        return d0^

    def diag(self, X: Dense64) raises -> Dense64:
        """Diagonal of k(X, X), shape `(n,)` (`(n, k)` for CompoundKernel)."""
        var Xd = as_dense64(X, "kernel.diag(X)")
        if self._nodes[self._root].kind != _Kind.COMPOUND:
            return self._diag_at(self._root, Xd)
        var children = self._nodes[self._root].children.copy()
        var n = Xd.shape[0]
        var count = len(children)
        var out = zeros[DType.float64](Shape(n, count))
        for t in range(count):
            var d = self._diag_at(children[t], Xd)
            for i in range(n):
                out.unsafe_set(i * count + t, d.unsafe_get(i))
        return out^

    def diag_of[dtype: DType](self, X: NDArray[dtype]) raises -> Dense64:
        """`diag` for Float32 or Float64 input (converted to Float64)."""
        return self.diag(as_dense64(X, "kernel.diag(X)"))

    # --- predicates -----------------------------------------------------------

    def _stationary_at(self, idx: Int) -> Bool:
        ref node = self._nodes[idx]
        if node.kind == _Kind.DOT_PRODUCT:
            return False
        if node.kind == _Kind.PAIRWISE:
            return node.metric == "rbf"
        if _is_leaf(node.kind):
            return True
        for c in node.children:
            if not self._stationary_at(c):
                return False
        return True

    def is_stationary(self) -> Bool:
        """Whether k(X, Y) depends only on X - Y."""
        return self._stationary_at(self._root)

    def _vector_at(self, idx: Int) -> Bool:
        ref node = self._nodes[idx]
        if node.kind == _Kind.CONSTANT or node.kind == _Kind.WHITE:
            return False
        if _is_leaf(node.kind):
            return True
        for c in node.children:
            if self._vector_at(c):
                return True
        return False

    def requires_vector_input(self) -> Bool:
        """Whether the kernel needs fixed-length feature vectors."""
        return self._vector_at(self._root)

    # --- equality and representation -------------------------------------------

    def _equal_at(self, idx: Int, other: Kernel, other_idx: Int) -> Bool:
        ref a = self._nodes[idx]
        ref b = other._nodes[other_idx]
        if a.kind != b.kind:
            return False
        if _is_leaf(a.kind):
            if len(a.values) != len(b.values):
                return False
            for i in range(len(a.values)):
                if a.values[i] != b.values[i]:
                    return False
            if a.bounds != b.bounds or a.metric != b.metric or a.has_kwargs != b.has_kwargs:
                return False
            if a.kind == _Kind.RATIONAL_QUADRATIC or a.kind == _Kind.EXP_SINE_SQUARED:
                if a.value2 != b.value2 or a.bounds2 != b.bounds2:
                    return False
            if a.kind == _Kind.MATERN and a.extra != b.extra:
                return False
            if a.has_kwargs:
                if len(a.kwargs) != len(b.kwargs):
                    return False
                for entry in a.kwargs.items():
                    var found = b.kwargs.get(entry.key)
                    if not found or found.value() != entry.value:
                        return False
            return True
        if a.kind == _Kind.EXPONENTIATION:
            return a.extra == b.extra and self._equal_at(a.children[0], other, b.children[0])
        if a.kind == _Kind.COMPOUND:
            if len(a.children) != len(b.children):
                return False
            for i in range(len(a.children)):
                if not self._equal_at(a.children[i], other, b.children[i]):
                    return False
            return True
        var straight = self._equal_at(a.children[0], other, b.children[0]) and self._equal_at(
            a.children[1], other, b.children[1]
        )
        return straight or (
            self._equal_at(a.children[0], other, b.children[1])
            and self._equal_at(a.children[1], other, b.children[0])
        )

    def __eq__(self, other: Kernel) -> Bool:
        """sklearn kernel equality (operators also match with swapped operands)."""
        return self._equal_at(self._root, other, other._root)

    def __ne__(self, other: Kernel) -> Bool:
        return not self == other

    def _write_at(self, idx: Int, mut writer: Some[Writer]):
        ref node = self._nodes[idx]
        var k = node.kind
        if k == _Kind.CONSTANT:
            writer.write(format_g3(sqrt(node.values[0])), "**2")
        elif k == _Kind.WHITE:
            writer.write("WhiteKernel(noise_level=", format_g3(node.values[0]), ")")
        elif k == _Kind.RBF or k == _Kind.MATERN:
            writer.write(_kind_name(k), "(length_scale=")
            if len(node.values) > 1:
                writer.write("[")
                for i in range(len(node.values)):
                    if i > 0:
                        writer.write(", ")
                    writer.write(format_g3(node.values[i]))
                writer.write("]")
            else:
                writer.write(format_g3(node.values[0]))
            if k == _Kind.MATERN:
                writer.write(", nu=", format_g3(node.extra))
            writer.write(")")
        elif k == _Kind.RATIONAL_QUADRATIC:
            writer.write(
                "RationalQuadratic(alpha=",
                format_g3(node.value2),
                ", length_scale=",
                format_g3(node.values[0]),
                ")",
            )
        elif k == _Kind.EXP_SINE_SQUARED:
            writer.write(
                "ExpSineSquared(length_scale=",
                format_g3(node.values[0]),
                ", periodicity=",
                format_g3(node.value2),
                ")",
            )
        elif k == _Kind.DOT_PRODUCT:
            writer.write("DotProduct(sigma_0=", format_g3(node.values[0]), ")")
        elif k == _Kind.PAIRWISE:
            writer.write(
                "PairwiseKernel(gamma=", _py_float(node.values[0]), ", metric=", node.metric, ")"
            )
        elif k == _Kind.SUM or k == _Kind.PRODUCT:
            self._write_at(node.children[0], writer)
            writer.write(" + " if k == _Kind.SUM else " * ")
            self._write_at(node.children[1], writer)
        elif k == _Kind.EXPONENTIATION:
            self._write_at(node.children[0], writer)
            writer.write(" ** ", _py_float(node.extra))
        else:
            writer.write("CompoundKernel(")
            var values = List[Float64]()
            self._theta_at(idx, values)
            for i in range(len(values)):
                if i > 0:
                    writer.write(", ")
                writer.write(format_g3(values[i]))
            writer.write(")")

    def write_to(self, mut writer: Some[Writer]):
        """sklearn `repr(kernel)`, e.g. `1**2 * RBF(length_scale=1)`."""
        self._write_at(self._root, writer)

    # --- operators ------------------------------------------------------------

    def __add__(self, other: Kernel) -> Kernel:
        return Sum(self, other)

    def __add__(self, other: Float64) -> Kernel:
        return Sum(self, ConstantKernel(other))

    def __radd__(self, other: Float64) -> Kernel:
        return Sum(ConstantKernel(other), self)

    def __mul__(self, other: Kernel) -> Kernel:
        return Product(self, other)

    def __mul__(self, other: Float64) -> Kernel:
        return Product(self, ConstantKernel(other))

    def __rmul__(self, other: Float64) -> Kernel:
        return Product(ConstantKernel(other), self)

    def __pow__(self, exponent: Float64) -> Kernel:
        return Exponentiation(self, exponent)


def _primary_name(kind: Int) -> String:
    if kind == _Kind.CONSTANT:
        return "constant_value"
    if kind == _Kind.WHITE:
        return "noise_level"
    if kind == _Kind.DOT_PRODUCT:
        return "sigma_0"
    if kind == _Kind.PAIRWISE:
        return "gamma"
    return "length_scale"


def _spec(name: String, n_elements: Int, bounds: Bounds) raises -> HyperparameterSpec:
    return HyperparameterSpec(
        name, n_elements, bounds.is_fixed, bounds.expanded(n_elements, name)
    )


# ===----------------------------------------------------------------------=== #
# Composite constructors
# ===----------------------------------------------------------------------=== #


def Sum(k1: Kernel, k2: Kernel) -> Kernel:
    """sklearn `Sum(k1, k2)`: k1(X, Y) + k2(X, Y)."""
    return Kernel._combine(_Kind.SUM, [k1.copy(), k2.copy()])


def Product(k1: Kernel, k2: Kernel) -> Kernel:
    """sklearn `Product(k1, k2)`: k1(X, Y) * k2(X, Y)."""
    return Kernel._combine(_Kind.PRODUCT, [k1.copy(), k2.copy()])


def Exponentiation(kernel: Kernel, exponent: Float64) -> Kernel:
    """sklearn `Exponentiation(kernel, exponent)`: k(X, Y) ** exponent."""
    return Kernel._combine(_Kind.EXPONENTIATION, [kernel.copy()], exponent)


def CompoundKernel(kernels: List[Kernel]) raises -> Kernel:
    """sklearn `CompoundKernel(kernels)`: children stacked on a last axis."""
    if len(kernels) == 0:
        raise Error("ValueError: CompoundKernel needs at least one kernel")
    return Kernel._combine(_Kind.COMPOUND, kernels)


# ===----------------------------------------------------------------------=== #
# Leaf kernels
# ===----------------------------------------------------------------------=== #


trait KernelLike(Copyable, Movable):
    """Typed leaf kernel that converts to `Kernel` and inherits its operations."""

    def as_kernel(self) -> Kernel:
        ...

    def theta(self) raises -> Dense64:
        return self.as_kernel().theta()

    def bounds(self) raises -> Dense64:
        return self.as_kernel().bounds()

    def n_dims(self) -> Int:
        return self.as_kernel().n_dims()

    def hyperparameters(self) raises -> List[HyperparameterSpec]:
        return self.as_kernel().hyperparameters()

    def get_params(self, deep: Bool = True) raises -> KernelParams:
        return self.as_kernel().get_params(deep)

    def clone_with_theta(self, theta: Dense64) raises -> Kernel:
        return self.as_kernel().clone_with_theta(theta)

    def evaluate(
        self, X: Dense64, Y: Optional[Dense64] = None, eval_gradient: Bool = False
    ) raises -> KernelEvaluation:
        return self.as_kernel().evaluate(X, Y, eval_gradient)

    def __call__[
        dtype: DType
    ](self, X: NDArray[dtype], Y: Optional[NDArray[dtype]] = None) raises -> Dense64:
        return self.as_kernel()(X, Y)

    def diag(self, X: Dense64) raises -> Dense64:
        return self.as_kernel().diag(X)

    def is_stationary(self) -> Bool:
        return self.as_kernel().is_stationary()

    def requires_vector_input(self) -> Bool:
        return self.as_kernel().requires_vector_input()

    def __add__(self, other: Kernel) -> Kernel:
        return Sum(self.as_kernel(), other)

    def __add__(self, other: Float64) -> Kernel:
        return Sum(self.as_kernel(), ConstantKernel(other))

    def __radd__(self, other: Float64) -> Kernel:
        return Sum(ConstantKernel(other), self.as_kernel())

    def __mul__(self, other: Kernel) -> Kernel:
        return Product(self.as_kernel(), other)

    def __mul__(self, other: Float64) -> Kernel:
        return Product(self.as_kernel(), ConstantKernel(other))

    def __rmul__(self, other: Float64) -> Kernel:
        return Product(ConstantKernel(other), self.as_kernel())

    def __pow__(self, exponent: Float64) -> Kernel:
        return Exponentiation(self.as_kernel(), exponent)


struct ConstantKernel(KernelLike, Writable):
    """k(x, y) = constant_value for all pairs (sklearn ConstantKernel)."""

    var constant_value: Float64
    var constant_value_bounds: Bounds

    def __init__(
        out self,
        constant_value: Float64 = 1.0,
        constant_value_bounds: Bounds = Bounds(1e-5, 1e5),
    ):
        self.constant_value = constant_value
        self.constant_value_bounds = constant_value_bounds.copy()

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.as_kernel())

    def as_kernel(self) -> Kernel:
        var node = _Node(_Kind.CONSTANT)
        node.values = [self.constant_value]
        node.bounds = self.constant_value_bounds.copy()
        return Kernel._leaf(node^)


struct WhiteKernel(KernelLike, Writable):
    """Independent noise: noise_level on the diagonal of k(X, X), zero otherwise."""

    var noise_level: Float64
    var noise_level_bounds: Bounds

    def __init__(
        out self,
        noise_level: Float64 = 1.0,
        noise_level_bounds: Bounds = Bounds(1e-5, 1e5),
    ):
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds.copy()

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.as_kernel())

    def as_kernel(self) -> Kernel:
        var node = _Node(_Kind.WHITE)
        node.values = [self.noise_level]
        node.bounds = self.noise_level_bounds.copy()
        return Kernel._leaf(node^)


struct RBF(KernelLike, Writable):
    """Squared-exponential kernel, isotropic or anisotropic length scale."""

    var length_scale: List[Float64]
    var length_scale_bounds: Bounds

    def __init__(
        out self,
        length_scale: Float64 = 1.0,
        length_scale_bounds: Bounds = Bounds(1e-5, 1e5),
    ):
        self.length_scale = [length_scale]
        self.length_scale_bounds = length_scale_bounds.copy()

    def __init__(
        out self,
        length_scale: List[Float64],
        length_scale_bounds: Bounds = Bounds(1e-5, 1e5),
    ):
        self.length_scale = length_scale.copy()
        self.length_scale_bounds = length_scale_bounds.copy()

    def anisotropic(self) -> Bool:
        return len(self.length_scale) > 1

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.as_kernel())

    def as_kernel(self) -> Kernel:
        var node = _Node(_Kind.RBF)
        node.values = self.length_scale.copy()
        node.bounds = self.length_scale_bounds.copy()
        return Kernel._leaf(node^)


struct Matern(KernelLike, Writable):
    """Matern kernel; nu in {0.5, 1.5, 2.5, inf} closed form, otherwise Bessel K_nu.
    """

    var length_scale: List[Float64]
    var length_scale_bounds: Bounds
    var nu: Float64

    def __init__(
        out self,
        length_scale: Float64 = 1.0,
        length_scale_bounds: Bounds = Bounds(1e-5, 1e5),
        nu: Float64 = 1.5,
    ):
        self.length_scale = [length_scale]
        self.length_scale_bounds = length_scale_bounds.copy()
        self.nu = nu

    def __init__(
        out self,
        length_scale: List[Float64],
        length_scale_bounds: Bounds = Bounds(1e-5, 1e5),
        nu: Float64 = 1.5,
    ):
        self.length_scale = length_scale.copy()
        self.length_scale_bounds = length_scale_bounds.copy()
        self.nu = nu

    def anisotropic(self) -> Bool:
        return len(self.length_scale) > 1

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.as_kernel())

    def as_kernel(self) -> Kernel:
        var node = _Node(_Kind.MATERN)
        node.values = self.length_scale.copy()
        node.bounds = self.length_scale_bounds.copy()
        node.extra = self.nu
        return Kernel._leaf(node^)


struct RationalQuadratic(KernelLike, Writable):
    """Scale mixture of RBF kernels: (1 + d^2 / (2 alpha l^2)) ** -alpha."""

    var length_scale: Float64
    var alpha: Float64
    var length_scale_bounds: Bounds
    var alpha_bounds: Bounds

    def __init__(
        out self,
        length_scale: Float64 = 1.0,
        alpha: Float64 = 1.0,
        length_scale_bounds: Bounds = Bounds(1e-5, 1e5),
        alpha_bounds: Bounds = Bounds(1e-5, 1e5),
    ):
        self.length_scale = length_scale
        self.alpha = alpha
        self.length_scale_bounds = length_scale_bounds.copy()
        self.alpha_bounds = alpha_bounds.copy()

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.as_kernel())

    def as_kernel(self) -> Kernel:
        var node = _Node(_Kind.RATIONAL_QUADRATIC)
        node.values = [self.length_scale]
        node.bounds = self.length_scale_bounds.copy()
        node.value2 = self.alpha
        node.bounds2 = self.alpha_bounds.copy()
        return Kernel._leaf(node^)


struct ExpSineSquared(KernelLike, Writable):
    """Periodic kernel: fexp(-2 sin^2(pi d / periodicity) / length_scale^2)."""

    var length_scale: Float64
    var periodicity: Float64
    var length_scale_bounds: Bounds
    var periodicity_bounds: Bounds

    def __init__(
        out self,
        length_scale: Float64 = 1.0,
        periodicity: Float64 = 1.0,
        length_scale_bounds: Bounds = Bounds(1e-5, 1e5),
        periodicity_bounds: Bounds = Bounds(1e-5, 1e5),
    ):
        self.length_scale = length_scale
        self.periodicity = periodicity
        self.length_scale_bounds = length_scale_bounds.copy()
        self.periodicity_bounds = periodicity_bounds.copy()

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.as_kernel())

    def as_kernel(self) -> Kernel:
        var node = _Node(_Kind.EXP_SINE_SQUARED)
        node.values = [self.length_scale]
        node.bounds = self.length_scale_bounds.copy()
        node.value2 = self.periodicity
        node.bounds2 = self.periodicity_bounds.copy()
        return Kernel._leaf(node^)


struct DotProduct(KernelLike, Writable):
    """Non-stationary linear kernel: sigma_0^2 + x . y."""

    var sigma_0: Float64
    var sigma_0_bounds: Bounds

    def __init__(
        out self, sigma_0: Float64 = 1.0, sigma_0_bounds: Bounds = Bounds(1e-5, 1e5)
    ):
        self.sigma_0 = sigma_0
        self.sigma_0_bounds = sigma_0_bounds.copy()

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.as_kernel())

    def as_kernel(self) -> Kernel:
        var node = _Node(_Kind.DOT_PRODUCT)
        node.values = [self.sigma_0]
        node.bounds = self.sigma_0_bounds.copy()
        return Kernel._leaf(node^)


struct PairwiseKernel(KernelLike, Writable):
    """Wrapper around sklearn.metrics.pairwise kernels with a tunable gamma.

    Supported dense metrics: linear, rbf, laplacian, poly/polynomial, sigmoid,
    cosine, chi2, additive_chi2, precomputed. `pairwise_kernels_kwargs` may hold `degree`
    and `coef0`; parameters a metric does not accept are ignored, as with
    sklearn's `filter_params=True`.
    """

    var gamma: Float64
    var gamma_bounds: Bounds
    var metric: String
    var pairwise_kernels_kwargs: Optional[Dict[String, Float64]]

    def __init__(
        out self,
        gamma: Float64 = 1.0,
        gamma_bounds: Bounds = Bounds(1e-5, 1e5),
        metric: String = "linear",
        pairwise_kernels_kwargs: Optional[Dict[String, Float64]] = None,
    ):
        self.gamma = gamma
        self.gamma_bounds = gamma_bounds.copy()
        self.metric = metric
        self.pairwise_kernels_kwargs = pairwise_kernels_kwargs.copy()

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.as_kernel())

    def as_kernel(self) -> Kernel:
        var node = _Node(_Kind.PAIRWISE)
        node.values = [self.gamma]
        node.bounds = self.gamma_bounds.copy()
        node.metric = self.metric
        if self.pairwise_kernels_kwargs:
            node.has_kwargs = True
            node.kwargs = self.pairwise_kernels_kwargs.value().copy()
        return Kernel._leaf(node^)


def to_float64[dtype: DType](X: NDArray[dtype]) raises -> Dense64:
    """Owned contiguous Float64 copy of Float32/Float64 dense input."""
    return as_dense64(X, "to_float64")
