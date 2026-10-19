# Adapted from scikit-learn sklearn/cross_decomposition/_pls.py at
# 4358542a1bd39af2e7e1cbace8b9fed067473c5b; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense two-block PLS engine and the PLSRegression, PLSCanonical, CCA estimators.

The engine (`PLSEngine`) is the single implementation of the family's
`TwoBlockEngine` contract: centering/scaling, NIPALS power iterations (modes A
and B), the SVD leading-vector solver, deflation, the PLSSVD projection weights
and pseudoinverse rotations. All fit arithmetic is Float64, as in scikit-learn.
Factorizations use mojo-linalg's economy Jacobi `svd` and its minimum-norm
`SVDFactor.solve`, which provides the pseudoinverse with an explicit cutoff.
"""
from std.math import sqrt, isfinite, isnan
import numojo as nm
from numojo import NDArray, Shape
from mojo_linalg import svd, SVDFactor
from moxie.cross_decomposition._base import (
    CrossDecompositionTransformer,
    PLSPredictor,
)
from moxie.cross_decomposition._contracts import (
    Matrix,
    KernelOptions,
    Loadings,
    PairedScores,
    ProjectionWeights,
    Scaling,
    TwoBlockEngine,
    WeightStep,
)

comptime EPS64: Float64 = 2.220446049250313e-16
"""NumPy `finfo(float64).eps`; every fit workspace is Float64."""


# ---------------------------------------------------------------------------
# Stride-aware element access. All engine inputs are valid NuMojo arrays; the
# helpers apply NuMojo strides so borrowed views are read without copying.
# ---------------------------------------------------------------------------


@always_inline
def _at(x: Matrix, i: Int, j: Int) raises -> Float64:
    return x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]))


@always_inline
def _put(mut x: Matrix, i: Int, j: Int, value: Float64) raises:
    x.unsafe_set(i * Int(x.strides[0]) + j * Int(x.strides[1]), value)


@always_inline
def _vat(v: Matrix, i: Int) raises -> Float64:
    """Element i of a vector stored as (n,) or as a single (n,1) column."""
    return v.unsafe_get(i * Int(v.strides[0]))


@always_inline
def _vput(mut v: Matrix, i: Int, value: Float64) raises:
    v.unsafe_set(i * Int(v.strides[0]), value)


def _vector(n: Int) raises -> Matrix:
    return nm.zeros[DType.float64](Shape(n))


def _matrix(rows: Int, cols: Int) raises -> Matrix:
    return nm.zeros[DType.float64](Shape(rows, cols))


def _dot(a: Matrix, b: Matrix) raises -> Float64:
    var sa = Int(a.strides[0])
    var sb = Int(b.strides[0])
    var total: Float64 = 0
    for i in range(a.shape[0]):
        total += a.unsafe_get(i * sa) * b.unsafe_get(i * sb)
    return total


def _matvec(x: Matrix, v: Matrix) raises -> Matrix:
    """x @ v for x (n,p) and v (p,)."""
    var rows = x.shape[0]
    var cols = x.shape[1]
    var s0 = Int(x.strides[0])
    var s1 = Int(x.strides[1])
    var sv = Int(v.strides[0])
    var out = _vector(rows)
    for i in range(rows):
        var total: Float64 = 0
        for j in range(cols):
            total += x.unsafe_get(i * s0 + j * s1) * v.unsafe_get(j * sv)
        out.unsafe_set(i, total)
    return out^


def _rmatvec(x: Matrix, v: Matrix) raises -> Matrix:
    """x.T @ v for x (n,p) and v (n,), accumulated row by row like BLAS."""
    var rows = x.shape[0]
    var cols = x.shape[1]
    var s0 = Int(x.strides[0])
    var s1 = Int(x.strides[1])
    var sv = Int(v.strides[0])
    var out = _vector(cols)
    for i in range(rows):
        var vi = v.unsafe_get(i * sv)
        for j in range(cols):
            out.unsafe_set(j, out.unsafe_get(j) + x.unsafe_get(i * s0 + j * s1) * vi)
    return out^


def _scale_vector(mut v: Matrix, factor: Float64) raises:
    for i in range(v.shape[0]):
        _vput(v, i, _vat(v, i) * factor)


def _divide_vector(mut v: Matrix, divisor: Float64) raises:
    for i in range(v.shape[0]):
        _vput(v, i, _vat(v, i) / divisor)


def compact_float64[
    dtype: DType
](x: NDArray[dtype]) raises -> Matrix:
    """Independent C-contiguous Float64 copy of a rank-1 or rank-2 array."""
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if x.ndim == 1:
        var out = _vector(x.shape[0])
        for i in range(x.shape[0]):
            out.unsafe_set(i, Float64(x.unsafe_get(i * Int(x.strides[0]))))
        return out^
    if x.ndim != 2:
        raise Error("ValueError: expected a rank-1 or rank-2 array")
    var rows = x.shape[0]
    var cols = x.shape[1]
    var out = _matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            out.unsafe_set(
                i * cols + j,
                Float64(
                    x.unsafe_get(
                        i * Int(x.strides[0]) + j * Int(x.strides[1])
                    )
                ),
            )
    return out^


def _cross_covariance(x: Matrix, y: Matrix) raises -> Matrix:
    """x.T @ y via NuMojo's approved rank-2 product."""
    return x.T() @ y


def _svd_factor(a: Matrix, context: String) raises -> SVDFactor[DType.float64]:
    try:
        return svd(a)
    except e:
        raise Error(
            "LinAlgError: SVD failed in "
            + context
            + " for a matrix of shape ("
            + String(a.shape[0])
            + ", "
            + String(a.shape[1])
            + "): "
            + String(e)
        )


def pinv(a: Matrix, rtol: Float64, context: String) raises -> Matrix:
    """Pseudoinverse keeping singular values strictly above rtol * max(s).

    Returns `V[:, :r] diag(1/s[:r]) U[:, :r].T` from mojo-linalg's economy
    SVD, with atol = 0 (the same retained spectrum as `SVDFactor.solve`).
    The two scikit-learn cutoffs are `max(shape) * eps` (SciPy `pinv`) and
    `1e6 * eps` (`_pinv2_old`).
    """
    var factor = _svd_factor(a, context)
    var u = factor.u()
    var singular = factor.singular_values()
    var vt = factor.vt()
    var rows = a.shape[0]
    var cols = a.shape[1]
    var k = singular.shape[0]
    var threshold = rtol * (singular.unsafe_get(0) if k else 0.0)
    var rank = 0
    for r in range(k):
        if singular.unsafe_get(r) > threshold:
            rank += 1
    # scaled[r, i] = U[i, r] / s[r]; out = Vt[:rank].T @ scaled
    var scaled = _matrix(rank, rows)
    for r in range(rank):
        var inv = 1.0 / singular.unsafe_get(r)
        for i in range(rows):
            scaled.unsafe_set(r * rows + i, _at(u, i, r) * inv)
    var out = _matrix(cols, rows)
    for j in range(cols):
        for r in range(rank):
            var v = _at(vt, r, j)
            for i in range(rows):
                out.unsafe_set(j * rows + i, out.unsafe_get(j * rows + i) + v * scaled.unsafe_get(r * rows + i))
    return out^


def svd_flip_1d(mut u: Matrix, mut v: Matrix) raises:
    """In-place sign flip making the first largest-|u| entry positive.

    Matches `_svd_flip_1d` (argmax returns the first maximum; sign(0)=0).
    """
    var best = 0
    var best_abs: Float64 = -1
    for i in range(u.shape[0]):
        var value = abs(_vat(u, i))
        if value > best_abs:
            best_abs = value
            best = i
    var pivot = _vat(u, best)
    var sign: Float64 = 1.0 if pivot > 0 else (-1.0 if pivot < 0 else 0.0)
    _scale_vector(u, sign)
    _scale_vector(v, sign)


def validate_options(options: KernelOptions) raises:
    """Validate private engine controls, naming the offending value."""
    try:
        options.validate()
    except e:
        raise Error(
            String(e)
            + " (mode="
            + options.mode
            + ", deflation_mode="
            + options.deflation_mode
            + ", algorithm="
            + options.algorithm
            + ", max_iter="
            + String(options.max_iter)
            + ", tol="
            + String(options.tol)
            + ")"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@fieldwise_init
struct PLSEngine(Copyable, ImplicitlyCopyable, Movable, TwoBlockEngine):
    """Stateless implementation of the family TwoBlockEngine contract."""

    def center_scale(
        self, mut x: Matrix, mut y: Matrix, scale: Bool
    ) raises -> Scaling:
        """Center columns in place and, if scale, divide by the ddof=1 std.

        Zero standard deviations become one. scale=False returns unit scales.
        Follows NumPy's `mean` and `std(ddof=1)` formulas (the std is taken of
        the centered data, including its residual mean).
        """
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise Error("ValueError: center_scale expects paired rank-2 blocks")
        if x.shape[0] < 2:
            raise Error("ValueError: center_scale needs at least two samples")
        var x_mean = _center_columns(x)
        var y_mean = _center_columns(y)
        var x_std: Matrix
        var y_std: Matrix
        if scale:
            x_std = _scale_columns(x)
            y_std = _scale_columns(y)
        else:
            x_std = nm.ones[DType.float64](Shape(x.shape[1]))
            y_std = nm.ones[DType.float64](Shape(y.shape[1]))
        return Scaling(x_mean^, y_mean^, x_std^, y_std^)

    def leading_weights(
        self, x: Matrix, y: Matrix, options: KernelOptions
    ) raises -> WeightStep:
        """First left/right singular vectors of x.T @ y, sign-flipped.

        'nipals' runs `_get_first_singular_vectors_power_method` (mode A or
        B; y weights are normalized for canonical deflation). 'svd' uses the
        full SVD of the cross-covariance and reports n_iter=0.
        """
        validate_options(options)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise Error("ValueError: leading_weights expects paired rank-2 blocks")
        if options.algorithm == "svd":
            var factor = _svd_factor(
                _cross_covariance(x, y), "PLS leading singular vectors"
            )
            var u = factor.u()
            var vt = factor.vt()
            var x_weights = _vector(u.shape[0])
            var y_weights = _vector(vt.shape[1])
            for i in range(u.shape[0]):
                _vput(x_weights, i, _at(u, i, 0))
            for j in range(vt.shape[1]):
                _vput(y_weights, j, _at(vt, 0, j))
            svd_flip_1d(x_weights, y_weights)
            return WeightStep(x_weights^, y_weights^, 0, True, False)
        var step = _power_method(
            x,
            y,
            options.mode,
            options.max_iter,
            options.tol,
            options.deflation_mode == "canonical",
        )
        if not step.constant_target:
            svd_flip_1d(step.x_weights, step.y_weights)
        return step^

    def deflate(
        self,
        mut x: Matrix,
        mut y: Matrix,
        scores: PairedScores,
        canonical: Bool,
    ) raises -> Loadings:
        """Subtract rank-one approximations and return the loadings.

        x loading is x.T @ t / (t.T @ t); canonical y regresses on its own
        score u, regression y on t. Zero denominators follow IEEE arithmetic
        (NaN/inf), as NumPy does.
        """
        var n = x.shape[0]
        if (
            scores.x_scores.shape[0] != n
            or scores.y_scores.shape[0] != n
            or y.shape[0] != n
        ):
            raise Error("ValueError: deflate score rows differ from block rows")
        var t = scores.x_scores.reshape(Shape(n)) if scores.x_scores.ndim == 2 else scores.x_scores.copy()
        var t_ss = _dot(t, t)
        var x_loadings = _rmatvec(x, t)
        _divide_vector(x_loadings, t_ss)
        _subtract_outer(x, t, x_loadings)
        var y_loadings: Matrix
        if canonical:
            var u = scores.y_scores.reshape(Shape(n)) if scores.y_scores.ndim == 2 else scores.y_scores.copy()
            var u_ss = _dot(u, u)
            y_loadings = _rmatvec(y, u)
            _divide_vector(y_loadings, u_ss)
            _subtract_outer(y, u, y_loadings)
        else:
            y_loadings = _rmatvec(y, t)
            _divide_vector(y_loadings, t_ss)
            _subtract_outer(y, t, y_loadings)
        return Loadings(x_loadings^, y_loadings^)

    def projection_weights(
        self, x: Matrix, y: Matrix, n_components: Int
    ) raises -> ProjectionWeights:
        """Return the k leading singular vectors of x.T @ y with u-based `svd_flip`."""
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise Error("ValueError: projection_weights expects paired rank-2 blocks")
        var p = x.shape[1]
        var q = y.shape[1]
        if n_components < 1 or n_components > min(p, q):
            raise Error(
                "ValueError: projection_weights n_components="
                + String(n_components)
                + " outside [1, "
                + String(min(p, q))
                + "]"
            )
        var factor = _svd_factor(_cross_covariance(x, y), "PLSSVD")
        var u = factor.u()
        var vt = factor.vt()
        var x_weights = _matrix(p, n_components)
        var y_weights = _matrix(q, n_components)
        for c in range(n_components):
            var best = 0
            var best_abs: Float64 = -1
            for i in range(p):
                var value = abs(_at(u, i, c))
                if value > best_abs:
                    best_abs = value
                    best = i
            var pivot = _at(u, best, c)
            var sign: Float64 = 1.0 if pivot > 0 else (
                -1.0 if pivot < 0 else 0.0
            )
            for i in range(p):
                _put(x_weights, i, c, _at(u, i, c) * sign)
            for j in range(q):
                _put(y_weights, j, c, _at(vt, c, j) * sign)
        return ProjectionWeights(x_weights^, y_weights^)

    def rotations(self, weights: Matrix, loadings: Matrix) raises -> Matrix:
        """Return weights @ pinv(loadings.T @ weights), SciPy's default pinv cutoff.
        """
        if weights.ndim != 2 or loadings.ndim != 2 or weights.shape[0] != loadings.shape[0] or weights.shape[1] != loadings.shape[1]:
            raise Error("ValueError: rotations expects equal (d,k) weights and loadings")
        var k = weights.shape[1]
        var inner = loadings.T() @ weights
        var inverse = pinv(inner, Float64(k) * EPS64, "PLS rotations")
        return weights @ inverse


def _center_columns(mut x: Matrix) raises -> Matrix:
    var n = x.shape[0]
    var mean = _vector(x.shape[1])
    for j in range(x.shape[1]):
        var total: Float64 = 0
        for i in range(n):
            total += _at(x, i, j)
        var m = total / Float64(n)
        _vput(mean, j, m)
        for i in range(n):
            _put(x, i, j, _at(x, i, j) - m)
    return mean^


def _scale_columns(mut x: Matrix) raises -> Matrix:
    var n = x.shape[0]
    var std = _vector(x.shape[1])
    for j in range(x.shape[1]):
        var total: Float64 = 0
        for i in range(n):
            total += _at(x, i, j)
        var residual_mean = total / Float64(n)
        var squares: Float64 = 0
        for i in range(n):
            var d = _at(x, i, j) - residual_mean
            squares += d * d
        var s = sqrt(squares / Float64(n - 1))
        if s == 0.0:
            s = 1.0
        _vput(std, j, s)
        for i in range(n):
            _put(x, i, j, _at(x, i, j) / s)
    return std^


def _subtract_outer(mut x: Matrix, left: Matrix, right: Matrix) raises:
    var s0 = Int(x.strides[0])
    var s1 = Int(x.strides[1])
    var sl = Int(left.strides[0])
    var sr = Int(right.strides[0])
    for i in range(x.shape[0]):
        var li = left.unsafe_get(i * sl)
        for j in range(x.shape[1]):
            var k = i * s0 + j * s1
            x.unsafe_set(k, x.unsafe_get(k) - li * right.unsafe_get(j * sr))


def _power_method(
    x: Matrix,
    y: Matrix,
    mode: String,
    max_iter: Int,
    tol: Float64,
    norm_y_weights: Bool,
) raises -> WeightStep:
    """`_get_first_singular_vectors_power_method` without the sign flip."""
    var n = x.shape[0]
    var p = x.shape[1]
    var q = y.shape[1]
    var y_score = _vector(n)
    var found = False
    for j in range(q):
        var nonconstant = False
        for i in range(n):
            if abs(_at(y, i, j)) > EPS64:
                nonconstant = True
                break
        if nonconstant:
            for i in range(n):
                _vput(y_score, i, _at(y, i, j))
            found = True
            break
    if not found:
        return WeightStep(_vector(p), _vector(q), 0, True, True)
    var x_pinv = _matrix(0, 0)
    var y_pinv = _matrix(0, 0)
    if mode == "B":
        # X_pinv = pinv(X) with the legacy pinv2 cutoff max(s) * 1e6 * eps.
        x_pinv = pinv(x, 1e6 * EPS64, "CCA X pseudoinverse")
        y_pinv = pinv(y, 1e6 * EPS64, "CCA y pseudoinverse")
    var x_weights = _vector(p)
    var y_weights = _vector(q)
    var x_weights_old = _vector(p)
    for i in range(p):
        _vput(x_weights_old, i, 100.0)
    var iteration = 0
    while iteration < max_iter:
        if mode == "B":
            x_weights = _matvec(x_pinv, y_score)
        else:
            x_weights = _rmatvec(x, y_score)
            _divide_vector(x_weights, _dot(y_score, y_score))
        _divide_vector(x_weights, sqrt(_dot(x_weights, x_weights)) + EPS64)
        var x_score = _matvec(x, x_weights)
        if mode == "B":
            y_weights = _matvec(y_pinv, x_score)
        else:
            y_weights = _rmatvec(y, x_score)
            _divide_vector(y_weights, _dot(x_score, x_score))
        if norm_y_weights:
            _divide_vector(y_weights, sqrt(_dot(y_weights, y_weights)) + EPS64)
        y_score = _matvec(y, y_weights)
        _divide_vector(y_score, _dot(y_weights, y_weights) + EPS64)
        var diff: Float64 = 0
        for i in range(p):
            var d = _vat(x_weights, i) - _vat(x_weights_old, i)
            diff += d * d
        if diff < tol or q == 1:
            break
        x_weights_old = x_weights.copy()
        iteration += 1
    var n_iter = min(iteration + 1, max_iter)
    return WeightStep(x_weights^, y_weights^, n_iter, n_iter != max_iter, False)


# ---------------------------------------------------------------------------
# Input validation with scikit-learn-shaped messages
# ---------------------------------------------------------------------------


def _shape_text[dtype: DType](x: NDArray[dtype]) raises -> String:
    if x.ndim == 1:
        return "(" + String(x.shape[0]) + ",)"
    var text = String("(")
    for d in range(x.ndim):
        if d:
            text += ", "
        text += String(x.shape[d])
    return text + ")"


def check_finite[dtype: DType](x: NDArray[dtype], name: String) raises:
    """Reject NaN/infinity like `check_array(ensure_all_finite=True)`."""
    var rows = x.shape[0]
    var cols = x.shape[1] if x.ndim == 2 else 1
    var s0 = Int(x.strides[0])
    var s1 = Int(x.strides[1]) if x.ndim == 2 else 0
    var has_inf = False
    for i in range(rows):
        for j in range(cols):
            var value = x.unsafe_get(i * s0 + j * s1)
            if isnan(value):
                raise Error("ValueError: Input " + name + " contains NaN.")
            if not isfinite(value):
                has_inf = True
    if has_inf:
        raise Error(
            "ValueError: Input "
            + name
            + " contains infinity or a value too large for dtype('"
            + ("float32" if dtype == DType.float32 else "float64")
            + "')."
        )


def check_matrix[
    dtype: DType
](x: NDArray[dtype], name: String, estimator: String, min_samples: Int) raises:
    """`check_array` for a required 2-D input with sample/feature minima."""
    if x.ndim != 2:
        raise Error(
            "ValueError: Expected 2D array, got "
            + String(x.ndim)
            + "D array instead for "
            + name
            + " with shape "
            + _shape_text(x)
            + "."
        )
    if x.shape[0] < min_samples:
        raise Error(
            "ValueError: Found array with "
            + String(x.shape[0])
            + " sample(s) (shape="
            + _shape_text(x)
            + ") while a minimum of "
            + String(min_samples)
            + " is required by "
            + estimator
            + "."
        )
    if x.shape[1] < 1:
        raise Error(
            "ValueError: Found array with 0 feature(s) (shape="
            + _shape_text(x)
            + ") while a minimum of 1 is required by "
            + estimator
            + "."
        )
    check_finite(x, name)


def check_target[dtype: DType](y: NDArray[dtype], estimator: String) raises:
    """`check_array(y, ensure_2d=False)`: rank 1 or 2, finite, nonempty."""
    if y.ndim != 1 and y.ndim != 2:
        raise Error(
            "ValueError: Found array with dim "
            + String(y.ndim)
            + ", while dim <= 2 is required by "
            + estimator
            + "."
        )
    if y.shape[0] < 1:
        raise Error(
            "ValueError: Found array with 0 sample(s) (shape="
            + _shape_text(y)
            + ") while a minimum of 1 is required by "
            + estimator
            + "."
        )
    if y.ndim == 2 and y.shape[1] < 1:
        raise Error(
            "ValueError: Found array with 0 feature(s) (shape="
            + _shape_text(y)
            + ") while a minimum of 1 is required by "
            + estimator
            + "."
        )
    check_finite(y, "y")


def check_consistent_length(n_x: Int, n_y: Int) raises:
    if n_x != n_y:
        raise Error(
            "ValueError: Found input variables with inconsistent numbers of"
            " samples: ["
            + String(n_x)
            + ", "
            + String(n_y)
            + "]"
        )


def as_target_matrix(y: Matrix) raises -> Matrix:
    """Float64 target as (n,q): rank-1 (n,) becomes an independent (n,1)."""
    if y.ndim == 1:
        return y.reshape(Shape(y.shape[0], 1))
    return y.copy()


def not_fitted_error(estimator: String) -> Error:
    return Error(
        "NotFittedError: This "
        + estimator
        + " instance is not fitted yet. Call 'fit' with appropriate arguments"
        " before using this estimator."
    )


def check_n_features(n_features: Int, expected: Int, estimator: String) raises:
    if n_features != expected:
        raise Error(
            "ValueError: X has "
            + String(n_features)
            + " features, but "
            + estimator
            + " is expecting "
            + String(expected)
            + " features as input."
        )


def check_n_components_bound(n_components: Int, bound: Int) raises:
    if n_components > bound:
        raise Error(
            "ValueError: `n_components` upper bound is "
            + String(bound)
            + ". Got "
            + String(n_components)
            + " instead. Reduce `n_components`."
        )


def check_positive_int(value: Int, name: String, estimator: String) raises:
    if value < 1:
        raise Error(
            "InvalidParameterError: The '"
            + name
            + "' parameter of "
            + estimator
            + " must be an int in the range [1, inf). Got "
            + String(value)
            + " instead."
        )


def normalized[
    dtype: DType
](x: NDArray[dtype], mean: Matrix, std: Matrix, divide: Bool) raises -> NDArray[dtype]:
    """Return (x - mean) [/ std] with input-dtype rounding, like NumPy's
    in-place `X -= mean; X /= std` on a Float32/Float64 copy."""
    var out = nm.zeros[dtype](Shape(x.shape[0], x.shape[1]))
    var cols = x.shape[1]
    for i in range(x.shape[0]):
        for j in range(cols):
            var value = x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]))
            var centered = Scalar[dtype](Float64(value) - _vat(mean, j))
            if divide:
                centered = Scalar[dtype](Float64(centered) / _vat(std, j))
            out.unsafe_set(i * cols + j, centered)
    return out^


def write_back[dtype: DType](mut target: NDArray[dtype], source: NDArray[dtype]) raises:
    """Copy source values into target's (possibly strided) logical layout."""
    if target.ndim == 1:
        for i in range(target.shape[0]):
            target.unsafe_set(
                i * Int(target.strides[0]),
                source.unsafe_get(i * Int(source.strides[0])),
            )
        return
    var cols = target.shape[1]
    var s_cols = source.shape[1] if source.ndim == 2 else 1
    for i in range(target.shape[0]):
        for j in range(cols):
            target.unsafe_set(
                i * Int(target.strides[0]) + j * Int(target.strides[1]),
                source.unsafe_get(i * Int(source.strides[0]) + (j * Int(source.strides[1]) if source.ndim == 2 else 0)),
            )
    _ = s_cols


def write_back_float64[dtype: DType](mut target: NDArray[dtype], source: Matrix) raises:
    """Store Float64 workspace values into a Float64 caller array."""
    comptime if dtype == DType.float64:
        write_back(target, rebind[NDArray[dtype]](source.copy()))


def r2_score(
    y_true: Matrix, y_pred: Matrix, sample_weight: Optional[Matrix]
) raises -> Float64:
    """Uniform-average R^2 with force_finite=True, as `RegressorMixin.score`.

    Inputs are (n,q) Float64 matrices. Fewer than two samples returns NaN
    (scikit-learn also emits UndefinedMetricWarning).
    """
    var n = y_true.shape[0]
    var q = y_true.shape[1]
    if y_pred.shape[0] != n or y_pred.shape[1] != q:
        raise Error(
            "ValueError: y_true and y_pred have different number of output ("
            + String(q)
            + "!="
            + String(y_pred.shape[1])
            + ")"
        )
    var weights = nm.ones[DType.float64](Shape(n))
    if sample_weight:
        var w = sample_weight.value().copy()
        if w.ndim != 1 or w.shape[0] != n:
            raise Error(
                "ValueError: sample_weight.shape == "
                + _shape_text(w)
                + ", expected ("
                + String(n)
                + ",)!"
            )
        check_finite(w, "sample_weight")
        weights = w^
    if n < 2:
        var zero: Float64 = 0
        return zero / zero
    var weight_sum: Float64 = 0
    for i in range(n):
        weight_sum += _vat(weights, i)
    if weight_sum == 0:
        raise Error(
            "ZeroDivisionError: Weights sum to zero, can't be normalized"
        )
    var total: Float64 = 0
    for t in range(q):
        var mean: Float64 = 0
        for i in range(n):
            mean += _vat(weights, i) * _at(y_true, i, t)
        mean /= weight_sum
        var numerator: Float64 = 0
        var denominator: Float64 = 0
        for i in range(n):
            var w = _vat(weights, i)
            var r = _at(y_true, i, t) - _at(y_pred, i, t)
            var d = _at(y_true, i, t) - mean
            numerator += w * r * r
            denominator += w * d * d
        var score: Float64 = 1.0
        if numerator != 0:
            score = 1.0 - numerator / denominator if denominator != 0 else 0.0
        total += score
    return total / Float64(q)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

comptime _REGRESSION = 0
comptime _CANONICAL = 1
comptime _CCA = 2


@fieldwise_init
struct PLSParams(Copyable, Movable, Writable):
    """Constructor configuration shared by PLSRegression, PLSCanonical, CCA.

    `algorithm` is a real parameter only for PLSCanonical; the other two
    estimators accept only its fixed value 'nipals'.
    """

    var n_components: Int
    var scale: Bool
    var algorithm: String
    var max_iter: Int
    var tol: Float64
    var copy_input: Bool


@fieldwise_init
struct PLSFit(Copyable, Movable):
    """Learned state of one successful `_PLS.fit` (all Float64)."""

    var x_weights: Matrix
    var y_weights: Matrix
    var x_loadings: Matrix
    var y_loadings: Matrix
    var x_scores: Matrix
    var y_scores: Matrix
    var x_rotations: Matrix
    var y_rotations: Matrix
    var coef: Matrix
    var scaling: Scaling
    var n_iter: List[Int]
    var diagnostics: List[String]


def fit_two_block(
    mut xk: Matrix,
    mut yk: Matrix,
    n_components: Int,
    scale: Bool,
    options: KernelOptions,
) raises -> PLSFit:
    """The `_PLS.fit` numerical core on validated Float64 (n,p)/(n,q) workspaces.

    Mutates xk/yk into the final deflated residuals (scikit-learn's copy=False
    end state). Warnings become `diagnostics` entries.
    """
    validate_options(options)
    var engine = PLSEngine()
    var n = xk.shape[0]
    var p = xk.shape[1]
    var q = yk.shape[1]
    var canonical = options.deflation_mode == "canonical"
    var scaling = engine.center_scale(xk, yk, scale)
    var x_weights = _matrix(p, n_components)
    var y_weights = _matrix(q, n_components)
    var x_scores = _matrix(n, n_components)
    var y_scores = _matrix(n, n_components)
    var x_loadings = _matrix(p, n_components)
    var y_loadings = _matrix(q, n_components)
    var n_iter = List[Int]()
    var diagnostics = List[String]()
    for k in range(n_components):
        var step: WeightStep
        if options.algorithm == "nipals":
            # Replace columns that are all close to zero with zeros.
            for j in range(q):
                var tiny = True
                for i in range(n):
                    if not (abs(_at(yk, i, j)) < 10 * EPS64):
                        tiny = False
                        break
                if tiny:
                    for i in range(n):
                        _put(yk, i, j, 0.0)
            step = engine.leading_weights(xk, yk, options)
            if step.constant_target:
                diagnostics.append(
                    "UserWarning: y residual is constant at iteration "
                    + String(k)
                )
                break
            n_iter.append(step.n_iter)
            if not step.converged:
                diagnostics.append(
                    "ConvergenceWarning: Maximum number of iterations reached"
                )
        else:
            step = engine.leading_weights(xk, yk, options)
        var t = _matvec(xk, step.x_weights)
        var y_ss: Float64 = 1.0 if canonical else _dot(
            step.y_weights, step.y_weights
        )
        var u = _matvec(yk, step.y_weights)
        _divide_vector(u, y_ss)
        var loadings = engine.deflate(
            xk,
            yk,
            PairedScores(t.reshape(Shape(n, 1)), u.reshape(Shape(n, 1))),
            canonical,
        )
        for i in range(p):
            _put(x_weights, i, k, _vat(step.x_weights, i))
            _put(x_loadings, i, k, _vat(loadings.x_loadings, i))
        for j in range(q):
            _put(y_weights, j, k, _vat(step.y_weights, j))
            _put(y_loadings, j, k, _vat(loadings.y_loadings, j))
        for i in range(n):
            _put(x_scores, i, k, _vat(t, i))
            _put(y_scores, i, k, _vat(u, i))
    var x_rotations = engine.rotations(x_weights, x_loadings)
    var y_rotations = engine.rotations(y_weights, y_loadings)
    # coef_ = (x_rotations @ y_loadings.T * y_std).T / x_std, shape (q, p).
    var product = x_rotations @ y_loadings.T()
    var coef = _matrix(q, p)
    for t in range(q):
        for j in range(p):
            _put(
                coef,
                t,
                j,
                (_at(product, j, t) * _vat(scaling.y_std, t))
                / _vat(scaling.x_std, j),
            )
    return PLSFit(
        x_weights^,
        y_weights^,
        x_loadings^,
        y_loadings^,
        x_scores^,
        y_scores^,
        x_rotations^,
        y_rotations^,
        coef^,
        scaling^,
        n_iter^,
        diagnostics^,
    )


struct _PLS[flavor: Int, dtype: DType = DType.float64](
    Copyable, CrossDecompositionTransformer, Movable, PLSPredictor
):
    """Shared implementation of scikit-learn's private `_PLS` base class.

    Use the public aliases `PLSRegression`, `PLSCanonical` and `CCA`.
    Inputs are NuMojo arrays of `dtype` (Float32 or Float64); fitting and all
    outputs are Float64, as in scikit-learn. See docs/api.md for the Mojo
    adaptations (copy=False overloads, diagnostics_, feature names).
    """

    comptime Params = PLSParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = Matrix
    comptime PairedTransform = PairedScores
    comptime Prediction = Matrix
    comptime SampleWeight = Matrix

    var n_components: Int
    var scale: Bool
    var algorithm: String
    var max_iter: Int
    var tol: Float64
    var copy_input: Bool

    var x_weights_: Matrix
    """(n_features, n_components) left singular vectors per component."""
    var y_weights_: Matrix
    """(n_targets, n_components) right singular vectors per component."""
    var x_loadings_: Matrix
    var y_loadings_: Matrix
    var x_scores_: Matrix
    """Training X scores (sklearn: public on PLSRegression, `_x_scores` otherwise)."""
    var y_scores_: Matrix
    var x_rotations_: Matrix
    var y_rotations_: Matrix
    var coef_: Matrix
    """(n_targets, n_features); y ~ X @ coef_.T + intercept_."""
    var intercept_: Matrix
    var n_iter_: List[Int]
    """Power-method iterations per extracted component; empty for 'svd'."""
    var n_features_in_: Int
    var diagnostics_: List[String]
    """Native replacement for Python warnings emitted by the last fit."""
    var _x_mean: Matrix
    var _y_mean: Matrix
    var _x_std: Matrix
    var _y_std: Matrix
    var _predict_1d: Bool
    var _fitted: Bool

    def __init__(
        out self,
        n_components: Int = 2,
        *,
        scale: Bool = True,
        max_iter: Int = 500,
        tol: Float64 = 1e-06,
        copy: Bool = True,
    ) raises where Self.flavor != _CANONICAL:
        """PLSRegression / CCA constructor (scikit-learn defaults)."""
        self = Self(
            params=PLSParams(n_components, scale, "nipals", max_iter, tol, copy)
        )

    def __init__(
        out self,
        n_components: Int = 2,
        *,
        scale: Bool = True,
        algorithm: String = "nipals",
        max_iter: Int = 500,
        tol: Float64 = 1e-06,
        copy: Bool = True,
    ) raises where Self.flavor == _CANONICAL:
        """PLSCanonical constructor; algorithm is 'nipals' or 'svd'."""
        self = Self(
            params=PLSParams(n_components, scale, algorithm, max_iter, tol, copy)
        )

    def __init__(out self, *, params: PLSParams) raises:
        """Construct from a typed parameter record (used by clone)."""
        comptime assert (
            Self.dtype == DType.float32 or Self.dtype == DType.float64
        ), "PLS estimators accept Float32 or Float64 inputs"
        self.n_components = params.n_components
        self.scale = params.scale
        self.algorithm = params.algorithm
        self.max_iter = params.max_iter
        self.tol = params.tol
        self.copy_input = params.copy_input
        self.x_weights_ = _matrix(0, 0)
        self.y_weights_ = _matrix(0, 0)
        self.x_loadings_ = _matrix(0, 0)
        self.y_loadings_ = _matrix(0, 0)
        self.x_scores_ = _matrix(0, 0)
        self.y_scores_ = _matrix(0, 0)
        self.x_rotations_ = _matrix(0, 0)
        self.y_rotations_ = _matrix(0, 0)
        self.coef_ = _matrix(0, 0)
        self.intercept_ = _vector(0)
        self.n_iter_ = List[Int]()
        self.n_features_in_ = 0
        self.diagnostics_ = List[String]()
        self._x_mean = _vector(0)
        self._y_mean = _vector(0)
        self._x_std = _vector(0)
        self._y_std = _vector(0)
        self._predict_1d = False
        self._fitted = False
        self._check_algorithm_parameter(params.algorithm)

    @staticmethod
    def name() -> String:
        """scikit-learn class name, used in messages and feature names."""
        comptime if Self.flavor == _REGRESSION:
            return "PLSRegression"
        elif Self.flavor == _CANONICAL:
            return "PLSCanonical"
        else:
            return "CCA"

    @staticmethod
    def _deflation_mode() -> String:
        comptime if Self.flavor == _REGRESSION:
            return "regression"
        else:
            return "canonical"

    @staticmethod
    def _mode() -> String:
        comptime if Self.flavor == _CCA:
            return "B"
        else:
            return "A"

    def _check_algorithm_parameter(self, algorithm: String) raises:
        comptime if Self.flavor != _CANONICAL:
            if algorithm != "nipals":
                raise Error(
                    "ValueError: Invalid parameter 'algorithm' for estimator "
                    + Self.name()
                    + "; only PLSCanonical accepts it."
                )

    # -- parameters and cloning ------------------------------------------

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return PLSParams(
            self.n_components,
            self.scale,
            self.algorithm,
            self.max_iter,
            self.tol,
            self.copy_input,
        )

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        self._check_algorithm_parameter(params.algorithm)
        self.n_components = params.n_components
        self.scale = params.scale
        self.algorithm = params.algorithm
        self.max_iter = params.max_iter
        self.tol = params.tol
        self.copy_input = params.copy_input
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self,
        *,
        n_components: Optional[Int] = None,
        scale: Optional[Bool] = None,
        algorithm: Optional[String] = None,
        max_iter: Optional[Int] = None,
        tol: Optional[Float64] = None,
        copy: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        """Update only the supplied parameters (validated at fit, as sklearn)."""
        var params = self.get_params()
        if n_components:
            params.n_components = n_components.value()
        if scale:
            params.scale = scale.value()
        if algorithm:
            params.algorithm = algorithm.value()
        if max_iter:
            params.max_iter = max_iter.value()
        if tol:
            params.tol = tol.value()
        if copy:
            params.copy_input = copy.value()
        _ = self.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(params=self.get_params())

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def _validate_params(self) raises:
        check_positive_int(self.n_components, "n_components", Self.name())
        check_positive_int(self.max_iter, "max_iter", Self.name())
        if isnan(self.tol) or self.tol < 0:
            raise Error(
                "InvalidParameterError: The 'tol' parameter of "
                + Self.name()
                + " must be a float in the range [0.0, inf). Got "
                + String(self.tol)
                + " instead."
            )
        if self.algorithm != "nipals" and self.algorithm != "svd":
            raise Error(
                "InvalidParameterError: The 'algorithm' parameter of "
                + Self.name()
                + " must be a str among {'nipals', 'svd'}. Got '"
                + self.algorithm
                + "' instead."
            )
        self._check_algorithm_parameter(self.algorithm)

    def _options(self) -> KernelOptions:
        return KernelOptions(
            mode=Self._mode(),
            deflation_mode=Self._deflation_mode(),
            algorithm=self.algorithm,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    # -- fit ---------------------------------------------------------------

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target
    ) raises -> ref[self] Self:
        """Fit on copies of X and y; the caller's arrays are never modified.

        y may be (n,) or (n,q). For copy=False mutation use `fit_inplace`.
        """
        self._validate_fit_inputs(X, y)
        var xk = compact_float64(X)
        var yk = as_target_matrix(compact_float64(y))
        self._fit_workspace(xk, yk, y.ndim == 1)
        return self

    def fit_inplace[origin: MutOrigin](
        ref[origin] self, mut X: Self.Input, mut y: Self.Target
    ) raises -> ref[self] Self:
        """scikit-learn `fit` honouring `copy`.

        With copy=False and Float64 inputs, X and y end holding the centered,
        scaled and deflated residuals, exactly like scikit-learn's in-place
        workspaces. Float32 inputs are converted, so they are never modified.
        """
        self._validate_fit_inputs(X, y)
        var xk = compact_float64(X)
        var yk = as_target_matrix(compact_float64(y))
        self._fit_workspace(xk, yk, y.ndim == 1)
        comptime if Self.dtype == DType.float64:
            if not self.copy_input:
                write_back_float64(X, xk)
                if y.ndim == 1:
                    write_back_float64(y, yk.reshape(Shape(yk.shape[0])))
                else:
                    write_back_float64(y, yk)
        return self

    def _validate_fit_inputs(self, X: Self.Input, y: Self.Target) raises:
        self._validate_params()
        check_consistent_length(X.shape[0], y.shape[0])
        check_matrix(X, "X", Self.name(), 2)
        check_target(y, Self.name())
        var n = X.shape[0]
        var p = X.shape[1]
        var q = 1 if y.ndim == 1 else y.shape[1]
        var bound = min(n, p) if Self.flavor == _REGRESSION else min(n, min(p, q))
        check_n_components_bound(self.n_components, bound)

    def _fit_workspace(mut self, mut xk: Matrix, mut yk: Matrix, was_1d: Bool) raises:
        var result = fit_two_block(
            xk, yk, self.n_components, self.scale, self._options()
        )
        # Commit only after the whole fit succeeded.
        self.x_weights_ = result.x_weights.copy()
        self.y_weights_ = result.y_weights.copy()
        self.x_loadings_ = result.x_loadings.copy()
        self.y_loadings_ = result.y_loadings.copy()
        self.x_scores_ = result.x_scores.copy()
        self.y_scores_ = result.y_scores.copy()
        self.x_rotations_ = result.x_rotations.copy()
        self.y_rotations_ = result.y_rotations.copy()
        self.coef_ = result.coef.copy()
        self.intercept_ = result.scaling.y_mean.copy()
        self._x_mean = result.scaling.x_mean.copy()
        self._y_mean = result.scaling.y_mean.copy()
        self._x_std = result.scaling.x_std.copy()
        self._y_std = result.scaling.y_std.copy()
        self.n_iter_ = result.n_iter.copy()
        self.diagnostics_ = result.diagnostics.copy()
        self.n_features_in_ = xk.shape[1]
        self._predict_1d = was_1d
        self._fitted = True

    # -- inference ---------------------------------------------------------

    def _check_inference_input(self, X: Self.Input) raises:
        if not self._fitted:
            raise not_fitted_error(Self.name())
        check_matrix(X, "X", Self.name(), 1)
        check_n_features(X.shape[1], self.n_features_in_, Self.name())

    def _normalized_y(self, y: Self.Target) raises -> NDArray[Self.dtype]:
        check_target(y, Self.name())
        var y2 = y.reshape(Shape(y.shape[0], 1)) if y.ndim == 1 else y.copy()
        if y2.shape[1] != self._y_mean.shape[0]:
            raise Error(
                "ValueError: y has "
                + String(y2.shape[1])
                + " targets, but "
                + Self.name()
                + " was fitted with "
                + String(self._y_mean.shape[0])
            )
        return normalized(y2, self._y_mean, self._y_std, True)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """x_scores = ((X - x_mean) / x_std) @ x_rotations_, shape (n, k)."""
        self._check_inference_input(X)
        var xr = normalized(X, self._x_mean, self._x_std, True)
        return compact_float64(xr) @ self.x_rotations_

    def transform(self, X: Self.Input, y: Self.Target) raises -> Self.PairedTransform:
        """Paired scores `(x_scores, y_scores)`; y may be (n,) or (n,q)."""
        var x_scores = self.transform(X)
        var yr = self._normalized_y(y)
        if yr.shape[0] != X.shape[0]:
            check_consistent_length(X.shape[0], yr.shape[0])
        return PairedScores(x_scores^, compact_float64(yr) @ self.y_rotations_)

    def transform_inplace(mut self, mut X: Self.Input) raises -> Self.Transformed:
        """transform(X, copy=False): X is normalized in place (input dtype)."""
        self._check_inference_input(X)
        var xr = normalized(X, self._x_mean, self._x_std, True)
        write_back(X, xr)
        return compact_float64(xr) @ self.x_rotations_

    def transform_inplace(
        mut self, mut X: Self.Input, mut y: Self.Target
    ) raises -> Self.PairedTransform:
        """transform(X, y, copy=False): X and y are normalized in place."""
        var x_scores = self.transform_inplace(X)
        var yr = self._normalized_y(y)
        write_back(y, yr)
        return PairedScores(x_scores^, compact_float64(yr) @ self.y_rotations_)

    def fit_transform(mut self, X: Self.Input, y: Self.Target) raises -> Self.PairedTransform:
        """fit(X, y) followed by the paired transform(X, y)."""
        _ = self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform[
        d: DType
    ](self, X: NDArray[d]) raises -> Matrix:
        """X_scores @ x_loadings_.T * x_std + x_mean (exact only when k = p)."""
        if not self._fitted:
            raise not_fitted_error(Self.name())
        return self._reconstruct(X, self.x_loadings_, self._x_std, self._x_mean, "X")

    def inverse_transform[
        d: DType
    ](self, X: NDArray[d], y: NDArray[d]) raises -> PairedScores:
        """Paired reconstruction; returned record holds (X_original, y_original)."""
        if not self._fitted:
            raise not_fitted_error(Self.name())
        var x_back = self._reconstruct(X, self.x_loadings_, self._x_std, self._x_mean, "X")
        var y_back = self._reconstruct(y, self.y_loadings_, self._y_std, self._y_mean, "y")
        return PairedScores(x_back^, y_back^)

    def _reconstruct[
        d: DType
    ](self, scores: NDArray[d], loadings: Matrix, std: Matrix, mean: Matrix, name: String) raises -> Matrix:
        comptime assert d == DType.float32 or d == DType.float64
        check_matrix(scores, name, Self.name(), 1)
        if scores.shape[1] != loadings.shape[1]:
            raise Error(
                "ValueError: "
                + name
                + " has "
                + String(scores.shape[1])
                + " columns, but "
                + Self.name()
                + " has "
                + String(loadings.shape[1])
                + " components."
            )
        var out = compact_float64(scores) @ loadings.T()
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                _put(out, i, j, _at(out, i, j) * _vat(std, j) + _vat(mean, j))
        return out^

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        """(X - x_mean) @ coef_.T + intercept_; rank-1 after a 1-D-target fit."""
        self._check_inference_input(X)
        var centered = normalized(X, self._x_mean, self._x_std, False)
        return self._predict_centered(centered)

    def predict_inplace(mut self, mut X: Self.Input) raises -> Self.Prediction:
        """predict(X, copy=False): X is centered in place (input dtype)."""
        self._check_inference_input(X)
        var centered = normalized(X, self._x_mean, self._x_std, False)
        write_back(X, centered)
        return self._predict_centered(centered)

    def _predict_centered(self, centered: NDArray[Self.dtype]) raises -> Matrix:
        var pred = compact_float64(centered) @ self.coef_.T()
        for i in range(pred.shape[0]):
            for t in range(pred.shape[1]):
                _put(pred, i, t, _at(pred, i, t) + _vat(self.intercept_, t))
        if self._predict_1d:
            return pred.reshape(Shape(pred.shape[0]))
        return pred^

    def score(
        self,
        X: Self.Input,
        y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        """Uniform-average R^2 of predict(X) against y (RegressorMixin.score).
        """
        check_target(y, Self.name())
        var pred = self.predict(X)
        var y_true = as_target_matrix(compact_float64(y))
        check_consistent_length(y_true.shape[0], pred.shape[0])
        var pred2 = as_target_matrix(pred)
        return r2_score(y_true, pred2, sample_weight)

    def get_feature_names_out(self) raises -> List[String]:
        """['<lowercase class name>0', ...] for each output component."""
        if not self._fitted:
            raise not_fitted_error(Self.name())
        var prefix = Self.name().lower()
        var names = List[String]()
        for k in range(self.x_rotations_.shape[1]):
            names.append(prefix + String(k))
        return names^


comptime PLSRegression[dtype: DType = DType.float64] = _PLS[_REGRESSION, dtype]
"""PLS regression (PLS1/PLS2): mode A, regression deflation, NIPALS."""

comptime PLSCanonical[dtype: DType = DType.float64] = _PLS[_CANONICAL, dtype]
"""Canonical PLS: mode A, canonical deflation, 'nipals' or 'svd'."""

comptime CCA[dtype: DType = DType.float64] = _PLS[_CCA, dtype]
"""Canonical correlation analysis: mode B, canonical deflation, NIPALS."""
