"""Dense FastICA (Hyvarinen et al., Independent Component Analysis, Tables 8.3/8.4).

Follows sklearn.decomposition._fastica (BSD-3-Clause; see LICENSES/scikit-learn.txt).
Whitening uses the shared SVD engine (whiten_solver="svd") or mojo-linalg's
symmetric eigen factor ("eigh"); symmetric decorrelation uses mojo-linalg
symmetric_eigh and the mixing matrix a mojo-linalg SVD pseudoinverse.
"""
from numojo import NDArray, Shape, zeros
from std.math import sqrt, tanh, abs, isfinite
from std.utils import Variant
from mojo_linalg import symmetric_eigh, svd
from moxie.base import UnsupervisedEstimator
from moxie._random import RandomState
from ._base import InvertibleDecomposition
from ._contracts import validate_matrix
from ._matrix import libm_exp as exp
from ._matrix import get2, set2, get1, set1, contiguous, identity, seeded_state, feature_names, machine_epsilon
from ._svd_engine import dense_svd

comptime WhitenOption = Variant[String, Bool]
"""sklearn `whiten`: "unit-variance", "arbitrary-variance" or False."""


@fieldwise_init
struct FastICAParams(Copyable, Movable):
    """Typed get_params/set_params record; field names follow sklearn."""
    var n_components: Optional[Int]
    var algorithm: String
    var whiten: WhitenOption
    var fun: String
    var fun_args: Optional[Dict[String, Float64]]
    var max_iter: Int
    var tol: Float64
    var w_init: Optional[NDArray[DType.float64]]
    var whiten_solver: String
    var random_state: Optional[Int]


@fieldwise_init
struct FastICAResult[dtype: DType](Movable):
    """Return record of `fastica`: sklearn's [K, W, S, X_mean, n_iter] list.

    K and X_mean are None when whiten=False; S is None when compute_sources=False,
    except under whiten="unit-variance", where sklearn returns the scaled sources.
    """
    var K: Optional[NDArray[Self.dtype]]
    var W: NDArray[Self.dtype]
    var S: Optional[NDArray[Self.dtype]]
    var X_mean: Optional[NDArray[Self.dtype]]
    var n_iter: Int


def _whitening_mode(whiten: WhitenOption) raises -> String:
    """Normalize `whiten` to "unit-variance", "arbitrary-variance" or "false"."""
    if whiten.isa[Bool]():
        if whiten[Bool]:
            raise Error("ValueError: FastICA whiten must be 'unit-variance', 'arbitrary-variance' or False, got True")
        return "false"
    var mode = whiten[String]
    if mode != "unit-variance" and mode != "arbitrary-variance":
        raise Error("ValueError: FastICA whiten must be 'unit-variance', 'arbitrary-variance' or False, got '" + mode + "'")
    return mode


def sym_decorrelation[dtype: DType](w: NDArray[dtype]) raises -> NDArray[dtype]:
    """W <- (W W^T)^{-1/2} W with eigenvalues clipped at finfo.tiny (basis invariant)."""
    var k = w.shape[0]
    var gram = w @ w.T()
    var factor = symmetric_eigh(gram)
    var values = factor.eigenvalues()
    var vectors = factor.eigenvectors()
    var tiny: Float64 = 1.1754943508222875e-38 if dtype == DType.float32 else 2.2250738585072014e-308
    var scaled = zeros[dtype](Shape(k, k))
    for j in range(k):
        var inv = 1.0 / sqrt(max(get1(values, j), tiny))
        for i in range(k):
            set2(scaled, i, j, get2(vectors, i, j) * inv)
    return (scaled @ vectors.T()) @ w


def _nonlinearity[dtype: DType](fun: String, alpha: Float64, wx: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """Return g(wx) (k, n) and the row means of g'(wx) (k,), like _logcosh/_exp/_cube."""
    var k = wx.shape[0]
    var n = wx.shape[1]
    var gx = zeros[dtype](Shape(k, n))
    var derivative = zeros[dtype](Shape(k))
    for r in range(k):
        var total: Float64 = 0
        for c in range(n):
            var x = get2(wx, r, c)
            var g: Float64
            var dg: Float64
            if fun == "logcosh":
                g = tanh(alpha * x)
                dg = alpha * (1 - g * g)
            elif fun == "exp":
                var e = exp(-(x * x) / 2)
                g = x * e
                dg = (1 - x * x) * e
            else:
                g = x * x * x
                dg = 3 * x * x
            set2(gx, r, c, g)
            total += dg
        set1(derivative, r, total / Float64(n))
    return (gx^, derivative^)


def _ica_par[dtype: DType](x: NDArray[dtype], tol: Float64, fun: String, alpha: Float64, max_iter: Int, w_init: NDArray[dtype]) raises -> Tuple[NDArray[dtype], Int]:
    """Parallel (symmetric) FastICA iterations on whitened x (k, n)."""
    var w = sym_decorrelation(w_init)
    var k = w.shape[0]
    var p = Float64(x.shape[1])
    var xt = x.T()
    var iterations = 0
    for ii in range(max_iter):
        iterations = ii + 1
        var nonlinear = _nonlinearity(fun, alpha, w @ x)
        var update = nonlinear[0] @ xt
        for r in range(k):
            for c in range(k):
                set2(update, r, c, get2(update, r, c) / p - get1(nonlinear[1], r) * get2(w, r, c))
        var w1 = sym_decorrelation(update)
        var limit: Float64 = 0
        for r in range(k):
            var dot: Float64 = 0
            for c in range(k):
                dot += get2(w1, r, c) * get2(w, r, c)
            limit = max(limit, abs(abs(dot) - 1))
        w = w1^
        if limit < tol:
            break
    # sklearn warns (ConvergenceWarning) when max_iter is exhausted.
    return (w^, iterations)


def _ica_def[dtype: DType](x: NDArray[dtype], tol: Float64, fun: String, alpha: Float64, max_iter: Int, w_init: NDArray[dtype]) raises -> Tuple[NDArray[dtype], Int]:
    """Deflationary FastICA with Gram-Schmidt decorrelation against found rows."""
    var k = w_init.shape[0]
    var m = x.shape[0]
    var n = x.shape[1]
    var w_all = zeros[dtype](Shape(k, k))
    var most = 0
    for j in range(k):
        var w = zeros[dtype](Shape(1, m))
        var norm: Float64 = 0
        for c in range(m):
            norm += get2(w_init, j, c) ** 2
        for c in range(m):
            set2(w, 0, c, get2(w_init, j, c) / sqrt(norm))
        var iterations = 0
        for i in range(max_iter):
            iterations = i + 1
            var nonlinear = _nonlinearity(fun, alpha, w @ x)
            var w1 = zeros[dtype](Shape(1, m))
            var projected = nonlinear[0] @ x.T()
            for c in range(m):
                set2(w1, 0, c, get2(projected, 0, c) / Float64(n) - get1(nonlinear[1], 0) * get2(w, 0, c))
            # w1 -= (w1 W[:j]^T) W[:j]
            var coefficients = List[Float64]()
            for r in range(j):
                var dot: Float64 = 0
                for c in range(m):
                    dot += get2(w1, 0, c) * get2(w_all, r, c)
                coefficients.append(dot)
            for c in range(m):
                var correction: Float64 = 0
                for r in range(j):
                    correction += coefficients[r] * get2(w_all, r, c)
                set2(w1, 0, c, get2(w1, 0, c) - correction)
            var length: Float64 = 0
            for c in range(m):
                length += get2(w1, 0, c) ** 2
            length = sqrt(length)
            var dot: Float64 = 0
            for c in range(m):
                set2(w1, 0, c, get2(w1, 0, c) / length)
                dot += get2(w1, 0, c) * get2(w, 0, c)
            var limit = abs(abs(dot) - 1)
            w = w1^
            if limit < tol:
                break
        most = max(most, iterations)
        for c in range(m):
            set2(w_all, j, c, get2(w, 0, c))
    return (w_all^, most)


def pseudo_inverse[dtype: DType](a: NDArray[dtype]) raises -> NDArray[dtype]:
    """scipy.linalg.pinv default cutoff: rtol = max(M, N) * eps, atol = 0."""
    var rtol = Float64(max(a.shape[0], a.shape[1])) * machine_epsilon[dtype]()
    try:
        var factor = svd(a)
        return factor.solve(identity[dtype](a.shape[0]), rtol=rtol, atol=0).solution.copy()
    except error:
        raise Error("LinAlgError: pseudoinverse failed: " + String(error))


struct FastICA[dtype: DType = DType.float64](InvertibleDecomposition, UnsupervisedEstimator):
    """FastICA on dense Float32/Float64 input; learned arrays keep the input dtype.

    algorithm "parallel"/"deflation"; fun "logcosh"/"exp"/"cube" with optional
    fun_args["alpha"] in [1, 2]; whiten "unit-variance", "arbitrary-variance"
    or False; whiten_solver "svd"/"eigh". Python callables for `fun` are
    binding work and not supported.
    """
    comptime Params = FastICAParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]

    var n_components: Optional[Int]
    var algorithm: String
    var whiten: WhitenOption
    var fun: String
    var fun_args: Optional[Dict[String, Float64]]
    var max_iter: Int
    var tol: Float64
    var w_init: Optional[NDArray[DType.float64]]
    var whiten_solver: String
    var random_state: Optional[Int]
    var components_: Optional[NDArray[Self.dtype]]
    var mixing_: Optional[NDArray[Self.dtype]]
    var mean_: Optional[NDArray[Self.dtype]]
    var whitening_: Optional[NDArray[Self.dtype]]
    var n_iter_: Int
    var n_features_in_: Int
    var _unmixing: Optional[NDArray[Self.dtype]]

    def __init__(
        out self, n_components: Optional[Int] = None, *, algorithm: String = "parallel",
        whiten: WhitenOption = String("unit-variance"), fun: String = "logcosh",
        fun_args: Optional[Dict[String, Float64]] = None, max_iter: Int = 200, tol: Float64 = 1e-4,
        w_init: Optional[NDArray[DType.float64]] = None, whiten_solver: String = "svd",
        random_state: Optional[Int] = None,
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten.copy()
        self.fun = fun
        self.fun_args = fun_args.copy()
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init.copy()
        self.whiten_solver = whiten_solver
        self.random_state = random_state
        self.components_ = None
        self.mixing_ = None
        self.mean_ = None
        self.whitening_ = None
        self.n_iter_ = 0
        self.n_features_in_ = 0
        self._unmixing = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.n_components, self.algorithm, self.whiten.copy(), self.fun,
            self.fun_args.copy(), self.max_iter, self.tol, self.w_init.copy(), self.whiten_solver,
            self.random_state)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.n_components = params.n_components
        self.algorithm = params.algorithm
        self.whiten = params.whiten.copy()
        self.fun = params.fun
        self.fun_args = params.fun_args.copy()
        self.max_iter = params.max_iter
        self.tol = params.tol
        self.w_init = params.w_init.copy()
        self.whiten_solver = params.whiten_solver
        self.random_state = params.random_state
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, n_components: Optional[Optional[Int]] = None, algorithm: Optional[String] = None,
        whiten: Optional[WhitenOption] = None, fun: Optional[String] = None,
        fun_args: Optional[Optional[Dict[String, Float64]]] = None, max_iter: Optional[Int] = None,
        tol: Optional[Float64] = None, w_init: Optional[Optional[NDArray[DType.float64]]] = None,
        whiten_solver: Optional[String] = None, random_state: Optional[Optional[Int]] = None,
    ) raises -> ref[self] Self:
        """Partial update; omitted keywords keep their values."""
        if n_components: self.n_components = n_components.value()
        if algorithm: self.algorithm = algorithm.value()
        if whiten: self.whiten = whiten.value().copy()
        if fun: self.fun = fun.value()
        if fun_args: self.fun_args = fun_args.value().copy()
        if max_iter: self.max_iter = max_iter.value()
        if tol: self.tol = tol.value()
        if w_init: self.w_init = w_init.value().copy()
        if whiten_solver: self.whiten_solver = whiten_solver.value()
        if random_state: self.random_state = random_state.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.n_components, algorithm=self.algorithm, whiten=self.whiten.copy(), fun=self.fun,
            fun_args=self.fun_args.copy(), max_iter=self.max_iter, tol=self.tol, w_init=self.w_init.copy(),
            whiten_solver=self.whiten_solver, random_state=self.random_state)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.components_)

    def _alpha(self) raises -> Float64:
        var alpha: Float64 = 1.0
        if self.fun_args:
            var arguments = self.fun_args.value().copy()
            # sklearn reads only "alpha" (fun_args.get("alpha", 1.0)) and ignores other keys.
            if "alpha" in arguments:
                alpha = arguments["alpha"]
        if not (alpha >= 1 and alpha <= 2):
            raise Error("ValueError: alpha must be in [1,2]")
        return alpha

    def _validate_params(self) raises:
        if self.n_components and self.n_components.value() < 1:
            raise Error("ValueError: FastICA n_components must be >= 1 or None, got " + String(self.n_components.value()))
        if self.algorithm != "parallel" and self.algorithm != "deflation":
            raise Error("ValueError: FastICA algorithm must be 'parallel' or 'deflation', got '" + self.algorithm + "'")
        _ = _whitening_mode(self.whiten)
        if self.fun != "logcosh" and self.fun != "exp" and self.fun != "cube":
            raise Error("ValueError: FastICA fun must be 'logcosh', 'exp' or 'cube' (callables are not supported natively), got '" + self.fun + "'")
        if self.max_iter < 1:
            raise Error("ValueError: FastICA max_iter must be >= 1, got " + String(self.max_iter))
        if not isfinite(self.tol) or self.tol < 0:
            raise Error("ValueError: FastICA tol must be a finite value >= 0")
        if self.whiten_solver != "svd" and self.whiten_solver != "eigh":
            raise Error("ValueError: FastICA whiten_solver must be 'svd' or 'eigh', got '" + self.whiten_solver + "'")

    def _fit_transform(mut self, X: Self.Input, compute_sources: Bool) raises -> Optional[NDArray[Self.dtype]]:
        self._validate_params()
        validate_matrix(X, min_samples=2)
        var alpha = self._alpha()
        var mode = _whitening_mode(self.whiten)
        var whitening = mode != "false"
        var n = X.shape[0]
        var p = X.shape[1]
        var rng = seeded_state(self.random_state, "FastICA")
        # sklearn ignores n_components when whiten=False and clips it to min(n, p).
        var k = min(n, p)
        if whitening and self.n_components and self.n_components.value() < k:
            k = self.n_components.value()
        var raw = contiguous(X)
        var xt = raw.T()
        var means = zeros[Self.dtype](Shape(p))
        var k_matrix = zeros[Self.dtype](Shape(k, p))
        var x1: NDArray[Self.dtype]
        if whitening:
            for f in range(p):
                var total: Float64 = 0
                for i in range(n): total += get2(xt, f, i)
                set1(means, f, total / Float64(n))
            for f in range(p):
                for i in range(n): set2(xt, f, i, get2(xt, f, i) - get1(means, f))
            var r = min(n, p)
            var u = zeros[Self.dtype](Shape(p, r))
            var d = zeros[Self.dtype](Shape(r))
            if self.whiten_solver == "eigh":
                # eigh(XT @ X) of the centered-by-raw product, as the reference.
                var gram = xt @ raw
                var factor = symmetric_eigh(gram)
                var values = factor.eigenvalues()
                var vectors = factor.eigenvectors()
                var eps = machine_epsilon[Self.dtype]() * 10
                u = zeros[Self.dtype](Shape(p, p))
                d = zeros[Self.dtype](Shape(p))
                for j in range(p):
                    var source = p - 1 - j  # ascending -> descending
                    var value = get1(values, source)
                    if value < eps: value = eps
                    set1(d, j, sqrt(value))
                    for f in range(p): set2(u, f, j, get2(vectors, f, source))
            else:
                var factors = dense_svd(xt, flip_sign=False)
                for j in range(r):
                    set1(d, j, get1(factors.singular_values, j))
                    for f in range(p): set2(u, f, j, get2(factors.U, f, j))
            for j in range(k):
                var first = get2(u, 0, j)
                var sign: Float64 = 1.0 if first > 0 else (-1.0 if first < 0 else 0.0)
                for f in range(p):
                    set2(k_matrix, j, f, sign * get2(u, f, j) / get1(d, j))
            x1 = k_matrix @ xt
            var root = sqrt(Float64(n))
            for j in range(k):
                for i in range(n): set2(x1, j, i, get2(x1, j, i) * root)
        else:
            x1 = xt.copy()
        if x1.shape[0] != k:
            raise Error("ValueError: FastICA whiten=False requires n_samples >= n_features (unmixing is square in the features); got shape ("
                + String(n) + ", " + String(p) + ")")
        var w_init = zeros[Self.dtype](Shape(k, k))
        if self.w_init:
            var given = self.w_init.value().copy()
            if given.ndim != 2 or given.shape[0] != k or given.shape[1] != k:
                raise Error("ValueError: w_init has invalid shape -- should be (" + String(k) + ", " + String(k) + ")")
            for i in range(k):
                for j in range(k): set2(w_init, i, j, get2(given, i, j))
        else:
            for i in range(k):
                for j in range(k): set2(w_init, i, j, rng.normal())
        var solved = _ica_par(x1, self.tol, self.fun, alpha, self.max_iter, w_init) if self.algorithm == "parallel" else _ica_def(x1, self.tol, self.fun, alpha, self.max_iter, w_init)
        var w = solved[0].copy()
        self.n_iter_ = solved[1]
        var sources: Optional[NDArray[Self.dtype]] = None
        if whitening:
            var unmix = w @ k_matrix
            if compute_sources or mode == "unit-variance":
                sources = (unmix @ xt).T()
            if mode == "unit-variance":
                var s = sources.value().copy()
                for j in range(k):
                    var total: Float64 = 0
                    for i in range(n): total += get2(s, i, j)
                    var average = total / Float64(n)
                    var squares: Float64 = 0
                    for i in range(n): squares += (get2(s, i, j) - average) ** 2
                    var std = sqrt(squares / Float64(n))
                    for i in range(n): set2(s, i, j, get2(s, i, j) / std)
                    for c in range(k): set2(w, j, c, get2(w, j, c) / std)
                sources = s^
            self.components_ = w @ k_matrix
            self.mean_ = means^
            self.whitening_ = k_matrix^
        else:
            if compute_sources:
                sources = (w @ xt).T()
            self.components_ = w.copy()
            self.mean_ = None
            self.whitening_ = None
        self.mixing_ = pseudo_inverse(self.components_.value())
        self._unmixing = w^
        self.n_features_in_ = p
        # sklearn computes S for unit-variance scaling even when compute_sources=False
        # and returns it, so fastica(..., compute_sources=False) still carries S there.
        if not compute_sources and mode != "unit-variance":
            return None
        return sources^

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        _ = self._fit_transform(X, False)
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        """Fit and return the estimated sources S, shape (n_samples, n_components)."""
        return self._fit_transform(X, True).value().copy()

    def _check(self) raises:
        if not self.components_:
            raise Error("NotFittedError: This FastICA instance is not fitted yet. Call 'fit' first.")

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """Recover sources: (X - mean_) @ components_.T (no centering if whiten=False)."""
        self._check()
        validate_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise Error("ValueError: X has " + String(X.shape[1]) + " features, but FastICA is expecting "
                + String(self.n_features_in_) + " features as input.")
        var x = contiguous(X)
        if self.mean_:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]): set2(x, i, j, get2(x, i, j) - get1(self.mean_.value(), j))
        return x @ self.components_.value().T()

    def transform(self, X: Self.Input, copy: Bool) raises -> Self.Transformed:
        """sklearn's `copy` keyword: native inputs are borrowed and never modified."""
        return self.transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        """Mix sources back: X @ mixing_.T + mean_."""
        self._check()
        validate_matrix(X)
        if X.shape[1] != self.mixing_.value().shape[1]:
            raise Error("ValueError: inverse_transform expects " + String(self.mixing_.value().shape[1])
                + " columns, got " + String(X.shape[1]))
        var result = contiguous(X) @ self.mixing_.value().T()
        if self.mean_:
            for i in range(result.shape[0]):
                for j in range(result.shape[1]): set2(result, i, j, get2(result, i, j) + get1(self.mean_.value(), j))
        return result^

    def inverse_transform(self, X: Self.Transformed, copy: Bool) raises -> Self.Input:
        return self.inverse_transform(X)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        self._check()
        return feature_names("fastica", self.components_.value().shape[0], input_features, self.n_features_in_)

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        if transform and transform.value() != "default":
            raise Error("NotImplementedError: FastICA supports native array output only")

    def get_metadata_routing(self) raises:
        raise Error("NotImplementedError: Python metadata routing is not available")


def fastica[dtype: DType](
    X: NDArray[dtype], n_components: Optional[Int] = None, *, algorithm: String = "parallel",
    whiten: WhitenOption = String("unit-variance"), fun: String = "logcosh",
    fun_args: Optional[Dict[String, Float64]] = None, max_iter: Int = 200, tol: Float64 = 1e-4,
    w_init: Optional[NDArray[DType.float64]] = None, whiten_solver: String = "svd",
    random_state: Optional[Int] = None, return_X_mean: Bool = False, compute_sources: Bool = True,
    return_n_iter: Bool = False,
) raises -> FastICAResult[dtype]:
    """Functional FastICA. Returns every value; `return_X_mean`/`return_n_iter`
    are accepted for signature parity (the record always carries both)."""
    var model = FastICA[dtype](n_components, algorithm=algorithm, whiten=whiten, fun=fun, fun_args=fun_args,
        max_iter=max_iter, tol=tol, w_init=w_init, whiten_solver=whiten_solver, random_state=random_state)
    var sources = model._fit_transform(X, compute_sources)
    return FastICAResult[dtype](model.whitening_.copy(), model._unmixing.value().copy(), sources^,
        model.mean_.copy(), model.n_iter_)
