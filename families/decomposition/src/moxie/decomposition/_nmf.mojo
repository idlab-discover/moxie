"""Dense non-negative matrix factorization: NMF, MiniBatchNMF, non_negative_factorization.

Follows sklearn.decomposition._nmf and _cdnmf_fast.pyx (BSD-3-Clause; see
LICENSES/scikit-learn.txt): NNDSVD/random initialization, coordinate descent
(Cichocki & Phan 2009) for the Frobenius loss, and multiplicative updates for
any beta-divergence (Fevotte & Idier 2011), including the online variant of
MiniBatchNMF (Lefevre et al. 2011). Only dense inputs are supported.
"""
from numojo import NDArray, Shape, zeros
from std.math import sqrt, log, abs, isfinite
from std.utils import Variant
from moxie.base import UnsupervisedEstimator
from moxie._random import RandomState
from ._base import InvertibleDecomposition
from ._contracts import validate_matrix
from ._matrix import get2, set2, get1, set1, contiguous, seeded_state, feature_names
from ._svd_engine import randomized_svd

comptime NMFComponents = Variant[Int, String, NoneType]
"""sklearn `n_components`: an int >= 1, "auto" or None."""
comptime BetaLoss = Variant[String, Float64]
"""sklearn `beta_loss`: "frobenius", "kullback-leibler", "itakura-saito" or a float."""
comptime AlphaH = Variant[Float64, String]
"""sklearn `alpha_H`: a float >= 0 or "same"."""

comptime EPSILON: Float64 = 1.1920928955078125e-07  # np.finfo(np.float32).eps
comptime EPS64: Float64 = 2.220446049250313e-16


def _beta_value(beta_loss: BetaLoss) raises -> Float64:
    if beta_loss.isa[Float64]():
        var beta = beta_loss[Float64]
        if not isfinite(beta):
            raise Error("ValueError: beta_loss must be finite")
        return beta
    var name = beta_loss[String]
    if name == "frobenius": return 2
    if name == "kullback-leibler": return 1
    if name == "itakura-saito": return 0
    raise Error("ValueError: beta_loss must be 'frobenius', 'kullback-leibler', 'itakura-saito' or a float, got '" + name + "'")


def _mean[dtype: DType](x: NDArray[dtype]) raises -> Float64:
    var total: Float64 = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]): total += get2(x, i, j)
    return total / Float64(x.shape[0] * x.shape[1])


def _pow(x: Float64, e: Float64) -> Float64:
    if e == 1: return x
    if e == 2: return x * x
    return x ** e


def beta_divergence[dtype: DType](X: NDArray[dtype], W: NDArray[dtype], H: NDArray[dtype], beta: Float64, *, square_root: Bool = False) raises -> Float64:
    """sklearn _beta_divergence for dense X; entries with X <= float32 eps are skipped
    in the X-weighted terms and W @ H is floored at that eps, as in the reference."""
    var wh = W @ H
    var n = X.shape[0]
    var p = X.shape[1]
    var result: Float64 = 0
    if beta == 2:
        for i in range(n):
            for j in range(p): result += (get2(X, i, j) - get2(wh, i, j)) ** 2
        result /= 2
        return sqrt(result * 2) if square_root else result
    if beta == 1:
        var sum_wh: Float64 = 0
        for i in range(n):
            for j in range(p): sum_wh += get2(wh, i, j)
        var x_sum: Float64 = 0
        for i in range(n):
            for j in range(p):
                var x = get2(X, i, j)
                if x > EPSILON:
                    var v = max(get2(wh, i, j), EPSILON)
                    result += x * log(x / v)
                    x_sum += x
        result += sum_wh - x_sum
    elif beta == 0:
        var ratio_sum: Float64 = 0
        var log_sum: Float64 = 0
        for i in range(n):
            for j in range(p):
                var x = get2(X, i, j)
                if x > EPSILON:
                    var d = x / max(get2(wh, i, j), EPSILON)
                    ratio_sum += d
                    log_sum += log(d)
        result = ratio_sum - Float64(n * p) - log_sum
    else:
        var sum_wh_beta: Float64 = 0
        var sum_x_wh: Float64 = 0
        var sum_x_beta: Float64 = 0
        for i in range(n):
            for j in range(p):
                var v = get2(wh, i, j)
                sum_wh_beta += _pow(v, beta)
                var x = get2(X, i, j)
                if x > EPSILON:
                    sum_x_wh += x * _pow(max(v, EPSILON), beta - 1)
                    sum_x_beta += _pow(x, beta)
        result = sum_x_beta - beta * sum_x_wh + sum_wh_beta * (beta - 1)
        result /= beta * (beta - 1)
    if square_root:
        return sqrt(2 * max(result, 0.0))
    return result


def _check_init[dtype: DType](a: NDArray[dtype], rows: Int, cols: Int, whom: String) raises:
    """sklearn _check_init: shape (rows/cols < 0 means 'auto'), non-negative, not all zero."""
    validate_matrix(a, min_samples=0)
    if rows >= 0 and a.shape[0] != rows:
        raise Error("ValueError: Array with wrong first dimension passed to " + whom + ". Expected "
            + String(rows) + ", but got " + String(a.shape[0]) + ".")
    if cols >= 0 and a.shape[1] != cols:
        raise Error("ValueError: Array with wrong second dimension passed to " + whom + ". Expected "
            + String(cols) + ", but got " + String(a.shape[1]) + ".")
    var largest: Float64 = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            var v = get2(a, i, j)
            if v < 0: raise Error("ValueError: Negative values in data passed to " + whom)
            largest = max(largest, v)
    if largest == 0:
        raise Error("ValueError: Array passed to " + whom + " is full of zeros.")


def _norm(values: List[Float64]) -> Float64:
    var total: Float64 = 0
    for v in values: total += v * v
    return sqrt(total)


def initialize_nmf[dtype: DType](X: NDArray[dtype], n_components: Int, init: Optional[String], random_state: Optional[Int], *, eps: Float64 = 1e-6) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """sklearn _initialize_nmf: 'random', 'nndsvd', 'nndsvda', 'nndsvdar' or None (auto).

    Each RNG use re-seeds from random_state, exactly like check_random_state(int).
    """
    var n = X.shape[0]
    var p = X.shape[1]
    var method = init.value() if init else String("nndsvda" if n_components <= min(n, p) else "random")
    if init and method != "random" and n_components > min(n, p):
        raise Error("ValueError: init = '" + method + "' can only be used when n_components <= min(n_samples, n_features)")
    if method != "random" and method != "nndsvd" and method != "nndsvda" and method != "nndsvdar":
        raise Error("ValueError: Invalid init parameter: got '" + method + "' instead of one of (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')")
    var w = zeros[dtype](Shape(n, n_components))
    var h = zeros[dtype](Shape(n_components, p))
    if method == "random":
        var avg = sqrt(_mean(X) / Float64(n_components))
        var rng = seeded_state(random_state, "NMF")
        for i in range(n_components):
            for j in range(p): set2(h, i, j, abs(avg * Float64(Scalar[dtype](rng.standard_normal()))))
        for i in range(n):
            for j in range(n_components): set2(w, i, j, abs(avg * Float64(Scalar[dtype](rng.standard_normal()))))
        return (w^, h^)
    var svd_rng = seeded_state(random_state, "NMF")
    var factors = randomized_svd(X, n_components, svd_rng)
    var s0 = sqrt(get1(factors.singular_values, 0))
    for i in range(n): set2(w, i, 0, s0 * abs(get2(factors.U, i, 0)))
    for j in range(p): set2(h, 0, j, s0 * abs(get2(factors.Vt, 0, j)))
    for c in range(1, n_components):
        var xp = List[Float64]()
        var xn = List[Float64]()
        var yp = List[Float64]()
        var yn = List[Float64]()
        for i in range(n):
            var v = get2(factors.U, i, c)
            xp.append(max(v, 0.0))
            xn.append(abs(min(v, 0.0)))
        for j in range(p):
            var v = get2(factors.Vt, c, j)
            yp.append(max(v, 0.0))
            yn.append(abs(min(v, 0.0)))
        var xpn = _norm(xp)
        var ypn = _norm(yp)
        var xnn = _norm(xn)
        var ynn = _norm(yn)
        var mp = xpn * ypn
        var mn = xnn * ynn
        var positive = mp > mn
        var sigma = mp if positive else mn
        var lbd = sqrt(get1(factors.singular_values, c) * sigma)
        for i in range(n): set2(w, i, c, lbd * (xp[i] / xpn if positive else xn[i] / xnn))
        for j in range(p): set2(h, c, j, lbd * (yp[j] / ypn if positive else yn[j] / ynn))
    for i in range(n):
        for c in range(n_components):
            if get2(w, i, c) < eps: set2(w, i, c, 0)
    for c in range(n_components):
        for j in range(p):
            if get2(h, c, j) < eps: set2(h, c, j, 0)
    if method == "nndsvda":
        var avg = _mean(X)
        for i in range(n):
            for c in range(n_components):
                if get2(w, i, c) == 0: set2(w, i, c, avg)
        for c in range(n_components):
            for j in range(p):
                if get2(h, c, j) == 0: set2(h, c, j, avg)
    elif method == "nndsvdar":
        var rng = seeded_state(random_state, "NMF")
        var avg = _mean(X)
        # Draw count follows the zero masks in C order, W first then H.
        for i in range(n):
            for c in range(n_components):
                if get2(w, i, c) == 0: set2(w, i, c, abs(avg * rng.standard_normal() / 100))
        for c in range(n_components):
            for j in range(p):
                if get2(h, c, j) == 0: set2(h, c, j, abs(avg * rng.standard_normal() / 100))
    return (w^, h^)


def _cd_update[dtype: DType](X: NDArray[dtype], mut W: NDArray[dtype], Ht: NDArray[dtype], l1_reg: Float64, l2_reg: Float64, shuffle: Bool, mut rng: RandomState) raises -> Float64:
    """One sweep of _update_coordinate_descent / _update_cdnmf_fast over W (rows of X)."""
    var k = Ht.shape[1]
    var hht = Ht.T() @ Ht
    var xht = X @ Ht
    var order = List[Int]()
    if shuffle:
        var permutation = rng.permutation(k)
        for s in range(k): order.append(Int(permutation.unsafe_get(s)))
    else:
        for s in range(k): order.append(s)
    var violation: Float64 = 0
    for s in range(k):
        var t = order[s]
        var hess = get2(hht, t, t) + l2_reg
        for i in range(W.shape[0]):
            var grad = -(get2(xht, i, t) - l1_reg)
            for r in range(k):
                grad += (get2(hht, t, r) + (l2_reg if r == t else 0.0)) * get2(W, i, r)
            var wit = get2(W, i, t)
            var pg = min(0.0, grad) if wit == 0 else grad
            violation += abs(pg)
            if hess != 0:
                set2(W, i, t, max(wit - grad / hess, 0.0))
    return violation


def fit_coordinate_descent[dtype: DType](X: NDArray[dtype], mut W: NDArray[dtype], mut H: NDArray[dtype], tol: Float64, max_iter: Int, l1_reg_W: Float64, l1_reg_H: Float64, l2_reg_W: Float64, l2_reg_H: Float64, update_H: Bool, shuffle: Bool, random_state: Optional[Int]) raises -> Int:
    """sklearn _fit_coordinate_descent (Frobenius loss); returns n_iter."""
    var ht = contiguous(H.T())
    var xt = contiguous(X.T())
    var rng = seeded_state(random_state, "NMF")
    var violation_init: Float64 = 0
    var n_iter = 0
    for iteration in range(1, max_iter + 1):
        n_iter = iteration
        var violation = _cd_update(X, W, ht, l1_reg_W, l2_reg_W, shuffle, rng)
        if update_H:
            violation += _cd_update(xt, ht, W, l1_reg_H, l2_reg_H, shuffle, rng)
        if iteration == 1:
            violation_init = violation
        if violation_init == 0:
            break
        if violation / violation_init <= tol:
            break
    H = contiguous(ht.T())
    return n_iter


def _mu_terms_w[dtype: DType](X: NDArray[dtype], W: NDArray[dtype], H: NDArray[dtype], beta: Float64, l1: Float64, l2: Float64, h_sum_offset: Float64 = 0) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """Numerator and (regularized, zero-safe) denominator of the W update, shape (n, k).

    h_sum_offset (KL only) reproduces the reference's cached H row sums, which
    its in-place `denominator += l1_reg_W` grows by l1 on every update_H=False
    iteration.
    """
    var n = X.shape[0]
    var k = W.shape[1]
    var numerator: NDArray[dtype]
    var denominator: NDArray[dtype]
    if beta == 2:
        numerator = X @ H.T()
        denominator = W @ (H @ H.T())
    else:
        var wh = W @ H
        var ratio = zeros[dtype](Shape(n, X.shape[1]))
        var powered = zeros[dtype](Shape(n, X.shape[1]))
        for i in range(n):
            for j in range(X.shape[1]):
                var v = get2(wh, i, j)
                var safe = max(v, EPSILON) if beta < 2 else v
                var x = get2(X, i, j)
                if beta == 1: set2(ratio, i, j, x / safe)
                elif beta == 0: set2(ratio, i, j, x * _pow(1 / safe, 2))
                else: set2(ratio, i, j, _pow(safe, beta - 2) * x)
                if beta != 1:
                    var base = max(v, EPSILON) if beta < 1 else v
                    set2(powered, i, j, _pow(base, beta - 1))
        numerator = ratio @ H.T()
        if beta == 1:
            denominator = zeros[dtype](Shape(n, k))
            for c in range(k):
                var total: Float64 = 0
                for j in range(H.shape[1]): total += get2(H, c, j)
                total += h_sum_offset
                for i in range(n): set2(denominator, i, c, total)
        else:
            denominator = powered @ H.T()
    for i in range(n):
        for c in range(k):
            var d = get2(denominator, i, c)
            if l1 > 0: d += l1
            if l2 > 0: d += l2 * get2(W, i, c)
            if d == 0: d = EPSILON
            set2(denominator, i, c, d)
    return (numerator^, denominator^)


def mu_update_w[dtype: DType](X: NDArray[dtype], W: NDArray[dtype], H: NDArray[dtype], beta: Float64, l1: Float64, l2: Float64, gamma: Float64, h_sum_offset: Float64 = 0) raises -> NDArray[dtype]:
    """sklearn _multiplicative_update_w; returns the updated W."""
    var terms = _mu_terms_w(X, W, H, beta, l1, l2, h_sum_offset)
    var result = contiguous(W)
    for i in range(W.shape[0]):
        for c in range(W.shape[1]):
            var delta = get2(terms[0], i, c) / get2(terms[1], i, c)
            if gamma != 1: delta = _pow(delta, gamma)
            set2(result, i, c, get2(W, i, c) * delta)
    return result^


def _mu_terms_h[dtype: DType](X: NDArray[dtype], W: NDArray[dtype], H: NDArray[dtype], beta: Float64, l1: Float64, l2: Float64) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """Numerator and (regularized, zero-safe) denominator of the H update, shape (k, p)."""
    var n = X.shape[0]
    var p = X.shape[1]
    var k = H.shape[0]
    var numerator: NDArray[dtype]
    var denominator: NDArray[dtype]
    if beta == 2:
        numerator = W.T() @ X
        denominator = (W.T() @ W) @ H
    else:
        var wh = W @ H
        var ratio = zeros[dtype](Shape(n, p))
        var powered = zeros[dtype](Shape(n, p))
        for i in range(n):
            for j in range(p):
                var v = get2(wh, i, j)
                var safe = max(v, EPSILON) if beta < 2 else v
                var x = get2(X, i, j)
                if beta == 1: set2(ratio, i, j, x / safe)
                elif beta == 0: set2(ratio, i, j, x * _pow(1 / safe, 2))
                else: set2(ratio, i, j, _pow(safe, beta - 2) * x)
                if beta != 1:
                    var base = max(v, EPSILON) if beta < 1 else v
                    set2(powered, i, j, _pow(base, beta - 1))
        numerator = W.T() @ ratio
        if beta == 1:
            denominator = zeros[dtype](Shape(k, p))
            for c in range(k):
                var total: Float64 = 0
                for i in range(n): total += get2(W, i, c)
                if total == 0: total = 1
                for j in range(p): set2(denominator, c, j, total)
        else:
            denominator = W.T() @ powered
    for c in range(k):
        for j in range(p):
            var d = get2(denominator, c, j)
            if l1 > 0: d += l1
            if l2 > 0: d += l2 * get2(H, c, j)
            if d == 0: d = EPSILON
            set2(denominator, c, j, d)
    return (numerator^, denominator^)


def mu_update_h[dtype: DType](X: NDArray[dtype], W: NDArray[dtype], H: NDArray[dtype], beta: Float64, l1: Float64, l2: Float64, gamma: Float64) raises -> NDArray[dtype]:
    """sklearn _multiplicative_update_h without the online accumulators."""
    var terms = _mu_terms_h(X, W, H, beta, l1, l2)
    var result = contiguous(H)
    for c in range(H.shape[0]):
        for j in range(H.shape[1]):
            var delta = get2(terms[0], c, j) / get2(terms[1], c, j)
            if gamma != 1: delta = _pow(delta, gamma)
            set2(result, c, j, get2(H, c, j) * delta)
    return result^


def mu_update_h_online[dtype: DType](X: NDArray[dtype], W: NDArray[dtype], H: NDArray[dtype], beta: Float64, l1: Float64, l2: Float64, gamma: Float64, mut A: NDArray[dtype], mut B: NDArray[dtype], rho: Float64) raises -> NDArray[dtype]:
    """sklearn _multiplicative_update_h with MiniBatchNMF's A/B accumulators (rho forgetting)."""
    var terms = _mu_terms_h(X, W, H, beta, l1, l2)
    var result = contiguous(H)
    for c in range(H.shape[0]):
        for j in range(H.shape[1]):
            var h = get2(H, c, j)
            if gamma != 1: h = _pow(h, 1 / gamma)
            var a = get2(A, c, j) * rho + get2(terms[0], c, j) * h
            var b = get2(B, c, j) * rho + get2(terms[1], c, j)
            set2(A, c, j, a)
            set2(B, c, j, b)
            var value = Float64(Scalar[dtype](a)) / Float64(Scalar[dtype](b))
            if gamma != 1: value = _pow(value, gamma)
            set2(result, c, j, value)
    return result^


def _gamma(beta: Float64) -> Float64:
    if beta < 1: return 1 / (2 - beta)
    if beta > 2: return 1 / (beta - 1)
    return 1


def fit_multiplicative_update[dtype: DType](X: NDArray[dtype], mut W: NDArray[dtype], mut H: NDArray[dtype], beta: Float64, max_iter: Int, tol: Float64, l1_reg_W: Float64, l1_reg_H: Float64, l2_reg_W: Float64, l2_reg_H: Float64, update_H: Bool) raises -> Int:
    """sklearn _fit_multiplicative_update; returns n_iter (error checked every 10 iterations)."""
    var gamma = _gamma(beta)
    var error_at_init = beta_divergence(X, W, H, beta, square_root=True)
    var previous = error_at_init
    var n_iter = 0
    for iteration in range(1, max_iter + 1):
        n_iter = iteration
        # With a fixed H the reference reuses its KL H_sum cache, mutated in place.
        var offset = Float64(iteration - 1) * l1_reg_W if (not update_H and beta == 1 and l1_reg_W > 0) else 0.0
        W = mu_update_w(X, W, H, beta, l1_reg_W, l2_reg_W, gamma, offset)
        if beta < 1:
            for i in range(W.shape[0]):
                for c in range(W.shape[1]):
                    if get2(W, i, c) < EPS64: set2(W, i, c, 0)
        if update_H:
            H = mu_update_h(X, W, H, beta, l1_reg_H, l2_reg_H, gamma)
            if beta <= 1:
                for c in range(H.shape[0]):
                    for j in range(H.shape[1]):
                        if get2(H, c, j) < EPS64: set2(H, c, j, 0)
        if tol > 0 and iteration % 10 == 0:
            var error = beta_divergence(X, W, H, beta, square_root=True)
            if (previous - error) / error_at_init < tol:
                break
            previous = error
    return n_iter


@fieldwise_init
struct NMFConfig(Copyable, Movable):
    """Parameters shared by NMF and MiniBatchNMF (sklearn _BaseNMF)."""
    var n_components: NMFComponents
    var init: Optional[String]
    var beta_loss: BetaLoss
    var tol: Float64
    var max_iter: Int
    var random_state: Optional[Int]
    var alpha_W: Float64
    var alpha_H: AlphaH
    var l1_ratio: Float64
    var verbose: Int

    def validate(self, estimator: String) raises:
        if self.n_components.isa[Int]() and self.n_components[Int] < 1:
            raise Error("ValueError: " + estimator + " n_components must be >= 1, 'auto' or None, got " + String(self.n_components[Int]))
        if self.n_components.isa[String]() and self.n_components[String] != "auto":
            raise Error("ValueError: " + estimator + " n_components must be an int, 'auto' or None, got '" + self.n_components[String] + "'")
        if self.init:
            var i = self.init.value()
            if i != "random" and i != "nndsvd" and i != "nndsvda" and i != "nndsvdar" and i != "custom":
                raise Error("ValueError: " + estimator + " init must be one of random, nndsvd, nndsvda, nndsvdar, custom or None, got '" + i + "'")
        _ = _beta_value(self.beta_loss)
        if not isfinite(self.tol) or self.tol < 0:
            raise Error("ValueError: " + estimator + " tol must be a finite value >= 0")
        if self.max_iter < 1:
            raise Error("ValueError: " + estimator + " max_iter must be >= 1, got " + String(self.max_iter))
        if not isfinite(self.alpha_W) or self.alpha_W < 0:
            raise Error("ValueError: " + estimator + " alpha_W must be >= 0")
        if self.alpha_H.isa[String]():
            if self.alpha_H[String] != "same":
                raise Error("ValueError: " + estimator + " alpha_H must be a float >= 0 or 'same'")
        elif not isfinite(self.alpha_H[Float64]) or self.alpha_H[Float64] < 0:
            raise Error("ValueError: " + estimator + " alpha_H must be >= 0")
        if not (self.l1_ratio >= 0 and self.l1_ratio <= 1):
            raise Error("ValueError: " + estimator + " l1_ratio must be in [0, 1]")
        if self.verbose < 0:
            raise Error("ValueError: " + estimator + " verbose must be >= 0")

    def regularization(self, n_samples: Int, n_features: Int) -> Tuple[Float64, Float64, Float64, Float64]:
        """(l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H), scaled by the data shape as in sklearn."""
        var alpha_H = self.alpha_W if self.alpha_H.isa[String]() else self.alpha_H[Float64]
        return (Float64(n_features) * self.alpha_W * self.l1_ratio, Float64(n_samples) * alpha_H * self.l1_ratio,
            Float64(n_features) * self.alpha_W * (1 - self.l1_ratio), Float64(n_samples) * alpha_H * (1 - self.l1_ratio))

    def resolve_components(self, n_features: Int) -> Int:
        """-1 encodes 'auto' (resolved later from H or n_features)."""
        if self.n_components.isa[NoneType](): return n_features
        if self.n_components.isa[String](): return -1
        return self.n_components[Int]


def _check_w_h[dtype: DType](config: NMFConfig, X: NDArray[dtype], W: Optional[NDArray[dtype]], H: Optional[NDArray[dtype]], update_H: Bool, mut k: Int, solver: String) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """sklearn _BaseNMF._check_w_h. `k` is the resolved component count (-1 = auto)."""
    var n = X.shape[0]
    var p = X.shape[1]
    var custom = config.init and config.init.value() == "custom"
    if custom and update_H:
        if not H or not W:
            raise Error("ValueError: init='custom' requires both W and H")
        _check_init(H.value(), k, p, "NMF (input H)")
        _check_init(W.value(), n, k, "NMF (input W)")
        if k < 0: k = H.value().shape[0]
        return (contiguous(W.value()), contiguous(H.value()))
    if not update_H:
        if not H:
            raise Error("ValueError: update_H=False requires H")
        _check_init(H.value(), k, p, "NMF (input H)")
        if k < 0: k = H.value().shape[0]
        var w = zeros[dtype](Shape(n, k))
        if solver == "mu":
            var avg = sqrt(_mean(X) / Float64(k))
            for i in range(n):
                for c in range(k): set2(w, i, c, avg)
        return (w^, contiguous(H.value()))
    # Provided W/H are ignored unless init='custom' (sklearn warns).
    if k < 0: k = p
    return initialize_nmf(X, k, config.init, config.random_state)


def _validate_input[dtype: DType](X: NDArray[dtype], beta: Float64, whom: String) raises:
    validate_matrix(X)
    var smallest: Float64 = get2(X, 0, 0)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            var v = get2(X, i, j)
            if v < 0: raise Error("ValueError: Negative values in data passed to " + whom)
            smallest = min(smallest, v)
    if smallest == 0 and beta <= 0:
        raise Error("ValueError: When beta_loss <= 0 and X contains zeros, the solver may diverge. Please add small values to X, or use a positive beta_loss.")


@fieldwise_init
struct NMFParams(Copyable, Movable):
    """Typed get_params/set_params record for NMF."""
    var config: NMFConfig
    var solver: String
    var shuffle: Bool


struct NMF[dtype: DType = DType.float64](InvertibleDecomposition, UnsupervisedEstimator):
    """Non-negative matrix factorization X ~ W H on dense non-negative input.

    solver "cd" (Frobenius only) or "mu" (any beta_loss). Learned arrays keep
    the input dtype. verbose is accepted but prints nothing.
    """
    comptime Params = NMFParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]

    var n_components: NMFComponents
    var init: Optional[String]
    var solver: String
    var beta_loss: BetaLoss
    var tol: Float64
    var max_iter: Int
    var random_state: Optional[Int]
    var alpha_W: Float64
    var alpha_H: AlphaH
    var l1_ratio: Float64
    var verbose: Int
    var shuffle: Bool
    var components_: Optional[NDArray[Self.dtype]]
    var n_components_: Int
    var reconstruction_err_: Float64
    var n_iter_: Int
    var n_features_in_: Int

    def __init__(
        out self, n_components: NMFComponents = String("auto"), *, init: Optional[String] = None,
        solver: String = "cd", beta_loss: BetaLoss = String("frobenius"), tol: Float64 = 1e-4,
        max_iter: Int = 200, random_state: Optional[Int] = None, alpha_W: Float64 = 0.0,
        alpha_H: AlphaH = String("same"), l1_ratio: Float64 = 0.0, verbose: Int = 0, shuffle: Bool = False,
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.n_components = n_components.copy()
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss.copy()
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H.copy()
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle
        self.components_ = None
        self.n_components_ = 0
        self.reconstruction_err_ = 0
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def _config(self) -> NMFConfig:
        return NMFConfig(self.n_components.copy(), self.init, self.beta_loss.copy(), self.tol, self.max_iter,
            self.random_state, self.alpha_W, self.alpha_H.copy(), self.l1_ratio, self.verbose)

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self._config(), self.solver, self.shuffle)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        var c = params.config.copy()
        self.n_components = c.n_components.copy()
        self.init = c.init
        self.beta_loss = c.beta_loss.copy()
        self.tol = c.tol
        self.max_iter = c.max_iter
        self.random_state = c.random_state
        self.alpha_W = c.alpha_W
        self.alpha_H = c.alpha_H.copy()
        self.l1_ratio = c.l1_ratio
        self.verbose = c.verbose
        self.solver = params.solver
        self.shuffle = params.shuffle
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, n_components: Optional[NMFComponents] = None, init: Optional[Optional[String]] = None,
        solver: Optional[String] = None, beta_loss: Optional[BetaLoss] = None, tol: Optional[Float64] = None,
        max_iter: Optional[Int] = None, random_state: Optional[Optional[Int]] = None,
        alpha_W: Optional[Float64] = None, alpha_H: Optional[AlphaH] = None, l1_ratio: Optional[Float64] = None,
        shuffle: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        """Partial update; omitted keywords keep their values."""
        if n_components: self.n_components = n_components.value().copy()
        if init: self.init = init.value()
        if solver: self.solver = solver.value()
        if beta_loss: self.beta_loss = beta_loss.value().copy()
        if tol: self.tol = tol.value()
        if max_iter: self.max_iter = max_iter.value()
        if random_state: self.random_state = random_state.value()
        if alpha_W: self.alpha_W = alpha_W.value()
        if alpha_H: self.alpha_H = alpha_H.value().copy()
        if l1_ratio: self.l1_ratio = l1_ratio.value()
        if shuffle: self.shuffle = shuffle.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.n_components.copy(), init=self.init, solver=self.solver, beta_loss=self.beta_loss.copy(),
            tol=self.tol, max_iter=self.max_iter, random_state=self.random_state, alpha_W=self.alpha_W,
            alpha_H=self.alpha_H.copy(), l1_ratio=self.l1_ratio, verbose=self.verbose, shuffle=self.shuffle)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.components_)

    def _fit_transform(self, X: Self.Input, W: Optional[Self.Input], H: Optional[Self.Input], update_H: Bool) raises -> Tuple[NDArray[Self.dtype], NDArray[Self.dtype], Int]:
        var config = self._config()
        config.validate("NMF")
        if self.solver != "cd" and self.solver != "mu":
            raise Error("ValueError: NMF solver must be 'cd' or 'mu', got '" + self.solver + "'")
        var beta = _beta_value(self.beta_loss)
        if self.solver != "mu" and beta != 2:
            raise Error("ValueError: Invalid beta_loss parameter: solver 'cd' does not handle beta_loss other than 'frobenius'")
        _validate_input(X, beta, "NMF (input X)")
        var x = contiguous(X)
        var k = config.resolve_components(x.shape[1])
        var factors = _check_w_h(config, x, W, H, update_H, k, self.solver)
        var w = factors[0].copy()
        var h = factors[1].copy()
        var regs = config.regularization(x.shape[0], x.shape[1])
        var n_iter: Int
        if self.solver == "cd":
            n_iter = fit_coordinate_descent(x, w, h, self.tol, self.max_iter, regs[0], regs[1], regs[2], regs[3],
                update_H, self.shuffle, self.random_state)
        else:
            n_iter = fit_multiplicative_update(x, w, h, beta, self.max_iter, self.tol, regs[0], regs[1], regs[2], regs[3], update_H)
        # sklearn warns (ConvergenceWarning) when n_iter == max_iter and tol > 0.
        return (w^, h^, n_iter)

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        _ = self.fit_transform(X)
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        return self.fit_transform(X, None, None)

    def fit_transform(mut self, X: Self.Input, W: Optional[Self.Input], H: Optional[Self.Input]) raises -> Self.Transformed:
        """Fit and return W; W and H are used only with init='custom'."""
        var result = self._fit_transform(X, W, H, True)
        var beta = _beta_value(self.beta_loss)
        self.reconstruction_err_ = beta_divergence(contiguous(X), result[0], result[1], beta, square_root=True)
        self.n_components_ = result[1].shape[0]
        self.components_ = result[1].copy()
        self.n_iter_ = result[2]
        self.n_features_in_ = X.shape[1]
        return result[0].copy()

    def _check(self) raises:
        if not self.components_:
            raise Error("NotFittedError: This NMF instance is not fitted yet. Call 'fit' first.")

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """Solve for W with components_ fixed (update_H=False), like sklearn."""
        self._check()
        validate_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise Error("ValueError: X has " + String(X.shape[1]) + " features, but NMF is expecting "
                + String(self.n_features_in_) + " features as input.")
        return self._fit_transform(X, None, self.components_.copy(), False)[0].copy()

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        """W @ components_."""
        self._check()
        validate_matrix(X)
        if X.shape[1] != self.components_.value().shape[0]:
            raise Error("ValueError: inverse_transform expects " + String(self.components_.value().shape[0]) + " columns")
        return contiguous(X) @ self.components_.value()

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        self._check()
        return feature_names("nmf", self.n_components_, input_features, self.n_features_in_)

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        if transform and transform.value() != "default":
            raise Error("NotImplementedError: NMF supports native array output only")

    def get_metadata_routing(self) raises:
        raise Error("NotImplementedError: Python metadata routing is not available")


@fieldwise_init
struct NMFResult[dtype: DType](Movable):
    """Return record of non_negative_factorization: (W, H, n_iter)."""
    var W: NDArray[Self.dtype]
    var H: NDArray[Self.dtype]
    var n_iter: Int


def non_negative_factorization[dtype: DType](
    X: NDArray[dtype], W: Optional[NDArray[dtype]] = None, H: Optional[NDArray[dtype]] = None,
    n_components: NMFComponents = String("auto"), *, init: Optional[String] = None, update_H: Bool = True,
    solver: String = "cd", beta_loss: BetaLoss = String("frobenius"), tol: Float64 = 1e-4, max_iter: Int = 200,
    alpha_W: Float64 = 0.0, alpha_H: AlphaH = String("same"), l1_ratio: Float64 = 0.0,
    random_state: Optional[Int] = None, verbose: Int = 0, shuffle: Bool = False,
) raises -> NMFResult[dtype]:
    """Functional NMF; update_H=False solves W for a fixed H."""
    var model = NMF[dtype](n_components, init=init, solver=solver, beta_loss=beta_loss, tol=tol, max_iter=max_iter,
        random_state=random_state, alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio, verbose=verbose, shuffle=shuffle)
    var result = model._fit_transform(X, W, H, update_H)
    return NMFResult[dtype](result[0].copy(), result[1].copy(), result[2])


@fieldwise_init
struct MiniBatchNMFParams(Copyable, Movable):
    """Typed get_params/set_params record for MiniBatchNMF."""
    var config: NMFConfig
    var batch_size: Int
    var max_no_improvement: Optional[Int]
    var forget_factor: Float64
    var fresh_restarts: Bool
    var fresh_restarts_max_iter: Int
    var transform_max_iter: Optional[Int]


def _frobenius_difference[dtype: DType](a: NDArray[dtype], b: NDArray[dtype]) raises -> Float64:
    """||a - b||_F / ||a||_F."""
    var diff: Float64 = 0
    var norm: Float64 = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            diff += (get2(a, i, j) - get2(b, i, j)) ** 2
            norm += get2(a, i, j) ** 2
    return sqrt(diff) / sqrt(norm)


def _rows[dtype: DType](x: NDArray[dtype], start: Int, stop: Int) raises -> NDArray[dtype]:
    var result = zeros[dtype](Shape(stop - start, x.shape[1]))
    for i in range(start, stop):
        for j in range(x.shape[1]): set2(result, i - start, j, get2(x, i, j))
    return result^


struct MiniBatchNMF[dtype: DType = DType.float64](InvertibleDecomposition, UnsupervisedEstimator):
    """Online NMF with multiplicative updates on cycling mini-batches (dense input)."""
    comptime Params = MiniBatchNMFParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]

    var n_components: NMFComponents
    var init: Optional[String]
    var batch_size: Int
    var beta_loss: BetaLoss
    var tol: Float64
    var max_no_improvement: Optional[Int]
    var max_iter: Int
    var alpha_W: Float64
    var alpha_H: AlphaH
    var l1_ratio: Float64
    var forget_factor: Float64
    var fresh_restarts: Bool
    var fresh_restarts_max_iter: Int
    var transform_max_iter: Optional[Int]
    var random_state: Optional[Int]
    var verbose: Int
    var components_: Optional[NDArray[Self.dtype]]
    var n_components_: Int
    var reconstruction_err_: Float64
    var n_iter_: Int
    var n_steps_: Int
    var n_features_in_: Int
    var _components_numerator: Optional[NDArray[Self.dtype]]
    var _components_denominator: Optional[NDArray[Self.dtype]]
    var _k: Int
    var _beta: Float64
    var _gamma: Float64
    var _rho: Float64
    var _transform_max_iter: Int

    def __init__(
        out self, n_components: NMFComponents = String("auto"), *, init: Optional[String] = None,
        batch_size: Int = 1024, beta_loss: BetaLoss = String("frobenius"), tol: Float64 = 1e-4,
        max_no_improvement: Optional[Int] = 10, max_iter: Int = 200, alpha_W: Float64 = 0.0,
        alpha_H: AlphaH = String("same"), l1_ratio: Float64 = 0.0, forget_factor: Float64 = 0.7,
        fresh_restarts: Bool = False, fresh_restarts_max_iter: Int = 30,
        transform_max_iter: Optional[Int] = None, random_state: Optional[Int] = None, verbose: Int = 0,
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.n_components = n_components.copy()
        self.init = init
        self.batch_size = batch_size
        self.beta_loss = beta_loss.copy()
        self.tol = tol
        self.max_no_improvement = max_no_improvement
        self.max_iter = max_iter
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H.copy()
        self.l1_ratio = l1_ratio
        self.forget_factor = forget_factor
        self.fresh_restarts = fresh_restarts
        self.fresh_restarts_max_iter = fresh_restarts_max_iter
        self.transform_max_iter = transform_max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.components_ = None
        self.n_components_ = 0
        self.reconstruction_err_ = 0
        self.n_iter_ = 0
        self.n_steps_ = 0
        self.n_features_in_ = 0
        self._components_numerator = None
        self._components_denominator = None
        self._k = 0
        self._beta = 2
        self._gamma = 1
        self._rho = 1
        self._transform_max_iter = max_iter

    def _config(self) -> NMFConfig:
        return NMFConfig(self.n_components.copy(), self.init, self.beta_loss.copy(), self.tol, self.max_iter,
            self.random_state, self.alpha_W, self.alpha_H.copy(), self.l1_ratio, self.verbose)

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self._config(), self.batch_size, self.max_no_improvement, self.forget_factor,
            self.fresh_restarts, self.fresh_restarts_max_iter, self.transform_max_iter)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        var c = params.config.copy()
        self.n_components = c.n_components.copy()
        self.init = c.init
        self.beta_loss = c.beta_loss.copy()
        self.tol = c.tol
        self.max_iter = c.max_iter
        self.random_state = c.random_state
        self.alpha_W = c.alpha_W
        self.alpha_H = c.alpha_H.copy()
        self.l1_ratio = c.l1_ratio
        self.verbose = c.verbose
        self.batch_size = params.batch_size
        self.max_no_improvement = params.max_no_improvement
        self.forget_factor = params.forget_factor
        self.fresh_restarts = params.fresh_restarts
        self.fresh_restarts_max_iter = params.fresh_restarts_max_iter
        self.transform_max_iter = params.transform_max_iter
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, n_components: Optional[NMFComponents] = None, init: Optional[Optional[String]] = None,
        batch_size: Optional[Int] = None, beta_loss: Optional[BetaLoss] = None, tol: Optional[Float64] = None,
        max_iter: Optional[Int] = None, forget_factor: Optional[Float64] = None,
        fresh_restarts: Optional[Bool] = None, random_state: Optional[Optional[Int]] = None,
    ) raises -> ref[self] Self:
        """Partial update of the most common settings; use the Params record for the rest."""
        if n_components: self.n_components = n_components.value().copy()
        if init: self.init = init.value()
        if batch_size: self.batch_size = batch_size.value()
        if beta_loss: self.beta_loss = beta_loss.value().copy()
        if tol: self.tol = tol.value()
        if max_iter: self.max_iter = max_iter.value()
        if forget_factor: self.forget_factor = forget_factor.value()
        if fresh_restarts: self.fresh_restarts = fresh_restarts.value()
        if random_state: self.random_state = random_state.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.n_components.copy(), init=self.init, batch_size=self.batch_size, beta_loss=self.beta_loss.copy(),
            tol=self.tol, max_no_improvement=self.max_no_improvement, max_iter=self.max_iter, alpha_W=self.alpha_W,
            alpha_H=self.alpha_H.copy(), l1_ratio=self.l1_ratio, forget_factor=self.forget_factor,
            fresh_restarts=self.fresh_restarts, fresh_restarts_max_iter=self.fresh_restarts_max_iter,
            transform_max_iter=self.transform_max_iter, random_state=self.random_state, verbose=self.verbose)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.components_)

    def _check_params(mut self, X: NDArray[Self.dtype]) raises -> Int:
        """sklearn MiniBatchNMF._check_params; returns the effective batch size."""
        self._config().validate("MiniBatchNMF")
        if self.batch_size < 1:
            raise Error("ValueError: MiniBatchNMF batch_size must be >= 1, got " + String(self.batch_size))
        if self.max_no_improvement and self.max_no_improvement.value() < 1:
            raise Error("ValueError: MiniBatchNMF max_no_improvement must be >= 1 or None")
        if not (self.forget_factor >= 0 and self.forget_factor <= 1):
            raise Error("ValueError: MiniBatchNMF forget_factor must be in [0, 1]")
        if self.fresh_restarts_max_iter < 1:
            raise Error("ValueError: MiniBatchNMF fresh_restarts_max_iter must be >= 1")
        if self.transform_max_iter and self.transform_max_iter.value() < 1:
            raise Error("ValueError: MiniBatchNMF transform_max_iter must be >= 1 or None")
        self._k = self._config().resolve_components(X.shape[1])
        self._beta = _beta_value(self.beta_loss)
        var batch = min(self.batch_size, X.shape[0])
        self._rho = self.forget_factor ** (Float64(batch) / Float64(X.shape[0]))
        self._gamma = _gamma(self._beta)
        self._transform_max_iter = self.transform_max_iter.value() if self.transform_max_iter else self.max_iter
        return batch

    def _solve_w(self, X: NDArray[Self.dtype], H: NDArray[Self.dtype], max_iter: Int) raises -> NDArray[Self.dtype]:
        var avg = sqrt(_mean(X) / Float64(self._k))
        var w = zeros[Self.dtype](Shape(X.shape[0], self._k))
        for i in range(X.shape[0]):
            for c in range(self._k): set2(w, i, c, avg)
        var buffer = w.copy()
        var regs = self._config().regularization(X.shape[0], X.shape[1])
        for _ in range(max_iter):
            w = mu_update_w(X, w, H, self._beta, regs[0], regs[2], self._gamma)
            if self.tol > 0 and _frobenius_difference(w, buffer) <= self.tol:
                break
            buffer = w.copy()
        return w^

    def _minibatch_step(mut self, X: NDArray[Self.dtype], mut W: Optional[NDArray[Self.dtype]], mut H: NDArray[Self.dtype], update_H: Bool) raises -> Float64:
        """One online step on batch X; W holds the batch rows (None -> solve afresh)."""
        var n = X.shape[0]
        var regs = self._config().regularization(n, X.shape[1])
        var w: NDArray[Self.dtype]
        if self.fresh_restarts or not W:
            w = self._solve_w(X, H, self.fresh_restarts_max_iter)
        else:
            w = mu_update_w(X, W.value(), H, self._beta, regs[0], regs[2], self._gamma)
        if self._beta < 1:
            for i in range(w.shape[0]):
                for c in range(w.shape[1]):
                    if get2(w, i, c) < EPS64: set2(w, i, c, 0)
        var w_sum: Float64 = 0
        var w_squares: Float64 = 0
        for i in range(w.shape[0]):
            for c in range(w.shape[1]):
                w_sum += get2(w, i, c)
                w_squares += get2(w, i, c) ** 2
        var h_sum: Float64 = 0
        var h_squares: Float64 = 0
        for c in range(H.shape[0]):
            for j in range(H.shape[1]):
                h_sum += get2(H, c, j)
                h_squares += get2(H, c, j) ** 2
        var cost = (beta_divergence(X, w, H, self._beta) + regs[0] * w_sum + regs[1] * h_sum
            + regs[2] * w_squares + regs[3] * h_squares) / Float64(n)
        if update_H:
            var a = self._components_numerator.value().copy()
            var b = self._components_denominator.value().copy()
            H = mu_update_h_online(X, w, H, self._beta, regs[1], regs[3], self._gamma, a, b, self._rho)
            self._components_numerator = a^
            self._components_denominator = b^
            if self._beta <= 1:
                for c in range(H.shape[0]):
                    for j in range(H.shape[1]):
                        if get2(H, c, j) < EPS64: set2(H, c, j, 0)
        W = w^
        return cost

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        _ = self.fit_transform(X)
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        return self.fit_transform(X, None, None)

    def fit_transform(mut self, X: Self.Input, W: Optional[Self.Input], H: Optional[Self.Input]) raises -> Self.Transformed:
        """Fit over max_iter passes of cycling batches; returns W for all samples."""
        _validate_input(X, _beta_value(self.beta_loss), "MiniBatchNMF (input X)")
        var x = contiguous(X)
        var batch = self._check_params(x)
        var n = x.shape[0]
        var config = self._config()
        var k = self._k
        var factors = _check_w_h(config, x, W, H, True, k, "mu")
        self._k = k
        var w = factors[0].copy()
        var h = factors[1].copy()
        var h_buffer = h.copy()
        self._components_numerator = h.copy()
        var ones = zeros[Self.dtype](Shape(h.shape[0], h.shape[1]))
        for c in range(h.shape[0]):
            for j in range(h.shape[1]): set2(ones, c, j, 1)
        self._components_denominator = ones^
        var ewa: Float64 = 0
        var has_ewa = False
        var ewa_min: Float64 = 0
        var has_min = False
        var no_improvement = 0
        var steps_per_iter = (n + batch - 1) // batch
        var total_steps = self.max_iter * steps_per_iter
        var last = 0
        for i in range(total_steps):
            last = i
            var start = (i % steps_per_iter) * batch
            var stop = min(start + batch, n)
            var xb = _rows(x, start, stop)
            var wb: Optional[NDArray[Self.dtype]] = _rows(w, start, stop)
            var cost = self._minibatch_step(xb, wb, h, True)
            if not self.fresh_restarts:
                # The reference updates the W row view in place.
                for r in range(stop - start):
                    for c in range(self._k): set2(w, start + r, c, get2(wb.value(), r, c))
            # _minibatch_convergence
            var step = i + 1
            var converged = False
            if step > 1:
                var size = Float64(stop - start)
                if not has_ewa:
                    ewa = cost
                    has_ewa = True
                else:
                    var alpha = min(size / Float64(n + 1), 1.0)
                    ewa = ewa * (1 - alpha) + cost * alpha
                if self.tol > 0 and _frobenius_difference(h, h_buffer) <= self.tol:
                    converged = True
                else:
                    if not has_min or ewa < ewa_min:
                        no_improvement = 0
                        ewa_min = ewa
                        has_min = True
                    else:
                        no_improvement += 1
                    if self.max_no_improvement and no_improvement >= self.max_no_improvement.value():
                        converged = True
            if converged:
                break
            h_buffer = h.copy()
        if self.fresh_restarts:
            w = self._solve_w(x, h, self._transform_max_iter)
        var n_steps = last + 1
        self.reconstruction_err_ = beta_divergence(x, w, h, self._beta, square_root=True)
        self.n_components_ = h.shape[0]
        self.components_ = h^
        self.n_iter_ = (n_steps + steps_per_iter - 1) // steps_per_iter
        self.n_steps_ = n_steps
        self.n_features_in_ = x.shape[1]
        return w^

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        self.partial_fit(X, None, None)
        return self

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, W: Optional[Self.Input], H: Optional[Self.Input]) raises -> ref[self] Self:
        """One online step on X; the first call initializes components from X."""
        var x = contiguous(X)
        var h: NDArray[Self.dtype]
        if not self.components_:
            _validate_input(x, _beta_value(self.beta_loss), "MiniBatchNMF (input X)")
            _ = self._check_params(x)
            var k = self._k
            var factors = _check_w_h(self._config(), x, W, H, True, k, "mu")
            self._k = k
            h = factors[1].copy()
            self._components_numerator = h.copy()
            var ones = zeros[Self.dtype](Shape(h.shape[0], h.shape[1]))
            for c in range(h.shape[0]):
                for j in range(h.shape[1]): set2(ones, c, j, 1)
            self._components_denominator = ones^
            self.n_steps_ = 0
            self.n_features_in_ = x.shape[1]
        else:
            validate_matrix(x)
            if x.shape[1] != self.n_features_in_:
                raise Error("ValueError: X has " + String(x.shape[1]) + " features, but MiniBatchNMF is expecting "
                    + String(self.n_features_in_) + " features as input.")
            h = self.components_.value().copy()
        var none: Optional[NDArray[Self.dtype]] = None
        _ = self._minibatch_step(x, none, h, True)
        self.n_components_ = h.shape[0]
        self.components_ = h^
        self.n_steps_ += 1
        return self

    def _check(self) raises:
        if not self.components_:
            raise Error("NotFittedError: This MiniBatchNMF instance is not fitted yet. Call 'fit' first.")

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """Solve W for fixed components_ with up to transform_max_iter updates."""
        self._check()
        validate_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise Error("ValueError: X has " + String(X.shape[1]) + " features, but MiniBatchNMF is expecting "
                + String(self.n_features_in_) + " features as input.")
        return self._solve_w(contiguous(X), self.components_.value(), self._transform_max_iter)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        self._check()
        validate_matrix(X)
        if X.shape[1] != self.components_.value().shape[0]:
            raise Error("ValueError: inverse_transform expects " + String(self.components_.value().shape[0]) + " columns")
        return contiguous(X) @ self.components_.value()

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        self._check()
        return feature_names("minibatchnmf", self.n_components_, input_features, self.n_features_in_)

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        if transform and transform.value() != "default":
            raise Error("NotImplementedError: MiniBatchNMF supports native array output only")

    def get_metadata_routing(self) raises:
        raise Error("NotImplementedError: Python metadata routing is not available")
