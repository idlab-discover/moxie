# Adapted from scikit-learn sklearn/linear_model/_cd_fast.pyx
# (enet_coordinate_descent_gram, gap_enet_gram, dual_gap_formulation_A);
# authors: scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Gram-form elastic-net coordinate descent with gap-safe screening (Float64).

Minimizes (1/2) wᵀQw - qᵀw + alpha |w|_1 + (beta/2)|w|² + (1/2) yᵀy by cyclic
coordinate descent, as sklearn's `enet_coordinate_descent_gram` with
random=False and positive=False. Candidate for consolidation with the
linear_model coordinate engine; it stays here until that engine exists.
"""
import numojo as nm
from numojo import NDArray, Shape
from std.math import sqrt, abs
from std.sys import stderr

comptime _EPS64: Float64 = 2.220446049250313e-16


@fieldwise_init
struct EnetResult(Copyable, Movable):
    var w: NDArray[DType.float64]
    var gap: Float64
    var tol: Float64
    var n_iter: Int


def _dot(a: NDArray[DType.float64], b: NDArray[DType.float64], n: Int) -> Float64:
    var total: Float64 = 0
    for k in range(n):
        total += a.unsafe_get(k) * b.unsafe_get(k)
    return total


def _axpy_row(mut target: NDArray[DType.float64], scale: Float64, Q: NDArray[DType.float64], row: Int, n: Int):
    """target += scale * Q[row, :]."""
    for k in range(n):
        target.unsafe_set(k, target.unsafe_get(k) + scale * Q.unsafe_get(row * n + k))


def _gap(n: Int, w: NDArray[DType.float64], alpha: Float64, beta: Float64, Qw: NDArray[DType.float64],
         q: NDArray[DType.float64], y_norm2: Float64, mut XtA: NDArray[DType.float64], gap_smaller_eps: Bool) -> Tuple[Float64, Float64]:
    """(gap, dual_norm_XtA) exactly as sklearn `gap_enet_gram` (positive=False)."""
    var w_l2_norm2: Float64 = 0
    if beta > 0:
        w_l2_norm2 = _dot(w, w, n)
    var q_dot_w = _dot(w, q, n)
    var wQw = _dot(w, Qw, n)
    var R_norm2 = y_norm2 + wQw - 2.0 * q_dot_w
    var Ry: Float64 = y_norm2 - q_dot_w
    if alpha == 0:
        for j in range(n):
            XtA.unsafe_set(j, q.unsafe_get(j) - Qw.unsafe_get(j))
        var dual_norm = _dot(XtA, XtA, n)
        if beta == 0:
            return (dual_norm, dual_norm)
        var primal_b = 0.5 * (R_norm2 + beta * w_l2_norm2)
        var dual_b = -0.5 * R_norm2 + Ry - 1.0 / (2.0 * beta) * dual_norm
        var gap_b = primal_b - dual_b
        if gap_smaller_eps and abs(gap_b) <= 2.0 * _EPS64 * primal_b:
            gap_b = 0.0
        return (gap_b, dual_norm)
    var dual_norm_XtA: Float64 = 0
    var w_l1_norm: Float64 = 0
    for j in range(n):
        var value = q.unsafe_get(j) - Qw.unsafe_get(j) - beta * w.unsafe_get(j)
        XtA.unsafe_set(j, value)
        dual_norm_XtA = max(dual_norm_XtA, abs(value))
        w_l1_norm += abs(w.unsafe_get(j))
    var primal = 0.5 * (R_norm2 + beta * w_l2_norm2) + alpha * w_l1_norm
    var scale: Float64 = alpha / dual_norm_XtA if dual_norm_XtA > alpha else 1.0
    var dual = -0.5 * (scale * scale) * (R_norm2 + beta * w_l2_norm2) + scale * Ry
    var gap = primal - dual
    if gap_smaller_eps and abs(gap) <= 2.0 * _EPS64 * primal:
        gap = 0.0
    return (gap, dual_norm_XtA)


def enet_coordinate_descent_gram(var w: NDArray[DType.float64], alpha: Float64, beta: Float64, Q: NDArray[DType.float64],
                                 q: NDArray[DType.float64], y: NDArray[DType.float64], max_iter: Int, tolerance: Float64,
                                 do_screening: Bool = True, early_stopping: Bool = True, *, emit_warnings: Bool = True) raises -> EnetResult:
    """Cyclic Gram coordinate descent; Q is contiguous (n, n), w/q/y contiguous (n,)."""
    var n = Q.shape[0]
    var XtA = nm.zeros[DType.float64](Shape(n))
    var y_norm2 = _dot(y, y, y.size)
    var screening = do_screening and alpha != 0
    var active = List[Int]()
    var excluded = List[Bool]()
    for _ in range(n):
        excluded.append(False)
    var Qw = nm.zeros[DType.float64](Shape(n))
    for a in range(n):
        var total: Float64 = 0
        for b in range(n):
            total += Q.unsafe_get(a * n + b) * w.unsafe_get(b)
        Qw.unsafe_set(a, total)
    var tol = tolerance * y_norm2
    var result = _gap(n, w, alpha, beta, Qw, q, y_norm2, XtA, False)
    var gap = result[0]
    var dual_norm_XtA = result[1]
    if early_stopping and gap >= 0 and gap <= tol:
        return EnetResult(w^, gap, tol, 0)
    var n_active = n
    if screening:
        var radius = sqrt(2.0 * abs(gap)) / alpha
        n_active = 0
        for j in range(n):
            var qjj = Q.unsafe_get(j * n + j)
            if qjj == 0:
                w.unsafe_set(j, 0)
                excluded[j] = True
                continue
            var theta = XtA.unsafe_get(j) / max(alpha, dual_norm_XtA)
            var d_j = (1.0 - abs(theta)) / sqrt(qjj + beta)
            if d_j <= radius:
                active.append(j)
                excluded[j] = False
                n_active += 1
            else:
                if w.unsafe_get(j) != 0:
                    _axpy_row(Qw, -w.unsafe_get(j), Q, j, n)
                    w.unsafe_set(j, 0)
                excluded[j] = True
    var n_iter = 0
    var converged = False
    for iteration in range(max_iter):
        n_iter = iteration
        var w_max: Float64 = 0
        var d_w_max: Float64 = 0
        for f in range(n_active):
            var j = active[f] if screening else f
            var qjj = Q.unsafe_get(j * n + j)
            if qjj == 0.0:
                continue
            var w_j = w.unsafe_get(j)
            var tmp = q.unsafe_get(j) - Qw.unsafe_get(j) + w_j * qjj
            var sign: Float64 = 0.0
            if tmp > 0:
                sign = 1.0
            elif tmp < 0:
                sign = -1.0
            var updated = sign * max(abs(tmp) - alpha, 0.0) / (qjj + beta)
            w.unsafe_set(j, updated)
            if updated != w_j:
                _axpy_row(Qw, updated - w_j, Q, j, n)
            d_w_max = max(d_w_max, abs(updated - w_j))
            w_max = max(w_max, abs(updated))
        if w_max == 0.0 or d_w_max / w_max <= tolerance or iteration == max_iter - 1 or n_active <= 1:
            result = _gap(n, w, alpha, beta, Qw, q, y_norm2, XtA, True)
            gap = result[0]
            dual_norm_XtA = result[1]
            if gap <= tol:
                converged = True
                break
            if screening:
                var radius = sqrt(2.0 * abs(gap)) / alpha
                var still = List[Int]()
                for j in range(n):
                    if excluded[j]:
                        continue
                    var theta = XtA.unsafe_get(j) / max(alpha, dual_norm_XtA)
                    var d_j = (1.0 - abs(theta)) / sqrt(Q.unsafe_get(j * n + j) + beta)
                    if d_j <= radius:
                        still.append(j)
                    else:
                        if w.unsafe_get(j) != 0:
                            _axpy_row(Qw, -w.unsafe_get(j), Q, j, n)
                            w.unsafe_set(j, 0)
                        excluded[j] = True
                active = still^
                n_active = len(active)
    if emit_warnings and not converged:
        print("ConvergenceWarning: Objective did not converge. Duality gap:", gap,
              "tolerance:", tol, file=stderr)
    return EnetResult(w^, gap, tol, n_iter + 1)
