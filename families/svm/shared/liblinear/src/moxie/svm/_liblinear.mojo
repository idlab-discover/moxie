"""Dense LIBLINEAR optimizer. Adapted from pinned sklearn: see LICENSES.

Private consumer boundary uses the family-owned options, models and arrays.
"""
import numojo as nm
from numojo.routines.sorting import quick_sort_inplace_1d
from std.math import abs, min, max, sqrt, isfinite, log, exp, inf
from moxie._dense import row_squared_norms
from moxie._random import RandomState
from moxie.svm._contracts import (
    Dense, Indices, LinearEngine, LinearModel, LinearOptions,
    validate_matrix, validate_vector,
)


def _bounded(mut rng: RandomState, bound: Int) -> Int:
    # sklearn newrand.h's Lemire mapping; reuse the shared MT19937 state.
    var width = UInt32(bound)
    var product = UInt64(rng.next_uint32()) * UInt64(width)
    var low = UInt32(product)
    if low < width:
        var threshold = (UInt32(0) - width) % width
        while low < threshold:
            product = UInt64(rng.next_uint32()) * UInt64(width)
            low = UInt32(product)
    return Int(product >> 32)


def _swap(mut order: Indices, i: Int, j: Int):
    var old = order.unsafe_get(i)
    order.unsafe_set(i, order.unsafe_get(j))
    order.unsafe_set(j, old)


def _shuffle(mut order: Indices, count: Int, mut rng: RandomState):
    for i in range(count):
        _swap(order, i, i + _bounded(rng, count - i))


def _row_value(x: Dense, row: Int, weights: Dense) raises -> Float64:
    var result = Float64(0)
    var p = x.shape[1]
    for j in range(p):
        result += weights.unsafe_get(j) * x.unsafe_get(row * p + j)
    return result


def _row_update(x: Dense, row: Int, amount: Float64, mut weights: Dense) raises:
    var p = x.shape[1]
    for j in range(p):
        weights.unsafe_set(j, weights.unsafe_get(j) + amount * x.unsafe_get(row * p + j))


def _svc_dual(x: Dense, y: Dense, costs: Dense, options: LinearOptions,
              mut rng: RandomState, mut weights: Dense) raises -> Int:
    var n = x.shape[0]
    var norms = row_squared_norms(x)
    validate_vector(norms, n)
    var alpha = nm.zeros[DType.float64](nm.Shape(n))
    var order = nm.zeros[DType.int64](nm.Shape(n))
    for i in range(n):
        order.unsafe_set(i, Int64(i))
    var active = n
    var pgmax_old = inf[DType.float64]()
    var pgmin_old = -inf[DType.float64]()
    var iteration = 0
    while iteration < options.max_iter:
        var pgmax = -inf[DType.float64]()
        var pgmin = inf[DType.float64]()
        _shuffle(order, active, rng)
        var s = 0
        while s < active:
            var i = Int(order.unsafe_get(s))
            var sign = y.unsafe_get(i)
            if options.solver == 1 and costs.unsafe_get(i) == 0:
                # C++'s infinite diagonal yields an inactive zero alpha here.
                pgmax = max(pgmax, Float64(0))
                pgmin = min(pgmin, Float64(0))
                s += 1
                continue
            var diagonal = Float64(0)
            var upper = costs.unsafe_get(i)
            if options.solver == 1:
                diagonal = 0.5 / upper
                upper = inf[DType.float64]()
            var a = alpha.unsafe_get(i)
            var gradient = _row_value(x, i, weights) * sign - 1 + a * diagonal
            var pg: Float64
            var shrink = False
            if a == 0:
                shrink = gradient > pgmax_old
                pg = min(gradient, Float64(0))
            elif a == upper:
                shrink = gradient < pgmin_old
                pg = max(gradient, Float64(0))
            else:
                pg = gradient
            if shrink:
                active -= 1
                _swap(order, s, active)
                continue
            pgmax = max(pgmax, pg)
            pgmin = min(pgmin, pg)
            if abs(pg) > 1e-12:
                var q = norms.unsafe_get(i) + diagonal
                # Zero feature vector: the hinge dual optimum is its bound.
                var updated = upper
                if q > 0:
                    updated = min(max(a - gradient / q, Float64(0)), upper)
                alpha.unsafe_set(i, updated)
                _row_update(x, i, (updated - a) * sign, weights)
            s += 1
        iteration += 1
        if pgmax - pgmin <= options.tol:
            if active == n:
                break
            active = n
            pgmax_old = inf[DType.float64]()
            pgmin_old = -inf[DType.float64]()
            continue
        pgmax_old = pgmax
        pgmin_old = pgmin
        if pgmax_old <= 0:
            pgmax_old = inf[DType.float64]()
        if pgmin_old >= 0:
            pgmin_old = -inf[DType.float64]()
    return iteration


def _svr_dual(x: Dense, y: Dense, costs: Dense, options: LinearOptions,
              mut rng: RandomState, mut weights: Dense) raises -> Int:
    var n = x.shape[0]
    var norms = row_squared_norms(x)
    validate_vector(norms, n)
    var beta = nm.zeros[DType.float64](nm.Shape(n))
    var order = nm.zeros[DType.int64](nm.Shape(n))
    for i in range(n):
        order.unsafe_set(i, Int64(i))
    var active = n
    var gmax_old = inf[DType.float64]()
    var initial = Float64(-1)
    var iteration = 0
    while iteration < options.max_iter:
        var gmax = Float64(0)
        var gnorm = Float64(0)
        _shuffle(order, active, rng)
        var s = 0
        while s < active:
            var i = Int(order.unsafe_get(s))
            var upper = costs.unsafe_get(i)
            var diagonal = Float64(0)
            if options.solver == 12:
                diagonal = 0.5 / upper
                upper = inf[DType.float64]()
            var b = beta.unsafe_get(i)
            var g = -y.unsafe_get(i) + diagonal * b
            for j in range(x.shape[1]):
                g += x.unsafe_get(i * x.shape[1] + j) * weights.unsafe_get(j)
            var h = norms.unsafe_get(i) + diagonal
            var gp = g + options.epsilon
            var gn = g - options.epsilon
            var violation = Float64(0)
            var shrink = False
            if b == 0:
                if gp < 0:
                    violation = -gp
                elif gn > 0:
                    violation = gn
                else:
                    shrink = gp > gmax_old and gn < -gmax_old
            elif b >= upper:
                violation = max(gp, Float64(0))
                shrink = gp < -gmax_old
            elif b <= -upper:
                violation = max(-gn, Float64(0))
                shrink = gn > gmax_old
            elif b > 0:
                violation = abs(gp)
            else:
                violation = abs(gn)
            if shrink:
                active -= 1
                _swap(order, s, active)
                continue
            gmax = max(gmax, violation)
            gnorm += violation
            var d = -b
            if h > 0:
                if gp < h * b:
                    d = -gp / h
                elif gn > h * b:
                    d = -gn / h
            elif gp < 0:
                d = upper - b
            elif gn > 0:
                d = -upper - b
            if abs(d) >= 1e-12:
                var updated = min(max(b + d, -upper), upper)
                beta.unsafe_set(i, updated)
                if updated != b:
                    _row_update(x, i, updated - b, weights)
            s += 1
        if iteration == 0:
            initial = gnorm
        iteration += 1
        if gnorm <= options.tol * initial:
            if active == n:
                break
            active = n
            gmax_old = inf[DType.float64]()
            continue
        gmax_old = gmax
    return iteration


def _lr_dual(x: Dense, y: Dense, costs: Dense, options: LinearOptions,
             mut rng: RandomState, mut weights: Dense) raises -> Int:
    var n = x.shape[0]
    var norms = row_squared_norms(x)
    validate_vector(norms, n)
    var alpha = nm.zeros[DType.float64](nm.Shape(2 * n))
    var order = nm.zeros[DType.int64](nm.Shape(n))
    for i in range(n):
        order.unsafe_set(i, Int64(i))
    var inner_tol = Float64(1e-2)
    var inner_min = min(Float64(1e-8), options.tol)
    for i in range(n):
        var a = min(0.001 * costs.unsafe_get(i), Float64(1e-8))
        alpha.unsafe_set(2 * i, a)
        alpha.unsafe_set(2 * i + 1, costs.unsafe_get(i) - a)
        _row_update(x, i, y.unsafe_get(i) * a, weights)
    var iteration = 0
    while iteration < options.max_iter:
        _shuffle(order, n, rng)
        var newton_iter = 0
        var gmax = Float64(0)
        for s in range(n):
            var i = Int(order.unsafe_get(s))
            var c = costs.unsafe_get(i)
            var a = norms.unsafe_get(i)
            var b = _row_value(x, i, weights) * y.unsafe_get(i)
            var first = 2 * i
            var second = first + 1
            var sign = Float64(1)
            if 0.5 * a * (alpha.unsafe_get(second) - alpha.unsafe_get(first)) + b < 0:
                first = 2 * i + 1
                second = 2 * i
                sign = -1
            var old = alpha.unsafe_get(first)
            var z = old
            if c - z < 0.5 * c:
                z *= 0.1
            var gp = a * (z - old) + sign * b + log(z / (c - z))
            gmax = max(gmax, abs(gp))
            var inner = 0
            while inner <= 100:
                if abs(gp) < inner_tol:
                    break
                var curvature = a + c / (c - z) / z
                var candidate = z - gp / curvature
                if candidate <= 0:
                    z *= 0.1
                else:
                    z = candidate
                gp = a * (z - old) + sign * b + log(z / (c - z))
                newton_iter += 1
                inner += 1
            if inner > 0:
                alpha.unsafe_set(first, z)
                alpha.unsafe_set(second, c - z)
                _row_update(x, i, sign * (z - old) * y.unsafe_get(i), weights)
        iteration += 1
        if gmax < options.tol:
            break
        if newton_iter <= n // 10:
            inner_tol = max(inner_min, 0.1 * inner_tol)
    return iteration


def _objective(x: Dense, y: Dense, costs: Dense, weights: Dense,
               solver: Int, epsilon: Float64) raises -> Float64:
    var value = Float64(0)
    for j in range(weights.shape[0]):
        value += weights.unsafe_get(j) * weights.unsafe_get(j)
    value *= 0.5
    for i in range(x.shape[0]):
        var score = _row_value(x, i, weights)
        var loss: Float64
        if solver == 0:
            var margin = y.unsafe_get(i) * score
            loss = log(1 + exp(-abs(margin))) + max(-margin, Float64(0))
        elif solver == 2:
            var residual = max(1 - y.unsafe_get(i) * score, Float64(0))
            loss = residual * residual
        else:
            var residual = max(abs(score - y.unsafe_get(i)) - epsilon, Float64(0))
            loss = residual * residual
        value += costs.unsafe_get(i) * loss
    return value


def _gradient(x: Dense, y: Dense, costs: Dense, weights: Dense,
              solver: Int, epsilon: Float64, mut g: Dense, mut curvature: Dense) raises:
    for j in range(weights.shape[0]):
        g.unsafe_set(j, weights.unsafe_get(j))
    for i in range(x.shape[0]):
        var score = _row_value(x, i, weights)
        var multiplier = Float64(0)
        var h = Float64(0)
        if solver == 0:
            var sigmoid = 1 / (1 + exp(-y.unsafe_get(i) * score))
            multiplier = costs.unsafe_get(i) * (sigmoid - 1) * y.unsafe_get(i)
            h = costs.unsafe_get(i) * sigmoid * (1 - sigmoid)
        elif solver == 2:
            var residual = 1 - y.unsafe_get(i) * score
            if residual > 0:
                multiplier = -2 * costs.unsafe_get(i) * y.unsafe_get(i) * residual
                h = 2 * costs.unsafe_get(i)
        else:
            var residual = score - y.unsafe_get(i)
            if residual > epsilon:
                multiplier = 2 * costs.unsafe_get(i) * (residual - epsilon)
                h = 2 * costs.unsafe_get(i)
            elif residual < -epsilon:
                multiplier = 2 * costs.unsafe_get(i) * (residual + epsilon)
                h = 2 * costs.unsafe_get(i)
        curvature.unsafe_set(i, h)
        _row_update(x, i, multiplier, g)


def _inner(first: Dense, second: Dense) raises -> Float64:
    # Private TRON scalar accumulation, not a general array dot operation.
    var result = Float64(0)
    for j in range(first.shape[0]):
        result += first.unsafe_get(j) * second.unsafe_get(j)
    return result


def _tron(x: Dense, y: Dense, costs: Dense, options: LinearOptions,
          mut weights: Dense) raises -> Int:
    var n = x.shape[0]
    var p = x.shape[1]
    var g = nm.zeros[DType.float64](nm.Shape(p))
    var curvature = nm.zeros[DType.float64](nm.Shape(n))
    var step = nm.zeros[DType.float64](nm.Shape(p))
    var residual = nm.zeros[DType.float64](nm.Shape(p))
    var direction = nm.zeros[DType.float64](nm.Shape(p))
    var hd = nm.zeros[DType.float64](nm.Shape(p))
    var candidate = nm.zeros[DType.float64](nm.Shape(p))
    var tolerance = options.tol
    if options.solver != 11:
        var positive = 0
        for i in range(n):
            if y.unsafe_get(i) > 0:
                positive += 1
        tolerance *= Float64(max(min(positive, n - positive), 1)) / Float64(n)
    var objective = _objective(x, y, costs, weights, options.solver, options.epsilon)
    _gradient(x, y, costs, weights, options.solver, options.epsilon, g, curvature)
    var initial_norm = sqrt(_inner(g, g))
    if not isfinite(initial_norm) or not isfinite(objective):
        raise Error("ValueError: numerical overflow in primal objective/gradient")
    var delta = initial_norm
    var iteration = 0
    while iteration < options.max_iter and sqrt(_inner(g, g)) > tolerance * initial_norm:
        for j in range(p):
            step.unsafe_set(j, 0)
            residual.unsafe_set(j, -g.unsafe_get(j))
            direction.unsafe_set(j, -g.unsafe_get(j))
        var cg_tol = 0.1 * sqrt(_inner(g, g))
        var rr = _inner(residual, residual)
        while sqrt(rr) > cg_tol:
            for j in range(p):
                hd.unsafe_set(j, direction.unsafe_get(j))
            for i in range(n):
                _row_update(x, i, curvature.unsafe_get(i) * _row_value(x, i, direction), hd)
            var alpha = rr / _inner(direction, hd)
            for j in range(p):
                step.unsafe_set(j, step.unsafe_get(j) + alpha * direction.unsafe_get(j))
            if sqrt(_inner(step, step)) > delta:
                for j in range(p):
                    step.unsafe_set(j, step.unsafe_get(j) - alpha * direction.unsafe_get(j))
                var sd = _inner(step, direction)
                var ss = _inner(step, step)
                var dd = _inner(direction, direction)
                var rad = sqrt(sd * sd + dd * (delta * delta - ss))
                if sd >= 0:
                    alpha = (delta * delta - ss) / (sd + rad)
                else:
                    alpha = (rad - sd) / dd
                for j in range(p):
                    step.unsafe_set(j, step.unsafe_get(j) + alpha * direction.unsafe_get(j))
                    residual.unsafe_set(j, residual.unsafe_get(j) - alpha * hd.unsafe_get(j))
                break
            for j in range(p):
                residual.unsafe_set(j, residual.unsafe_get(j) - alpha * hd.unsafe_get(j))
            var new_rr = _inner(residual, residual)
            var beta = new_rr / rr
            for j in range(p):
                direction.unsafe_set(j, beta * direction.unsafe_get(j) + residual.unsafe_get(j))
            rr = new_rr
        for j in range(p):
            candidate.unsafe_set(j, weights.unsafe_get(j) + step.unsafe_get(j))
        var gs = _inner(g, step)
        var predicted = -0.5 * (gs - _inner(step, residual))
        var next_objective = _objective(x, y, costs, candidate, options.solver, options.epsilon)
        var actual = objective - next_objective
        var snorm = sqrt(_inner(step, step))
        if iteration == 0:
            delta = min(delta, snorm)
        var alpha = Float64(4)
        if next_objective - objective - gs > 0:
            alpha = max(Float64(0.25), -0.5 * gs / (next_objective - objective - gs))
        if actual < 1e-4 * predicted:
            delta = min(max(alpha, Float64(0.25)) * snorm, 0.5 * delta)
        elif actual < 0.25 * predicted:
            delta = max(0.25 * delta, min(alpha * snorm, 0.5 * delta))
        elif actual < 0.75 * predicted:
            delta = max(0.25 * delta, min(alpha * snorm, 4 * delta))
        else:
            delta = max(delta, min(alpha * snorm, 4 * delta))
        if actual > 1e-4 * predicted:
            iteration += 1
            for j in range(p):
                weights.unsafe_set(j, candidate.unsafe_get(j))
            objective = next_objective
            _gradient(x, y, costs, weights, options.solver, options.epsilon, g, curvature)
            if sqrt(_inner(g, g)) <= tolerance * initial_norm:
                break
        if (abs(actual) <= 0 and predicted <= 0) or (abs(actual) <= 1e-12 * abs(objective) and abs(predicted) <= 1e-12 * abs(objective)):
            break
    return iteration


def _l1_violation(g: Float64, w: Float64) -> Float64:
    if w > 0:
        return abs(g + 1)
    if w < 0:
        return abs(g - 1)
    return max(max(-g - 1, g - 1), Float64(0))


def _l1_direction(g: Float64, h: Float64, w: Float64) -> Float64:
    if g + 1 < h * w:
        return -(g + 1) / h
    if g - 1 > h * w:
        return -(g - 1) / h
    return -w


def _l1_svc(x: Dense, y: Dense, costs: Dense, options: LinearOptions,
            mut rng: RandomState, mut weights: Dense) raises -> Int:
    var n = x.shape[0]
    var p = x.shape[1]
    var signed = nm.zeros[DType.float64](nm.Shape(n, p))
    var residual = nm.ones[DType.float64](nm.Shape(n))
    var norms = nm.zeros[DType.float64](nm.Shape(p))
    var order = nm.zeros[DType.int64](nm.Shape(p))
    var positive = 0
    for i in range(n):
        if y.unsafe_get(i) > 0:
            positive += 1
        for j in range(p):
            signed.unsafe_set(i * p + j, y.unsafe_get(i) * x.unsafe_get(i * p + j))
    for j in range(p):
        order.unsafe_set(j, Int64(j))
        var total = Float64(0)
        for i in range(n):
            var value = signed.unsafe_get(i * p + j)
            total += costs.unsafe_get(i) * value * value
        norms.unsafe_set(j, total)
    var tolerance = options.tol * Float64(max(min(positive, n - positive), 1)) / Float64(n)
    var active = p
    var gmax_old = inf[DType.float64]()
    var initial = Float64(-1)
    var iteration = 0
    while iteration < options.max_iter:
        var gmax = Float64(0)
        var gnorm = Float64(0)
        _shuffle(order, active, rng)
        var s = 0
        while s < active:
            var j = Int(order.unsafe_get(s))
            var g = Float64(0)
            var h = Float64(0)
            for i in range(n):
                if residual.unsafe_get(i) > 0:
                    var value = signed.unsafe_get(i * p + j)
                    var temp = costs.unsafe_get(i) * value
                    g -= temp * residual.unsafe_get(i)
                    h += temp * value
            g *= 2
            h = max(2 * h, Float64(1e-12))
            var w = weights.unsafe_get(j)
            var violation = _l1_violation(g, w)
            if w == 0 and g + 1 > gmax_old / Float64(n) and g - 1 < -gmax_old / Float64(n):
                active -= 1
                _swap(order, s, active)
                continue
            gmax = max(gmax, violation)
            gnorm += violation
            var d = _l1_direction(g, h, w)
            if abs(d) >= 1e-12:
                var delta = abs(w + d) - abs(w) + g * d
                var old_d = Float64(0)
                var old_loss = Float64(0)
                var accepted = False
                for line in range(20):
                    var diff = old_d - d
                    var condition = abs(w + d) - abs(w) - 0.01 * delta
                    if norms.unsafe_get(j) * d * d + g * d + condition <= 0:
                        for i in range(n):
                            residual.unsafe_set(i, residual.unsafe_get(i) + diff * signed.unsafe_get(i * p + j))
                        accepted = True
                        break
                    var new_loss = Float64(0)
                    for i in range(n):
                        var value = signed.unsafe_get(i * p + j)
                        # C++ traverses only nonzero entries of this column.
                        if value == 0:
                            continue
                        var b = residual.unsafe_get(i)
                        if line == 0 and b > 0:
                            old_loss += costs.unsafe_get(i) * b * b
                        b += diff * value
                        residual.unsafe_set(i, b)
                        if b > 0:
                            new_loss += costs.unsafe_get(i) * b * b
                    if condition + new_loss - old_loss <= 0:
                        accepted = True
                        break
                    old_d = d
                    d *= 0.5
                    delta *= 0.5
                weights.unsafe_set(j, w + d)
                if not accepted:
                    for i in range(n):
                        residual.unsafe_set(i, 1 - _row_value(signed, i, weights))
            s += 1
        if iteration == 0:
            initial = gnorm
        iteration += 1
        if gnorm <= tolerance * initial:
            if active == p:
                break
            active = p
            gmax_old = inf[DType.float64]()
            continue
        gmax_old = gmax
    return iteration


def _l1_lr(x: Dense, y: Dense, costs: Dense, options: LinearOptions,
           mut rng: RandomState, mut weights: Dense) raises -> Int:
    var n = x.shape[0]
    var p = x.shape[1]
    var order = nm.zeros[DType.int64](nm.Shape(p))
    var hdiag = nm.zeros[DType.float64](nm.Shape(p))
    var gradient = nm.zeros[DType.float64](nm.Shape(p))
    var candidate = nm.zeros[DType.float64](nm.Shape(p))
    var negative_sum = nm.zeros[DType.float64](nm.Shape(p))
    var xd = nm.zeros[DType.float64](nm.Shape(n))
    var exponential = nm.ones[DType.float64](nm.Shape(n))
    var next_exp = nm.ones[DType.float64](nm.Shape(n))
    var tau = nm.zeros[DType.float64](nm.Shape(n))
    var curvature = nm.zeros[DType.float64](nm.Shape(n))
    var positive = 0
    for i in range(n):
        if y.unsafe_get(i) > 0:
            positive += 1
        tau.unsafe_set(i, costs.unsafe_get(i) * 0.5)
        curvature.unsafe_set(i, costs.unsafe_get(i) * 0.25)
    for j in range(p):
        order.unsafe_set(j, Int64(j))
        var value = Float64(0)
        for i in range(n):
            if y.unsafe_get(i) < 0:
                value += costs.unsafe_get(i) * x.unsafe_get(i * p + j)
        negative_sum.unsafe_set(j, value)
    var tolerance = options.tol * Float64(max(min(positive, n - positive), 1)) / Float64(n)
    var initial = Float64(-1)
    var inner_tol = Float64(1)
    var gmax_old = inf[DType.float64]()
    var no_change = 0
    var iteration = 0
    var wnorm = Float64(0)
    while iteration < options.max_iter:
        var gmax = Float64(0)
        var gnorm = Float64(0)
        var active = p
        var s = 0
        while s < active:
            var j = Int(order.unsafe_get(s))
            var h = Float64(1e-12)
            var temp = Float64(0)
            for i in range(n):
                var value = x.unsafe_get(i * p + j)
                h += value * value * curvature.unsafe_get(i)
                temp += value * tau.unsafe_get(i)
            var g = -temp + negative_sum.unsafe_get(j)
            hdiag.unsafe_set(j, h)
            gradient.unsafe_set(j, g)
            var w = weights.unsafe_get(j)
            if w == 0 and g + 1 > gmax_old / Float64(n) and g - 1 < -gmax_old / Float64(n):
                active -= 1
                _swap(order, s, active)
                continue
            var violation = _l1_violation(g, w)
            gmax = max(gmax, violation)
            gnorm += violation
            s += 1
        if iteration == 0:
            initial = gnorm
        if gnorm <= tolerance * initial or no_change >= 10:
            break
        no_change += 1
        var qp_active = active
        var qp_max_old = inf[DType.float64]()
        var inner = 0
        for i in range(n):
            xd.unsafe_set(i, 0)
        while inner < 1000:
            var qp_max = Float64(0)
            var qp_norm = Float64(0)
            _shuffle(order, qp_active, rng)
            s = 0
            while s < qp_active:
                var j = Int(order.unsafe_get(s))
                var w = candidate.unsafe_get(j)
                var g = gradient.unsafe_get(j) + (w - weights.unsafe_get(j)) * 1e-12
                for i in range(n):
                    g += x.unsafe_get(i * p + j) * curvature.unsafe_get(i) * xd.unsafe_get(i)
                if w == 0 and g + 1 > qp_max_old / Float64(n) and g - 1 < -qp_max_old / Float64(n):
                    qp_active -= 1
                    _swap(order, s, qp_active)
                    continue
                var violation = _l1_violation(g, w)
                var z = _l1_direction(g, hdiag.unsafe_get(j), w)
                if abs(z) >= 1e-12:
                    z = min(max(z, Float64(-10)), Float64(10))
                    no_change = 0
                    qp_max = max(qp_max, violation)
                    qp_norm += violation
                    candidate.unsafe_set(j, w + z)
                    for i in range(n):
                        xd.unsafe_set(i, xd.unsafe_get(i) + x.unsafe_get(i * p + j) * z)
                s += 1
            inner += 1
            if qp_norm <= inner_tol * initial:
                if qp_active == active:
                    break
                qp_active = active
                qp_max_old = inf[DType.float64]()
                continue
            qp_max_old = qp_max
        var delta = Float64(0)
        var next_norm = Float64(0)
        for j in range(p):
            delta += gradient.unsafe_get(j) * (candidate.unsafe_get(j) - weights.unsafe_get(j))
            next_norm += abs(candidate.unsafe_get(j))
        delta += next_norm - wnorm
        var negative_xd = Float64(0)
        for i in range(n):
            if y.unsafe_get(i) < 0:
                negative_xd += costs.unsafe_get(i) * xd.unsafe_get(i)
        var accepted = False
        for _ in range(20):
            var condition = next_norm - wnorm + negative_xd - 0.01 * delta
            for i in range(n):
                var exp_xd = exp(xd.unsafe_get(i))
                var value = exponential.unsafe_get(i) * exp_xd
                next_exp.unsafe_set(i, value)
                condition += costs.unsafe_get(i) * log((1 + value) / (exp_xd + value))
            if condition <= 0:
                wnorm = next_norm
                for j in range(p):
                    weights.unsafe_set(j, candidate.unsafe_get(j))
                for i in range(n):
                    var value = next_exp.unsafe_get(i)
                    exponential.unsafe_set(i, value)
                    var temp = 1 / (1 + value)
                    tau.unsafe_set(i, costs.unsafe_get(i) * temp)
                    curvature.unsafe_set(i, costs.unsafe_get(i) * value * temp * temp)
                accepted = True
                break
            next_norm = 0
            for j in range(p):
                var value = (weights.unsafe_get(j) + candidate.unsafe_get(j)) * 0.5
                candidate.unsafe_set(j, value)
                next_norm += abs(value)
            delta *= 0.5
            negative_xd *= 0.5
            for i in range(n):
                xd.unsafe_set(i, xd.unsafe_get(i) * 0.5)
        if not accepted:
            for i in range(n):
                exponential.unsafe_set(i, exp(_row_value(x, i, weights)))
        if inner == 1:
            inner_tol *= 0.25
        iteration += 1
        gmax_old = gmax
    return iteration


def _cs_shrink(m: Int, yi: Int, a: Float64, c: Float64, g: Float64, minimum: Float64) -> Bool:
    var bound = Float64(0)
    if m == yi:
        bound = c
    return a == bound and g < minimum


def _crammer_singer(x: Dense, labels: Dense, sw: Dense, cw: Dense,
                    options: LinearOptions, p: Int) raises -> LinearModel:
    var n = x.shape[0]
    var q = x.shape[1]
    var k = cw.shape[0]
    var weights = nm.zeros[DType.float64](nm.Shape(q, k))
    var alpha = nm.zeros[DType.float64](nm.Shape(n, k))
    var class_index = nm.zeros[DType.int64](nm.Shape(n, k))
    var label_index = nm.zeros[DType.int64](nm.Shape(n))
    var active_classes = nm.zeros[DType.int64](nm.Shape(n))
    var order = nm.zeros[DType.int64](nm.Shape(n))
    var gradient = nm.zeros[DType.float64](nm.Shape(k))
    var b = nm.zeros[DType.float64](nm.Shape(k))
    var differences = nm.zeros[DType.float64](nm.Shape(k))
    var norms = row_squared_norms(x)
    validate_vector(norms, n)
    var rng = RandomState(options.seed)
    for i in range(n):
        order.unsafe_set(i, Int64(i))
        active_classes.unsafe_set(i, Int64(k))
        label_index.unsafe_set(i, Int64(labels.unsafe_get(i)))
        for m in range(k):
            class_index.unsafe_set(i * k + m, Int64(m))
    var active = n
    var shrink_tol = max(10 * options.tol, Float64(1))
    var all_active = True
    var iteration = 0
    # Pinned sklearn does not pass max_iter to this solver (default 100000).
    while iteration < 100000:
        var stopping = -inf[DType.float64]()
        _shuffle(order, active, rng)
        var s = 0
        while s < active:
            var i = Int(order.unsafe_get(s))
            var a = norms.unsafe_get(i)
            if a <= 0:
                s += 1
                continue
            var label = Int(labels.unsafe_get(i))
            var c = options.c * sw.unsafe_get(i) * cw.unsafe_get(label * Int(cw.strides[0]))
            if not isfinite(c) or c < 0:
                raise Error("ValueError: effective class cost must be nonnegative and finite")
            var count = Int(active_classes.unsafe_get(i))
            var yi = Int(label_index.unsafe_get(i))
            for m in range(count):
                var value = Float64(1)
                if m == yi:
                    value = 0
                var cls = Int(class_index.unsafe_get(i * k + m))
                for j in range(q):
                    value += weights.unsafe_get(j * k + cls) * x.unsafe_get(i * q + j)
                gradient.unsafe_set(m, value)
            var minimum = inf[DType.float64]()
            var maximum = -inf[DType.float64]()
            for m in range(count):
                var cls = Int(class_index.unsafe_get(i * k + m))
                var g = gradient.unsafe_get(m)
                if alpha.unsafe_get(i * k + cls) < 0:
                    minimum = min(minimum, g)
                maximum = max(maximum, g)
            if yi < count and alpha.unsafe_get(i * k + label) < c:
                minimum = min(minimum, gradient.unsafe_get(yi))
            var m = 0
            while m < count:
                var cls = Int(class_index.unsafe_get(i * k + m))
                if _cs_shrink(m, yi, alpha.unsafe_get(i * k + cls), c, gradient.unsafe_get(m), minimum):
                    count -= 1
                    while count > m:
                        var other = Int(class_index.unsafe_get(i * k + count))
                        if not _cs_shrink(count, yi, alpha.unsafe_get(i * k + other), c, gradient.unsafe_get(count), minimum):
                            _swap(class_index, i * k + m, i * k + count)
                            var old = gradient.unsafe_get(m)
                            gradient.unsafe_set(m, gradient.unsafe_get(count))
                            gradient.unsafe_set(count, old)
                            if yi == count:
                                yi = m
                            elif yi == m:
                                yi = count
                            break
                        count -= 1
                m += 1
            active_classes.unsafe_set(i, Int64(count))
            label_index.unsafe_set(i, Int64(yi))
            if count <= 1:
                active -= 1
                _swap(order, s, active)
                continue
            if maximum - minimum <= 1e-12:
                s += 1
                continue
            stopping = max(maximum - minimum, stopping)
            var d = nm.zeros[DType.float64](nm.Shape(count))
            for m in range(count):
                var cls = Int(class_index.unsafe_get(i * k + m))
                var value = gradient.unsafe_get(m) - a * alpha.unsafe_get(i * k + cls)
                b.unsafe_set(m, value)
                if m == yi:
                    value += a * c
                d.unsafe_set(m, value)
            # NuMojo supplies the small finite vector sort; traverse descending.
            quick_sort_inplace_1d(d)
            var sorted_d = d^
            var beta = sorted_d.unsafe_get(count - 1) - a * c
            var r = 1
            while r < count and beta < Float64(r) * sorted_d.unsafe_get(count - 1 - r):
                beta += sorted_d.unsafe_get(count - 1 - r)
                r += 1
            beta /= Float64(r)
            for m in range(count):
                var cls = Int(class_index.unsafe_get(i * k + m))
                var bound = Float64(0)
                if m == yi:
                    bound = c
                var updated = min(bound, (beta - b.unsafe_get(m)) / a)
                differences.unsafe_set(m, updated - alpha.unsafe_get(i * k + cls))
                alpha.unsafe_set(i * k + cls, updated)
            for j in range(q):
                var value = x.unsafe_get(i * q + j)
                for m in range(count):
                    var diff = differences.unsafe_get(m)
                    if abs(diff) >= 1e-12:
                        var cls = Int(class_index.unsafe_get(i * k + m))
                        weights.unsafe_set(j * k + cls, weights.unsafe_get(j * k + cls) + diff * value)
            s += 1
        iteration += 1
        if stopping < shrink_tol:
            if stopping < options.tol and all_active:
                break
            active = n
            for i in range(n):
                active_classes.unsafe_set(i, Int64(k))
            shrink_tol = max(shrink_tol / 2, options.tol)
            all_active = True
        else:
            all_active = False
    var coef = nm.zeros[DType.float64](nm.Shape(k, p))
    var intercept = nm.zeros[DType.float64](nm.Shape(k))
    var iterations = nm.zeros[DType.int64](nm.Shape(1))
    iterations.unsafe_set(0, Int64(iteration))
    for m in range(k):
        for j in range(p):
            coef.unsafe_set(m * p + j, weights.unsafe_get(j * k + m))
        if options.fit_intercept:
            intercept.unsafe_set(m, weights.unsafe_get(p * k + m) * options.intercept_scaling)
    validate_matrix(coef)
    validate_vector(intercept, k)
    return LinearModel(coef^, intercept^, iterations^, Int(iteration >= options.max_iter))


def _validate_options(options: LinearOptions) raises:
    if options.solver != 0 and options.solver != 1 and options.solver != 2 and options.solver != 3 and options.solver != 4 and options.solver != 5 and options.solver != 6 and options.solver != 7 and options.solver != 11 and options.solver != 12 and options.solver != 13:
        raise Error("NotImplementedError: solver ID is not implemented")
    if not isfinite(options.c) or options.c <= 0:
        raise Error("ValueError: C must be positive and finite")
    if not isfinite(options.tol) or options.tol <= 0:
        raise Error("ValueError: tol must be positive and finite")
    if not isfinite(options.epsilon) or options.epsilon < 0:
        raise Error("ValueError: epsilon must be nonnegative and finite")
    if options.max_iter < 0:
        raise Error("ValueError: max_iter must be nonnegative")
    if options.fit_intercept and (not isfinite(options.intercept_scaling) or options.intercept_scaling <= 0):
        raise Error("ValueError: intercept_scaling must be positive and finite")
    if options.seed < 0 or options.seed > 4294967295:
        raise Error("ValueError: seed must be uint32")


@fieldwise_init
struct LibLinear(LinearEngine):
    """Stateless engine; each solve owns its result and its optimization state."""
    def solve(self, x: Dense, y: Dense, sample_weight: Dense,
              class_weight: Dense, options: LinearOptions) raises -> LinearModel:
        _validate_options(options)
        validate_matrix(x)
        var n = x.shape[0]
        var p = x.shape[1]
        validate_vector(y, n)
        validate_vector(sample_weight, n)
        if class_weight.ndim != 1:
            raise Error("ValueError: class_weight must be a vector")
        validate_vector(class_weight, class_weight.shape[0])
        var regression = options.solver >= 11
        var classes = class_weight.shape[0]
        if not regression and classes < 2:
            raise Error("ValueError: classification needs at least two class weights")
        if regression and classes != 0:
            raise Error("ValueError: regression requires empty class_weight")
        var counts = nm.zeros[DType.int64](nm.Shape(max(classes, 1)))
        var kept = 0
        for i in range(n):
            var label = y.unsafe_get(i * Int(y.strides[0]))
            if not regression and (label < 0 or label >= Float64(classes) or label != Float64(Int(label))):
                raise Error("ValueError: y must contain contiguous class codes")
            if sample_weight.unsafe_get(i * Int(sample_weight.strides[0])) > 0:
                kept += 1
                if not regression:
                    var k = Int(label)
                    counts.unsafe_set(k, counts.unsafe_get(k) + 1)
        if kept == 0:
            raise Error("ValueError: no positive sample weights")
        var retained = 0
        for k in range(classes):
            if counts.unsafe_get(k) > 0:
                retained += 1
            if class_weight.unsafe_get(k * Int(class_weight.strides[0])) < 0:
                raise Error("ValueError: class weights must be nonnegative")
        var active_weights = nm.zeros[DType.float64](nm.Shape(retained))
        var class_code = 0
        for k in range(classes):
            if counts.unsafe_get(k) > 0:
                active_weights.unsafe_set(class_code, class_weight.unsafe_get(k * Int(class_weight.strides[0])))
                class_code += 1
        var augmented = p + Int(options.fit_intercept)
        var train = nm.zeros[DType.float64](nm.Shape(kept, augmented))
        var target = nm.zeros[DType.float64](nm.Shape(kept))
        var sw = nm.zeros[DType.float64](nm.Shape(kept))
        var destination = 0
        # Ascending class grouping is part of sklearn's seeded trajectory.
        class_code = 0
        for k in range(max(classes, 1)):
            if not regression and counts.unsafe_get(k) == 0:
                continue
            for i in range(n):
                var label = y.unsafe_get(i * Int(y.strides[0]))
                var weight = sample_weight.unsafe_get(i * Int(sample_weight.strides[0]))
                if weight <= 0 or (not regression and label != Float64(k)):
                    continue
                for j in range(p):
                    train.unsafe_set(destination * augmented + j, x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1])))
                if options.fit_intercept:
                    train.unsafe_set(destination * augmented + p, options.intercept_scaling)
                if regression:
                    target.unsafe_set(destination, label)
                else:
                    target.unsafe_set(destination, Float64(class_code))
                sw.unsafe_set(destination, weight)
                destination += 1
            class_code += 1
        classes = retained
        if options.solver == 4:
            return _crammer_singer(train, target, sw, active_weights, options, p)
        var rows = 1
        if classes > 2:
            rows = classes
        var coef = nm.zeros[DType.float64](nm.Shape(rows, p))
        var intercept = nm.zeros[DType.float64](nm.Shape(rows))
        var iterations = nm.zeros[DType.int64](nm.Shape(rows))
        var rng = RandomState(options.seed)
        var fit_status = 0
        for row in range(rows):
            var weights = nm.zeros[DType.float64](nm.Shape(augmented))
            var labels = nm.zeros[DType.float64](nm.Shape(kept))
            var costs = nm.zeros[DType.float64](nm.Shape(kept))
            for i in range(kept):
                var label = target.unsafe_get(i)
                var factor = Float64(1)
                if not regression:
                    var positive = row
                    if classes == 2:
                        positive = 1
                    if Int(label) == positive:
                        label = 1
                        factor = active_weights.unsafe_get(positive)
                    else:
                        label = -1
                        if classes == 2:
                            factor = active_weights.unsafe_get(0)
                labels.unsafe_set(i, label)
                var cost = options.c * factor * sw.unsafe_get(i)
                if not isfinite(cost) or cost < 0:
                    raise Error("ValueError: effective sample cost must be nonnegative and finite")
                costs.unsafe_set(i, cost)
            var count: Int
            if options.solver == 1 or options.solver == 3:
                count = _svc_dual(train, labels, costs, options, rng, weights)
            elif options.solver == 12 or options.solver == 13:
                count = _svr_dual(train, labels, costs, options, rng, weights)
            elif options.solver == 5:
                count = _l1_svc(train, labels, costs, options, rng, weights)
            elif options.solver == 6:
                count = _l1_lr(train, labels, costs, options, rng, weights)
            elif options.solver == 7:
                count = _lr_dual(train, labels, costs, options, rng, weights)
            else:
                count = _tron(train, labels, costs, options, weights)
            iterations.unsafe_set(row, Int64(count))
            if count >= options.max_iter:
                fit_status = 1
            for j in range(p):
                var value = weights.unsafe_get(j)
                if not isfinite(value):
                    raise Error("ValueError: solver produced nonfinite coefficients")
                coef.unsafe_set(row * p + j, value)
            if options.fit_intercept:
                intercept.unsafe_set(row, weights.unsafe_get(p) * options.intercept_scaling)
        validate_vector(intercept, rows)
        return LinearModel(coef^, intercept^, iterations^, fit_status)

    def decision_values(self, x: Dense, model: LinearModel) raises -> Dense:
        if model.coef.ndim != 2 or model.coef.shape[0] == 0:
            raise Error("ValueError: malformed linear model")
        validate_matrix(model.coef)
        validate_matrix(x, model.coef.shape[1])
        var rows = model.coef.shape[0]
        validate_vector(model.intercept, rows)
        var result = x @ model.coef.T()
        for i in range(x.shape[0]):
            for r in range(rows):
                result.unsafe_set(i * rows + r, result.unsafe_get(i * rows + r) + model.intercept.unsafe_get(r * Int(model.intercept.strides[0])))
        return result^
