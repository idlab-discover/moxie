"""Dense LIBSVM SMO, adapted from sklearn's BSD-licensed LIBSVM.

See LICENSES/LIBSVM.txt and LICENSES/scikit-learn.txt. Arrays retain the
reference Float64 optimizer / Float32 Q-column boundary.
"""
import numojo as nm
from std.math import abs, min, max, inf, isfinite, exp, log
from moxie._random import RandomState
from moxie.svm._cache import ColumnCache
from moxie.svm._kernels import prediction_kernel_matrix
from moxie.svm._contracts import Dense, Indices, KernelOptions, KernelModel, KernelSpec, KernelEngine, validate_matrix, validate_vector


def _z(n: Int) raises -> Dense:
    return nm.zeros[DType.float64](nm.Shape(n))

def _iz(n: Int) raises -> Indices:
    return nm.zeros[DType.int64](nm.Shape(n))

def _swap(mut a: Dense, i: Int, j: Int):
    var old = a.unsafe_get(i)
    a.unsafe_set(i, a.unsafe_get(j))
    a.unsafe_set(j, old)

def _iswap(mut a: Indices, i: Int, j: Int):
    var old = a.unsafe_get(i)
    a.unsafe_set(i, a.unsafe_get(j))
    a.unsafe_set(j, old)

@fieldwise_init
struct _Solution(Movable):
    var alpha: Dense
    var rho: Float64
    var r: Float64
    var iterations: Int
    var status: Int

struct _Solver(Movable):
    var cache: ColumnCache
    var rows: Indices
    var sign: Dense
    var alpha: Dense
    var cost: Dense
    var gradient: Dense
    var bar: Dense
    var linear: Dense
    var diagonal: Dense
    var order: Indices
    var active: Int
    var unshrunk: Bool
    var nu: Bool
    var tolerance: Float64

    def __init__(out self, x: Dense, rows: Indices, signs: Dense, alpha: Dense, cost: Dense, linear: Dense, options: KernelOptions) raises:
        self.cache = ColumnCache(x, options.kernel, options.cache_mb)
        self.rows = rows.copy()
        self.sign = signs.copy()
        self.alpha = alpha.copy()
        self.cost = cost.copy()
        self.linear = linear.copy()
        self.gradient = linear.copy()
        self.bar = _z(alpha.size)
        self.diagonal = _z(alpha.size)
        self.order = _iz(alpha.size)
        self.active = alpha.size
        self.unshrunk = False
        self.nu = options.formulation == "nu_svc" or options.formulation == "nu_svr"
        self.tolerance = options.tol
        for i in range(alpha.size):
            self.order.unsafe_set(i, Int64(i))
            var row = Int(rows.unsafe_get(i))
            self.diagonal.unsafe_set(i, self.cache.diagonal(row))
        for i in range(alpha.size):
            if not self.lower(i):
                var q = self.q(i, alpha.size)
                for j in range(alpha.size):
                    self.gradient.unsafe_set(j, self.gradient.unsafe_get(j) + alpha.unsafe_get(i) * q.unsafe_get(j))
                    if self.upper(i):
                        self.bar.unsafe_set(j, self.bar.unsafe_get(j) + cost.unsafe_get(i) * q.unsafe_get(j))

    def upper(self, i: Int) -> Bool:
        return self.alpha.unsafe_get(i) >= self.cost.unsafe_get(i)

    def lower(self, i: Int) -> Bool:
        return not self.upper(i) and self.alpha.unsafe_get(i) <= 0

    def free(self, i: Int) -> Bool:
        return not self.upper(i) and not self.lower(i)

    def q(mut self, i: Int, length: Int) raises -> Dense:
        var raw = self.cache.column(Int(self.rows.unsafe_get(i)))
        var outcol = _z(length)
        for j in range(length):
            outcol.unsafe_set(j, self.sign.unsafe_get(i) * self.sign.unsafe_get(j) * Float64(raw.unsafe_get(Int(self.rows.unsafe_get(j)))))
        return outcol^

    def swap(mut self, i: Int, j: Int):
        _iswap(self.rows, i, j)
        _iswap(self.order, i, j)
        _swap(self.sign, i, j)
        _swap(self.alpha, i, j)
        _swap(self.cost, i, j)
        _swap(self.gradient, i, j)
        _swap(self.bar, i, j)
        _swap(self.linear, i, j)
        _swap(self.diagonal, i, j)

    def reconstruct(mut self) raises:
        var n = self.alpha.size
        if self.active == n:
            return
        var free_count = 0
        for j in range(self.active, n):
            self.gradient.unsafe_set(j, self.bar.unsafe_get(j) + self.linear.unsafe_get(j))
        for j in range(self.active):
            free_count += Int(self.free(j))
        if free_count * n > 2 * self.active * (n - self.active):
            for i in range(self.active, n):
                var q = self.q(i, self.active)
                for j in range(self.active):
                    if self.free(j):
                        self.gradient.unsafe_set(i, self.gradient.unsafe_get(i) + self.alpha.unsafe_get(j) * q.unsafe_get(j))
        else:
            for i in range(self.active):
                if self.free(i):
                    var q = self.q(i, n)
                    for j in range(self.active, n):
                        self.gradient.unsafe_set(j, self.gradient.unsafe_get(j) + self.alpha.unsafe_get(i) * q.unsafe_get(j))

    def select(mut self) raises -> Tuple[Int, Int]:
        var gp = -inf[DType.float64]()
        var gn = -inf[DType.float64]()
        var gp2 = -inf[DType.float64]()
        var gn2 = -inf[DType.float64]()
        var ip = -1
        var inn = -1
        for i in range(self.active):
            var g = self.gradient.unsafe_get(i)
            if self.sign.unsafe_get(i) == 1:
                if not self.upper(i) and -g >= gp:
                    gp = -g
                    ip = i
            elif not self.lower(i):
                if self.nu:
                    if g >= gn:
                        gn = g
                        inn = i
                elif g >= gp:
                    gp = g
                    ip = i
        var qp = _z(self.active)
        var qn = _z(self.active)
        if ip >= 0:
            qp = self.q(ip, self.active)
        if inn >= 0:
            qn = self.q(inn, self.active)
        var best = inf[DType.float64]()
        var outi = -1
        var outj = -1
        for j in range(self.active):
            var g = self.gradient.unsafe_get(j)
            var positive = self.sign.unsafe_get(j) == 1
            if (positive and not self.lower(j)) or (not positive and not self.upper(j)):
                var i = ip
                var diff = gp + g if positive else gp - g
                var cross = qp.unsafe_get(j)
                if not positive and self.nu:
                    diff = gn - g
                    i = inn
                    cross = qn.unsafe_get(j)
                    gn2 = max(gn2, -g)
                else:
                    gp2 = max(gp2, g if positive else -g)
                if diff > 0 and i >= 0:
                    var quad = self.diagonal.unsafe_get(i) + self.diagonal.unsafe_get(j)
                    if self.nu:
                        quad -= 2 * cross
                    else:
                        quad -= 2 * self.sign.unsafe_get(i) * self.sign.unsafe_get(j) * cross
                    if quad <= 0:
                        quad = 1e-12
                    var change = -diff * diff / quad
                    if change <= best:
                        best = change
                        outi = i
                        outj = j
        var violation = max(gp + gp2, gn + gn2) if self.nu else gp + gp2
        if violation < self.tolerance:
            return (-1, -1)
        return (outi, outj)

    def shrunk(self, i: Int, g1: Float64, g2: Float64, g3: Float64, g4: Float64) -> Bool:
        var positive = self.sign.unsafe_get(i) == 1
        var g = self.gradient.unsafe_get(i)
        if self.upper(i):
            return -g > (g1 if positive else g4)
        if self.lower(i):
            return g > (g2 if positive else g3)
        return False

    def shrink(mut self) raises:
        var g1 = -inf[DType.float64]()
        var g2 = g1
        var g3 = g1
        var g4 = g1
        for i in range(self.active):
            var g = self.gradient.unsafe_get(i)
            if self.sign.unsafe_get(i) == 1:
                if not self.upper(i):
                    g1 = max(g1, -g)
                if not self.lower(i):
                    g2 = max(g2, g)
            else:
                if not self.upper(i):
                    g4 = max(g4, -g)
                if not self.lower(i):
                    g3 = max(g3, g)
        if not self.nu:
            g1 = max(g1, g3)
            g2 = max(g2, g4)
            g3 = g1
            g4 = g2
        if not self.unshrunk and max(g1 + g2, g3 + g4) <= self.tolerance * 10:
            self.unshrunk = True
            self.reconstruct()
            self.active = self.alpha.size
        var i = 0
        while i < self.active:
            if self.shrunk(i, g1, g2, g3, g4):
                self.active -= 1
                while self.active > i:
                    if not self.shrunk(self.active, g1, g2, g3, g4):
                        self.swap(i, self.active)
                        break
                    self.active -= 1
            i += 1

    def thresholds(self) -> Tuple[Float64, Float64]:
        var ub1 = inf[DType.float64]()
        var ub2 = ub1
        var lb1 = -ub1
        var lb2 = -ub1
        var sum1 = Float64(0)
        var sum2 = Float64(0)
        var count1 = 0
        var count2 = 0
        for i in range(self.active):
            var positive = self.sign.unsafe_get(i) == 1
            var g = self.gradient.unsafe_get(i)
            if self.nu:
                if positive:
                    if self.upper(i):
                        lb1 = max(lb1, g)
                    elif self.lower(i):
                        ub1 = min(ub1, g)
                    else:
                        sum1 += g
                        count1 += 1
                else:
                    if self.upper(i):
                        lb2 = max(lb2, g)
                    elif self.lower(i):
                        ub2 = min(ub2, g)
                    else:
                        sum2 += g
                        count2 += 1
            else:
                g *= self.sign.unsafe_get(i)
                if self.upper(i):
                    if positive:
                        lb1 = max(lb1, g)
                    else:
                        ub1 = min(ub1, g)
                elif self.lower(i):
                    if positive:
                        ub1 = min(ub1, g)
                    else:
                        lb1 = max(lb1, g)
                else:
                    sum1 += g
                    count1 += 1
        var r1 = sum1 / Float64(count1) if count1 > 0 else (ub1 + lb1) / 2
        if not self.nu:
            return (r1, Float64(0))
        var r2 = sum2 / Float64(count2) if count2 > 0 else (ub2 + lb2) / 2
        return ((r1 - r2) / 2, (r1 + r2) / 2)

    def run(mut self, options: KernelOptions) raises -> _Solution:
        var iteration = 0
        var status = 0
        var n = self.alpha.size
        var counter = min(n, 1000) + 1
        while True:
            if options.max_iter != -1 and iteration >= options.max_iter:
                status = 1
                break
            counter -= 1
            if counter == 0:
                counter = min(n, 1000)
                if options.shrinking:
                    self.shrink()
            var chosen = self.select()
            if chosen[0] < 0:
                self.reconstruct()
                self.active = n
                chosen = self.select()
                if chosen[0] < 0:
                    break
                counter = 1
            var i = chosen[0]
            var j = chosen[1]
            iteration += 1
            var qi = self.q(i, self.active)
            var qj = self.q(j, self.active)
            var ci = self.cost.unsafe_get(i)
            var cj = self.cost.unsafe_get(j)
            var ai = self.alpha.unsafe_get(i)
            var aj = self.alpha.unsafe_get(j)
            var oldi = ai
            var oldj = aj
            if self.sign.unsafe_get(i) != self.sign.unsafe_get(j):
                var quad = self.diagonal.unsafe_get(i) + self.diagonal.unsafe_get(j) + 2 * qi.unsafe_get(j)
                if quad <= 0:
                    quad = 1e-12
                var delta = (-self.gradient.unsafe_get(i) - self.gradient.unsafe_get(j)) / quad
                var diff = ai - aj
                ai += delta
                aj += delta
                if diff > 0:
                    if aj < 0:
                        aj = 0
                        ai = diff
                elif ai < 0:
                    ai = 0
                    aj = -diff
                if diff > ci - cj:
                    if ai > ci:
                        ai = ci
                        aj = ci - diff
                elif aj > cj:
                    aj = cj
                    ai = cj + diff
            else:
                var quad = self.diagonal.unsafe_get(i) + self.diagonal.unsafe_get(j) - 2 * qi.unsafe_get(j)
                if quad <= 0:
                    quad = 1e-12
                var delta = (self.gradient.unsafe_get(i) - self.gradient.unsafe_get(j)) / quad
                var total = ai + aj
                ai -= delta
                aj += delta
                if total > ci:
                    if ai > ci:
                        ai = ci
                        aj = total - ci
                elif aj < 0:
                    aj = 0
                    ai = total
                if total > cj:
                    if aj > cj:
                        aj = cj
                        ai = total - cj
                elif ai < 0:
                    ai = 0
                    aj = total
            var was_ui = self.upper(i)
            var was_uj = self.upper(j)
            self.alpha.unsafe_set(i, ai)
            self.alpha.unsafe_set(j, aj)
            for k in range(self.active):
                self.gradient.unsafe_set(k, self.gradient.unsafe_get(k) + (qi.unsafe_get(k) * (ai - oldi) + qj.unsafe_get(k) * (aj - oldj)))
            if was_ui != self.upper(i):
                qi = self.q(i, n)
                for k in range(n):
                    self.bar.unsafe_set(k, self.bar.unsafe_get(k) + (-ci if was_ui else ci) * qi.unsafe_get(k))
            if was_uj != self.upper(j):
                qj = self.q(j, n)
                for k in range(n):
                    self.bar.unsafe_set(k, self.bar.unsafe_get(k) + (-cj if was_uj else cj) * qj.unsafe_get(k))
        var threshold = self.thresholds()
        var alpha = _z(n)
        for i in range(n):
            alpha.unsafe_set(Int(self.order.unsafe_get(i)), self.alpha.unsafe_get(i))
        return _Solution(alpha^, threshold[0], threshold[1], iteration, status)


def _train(x: Dense, rows: Indices, labels: Dense, weights: Dense, options: KernelOptions, cp: Float64, cn: Float64) raises -> _Solution:
    var n = rows.size
    var regression = options.formulation == "epsilon_svr" or options.formulation == "nu_svr"
    var l = 2 * n if regression else n
    var mapping = _iz(l)
    var signs = _z(l)
    var alpha = _z(l)
    var cost = _z(l)
    var linear = _z(l)
    var mass = Float64(0)
    for i in range(n):
        var bound = weights.unsafe_get(i) * options.c if regression else weights.unsafe_get(i)
        mass += bound * options.nu
    var pos = mass / 2
    var neg = pos
    if regression:
        mass /= 2
    for i in range(n):
        mapping.unsafe_set(i, rows.unsafe_get(i))
        var w = weights.unsafe_get(i)
        var label = labels.unsafe_get(i)
        if regression:
            mapping.unsafe_set(i + n, rows.unsafe_get(i))
            signs.unsafe_set(i, 1)
            signs.unsafe_set(i + n, -1)
            cost.unsafe_set(i, w * options.c)
            cost.unsafe_set(i + n, w * options.c)
            var epsilon = options.epsilon if options.formulation == "epsilon_svr" else Float64(0)
            linear.unsafe_set(i, epsilon - label)
            linear.unsafe_set(i + n, epsilon + label)
            if options.formulation == "nu_svr":
                var a = min(mass, w * options.c)
                alpha.unsafe_set(i, a)
                alpha.unsafe_set(i + n, a)
                mass -= a
        elif options.formulation == "one_class":
            signs.unsafe_set(i, 1)
            cost.unsafe_set(i, w)
            var a = min(w, max(Float64(0), mass))
            alpha.unsafe_set(i, a)
            mass -= a
        else:
            var sign = Float64(1) if label > 0 else Float64(-1)
            signs.unsafe_set(i, sign)
            if options.formulation == "c_svc":
                linear.unsafe_set(i, -1)
                cost.unsafe_set(i, w * (cp if label > 0 else cn))
            else:
                cost.unsafe_set(i, w)
                if label > 0:
                    var a = min(w, pos)
                    alpha.unsafe_set(i, a)
                    pos -= a
                else:
                    var a = min(w, neg)
                    alpha.unsafe_set(i, a)
                    neg -= a
    var solver = _Solver(x, mapping, signs, alpha, cost, linear, options)
    var result = solver.run(options)
    var signed_alpha = _z(n)
    for i in range(n):
        var a = result.alpha.unsafe_get(i)
        if regression:
            a -= result.alpha.unsafe_get(i + n)
        elif options.formulation != "one_class":
            if options.formulation == "nu_svc":
                a *= signs.unsafe_get(i) / result.r
            else:
                a *= signs.unsafe_get(i)
        signed_alpha.unsafe_set(i, a)
    if options.formulation == "nu_svc":
        result.rho /= result.r
    return _Solution(signed_alpha^, result.rho, result.r, result.iterations, result.status)


def _bounded(mut rng: RandomState, bound: Int) -> Int:
    var width = UInt32(bound)
    var product = UInt64(rng.next_uint32()) * UInt64(width)
    var low = UInt32(product)
    if low < width:
        var threshold = (UInt32(0) - width) % width
        while low < threshold:
            product = UInt64(rng.next_uint32()) * UInt64(width)
            low = UInt32(product)
    return Int(product >> 32)


def _sigmoid(decisions: Dense, labels: Dense) raises -> Tuple[Float64, Float64]:
    var positives = Float64(0)
    for i in range(labels.size):
        positives += Float64(labels.unsafe_get(i) > 0)
    var negatives = Float64(labels.size) - positives
    var high = (positives + 1) / (positives + 2)
    var low = 1 / (negatives + 2)
    var a = Float64(0)
    var b = log((negatives + 1) / (positives + 1))
    var targets = _z(labels.size)
    var objective = Float64(0)
    for i in range(labels.size):
        var t = high if labels.unsafe_get(i) > 0 else low
        targets.unsafe_set(i, t)
        objective += t * b + log(1 + exp(-b)) if b >= 0 else (t - 1) * b + log(1 + exp(b))
    for _ in range(100):
        var h11 = Float64(1e-12)
        var h22 = Float64(1e-12)
        var h21 = Float64(0)
        var g1 = Float64(0)
        var g2 = Float64(0)
        for i in range(labels.size):
            var d = decisions.unsafe_get(i)
            var f = d * a + b
            var p: Float64
            var q: Float64
            if f >= 0:
                p = exp(-f) / (1 + exp(-f))
                q = 1 / (1 + exp(-f))
            else:
                p = 1 / (1 + exp(f))
                q = exp(f) / (1 + exp(f))
            var second = p * q
            h11 += d * d * second
            h22 += second
            h21 += d * second
            var first = targets.unsafe_get(i) - p
            g1 += d * first
            g2 += first
        if abs(g1) < 1e-5 and abs(g2) < 1e-5:
            break
        var det = h11 * h22 - h21 * h21
        var da = -(h22 * g1 - h21 * g2) / det
        var db = -(-h21 * g1 + h11 * g2) / det
        var gd = g1 * da + g2 * db
        var step = Float64(1)
        while step >= 1e-10:
            var na = a + step * da
            var nb = b + step * db
            var candidate = Float64(0)
            for i in range(labels.size):
                var f = decisions.unsafe_get(i) * na + nb
                var t = targets.unsafe_get(i)
                candidate += t * f + log(1 + exp(-f)) if f >= 0 else (t - 1) * f + log(1 + exp(f))
            if candidate < objective + 0.0001 * step * gd:
                a = na
                b = nb
                objective = candidate
                break
            step /= 2
        if step < 1e-10:
            break
    return (a, b)


def _calibrate(x: Dense, rows: Indices, labels: Dense, weights: Dense, options: KernelOptions, cp: Float64, cn: Float64, mut rng: RandomState, mut status: Int) raises -> Tuple[Float64, Float64]:
    var n = rows.size
    var order = _iz(n)
    for i in range(n):
        order.unsafe_set(i, Int64(i))
    for i in range(n):
        _iswap(order, i, i + _bounded(rng, n - i))
    var decisions = _z(n)
    var cache = ColumnCache(x, options.kernel, options.cache_mb)
    for fold in range(5):
        var begin = fold * n // 5
        var end = (fold + 1) * n // 5
        var count = n - (end - begin)
        var subrows = _iz(count)
        var sublabels = _z(count)
        var subweights = _z(count)
        var positives = 0
        var negatives = 0
        var cursor = 0
        # LIBSVM's sorted label grouping places -1 before +1 in CV submodels.
        for group in range(2):
            for t in range(n):
                if t >= begin and t < end:
                    continue
                var i = Int(order.unsafe_get(t))
                var positive = labels.unsafe_get(i) > 0
                if positive != (group == 1):
                    continue
                subrows.unsafe_set(cursor, rows.unsafe_get(i))
                sublabels.unsafe_set(cursor, labels.unsafe_get(i))
                subweights.unsafe_set(cursor, weights.unsafe_get(i))
                positives += Int(positive)
                negatives += Int(not positive)
                cursor += 1
        if positives == 0 or negatives == 0:
            var value = Float64(1) if positives > 0 else Float64(-1)
            if positives == 0 and negatives == 0:
                value = 0
            for t in range(begin, end):
                decisions.unsafe_set(Int(order.unsafe_get(t)), value)
        else:
            # Train in sorted-label orientation, then reverse decisions back to +1.
            for i in range(count):
                sublabels.unsafe_set(i, -sublabels.unsafe_get(i))
            # Each reference CV submodel enters svm_train and resets its seed.
            rng = RandomState(options.seed)
            var fitted = _train(x, subrows, sublabels, subweights, options, cn, cp)
            status |= fitted.status
            for t in range(begin, end):
                var i = Int(order.unsafe_get(t))
                var raw = cache.raw(Int(rows.unsafe_get(i)), prediction=True)
                var decision = Float64(0)
                for j in range(count):
                    decision += fitted.alpha.unsafe_get(j) * raw.unsafe_get(Int(subrows.unsafe_get(j)))
                decisions.unsafe_set(i, -(decision - fitted.rho))
    return _sigmoid(decisions, labels)


@fieldwise_init
struct LibSVM(KernelEngine):
    def solve(self, x: Dense, y: Dense, sample_weight: Dense, class_weight: Dense, options: KernelOptions) raises -> KernelModel:
        validate_matrix(x)
        validate_vector(y, x.shape[0])
        validate_vector(sample_weight, x.shape[0])
        options.kernel.validate()
        var classification = options.formulation == "c_svc" or options.formulation == "nu_svc"
        var regression = options.formulation == "epsilon_svr" or options.formulation == "nu_svr"
        if not classification and not regression and options.formulation != "one_class":
            raise Error("ValueError: unknown SVM formulation")
        if options.kernel.kind == "callable":
            raise Error("ValueError: callable kernels must be evaluated before solving")
        if options.kernel.kind == "precomputed" and x.shape[0] != x.shape[1]:
            raise Error("ValueError: precomputed training kernel must be square")
        if not isfinite(options.tol) or options.tol <= 0 or not isfinite(options.cache_mb) or options.cache_mb <= 0 or options.max_iter < -1:
            raise Error("ValueError: invalid tolerance, cache size, or iteration limit")
        if (options.formulation == "c_svc" or regression) and not (options.c > 0):
            raise Error("ValueError: C must be positive")
        if options.formulation != "c_svc" and options.formulation != "epsilon_svr" and not (options.nu > 0 and options.nu <= 1):
            raise Error("ValueError: nu must be in (0,1]")
        if options.formulation == "epsilon_svr" and not (options.epsilon >= 0 and isfinite(options.epsilon)):
            raise Error("ValueError: epsilon must be nonnegative and finite")
        if options.probability and not classification:
            raise Error("ValueError: probability calibration is classifier-only")
        var n = x.shape[0]
        var classes = class_weight.size if classification else 1
        if classification:
            if classes < 2:
                raise Error("ValueError: classification requires two or more classes")
            validate_vector(class_weight, classes)
        var surviving = 0
        var counts = _iz(classes)
        for i in range(n):
            if classification:
                var label = y.unsafe_get(i * Int(y.strides[0]))
                if label < 0 or label >= Float64(classes) or label != Float64(Int(label)):
                    raise Error("ValueError: classifier labels must be contiguous class codes")
            if sample_weight.unsafe_get(i * Int(sample_weight.strides[0])) > 0:
                surviving += 1
                var label = Int(y.unsafe_get(i * Int(y.strides[0]))) if classification else 0
                counts.unsafe_set(label, counts.unsafe_get(label) + 1)
        if options.formulation == "nu_svc":
            var mass = _z(classes)
            for i in range(n):
                var c = Int(y.unsafe_get(i * Int(y.strides[0])))
                mass.unsafe_set(c, mass.unsafe_get(c) + sample_weight.unsafe_get(i * Int(sample_weight.strides[0])))
            for c in range(classes):
                for d in range(c + 1, classes):
                    var first = mass.unsafe_get(c)
                    var second = mass.unsafe_get(d)
                    if options.nu * (first + second) / 2 > min(first, second):
                        raise Error("ValueError: specified nu is infeasible")
        if surviving == 0:
            raise Error("ValueError: all samples have nonpositive weights")
        for c in range(classes):
            if counts.unsafe_get(c) == 0:
                raise Error("ValueError: a class has no positive-weight samples")
        var rows = _iz(surviving)
        var labels = _z(surviving)
        var weights = _z(surviving)
        var cursor = 0
        for c in range(classes):
            for i in range(n):
                if sample_weight.unsafe_get(i * Int(sample_weight.strides[0])) <= 0:
                    continue
                if classification and Int(y.unsafe_get(i * Int(y.strides[0]))) != c:
                    continue
                rows.unsafe_set(cursor, Int64(i))
                labels.unsafe_set(cursor, y.unsafe_get(i * Int(y.strides[0])))
                weights.unsafe_set(cursor, sample_weight.unsafe_get(i * Int(sample_weight.strides[0])))
                cursor += 1
        var pairs = classes * (classes - 1) // 2 if classification else 1
        var solutions = List[_Solution]()
        var pair_rows = List[Indices]()
        var rho = _z(pairs)
        var iterations = _iz(pairs)
        var prob_a = _z(pairs if options.probability else 0)
        var prob_b = _z(pairs if options.probability else 0)
        var used = _iz(n)
        var status = 0
        var rng = RandomState(options.seed)
        var pair = 0
        for c in range(classes if classification else 1):
            for d in range(c + 1 if classification else 0, classes):
                var count = Int(counts.unsafe_get(c) + counts.unsafe_get(d)) if classification else surviving
                var selected = _iz(count)
                var targets = _z(count)
                var costs = _z(count)
                cursor = 0
                for i in range(surviving):
                    var label = Int(labels.unsafe_get(i))
                    if classification and label != c and label != d:
                        continue
                    selected.unsafe_set(cursor, rows.unsafe_get(i))
                    targets.unsafe_set(cursor, (Float64(1) if label == c else Float64(-1)) if classification else labels.unsafe_get(i))
                    costs.unsafe_set(cursor, weights.unsafe_get(i))
                    cursor += 1
                var cp = options.c * class_weight.unsafe_get(c) if classification else options.c
                var cn = options.c * class_weight.unsafe_get(d) if classification else options.c
                if options.probability:
                    var calibration = _calibrate(x, selected, targets, costs, options, cp, cn, rng, status)
                    prob_a.unsafe_set(pair, calibration[0])
                    prob_b.unsafe_set(pair, calibration[1])
                var fitted = _train(x, selected, targets, costs, options, cp, cn)
                status |= fitted.status
                rho.unsafe_set(pair, fitted.rho)
                iterations.unsafe_set(pair, Int64(fitted.iterations))
                for i in range(count):
                    if abs(fitted.alpha.unsafe_get(i)) > 0:
                        used.unsafe_set(Int(selected.unsafe_get(i)), 1)
                pair_rows.append(selected^)
                solutions.append(fitted^)
                pair += 1
        var support_count = 0
        var n_support = _iz(classes)
        for i in range(surviving):
            var row = Int(rows.unsafe_get(i))
            if used.unsafe_get(row) != 0:
                support_count += 1
                var c = Int(labels.unsafe_get(i)) if classification else 0
                n_support.unsafe_set(c, n_support.unsafe_get(c) + 1)
        var support = _iz(support_count)
        var positions = _iz(n)
        var precomputed = options.kernel.kind == "precomputed"
        var vectors = nm.zeros[DType.float64](nm.Shape(0 if precomputed else support_count, 0 if precomputed else x.shape[1]))
        cursor = 0
        for i in range(surviving):
            var row = Int(rows.unsafe_get(i))
            if used.unsafe_get(row) == 0:
                continue
            support.unsafe_set(cursor, Int64(row))
            positions.unsafe_set(row, Int64(cursor))
            if not precomputed:
                for j in range(x.shape[1]):
                    vectors.unsafe_set(cursor * x.shape[1] + j, x.unsafe_get(row * Int(x.strides[0]) + j * Int(x.strides[1])))
            cursor += 1
        var dual = nm.zeros[DType.float64](nm.Shape(classes - 1 if classification else 1, support_count))
        pair = 0
        for c in range(classes if classification else 1):
            for d in range(c + 1 if classification else 0, classes):
                for i in range(pair_rows[pair].size):
                    var row = Int(pair_rows[pair].unsafe_get(i))
                    if used.unsafe_get(row) == 0:
                        continue
                    var dual_row = (d - 1 if Int(y.unsafe_get(row * Int(y.strides[0]))) == c else c) if classification else 0
                    dual.unsafe_set(dual_row * support_count + Int(positions.unsafe_get(row)), solutions[pair].alpha.unsafe_get(i))
                pair += 1
        for i in range(rho.size):
            if not isfinite(rho.unsafe_get(i)):
                raise Error("ValueError: dual coefficients or intercepts are not finite")
        for i in range(dual.size):
            if not isfinite(dual.unsafe_get(i)):
                raise Error("ValueError: dual coefficients or intercepts are not finite")
        return KernelModel(support^, vectors^, n_support^, dual^, rho^, iterations^, status, prob_a^, prob_b^)

    def decision_values(self, x: Dense, model: KernelModel, kernel: KernelSpec) raises -> Dense:
        validate_matrix(x)
        var n = x.shape[0]
        var s = model.support.size
        var k = model.n_support.size
        var q = model.rho.size
        var values = nm.zeros[DType.float64](nm.Shape(n, s))
        if kernel.kind == "precomputed":
            for i in range(n):
                for j in range(s):
                    var col = Int(model.support.unsafe_get(j))
                    if col >= x.shape[1]:
                        raise Error("ValueError: insufficient precomputed columns")
                    values.unsafe_set(i * s + j, x.unsafe_get(i * Int(x.strides[0]) + col * Int(x.strides[1])))
        elif s > 0:
            values = prediction_kernel_matrix(x, model.support_vectors, kernel)
        var result = nm.zeros[DType.float64](nm.Shape(n, q))
        for row in range(n):
            if k == 1:
                var value = Float64(0)
                for t in range(s):
                    value += model.dual_coef.unsafe_get(t) * values.unsafe_get(row * s + t)
                result.unsafe_set(row, value - model.rho.unsafe_get(0))
            else:
                var pair = 0
                var start_i = 0
                for i in range(k):
                    var ci = Int(model.n_support.unsafe_get(i))
                    var start_j = start_i + ci
                    for j in range(i + 1, k):
                        var cj = Int(model.n_support.unsafe_get(j))
                        var value = Float64(0)
                        for t in range(ci):
                            value += model.dual_coef.unsafe_get((j - 1) * s + start_i + t) * values.unsafe_get(row * s + start_i + t)
                        for t in range(cj):
                            value += model.dual_coef.unsafe_get(i * s + start_j + t) * values.unsafe_get(row * s + start_j + t)
                        result.unsafe_set(row * q + pair, value - model.rho.unsafe_get(pair))
                        start_j += cj
                        pair += 1
                    start_i += ci
        return result^

    def class_probabilities(self, decisions: Dense, model: KernelModel) raises -> Dense:
        var k = model.n_support.size
        var pairs = k * (k - 1) // 2
        if k < 2 or model.prob_a.size != pairs or model.prob_b.size != pairs:
            raise Error("ValueError: classifier probability calibration is absent")
        validate_matrix(decisions, pairs)
        var result = nm.zeros[DType.float64](nm.Shape(decisions.shape[0], k))
        for row in range(decisions.shape[0]):
            var r = _z(k * k)
            var pair = 0
            for i in range(k):
                for j in range(i + 1, k):
                    var f = decisions.unsafe_get(row * Int(decisions.strides[0]) + pair * Int(decisions.strides[1])) * model.prob_a.unsafe_get(pair) + model.prob_b.unsafe_get(pair)
                    var value = exp(-f) / (1 + exp(-f)) if f >= 0 else 1 / (1 + exp(f))
                    value = min(max(value, Float64(1e-7)), Float64(1 - 1e-7))
                    r.unsafe_set(i * k + j, value)
                    r.unsafe_set(j * k + i, 1 - value)
                    pair += 1
            var p = _z(k)
            var matrix = _z(k * k)
            var qp = _z(k)
            for t in range(k):
                p.unsafe_set(t, 1 / Float64(k))
                for j in range(k):
                    if t != j:
                        var value = r.unsafe_get(j * k + t)
                        matrix.unsafe_set(t * k + t, matrix.unsafe_get(t * k + t) + value * value)
                        matrix.unsafe_set(t * k + j, -value * r.unsafe_get(t * k + j))
            for _ in range(max(100, k)):
                var pqp = Float64(0)
                for t in range(k):
                    var value = Float64(0)
                    for j in range(k):
                        value += matrix.unsafe_get(t * k + j) * p.unsafe_get(j)
                    qp.unsafe_set(t, value)
                    pqp += p.unsafe_get(t) * value
                var error = Float64(0)
                for t in range(k):
                    error = max(error, abs(qp.unsafe_get(t) - pqp))
                if error < 0.005 / Float64(k):
                    break
                for t in range(k):
                    var diff = (-qp.unsafe_get(t) + pqp) / matrix.unsafe_get(t * k + t)
                    p.unsafe_set(t, p.unsafe_get(t) + diff)
                    pqp = (pqp + diff * (diff * matrix.unsafe_get(t * k + t) + 2 * qp.unsafe_get(t))) / (1 + diff) / (1 + diff)
                    for j in range(k):
                        qp.unsafe_set(j, (qp.unsafe_get(j) + diff * matrix.unsafe_get(t * k + j)) / (1 + diff))
                        p.unsafe_set(j, p.unsafe_get(j) / (1 + diff))
            for t in range(k):
                result.unsafe_set(row * k + t, p.unsafe_get(t))
        return result^
