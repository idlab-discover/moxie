# Port of SciPy `scipy/optimize/_optimize.py` (`bracket`, `Brent.optimize`,
# `_minimize_scalar_bounded`) from SciPy 1.18.1.
# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
# SPDX-License-Identifier: BSD-3-Clause (see LICENSES/scipy.txt)
"""Scalar minimizers used by PowerTransformer's lambda estimation.

Family-private: SciPy's `optimize.brent` (bracket search plus Brent's
parabolic/golden-section method) for Box-Cox and `optimize.fminbound` for
Yeo-Johnson, with SciPy's constants, update order and termination tests so
the iterates follow the same path. Objectives implement `ScalarObjective`.
"""
from std.math import sqrt, isnan, isfinite, abs, inf


trait ScalarObjective:
    def __call__(mut self, x: Float64) raises -> Float64:
        ...


@always_inline
def _sign(x: Float64) -> Float64:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return x


@fieldwise_init
struct Bracket(Copyable, Movable):
    var xa: Float64
    var xb: Float64
    var xc: Float64
    var fa: Float64
    var fb: Float64
    var fc: Float64


def bracket[O: ScalarObjective](mut func: O, xa_start: Float64, xb_start: Float64, grow_limit: Float64 = 110.0, maxiter: Int = 1000) raises -> Bracket:
    """`scipy.optimize.bracket`; raises SciPy's BracketError as RuntimeError."""
    comptime _gold = 1.618034
    comptime _verysmall_num = 1e-21
    var xa = xa_start
    var xb = xb_start
    var fa = func(xa)
    var fb = func(xb)
    if fa < fb:
        var t = xa
        xa = xb
        xb = t
        var ft = fa
        fa = fb
        fb = ft
    var xc = xb + _gold * (xb - xa)
    var fc = func(xc)
    var iter = 0
    while fc < fb:
        var tmp1 = (xb - xa) * (fb - fc)
        var tmp2 = (xb - xc) * (fb - fa)
        var val = tmp2 - tmp1
        var denom: Float64
        if abs(val) < _verysmall_num:
            denom = 2.0 * _verysmall_num
        else:
            denom = 2.0 * val
        var w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        var wlim = xb + grow_limit * (xc - xb)
        if iter > maxiter:
            raise Error("RuntimeError: No valid bracket was found before the iteration limit was reached. Consider trying different initial points or increasing `maxiter`.")
        iter += 1
        var fw: Float64
        if (w - xc) * (xb - w) > 0.0:
            fw = func(w)
            if fw < fc:
                xa = xb
                xb = w
                fa = fb
                fb = fw
                break
            elif fw > fb:
                xc = w
                fc = fw
                break
            w = xc + _gold * (xc - xb)
            fw = func(w)
        elif (w - wlim) * (wlim - xc) >= 0.0:
            w = wlim
            fw = func(w)
        elif (w - wlim) * (xc - w) > 0.0:
            fw = func(w)
            if fw < fc:
                xb = xc
                xc = w
                w = xc + _gold * (xc - xb)
                fb = fc
                fc = fw
                fw = func(w)
        else:
            w = xc + _gold * (xc - xb)
            fw = func(w)
        xa = xb
        xb = xc
        xc = w
        fa = fb
        fb = fc
        fc = fw
    var cond1 = (fb < fc and fb <= fa) or (fb < fa and fb <= fc)
    var cond2 = (xa < xb and xb < xc) or (xc < xb and xb < xa)
    var cond3 = isfinite(xa) and isfinite(xb) and isfinite(xc)
    if not (cond1 and cond2 and cond3):
        raise Error("RuntimeError: The algorithm terminated without finding a valid bracket. Consider trying different initial points.")
    return Bracket(xa, xb, xc, fa, fb, fc)


def brent[O: ScalarObjective](mut func: O, xa_start: Float64, xb_start: Float64, tol: Float64 = 1.48e-8, maxiter: Int = 500) raises -> Float64:
    """`scipy.optimize.brent(func, brack=(xa, xb), tol, maxiter)`: the minimizer."""
    var br = bracket(func, xa_start, xb_start)
    comptime _mintol = 1.0e-11
    comptime _cg = 0.3819660
    var x = br.xb
    var w = br.xb
    var v = br.xb
    var fw = br.fb
    var fv = br.fb
    var fx = br.fb
    var a: Float64
    var b: Float64
    if br.xa < br.xc:
        a = br.xa
        b = br.xc
    else:
        a = br.xc
        b = br.xa
    var deltax = 0.0
    var rat = 0.0
    var iter = 0
    while iter < maxiter:
        var tol1 = tol * abs(x) + _mintol
        var tol2 = 2.0 * tol1
        var xmid = 0.5 * (a + b)
        if abs(x - xmid) < (tol2 - 0.5 * (b - a)):
            break
        if abs(deltax) <= tol1:
            if x >= xmid:
                deltax = a - x
            else:
                deltax = b - x
            rat = _cg * deltax
        else:
            var tmp1 = (x - w) * (fx - fv)
            var tmp2 = (x - v) * (fx - fw)
            var p = (x - v) * tmp2 - (x - w) * tmp1
            tmp2 = 2.0 * (tmp2 - tmp1)
            if tmp2 > 0.0:
                p = -p
            tmp2 = abs(tmp2)
            var dx_temp = deltax
            deltax = rat
            if (p > tmp2 * (a - x)) and (p < tmp2 * (b - x)) and (abs(p) < abs(0.5 * tmp2 * dx_temp)):
                rat = p * 1.0 / tmp2
                var u = x + rat
                if (u - a) < tol2 or (b - u) < tol2:
                    if xmid - x >= 0:
                        rat = tol1
                    else:
                        rat = -tol1
            else:
                if x >= xmid:
                    deltax = a - x
                else:
                    deltax = b - x
                rat = _cg * deltax
        var u: Float64
        if abs(rat) < tol1:
            if rat >= 0:
                u = x + tol1
            else:
                u = x - tol1
        else:
            u = x + rat
        var fu = func(u)
        if fu > fx:
            if u < x:
                a = u
            else:
                b = u
            if (fu <= fw) or (w == x):
                v = w
                w = u
                fv = fw
                fw = fu
            elif (fu <= fv) or (v == x) or (v == w):
                v = u
                fv = fu
        else:
            if u >= x:
                a = x
            else:
                b = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        iter += 1
    return x


@always_inline
def _f32(x: Float64) -> Float64:
    return Float64(Float32(x))


def fminbound[O: ScalarObjective](mut func: O, x1: Float64, x2: Float64, xatol: Float64 = 1e-5, maxfun: Int = 500, float32_bounds: Bool = False) raises -> Float64:
    """`scipy.optimize.fminbound(func, x1, x2, xtol=xatol, maxfun)`: the minimizer.

    `float32_bounds` reproduces NumPy scalar promotion when SciPy passes
    float32 bounds: arithmetic between the two bounds alone stays in float32
    until an iterate (always float64) replaces one of them.
    """
    if not (isfinite(x1) and isfinite(x2)):
        raise Error("ValueError: Optimization bounds must be finite scalars.")
    if x1 > x2:
        raise Error("ValueError: The lower bound exceeds the upper bound.")
    var sqrt_eps = sqrt(Float64(2.2e-16))
    var golden_mean = 0.5 * (3.0 - sqrt(Float64(5.0)))
    var a = x1
    var b = x2
    var a32 = float32_bounds
    var b32 = float32_bounds

    @always_inline
    def width(a: Float64, b: Float64, both32: Bool) -> Float64:
        return _f32(b - a) if both32 else b - a

    @always_inline
    def total(a: Float64, b: Float64, both32: Bool) -> Float64:
        return _f32(a + b) if both32 else a + b

    var fulc = a + golden_mean * width(a, b, a32 and b32)
    var nfc = fulc
    var xf = fulc
    var rat = 0.0
    var e = 0.0
    var x = xf
    var fx = func(x)
    var num = 1
    var ffulc = fx
    var fnfc = fx
    var xm = 0.5 * total(a, b, a32 and b32)
    var tol1 = sqrt_eps * abs(xf) + xatol / 3.0
    var tol2 = 2.0 * tol1
    while abs(xf - xm) > (tol2 - 0.5 * width(a, b, a32 and b32)):
        var golden = True
        if abs(e) > tol1:
            golden = False
            var r = (xf - nfc) * (fx - ffulc)
            var q = (xf - fulc) * (fx - fnfc)
            var p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = rat
            if (abs(p) < abs(0.5 * q * r)) and (p > q * (a - xf)) and (p < q * (b - xf)):
                rat = (p + 0.0) / q
                x = xf + rat
                if ((x - a) < tol2) or ((b - x) < tol2):
                    var si = _sign(xm - xf) + (1.0 if (xm - xf) == 0 else 0.0)
                    rat = tol1 * si
            else:
                golden = True
        if golden:
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean * e
        var si = _sign(rat) + (1.0 if rat == 0 else 0.0)
        var step = abs(rat)
        if isnan(step) or isnan(tol1):
            step = Float64("nan")
        elif tol1 > step:
            step = tol1
        x = xf + si * step
        var fu = func(x)
        num += 1
        if fu <= fx:
            if x >= xf:
                a = xf
                a32 = False
            else:
                b = xf
                b32 = False
            fulc = nfc
            ffulc = fnfc
            nfc = xf
            fnfc = fx
            xf = x
            fx = fu
        else:
            if x < xf:
                a = x
                a32 = False
            else:
                b = x
                b32 = False
            if (fu <= fnfc) or (nfc == xf):
                fulc = nfc
                ffulc = fnfc
                nfc = x
                fnfc = fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc = x
                ffulc = fu
        xm = 0.5 * total(a, b, a32 and b32)
        tol1 = sqrt_eps * abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        if num >= maxfun:
            break
    return xf
