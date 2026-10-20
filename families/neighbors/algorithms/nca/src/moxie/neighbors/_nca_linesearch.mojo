"""NCA's More–Thuente search, translated from SciPy 1.18.1 dcsrch/dcstep.

SciPy BSD-3-Clause notice: LICENSES/scipy.txt. Scalar reverse communication
keeps the reference interpolation, safeguards, warning and convergence order.
"""
from std.math import abs, sqrt


def fmax(a: Float64, b: Float64) -> Float64:
    """C99 fmax: a NaN argument yields the other argument."""
    if a != a:
        return b
    if b != b:
        return a
    return a if a > b else b


def fmin(a: Float64, b: Float64) -> Float64:
    """C99 fmin: a NaN argument yields the other argument."""
    if a != a:
        return b
    if b != b:
        return a
    return a if a < b else b


struct SearchTask:
    """dcsrch task codes (subset of the reference's enum)."""

    comptime start = 0
    comptime fg = 3
    comptime convergence = 4
    comptime warning = 6
    comptime error = 7


struct MoreThuente(Copyable, Movable):
    """Saved dcsrch state; fields are updated only where the reference saves."""

    var brackt: Bool
    var stage: Int
    var ginit: Float64
    var gtest: Float64
    var gx: Float64
    var gy: Float64
    var finit: Float64
    var fx: Float64
    var fy: Float64
    var stx: Float64
    var sty: Float64
    var stmin: Float64
    var stmax: Float64
    var width: Float64
    var width1: Float64
    var task: Int
    var message: Int

    def __init__(out self):
        self.brackt = False
        self.stage = 0
        self.ginit = 0
        self.gtest = 0
        self.gx = 0
        self.gy = 0
        self.finit = 0
        self.fx = 0
        self.fy = 0
        self.stx = 0
        self.sty = 0
        self.stmin = 0
        self.stmax = 0
        self.width = 0
        self.width1 = 0
        self.task = SearchTask.start
        self.message = 0

    def search(
        mut self,
        f: Float64,
        g: Float64,
        mut stp: Float64,
        ftol: Float64,
        gtol: Float64,
        xtol: Float64,
        stpmin: Float64,
        stpmax: Float64,
    ):
        """One reverse-communication step of Moré–Thuente dcsrch."""
        comptime xtrapl = 1.1
        comptime xtrapu = 4.0
        if self.task == SearchTask.start:
            if stp < stpmin:
                self.task = SearchTask.error
                self.message = 706
            if stp > stpmax:
                self.task = SearchTask.error
                self.message = 707
            if g >= 0.0:
                self.task = SearchTask.error
                self.message = 710
            if ftol < 0.0:
                self.task = SearchTask.error
                self.message = 703
            if gtol < 0.0:
                self.task = SearchTask.error
                self.message = 704
            if xtol < 0.0:
                self.task = SearchTask.error
                self.message = 705
            if stpmin < 0.0:
                self.task = SearchTask.error
                self.message = 708
            if stpmax < stpmin:
                self.task = SearchTask.error
                self.message = 709
            if self.task == SearchTask.error:
                return
            self.brackt = False
            self.stage = 1
            self.finit = f
            self.ginit = g
            self.gtest = ftol * self.ginit
            self.width = stpmax - stpmin
            self.width1 = self.width / 0.5
            self.stx = 0.0
            self.fx = self.finit
            self.gx = self.ginit
            self.sty = 0.0
            self.fy = self.finit
            self.gy = self.ginit
            self.stmin = 0.0
            self.stmax = stp + xtrapu * stp
            self.task = SearchTask.fg
            self.message = 0
            return
        var brackt = self.brackt
        var stage = self.stage
        var ginit = self.ginit
        var gtest = self.gtest
        var gx = self.gx
        var gy = self.gy
        var finit = self.finit
        var fx = self.fx
        var fy = self.fy
        var stx = self.stx
        var sty = self.sty
        var stmin = self.stmin
        var stmax = self.stmax
        var width = self.width
        var width1 = self.width1
        var ftest = finit + stp * gtest
        if stage == 1 and f <= ftest and g >= 0.0:
            stage = 2
        if brackt and (stp <= stmin or stp >= stmax):
            self.task = SearchTask.warning
            self.message = 601
        if brackt and (stmax - stmin) <= xtol * stmax:
            self.task = SearchTask.warning
            self.message = 604
        if stp == stpmax and f <= ftest and g <= gtest:
            self.task = SearchTask.warning
            self.message = 602
        if stp == stpmin and (f > ftest or g >= gtest):
            self.task = SearchTask.warning
            self.message = 603
        if f <= ftest and abs(g) <= gtol * (-ginit):
            self.task = SearchTask.convergence
        if (
            self.task == SearchTask.warning
            or self.task == SearchTask.convergence
        ):
            self._save(
                brackt, stage, ginit, gtest, gx, gy, finit, fx, fy, stx, sty,
                stmin, stmax, width, width1,
            )
            return
        if stage == 1 and f <= fx and f > ftest:
            var fm = f - stp * gtest
            var fxm = fx - stx * gtest
            var fym = fy - sty * gtest
            var gm = g - gtest
            var gxm = gx - gtest
            var gym = gy - gtest
            dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax)
            fx = fxm + stx * gtest
            fy = fym + sty * gtest
            gx = gxm + gtest
            gy = gym + gtest
        else:
            dcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax)
        if brackt:
            if abs(sty - stx) >= 0.66 * width1:
                stp = stx + 0.5 * (sty - stx)
            width1 = width
            width = abs(sty - stx)
        if brackt:
            stmin = fmin(stx, sty)
            stmax = fmax(stx, sty)
        else:
            stmin = stp + xtrapl * (stp - stx)
            stmax = stp + xtrapu * (stp - stx)
        stp = fmax(stp, stpmin)
        stp = fmin(stp, stpmax)
        if (brackt and (stp <= stmin or stp >= stmax)) or (
            brackt and stmax - stmin <= xtol * stmax
        ):
            stp = stx
        self.task = SearchTask.fg
        self.message = 0
        self._save(
            brackt, stage, ginit, gtest, gx, gy, finit, fx, fy, stx, sty,
            stmin, stmax, width, width1,
        )

    def _save(
        mut self,
        brackt: Bool,
        stage: Int,
        ginit: Float64,
        gtest: Float64,
        gx: Float64,
        gy: Float64,
        finit: Float64,
        fx: Float64,
        fy: Float64,
        stx: Float64,
        sty: Float64,
        stmin: Float64,
        stmax: Float64,
        width: Float64,
        width1: Float64,
    ):
        self.brackt = brackt
        self.stage = stage
        self.ginit = ginit
        self.gtest = gtest
        self.gx = gx
        self.gy = gy
        self.finit = finit
        self.fx = fx
        self.fy = fy
        self.stx = stx
        self.sty = sty
        self.stmin = stmin
        self.stmax = stmax
        self.width = width
        self.width1 = width1


def dcstep(
    mut stx: Float64,
    mut fx: Float64,
    mut dx: Float64,
    mut sty: Float64,
    mut fy: Float64,
    mut dy: Float64,
    mut stp: Float64,
    fp: Float64,
    dp: Float64,
    mut brackt: Bool,
    stpmin: Float64,
    stpmax: Float64,
):
    """Safeguarded cubic/quadratic step of Moré and Thuente (1994)."""
    var gamma: Float64
    var p: Float64
    var q: Float64
    var r: Float64
    var s: Float64
    var stpc: Float64
    var stpf: Float64
    var stpq: Float64
    var theta: Float64
    var sgnd = dp * (dx / abs(dx))
    if fp > fx:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = fmax(abs(theta), fmax(abs(dx), abs(dp)))
        gamma = s * sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s))
        if stp < stx:
            gamma = -gamma
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx)
        if abs(stpc - stx) < abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2.0
        brackt = True
    elif sgnd < 0.0:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = fmax(abs(theta), fmax(abs(dx), abs(dp)))
        gamma = s * sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    elif abs(dp) < abs(dx):
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = fmax(abs(theta), fmax(abs(dx), abs(dp)))
        gamma = s * sqrt(
            fmax(0.0, (theta / s) * (theta / s) - (dx / s) * (dp / s))
        )
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        if r < 0.0 and gamma != 0.0:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if brackt:
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            if stp > stx:
                stpf = fmin(stp + 0.66 * (sty - stp), stpf)
            else:
                stpf = fmax(stp + 0.66 * (sty - stp), stpf)
        else:
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = fmin(stpmax, stpf)
            stpf = fmax(stpmin, stpf)
    else:
        if brackt:
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
            s = fmax(abs(theta), fmax(abs(dy), abs(dp)))
            gamma = s * sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpc = stp + r * (sty - stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin
    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0.0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp
    stp = stpf
