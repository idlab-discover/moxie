from numojo import zeros, Shape
from moxie.linear_model import ElasticNet, Lasso
from parity_io import Fixture, ResultLine, fixture_lines, report_error


def run[dtype: DType](fx: Fixture) raises:
    var X = fx.array[dtype]("X")
    var y = fx.array[dtype]("y")
    var model = ElasticNet[dtype](alpha=fx.number("alpha"), l1_ratio=fx.number("ratio"), fit_intercept=fx.flag("intercept"), tol=fx.number("tol"), max_iter=fx.integer("max_iter"), positive=fx.flag("positive"), warm_start=fx.flag("warm"))
    if fx.has("sw"):
        model.fit(X, y, sample_weight=fx.array[dtype]("sw"))
    else:
        model.fit(X, y)
    if fx.flag("warm"):
        model.alpha *= 0.5
        if fx.has("sw"):
            model.fit(X, y, sample_weight=fx.array[dtype]("sw"))
        else:
            model.fit(X, y)
    var out = ResultLine(fx.id)
    out.add("coef", model.coef_)
    out.add("intercept", model.intercept_)
    out.add("gap", model.dual_gap_)
    out.scalar("score", model.score(X, y))
    var unconverged = 0
    for converged in model.converged_:
        if not converged:
            unconverged += 1
    out.integer("unconverged", unconverged)
    out.add("pred", model.predict(fx.array[dtype]("Z")))
    var iters = zeros[DType.float64](Shape(len(model.n_iter_)))
    for i in range(len(model.n_iter_)):
        iters.unsafe_set(i, Float64(model.n_iter_[i]))
    out.add("iterations", iters)
    out.emit()


def run_lasso[dtype: DType](fx: Fixture) raises:
    var X = fx.array[dtype]("X")
    var y = fx.array[dtype]("y")
    var model = Lasso[dtype](alpha=fx.number("alpha"), fit_intercept=fx.flag("intercept"), tol=fx.number("tol"), max_iter=fx.integer("max_iter"), positive=fx.flag("positive"), warm_start=fx.flag("warm"))
    if fx.has("sw"):
        model.fit(X, y, sample_weight=fx.array[dtype]("sw"))
    else:
        model.fit(X, y)
    if fx.flag("warm"):
        model.alpha *= 0.5
        if fx.has("sw"):
            model.fit(X, y, sample_weight=fx.array[dtype]("sw"))
        else:
            model.fit(X, y)
    var out = ResultLine(fx.id)
    out.add("coef", model.coef_)
    out.add("intercept", model.intercept_)
    out.add("gap", model.dual_gap_)
    out.scalar("score", model.score(X, y))
    var unconverged = 0
    for converged in model.converged_:
        if not converged:
            unconverged += 1
    out.integer("unconverged", unconverged)
    out.add("pred", model.predict(fx.array[dtype]("Z")))
    var iters = zeros[DType.float64](Shape(len(model.n_iter_)))
    for i in range(len(model.n_iter_)):
        iters.unsafe_set(i, Float64(model.n_iter_[i]))
    out.add("iterations", iters)
    out.emit()


def main() raises:
    var lines = fixture_lines()
    for line in lines:
        var fx = Fixture(line)
        try:
            if fx.bits == 32:
                if fx.number("ratio") == 1:
                    run_lasso[DType.float32](fx)
                else:
                    run[DType.float32](fx)
            else:
                if fx.number("ratio") == 1:
                    run_lasso[DType.float64](fx)
                else:
                    run[DType.float64](fx)
        except e:
            report_error(fx.id, e)
