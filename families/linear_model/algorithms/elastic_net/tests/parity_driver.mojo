from numojo import NDArray, zeros, Shape
from moxie.linear_model import CoordinatePathResult, ElasticNet, Lasso
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


def run_path[dtype: DType](fx: Fixture) raises:
    var X = fx.array[dtype]("X")
    var y = fx.array[dtype]("y")
    var initial = Optional[NDArray[dtype]](None)
    if fx.has("initial"):
        initial = fx.array[dtype]("initial")
    var alphas = Optional[NDArray[DType.float64]](None)
    if fx.has("alphas"):
        alphas = fx.array[DType.float64]("alphas")
    var ratio = fx.number("ratio")
    var count = fx.integer("count")
    var max_iter = fx.integer("max_iter")
    var positive = fx.flag("positive")
    var eps = fx.number("eps")
    var result: CoordinatePathResult[dtype]
    if ratio == 1:
        result = Lasso[dtype].path(
            X, y, eps=eps, n_alphas=count, alphas=alphas, coef_init=initial,
            max_iter=max_iter, positive=positive, return_n_iter=True,
        )
    else:
        result = ElasticNet[dtype].path(
            X, y, l1_ratio=ratio, eps=eps, n_alphas=count, alphas=alphas,
            coef_init=initial, max_iter=max_iter, positive=positive,
            return_n_iter=True,
        )
    var out = ResultLine(fx.id)
    out.add("alphas", result.alphas)
    out.add("coefs", result.coefs)
    out.add("dual_gaps", result.dual_gaps)
    ref counts = result.n_iters.value()
    var iterations = zeros[DType.float64](Shape(len(counts)))
    for i in range(len(counts)):
        iterations.unsafe_set(i, Float64(counts[i]))
    out.add("iterations", iterations)
    out.emit()


def main() raises:
    var lines = fixture_lines()
    for line in lines:
        var fx = Fixture(line)
        try:
            if fx.has("path"):
                if fx.bits == 32:
                    run_path[DType.float32](fx)
                else:
                    run_path[DType.float64](fx)
            elif fx.bits == 32:
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
