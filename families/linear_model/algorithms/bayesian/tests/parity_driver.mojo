"""Compare public BayesianRidge learned state and predictive uncertainty."""
from moxie.linear_model import BayesianRidge
from parity_io import Fixture, ResultLine, fixture_lines, report_error


def run[dtype: DType](fx: Fixture) raises:
    var initial = Optional[Float64](None)
    if fx.number("alpha_init") >= 0:
        initial = fx.number("alpha_init")
    var penalty = Optional[Float64](None)
    if fx.number("lambda_init") >= 0:
        penalty = fx.number("lambda_init")
    var model = BayesianRidge[dtype](fit_intercept=fx.flag("fit_intercept"), compute_score=fx.flag("compute_score"),
        max_iter=Int(fx.number("max_iter")), tol=fx.number("tol"),
        alpha_init=initial, lambda_init=penalty,
        alpha_1=fx.number("alpha_1"), alpha_2=fx.number("alpha_2"),
        lambda_1=fx.number("lambda_1"), lambda_2=fx.number("lambda_2"))
    var X = fx.array[dtype]("X")
    var y = fx.array[dtype]("y")
    if fx.has("sw"):
        _ = model.fit(X, y, sample_weight=fx.array[dtype]("sw"))
    else:
        _ = model.fit(X, y)
    var out = ResultLine(fx.id)
    out.add("coef", model.coef_)
    out.add("intercept", model.intercept_)
    out.add("sigma", model.sigma_)
    out.add("scores", model.scores_)
    out.add("offset", model.X_offset_)
    out.add("scale", model.X_scale_)
    out.scalar("alpha", model.alpha_)
    out.scalar("lambda", model.lambda_)
    out.integer("n_iter", model.n_iter_)
    var Z = fx.array[dtype]("Z")
    var predictions = model.predict_std(Z)
    out.add("pred", predictions[0])
    out.add("std", predictions[1])
    out.emit()


def main() raises:
    for line in fixture_lines():
        var fx = Fixture(line)
        try:
            if fx.bits == 32:
                run[DType.float32](fx)
            else:
                run[DType.float64](fx)
        except error:
            report_error(fx.id, error)
