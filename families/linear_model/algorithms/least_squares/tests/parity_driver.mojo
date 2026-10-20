"""Parity driver: fit LinearRegression per fixture line and print learned state."""
from numojo import NDArray
from moxie.linear_model import LinearRegression
from parity_io import Fixture, ResultLine, fixture_lines, report_error


def run[dtype: DType](fx: Fixture) raises:
    var model = LinearRegression[dtype](fit_intercept=fx.flag("fit_intercept"), tol=fx.number("tol"), positive=fx.flag("positive"))
    var X = fx.array[dtype]("X")
    var y = fx.array[dtype]("y")
    if fx.has("sw"):
        model.fit(X, y, sample_weight=fx.array[dtype]("sw"))
    else:
        model.fit(X, y)
    var out = ResultLine(fx.id)
    out.add("coef", model.coef_)
    out.add("intercept", model.intercept_)
    if model.rank_:
        out.integer("rank", model.rank_.value())
        out.add("singular", model.singular_.value())
    var Z = fx.array[dtype]("Z")
    out.add("pred", model.predict(Z))
    out.scalar("score", model.score(Z, fx.array[dtype]("yz")))
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
