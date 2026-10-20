from numojo import zeros, Shape
from moxie.linear_model._cd_fast import coordinate_descent
from parity_io import Fixture, ResultLine, fixture_lines, report_error


def run[dtype: DType](fx: Fixture) raises:
    var result = coordinate_descent(fx.array[dtype]("X"), fx.array[dtype]("y"), fx.array[dtype]("initial"), fx.number("alpha"), fx.number("beta"), fx.integer("max_iter"), fx.number("tol"), fx.flag("positive"))
    var out = ResultLine(fx.id)
    out.add("coef", result.coef)
    out.scalar("gap", result.dual_gap)
    out.integer("iterations", result.iterations)
    out.integer("converged", Int(result.converged))
    out.emit()


def main() raises:
    var lines = fixture_lines()
    for line in lines:
        var fx = Fixture(line)
        try:
            if fx.bits == 32:
                run[DType.float32](fx)
            else:
                run[DType.float64](fx)
        except e:
            report_error(fx.id, e)
