"""Parity driver: fit TransformedTargetRegressor per fixture line and print results.

Functions are selected by name (`func` parameter): identity, log, sqrt, shift
(non-inverse: y + 1 / identity) and clip (min(y, 5) / identity). Warnings
are printed as `warn|<id>|<text>` lines before the result line.
"""
from numojo import NDArray
from std.math import log, exp, sqrt, min
from moxie.compose import TransformedTargetRegressor, TargetFunction
from parity_io import Fixture, ResultLine, fixture_lines, report_error


def _scalar[op: Int](v: Float64) -> Float64:
    comptime if op == 0:
        return log(v)
    elif op == 1:
        return exp(v)
    elif op == 2:
        return sqrt(v)
    elif op == 3:
        return v * v
    elif op == 4:
        return v + 1
    elif op == 5:
        return min(v, 5.0)
    return v


def _scalar32[op: Int](v: Float32) -> Float32:
    comptime if op == 0:
        return log(v)
    elif op == 1:
        return exp(v)
    elif op == 2:
        return sqrt(v)
    elif op == 3:
        return v * v
    elif op == 4:
        return v + 1
    elif op == 5:
        return min(v, 5.0)
    return v


def _map[dtype: DType, op: Int](x: NDArray[dtype]) raises -> NDArray[dtype]:
    """Elementwise op evaluated in dtype, as NumPy ufuncs do."""
    var out = x.copy()
    for i in range(out.size):
        comptime if dtype == DType.float32:
            out.unsafe_set(i, _scalar32[op](Float32(out.unsafe_get(i))).cast[dtype]())
        else:
            out.unsafe_set(i, _scalar[op](Float64(out.unsafe_get(i))).cast[dtype]())
    return out^


def forward[dtype: DType](name: String) raises -> Optional[TargetFunction[dtype]]:
    if name == "log":
        return _map[dtype, 0]
    if name == "sqrt":
        return _map[dtype, 2]
    if name == "shift":
        return _map[dtype, 4]
    if name == "clip":
        return _map[dtype, 5]
    if name == "forward_only":
        return _map[dtype, 0]
    return None


def inverse[dtype: DType](name: String) raises -> Optional[TargetFunction[dtype]]:
    if name == "log":
        return _map[dtype, 1]
    if name == "sqrt":
        return _map[dtype, 3]
    if name == "shift" or name == "clip":
        return _map[dtype, 6]
    return None


def run[dtype: DType](fx: Fixture) raises:
    var name = fx.text("func")
    var model = TransformedTargetRegressor[dtype](
        func=forward[dtype](name), inverse_func=inverse[dtype](name), check_inverse=fx.flag("check_inverse"))
    var X = fx.array[dtype]("X")
    var y = fx.array[dtype]("y")
    if fx.has("sw"):
        _ = model.fit(X, y, sample_weight=fx.array[dtype]("sw"))
    else:
        _ = model.fit(X, y)
    for message in model.warnings_:
        print(String("warn|", fx.id, "|", message))
    var out = ResultLine(fx.id)
    out.add("coef", model.regressor_.value().coef_)
    out.add("intercept", model.regressor_.value().intercept_)
    var Z = fx.array[dtype]("Z")
    out.add("pred", model.predict(Z))
    out.scalar("score", model.score(Z, fx.array[dtype]("yz")))
    if fx.has("sw"):
        out.scalar("train_score", model.score(X, y, fx.array[dtype]("sw")))
    out.integer("n_features_in", model.n_features_in_.value())
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
