"""Parity driver: Ridge / RidgeClassifier / ridge_regression per fixture line."""
import numojo as nm
from numojo import NDArray, Shape
from moxie.linear_model import Ridge, RidgeClassifier, ridge_regression, ClassWeight
from moxie.linear_model._ridge import ScalarOrVector
from parity_io import Fixture, ResultLine, fixture_lines, report_error


def solver_code(name: String) -> Int:
    var names: List[String] = ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]
    for i in range(len(names)):
        if names[i] == name:
            return i
    return -1


def alpha_of(fx: Fixture) raises -> ScalarOrVector:
    var values = fx.arrays["alpha"].values.copy()
    if fx.arrays["alpha"].cols == 0 and fx.flag("scalar_alpha"):
        return ScalarOrVector(values[0])
    return ScalarOrVector(values^)


def max_iter_of(fx: Fixture) raises -> Optional[Int]:
    if fx.integer("max_iter") < 0:
        return None
    return fx.integer("max_iter")


def iterations(n_iter: Optional[List[Int]]) raises -> NDArray[DType.float64]:
    if not n_iter:
        return nm.zeros[DType.float64](Shape(0))
    var values = n_iter.value().copy()
    var a = nm.zeros[DType.float64](Shape(len(values)))
    for i in range(len(values)):
        a.unsafe_set(i, Float64(values[i]))
    return a^


def labels(fx: Fixture, key: String) raises -> NDArray[DType.int64]:
    var source = fx.array[DType.float64](key)
    var result = nm.zeros[DType.int64](source.shape)
    for i in range(source.size):
        result.unsafe_set(i, Int64(Int(source.unsafe_get(i))))
    return result^


def class_weight_of(fx: Fixture) raises -> Optional[ClassWeight]:
    var text = fx.text("class_weight")
    if text == "none":
        return None
    if text == "balanced":
        return ClassWeight.balanced_weights()
    var weights = Dict[Int, Float64]()
    for item in text.split(","):
        var pair = item.split(":")
        weights[Int(String(pair[0]))] = atof(pair[1])
    return ClassWeight.explicit(weights^)


def run[dtype: DType](fx: Fixture) raises:
    var kind = fx.text("kind")
    var X = fx.array[dtype]("X")
    var out = ResultLine(fx.id)
    if kind == "function":
        var sw = Optional[NDArray[dtype]](None)
        if fx.has("sw"):
            sw = fx.array[dtype]("sw")
        var result = ridge_regression(X, fx.array[dtype]("y"), alpha_of(fx), sample_weight=sw, solver=fx.text("solver"), max_iter=max_iter_of(fx), tol=fx.number("tol"), positive=fx.flag("positive"), return_n_iter=True)
        out.add("coef", result.coef)
        out.add("n_iter", iterations(result.n_iter))
    elif kind == "ridge":
        var model = Ridge[dtype](alpha_of(fx), fit_intercept=fx.flag("fit_intercept"), max_iter=max_iter_of(fx), tol=fx.number("tol"), solver=fx.text("solver"), positive=fx.flag("positive"))
        if fx.has("sw"):
            model.fit(X, fx.array[dtype]("y"), sample_weight=fx.array[dtype]("sw"))
        else:
            model.fit(X, fx.array[dtype]("y"))
        out.add("coef", model.coef_)
        out.add("intercept", model.intercept_)
        out.add("n_iter", iterations(model.n_iter_))
        out.integer("solver", solver_code(model.solver_))
        var Z = fx.array[dtype]("Z")
        out.add("pred", model.predict(Z))
        out.scalar("score", model.score(Z, fx.array[dtype]("yz")))
    else:
        var model = RidgeClassifier[dtype](alpha_of(fx), fit_intercept=fx.flag("fit_intercept"), max_iter=max_iter_of(fx), tol=fx.number("tol"), class_weight=class_weight_of(fx), solver=fx.text("solver"), positive=fx.flag("positive"))
        var y = labels(fx, "y")
        if fx.has("sw"):
            model.fit(X, y, sample_weight=fx.array[dtype]("sw"))
        else:
            model.fit(X, y)
        out.add("coef", model.coef_)
        out.add("intercept", model.intercept_)
        out.integer("solver", solver_code(model.solver_))
        var Z = fx.array[dtype]("Z")
        var classes = model.classes_.copy()
        var c = nm.zeros[DType.float64](Shape(classes.shape[0]))
        for i in range(classes.shape[0]):
            c.unsafe_set(i, Float64(classes.unsafe_get(i)))
        out.add("classes", c)
        out.add("decision", model.decision_function(Z))
        var predicted = model.predict(Z)
        var p = nm.zeros[DType.float64](predicted.shape)
        for i in range(predicted.size):
            p.unsafe_set(i, Float64(predicted.unsafe_get(i)))
        out.add("pred", p)
        out.scalar("score", model.score(Z, labels(fx, "yz")))
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
