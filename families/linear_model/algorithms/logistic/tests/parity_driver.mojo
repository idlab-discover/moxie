from std.math import inf
from numojo import NDArray
from moxie.linear_model import ClassWeight, LogisticRegression
from parity_io import Fixture, ResultLine, fixture_lines, report_error


def run[dtype: DType](fx: Fixture) raises:
    var C = inf[DType.float64]() if fx.flag("unpenalized") else fx.number("C")
    var class_weight = Optional[ClassWeight](None)
    if fx.text("weight") == "balanced":
        class_weight = ClassWeight.balanced_weights()
    elif fx.text("weight") == "explicit":
        var weights = Dict[Int, Float64]()
        weights[-3] = 2.0
        class_weight = ClassWeight.explicit(weights^)
    var model = LogisticRegression[dtype](
        C=C, class_weight=class_weight^, fit_intercept=fx.flag("intercept"),
        max_iter=fx.integer("max_iter"), tol=fx.number("tol"), warm_start=fx.has("refit_C"),
    )
    var X = fx.array[dtype]("X")
    var y = fx.array[DType.int64]("y")
    var sw = Optional[NDArray[dtype]](None)
    if fx.has("sw"):
        sw = fx.array[dtype]("sw")
        _ = model.fit(X, y, sample_weight=sw.value())
    else:
        _ = model.fit(X, y)
    if fx.has("refit_C"):
        _ = model.set_params(C=fx.number("refit_C"))
        _ = model.fit(X, y)
    var Z = fx.array[dtype]("Z")
    var result = ResultLine(fx.id)
    result.add("classes", model.classes_)
    result.add("coef", model.coef_)
    result.add("intercept", model.intercept_)
    result.add("iterations", model.n_iter_)
    result.add("decision", model.decision_function(Z))
    result.add("predictions", model.predict(Z))
    result.add("probability", model.predict_proba(Z))
    result.add("log_probability", model.predict_log_proba(Z))
    result.scalar("score", model.score(X, y, sample_weight=sw))
    result.integer("features", model.n_features_in_)
    result.emit()


def main() raises:
    for line in fixture_lines():
        var case_data = Fixture(line)
        try:
            if case_data.bits == 32:
                run[DType.float32](case_data)
            else:
                run[DType.float64](case_data)
        except error:
            report_error(case_data.id, error)
