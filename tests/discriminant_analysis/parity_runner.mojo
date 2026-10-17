"""Native side of the sklearn parity check: fit every fixture case in Mojo and compare.

Usage: parity_runner FIXTURES. Python only generated the fixtures (tests/parity.py).
"""
from std.sys import argv
from std.math import abs, exp
from numojo import NDArray
from moxie.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    Shrinkage,
    ClassLabel,
)
from support.fixtures import (
    FixtureCovariance,
    parse_array,
    parse_numbers,
    assert_close,
    assert_columns_close_up_to_sign,
)

comptime Fields = Dict[String, String]


def parse_shrinkage(fields: Fields) raises -> Shrinkage:
    var text = fields.get("param shrinkage", "none")
    if text == "none":
        return Shrinkage()
    if text == "auto":
        return Shrinkage("auto")
    return Shrinkage(Float64(text))


def parse_priors(fields: Fields) raises -> Optional[List[Float64]]:
    var text = fields.get("param priors", "none")
    if text == "none":
        return None
    return parse_numbers(text)


def parse_estimator(fields: Fields) raises -> Optional[FixtureCovariance]:
    var text = fields["cov"]
    if text == "none":
        return None
    if text.startswith("shrunk:"):
        return FixtureCovariance("shrunk", Float64(String(text.split(":")[1])))
    return FixtureCovariance(text)


def assert_log_proba_close[
    dtype: DType
](actual: NDArray[dtype], expected: NDArray[DType.float64], tol: Float64, label: String) raises:
    """Compare log-probabilities (Float64) or their exponentials (Float32).

    For Float32 input sklearn evaluates parts of predict_log_proba in single
    precision: probabilities below FLT_MIN are clamped to log(FLT_MIN) ~ -87.3 and
    binary `1 - expit(d)` suffers cancellation, so log-space errors of sklearn
    itself are unbounded for tiny probabilities. The native port keeps Float64
    probabilities. Float32 cases therefore compare exp(log_proba) with the
    absolute tolerance used for probabilities; Float64 compares log values.
    """
    comptime if dtype == DType.float32:
        for i in range(actual.size):
            var a = exp(Float64(actual.unsafe_get(i)))
            var b = exp(expected.unsafe_get(i))
            if not (abs(a - b) <= tol):
                raise Error(label + ": element " + String(i) + " exp(actual)=" + String(a) + " exp(expected)=" + String(b))
    else:
        assert_close(actual, expected, tol, label)


def labels_equal[L: ClassLabel](actual: List[L], expected: List[L], label: String) raises:
    if len(actual) != len(expected):
        raise Error(label + ": label count " + String(len(actual)) + " != " + String(len(expected)))
    for i in range(len(actual)):
        if not (actual[i] == expected[i]):
            raise Error(label + ": label " + String(i) + " differs")


def expect_error(message: String, fields: Fields, label: String) raises:
    var kind = fields["error"]
    if not message.startswith(kind + ":"):
        raise Error(label + ": expected " + kind + " but got: " + message)


def compare_optional[
    dtype: DType
](actual: Optional[NDArray[dtype]], fields: Fields, key: String, tol: Float64, label: String) raises:
    var text = fields[key]
    if text == "absent":
        if actual:
            raise Error(label + ": " + key + " should be absent")
        return
    if not actual:
        raise Error(label + ": " + key + " missing")
    assert_close(actual.value(), parse_array[DType.float64](text), tol, label + " " + key)


def run_lda[
    dtype: DType, L: ClassLabel
](fields: Fields, y: List[L], classes: List[L], predicted: List[L], label: String) raises:
    var X = parse_array[dtype](fields["X"])
    var T = parse_array[dtype](fields["T"])
    var n_components: Optional[Int] = None
    var nc = fields.get("param n_components", "none")
    if nc != "none":
        n_components = Int(nc)
    var model = LinearDiscriminantAnalysis[dtype, L, FixtureCovariance](
        solver=fields.get("param solver", "svd"),
        shrinkage=parse_shrinkage(fields),
        priors=parse_priors(fields),
        n_components=n_components,
        store_covariance=fields.get("param store_covariance", "0") == "1",
        tol=Float64(fields.get("param tol", "0.0001")),
        covariance_estimator=parse_estimator(fields),
    )
    if "error" in fields:
        try:
            model.fit(X, y)
        except e:
            expect_error(String(e), fields, label)
            if model.__sklearn_is_fitted__():
                raise Error(label + ": failed fit left the model fitted")
            return
        raise Error(label + ": expected " + fields["error"] + " but fit succeeded")
    model.fit(X, y)
    var tol = Float64(fields["tol_check"])
    labels_equal(model.classes_, classes, label + " classes_")
    assert_close(model.means_, parse_array[DType.float64](fields["means_"]), tol, label + " means_")
    assert_close(model.priors_, parse_array[DType.float64](fields["priors_"]), tol, label + " priors_")
    assert_close(model.coef_, parse_array[DType.float64](fields["coef_"]), tol, label + " coef_")
    assert_close(model.intercept_, parse_array[DType.float64](fields["intercept_"]), tol, label + " intercept_")
    compare_optional(model.covariance_, fields, "covariance_", tol, label)
    compare_optional(model.explained_variance_ratio_, fields, "explained_variance_ratio_", tol, label)
    compare_optional(model.xbar_, fields, "xbar_", tol, label)
    assert_close(model.decision_function(T), parse_array[DType.float64](fields["decision"]), tol, label + " decision")
    assert_close(model.predict_proba(T), parse_array[DType.float64](fields["proba"]), tol, label + " proba")
    assert_log_proba_close(model.predict_log_proba(T), parse_array[DType.float64](fields["log_proba"]), tol, label + " log_proba")
    labels_equal(model.predict(T), predicted, label + " predict")
    var score = model.score(X, y)
    if abs(score - Float64(fields["score"])) > 1e-12:
        raise Error(label + ": score " + String(score) + " != " + fields["score"])
    if len(model.get_feature_names_out()) != Int(fields["feature_names"]):
        raise Error(label + ": feature name count differs")
    if fields["scalings_"] == "absent":
        if model.scalings_:
            raise Error(label + ": scalings_ should be absent")
        try:
            _ = model.transform(T)
        except e:
            if not String(e).startswith("NotImplementedError:"):
                raise Error(label + ": lsqr transform raised " + String(e))
            return
        raise Error(label + ": lsqr transform should raise")
    var columns = Int(fields["comparable"])
    assert_columns_close_up_to_sign(
        model.scalings_.value(), parse_array[DType.float64](fields["scalings_"]), columns, tol, label + " scalings_"
    )
    assert_columns_close_up_to_sign(
        model.transform(T), parse_array[DType.float64](fields["transform"]), columns, tol, label + " transform"
    )


def run_qda[
    dtype: DType, L: ClassLabel
](fields: Fields, y: List[L], classes: List[L], predicted: List[L], label: String) raises:
    var X = parse_array[dtype](fields["X"])
    var T = parse_array[dtype](fields["T"])
    var model = QuadraticDiscriminantAnalysis[dtype, L, FixtureCovariance](
        solver=fields.get("param solver", "svd"),
        shrinkage=parse_shrinkage(fields),
        priors=parse_priors(fields),
        reg_param=Float64(fields.get("param reg_param", "0.0")),
        store_covariance=fields.get("param store_covariance", "0") == "1",
        tol=Float64(fields.get("param tol", "0.0001")),
        covariance_estimator=parse_estimator(fields),
    )
    if "error" in fields:
        try:
            model.fit(X, y)
        except e:
            expect_error(String(e), fields, label)
            if model.__sklearn_is_fitted__():
                raise Error(label + ": failed fit left the model fitted")
            return
        raise Error(label + ": expected " + fields["error"] + " but fit succeeded")
    model.fit(X, y)
    var tol = Float64(fields["tol_check"])
    labels_equal(model.classes_, classes, label + " classes_")
    assert_close(model.means_, parse_array[DType.float64](fields["means_"]), tol, label + " means_")
    assert_close(model.priors_, parse_array[DType.float64](fields["priors_"]), tol, label + " priors_")
    var k = Int(fields["n_scalings"])
    if len(model.scalings_) != k or len(model.rotations_) != k:
        raise Error(label + ": per-class factor count differs")
    for c in range(k):
        var suffix = String(c)
        assert_close(model.scalings_[c], parse_array[DType.float64](fields["q_scaling_" + suffix]), tol, label + " scalings_")
        assert_columns_close_up_to_sign(
            model.rotations_[c],
            parse_array[DType.float64](fields["q_rotation_" + suffix]),
            Int(fields["q_comparable_" + suffix]),
            tol,
            label + " rotations_",
        )
    if "q_covariance" in fields:
        if model.covariance_:
            raise Error(label + ": covariance_ should be absent")
    else:
        if not model.covariance_:
            raise Error(label + ": covariance_ missing")
        for c in range(k):
            assert_close(
                model.covariance_.value()[c],
                parse_array[DType.float64](fields["q_covariance_" + String(c)]),
                tol,
                label + " covariance_",
            )
    assert_close(model.decision_function(T), parse_array[DType.float64](fields["decision"]), tol, label + " decision")
    assert_close(model.predict_proba(T), parse_array[DType.float64](fields["proba"]), tol, label + " proba")
    assert_log_proba_close(model.predict_log_proba(T), parse_array[DType.float64](fields["log_proba"]), tol, label + " log_proba")
    labels_equal(model.predict(T), predicted, label + " predict")
    var score = model.score(X, y)
    if abs(score - Float64(fields["score"])) > 1e-12:
        raise Error(label + ": score " + String(score) + " != " + fields["score"])


def int_labels(text: String) raises -> List[Int]:
    var out = List[Int]()
    for token in text.split(","):
        out.append(Int(String(token)))
    return out^


def str_labels(text: String) raises -> List[String]:
    var out = List[String]()
    for token in text.split(","):
        out.append(String(token))
    return out^


def run_typed[dtype: DType](fields: Fields, label: String) raises:
    var lda = fields["est"] == "lda"
    if fields["labels"] == "int":
        var y = int_labels(fields["y"])
        var classes = int_labels(fields.get("classes", "0"))
        var predicted = int_labels(fields.get("predict", "0"))
        if lda:
            run_lda[dtype, Int](fields, y, classes, predicted, label)
        else:
            run_qda[dtype, Int](fields, y, classes, predicted, label)
    else:
        var y = str_labels(fields["y"])
        var classes = str_labels(fields.get("classes", "x"))
        var predicted = str_labels(fields.get("predict", "x"))
        if lda:
            run_lda[dtype, String](fields, y, classes, predicted, label)
        else:
            run_qda[dtype, String](fields, y, classes, predicted, label)


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: parity_runner FIXTURES")
    var text = open(String(args[1]), "r").read()
    var fields = Fields()
    var label = String("")
    var passed = 0
    var failed = 0
    var errors_checked = 0
    for raw in text.split("\n"):
        var line = String(raw)
        if line.byte_length() == 0:
            continue
        if line == "end":
            var is64 = fields.get("dtype", "64") == "64"
            try:
                if is64:
                    run_typed[DType.float64](fields, label)
                else:
                    run_typed[DType.float32](fields, label)
                passed += 1
                if "error" in fields:
                    errors_checked += 1
            except e:
                failed += 1
                print("FAIL", e)
            fields = Fields()
            continue
        var split = line.find(" ")
        var key = String(line[byte=0:split])
        var value = String(line[byte=split + 1 :])
        if key == "case":
            label = value
        elif key == "param":
            var inner = value.find(" ")
            fields["param " + String(value[byte=0:inner])] = String(value[byte=inner + 1 :])
        else:
            fields[key] = value
    print("parity cases passed:", passed, "failed:", failed, "(expected-error cases:", String(errors_checked) + ")")
    if failed != 0:
        raise Error("parity failures: " + String(failed))
