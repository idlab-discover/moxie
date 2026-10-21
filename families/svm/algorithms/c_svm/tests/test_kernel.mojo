import numojo as nm
from std.math import abs
from std.testing import assert_equal, assert_true, assert_raises
from moxie.svm import SVC, SVR, NuSVC, NuSVR, OneClassSVM, Probability
from moxie.svm._contracts import Dense, KernelProvider


def features() raises -> Dense:
    var result = nm.zeros[DType.float64](nm.Shape(6, 2))
    var values: List[Float64] = [-2,-1,-1,-1,-1,-2,1,1,1,2,2,1]
    for i in range(12):
        result.unsafe_set(i, values[i])
    return result^


def labels() -> List[Float64]:
    return [1,1,1,2,2,2]


def targets() raises -> Dense:
    var result = nm.zeros[DType.float64](nm.Shape(6))
    var y = labels()
    for i in range(6):
        result.unsafe_set(i, y[i])
    return result^


struct LinearProvider(KernelProvider):
    def __init__(out self):
        pass

    def evaluate(self, x: Dense, train: Dense) raises -> Dense:
        return x @ train.T()


@no_inline
def classifier_lifecycle() raises:
    var X = features()
    var y = labels()
    var svc = SVC(kernel="linear")
    _ = svc.fit(X, y)
    assert_equal(svc.predict(X)[0], Float64(1))
    assert_equal(svc.predict(X)[5], Float64(2))
    assert_equal(svc.coef_().shape[0], 1)
    var clone = svc.__sklearn_clone__()
    assert_true(not clone.__sklearn_is_fitted__())
    with assert_raises():
        _ = clone.predict(X)
    _ = clone.set_params(C=2.0).fit(X, y)
    var params = clone.get_params()
    _ = clone.set_params(params)
    var f32 = X.astype[DType.float32]()
    _ = clone.fit(f32, y)
    assert_equal(clone.predict(f32)[0], Float64(1))
    var target = targets()
    _ = clone.fit(X, target)
    _ = clone.fit(f32, target)
    var old_prediction = clone.predict(X)
    _ = clone.set_params(C=-1.0)
    with assert_raises():
        _ = clone.fit(X, y)
    assert_equal(clone.predict(X)[0], old_prediction[0])
    _ = clone.set_params(C=1.0).fit(X, y)
    var bad = nm.zeros[DType.float64](nm.Shape(2, 3))
    with assert_raises():
        _ = svc.predict(bad)
    with assert_raises():
        _ = svc.predict_proba(X)
    var before = svc.decision_function(X)
    svc.dual_coef_.unsafe_set(0, 999)
    svc.intercept_.unsafe_set(0, 999)
    assert_equal(before.unsafe_get(0), svc.decision_function(X).unsafe_get(0))
    var weights = nm.ones[DType.float64](nm.Shape(6))
    assert_equal(svc.score(X, y, weights.copy()), Float64(1))


@no_inline
def classifier_labels_probability() raises:
    var X = features()
    var y = labels()
    var strings = SVC[String](kernel="linear")
    var text: List[String] = ["a", "a", "a", "z", "z", "z"]
    _ = strings.fit(X, text)
    assert_equal(strings.predict(X)[5], String("z"))
    var prob = SVC(probability=True, random_state=0)
    _ = prob.fit(X, y)
    assert_equal(prob.predict_proba(X).shape[1], 2)
    _ = prob.predict_log_proba(X)


def raises_with(prefix: String, message: String) -> Bool:
    return message.startswith(prefix)


@no_inline
def probability_and_coef_gating() raises:
    """Pinned BaseSVC.predict_proba and BaseLibSVM.coef_ error behavior."""
    var X = features()
    var y = labels()
    # Fitted with the deprecated/False default, then enabled: sklearn raises
    # NotFittedError("... fitted with probability=False") instead of predicting.
    for nu in range(2):
        var message = String("")
        if nu == 0:
            var svc = SVC(kernel="linear")
            _ = svc.fit(X, y)
            _ = svc.set_params(probability=Probability(True))
            try:
                _ = svc.predict_proba(X)
            except e:
                message = String(e)
        else:
            var nusvc = NuSVC(kernel="linear")
            _ = nusvc.fit(X, y)
            _ = nusvc.set_params(probability=Probability(True))
            try:
                _ = nusvc.predict_log_proba(X)
            except e:
                message = String(e)
        assert_true(raises_with("NotFittedError: predict_proba is not available when fitted with probability=False", message))
    # The coef_ property checks the current kernel parameter first, also unfitted.
    var linear = SVC(kernel="linear")
    _ = linear.fit(X, y)
    assert_equal(linear.coef_().shape[1], 2)
    _ = linear.set_params(kernel="rbf")
    var message = String("")
    try:
        _ = linear.coef_()
    except e:
        message = String(e)
    assert_true(raises_with("AttributeError: coef_ is only available when using a linear kernel", message))
    message = ""
    try:
        _ = SVR().coef_()
    except e:
        message = String(e)
    assert_true(raises_with("AttributeError", message))
    message = ""
    try:
        _ = OneClassSVM(kernel="poly").coef_()
    except e:
        message = String(e)
    assert_true(raises_with("AttributeError", message))
    message = ""
    try:
        _ = SVR(kernel="linear").coef_()
    except e:
        message = String(e)
    assert_true(raises_with("NotFittedError", message))


@no_inline
def regression_lifecycle() raises:
    var X = features()
    var y = targets()
    var svr = SVR(kernel="linear")
    _ = svr.fit(X, y)
    assert_true(svr.score(X, y) > 0.8)
    _ = svr.fit(X.astype[DType.float32](), y)
    var nr = NuSVR(kernel="linear")
    _ = nr.fit(X, y)
    assert_true(nr.score(X, y) > 0.8)
    _ = nr.fit(X.astype[DType.float32](), y)
    var nu = NuSVC(kernel="linear")
    var classes = labels()
    _ = nu.fit(X, classes)
    assert_equal(nu.predict(X)[0], Float64(1))
    _ = nu.fit(X.astype[DType.float32](), classes)


@no_inline
def one_class_lifecycle() raises:
    var X = features()
    var oc = OneClassSVM()
    _ = oc.fit(X)
    assert_equal(oc.predict(X).shape[0], 6)
    var clone = oc.__sklearn_clone__()
    assert_true(not clone.__sklearn_is_fitted__())
    _ = oc.fit(X.astype[DType.float32]())
    _ = oc.fit_predict(X)
    assert_equal(oc.fit_predict(X.astype[DType.float32]()).shape[0], 6)
    var weights = nm.ones[DType.float64](nm.Shape(6))
    _ = oc.fit(X, sample_weight=weights.copy())
    var samples = oc.score_samples(X)
    var decisions = oc.decision_function(X)
    assert_true(abs(samples.unsafe_get(0) - decisions.unsafe_get(0) - oc.offset_.unsafe_get(0)) < 1e-12)


@no_inline
def callable_classifiers() raises:
    var X = features()
    var y = labels()
    var provider = LinearProvider()
    var callable = SVC(kernel="callable")
    _ = callable.fit_with_kernel(X, y, provider)
    assert_equal(callable.predict(X, provider)[0], Float64(1))
    assert_equal(callable.predict(X, provider)[5], Float64(2))
    assert_equal(callable.score(X, y, provider), Float64(1))
    var changed = X.copy()
    _ = callable.fit_with_kernel(changed, y, provider)
    var saved = callable.decision_function(X, provider)
    changed.unsafe_set(0, 500)
    assert_equal(callable.decision_function(X, provider).unsafe_get(0), saved.unsafe_get(0))
    var bad = nm.zeros[DType.float64](nm.Shape(2, 3))
    with assert_raises():
        _ = callable.predict(bad, provider)
    var nucall = NuSVC(kernel="callable")
    _ = nucall.fit_with_kernel(X, y, provider)
    assert_equal(nucall.predict(X, provider)[5], Float64(2))
    var gram = provider.evaluate(X, X)
    var precomputed = SVC(kernel="precomputed")
    _ = precomputed.fit(gram, y)
    assert_equal(precomputed.predict(gram)[0], Float64(1))


@no_inline
def callable_regressors() raises:
    var X = features()
    var y = targets()
    var provider = LinearProvider()
    var regcall = SVR(kernel="callable")
    _ = regcall.fit_with_kernel(X, y, provider)
    assert_true(regcall.score(X, y, provider) > 0.8)
    var nrcall = NuSVR(kernel="callable")
    _ = nrcall.fit_with_kernel(X, y, provider)
    assert_equal(nrcall.predict(X, provider).shape[0], 6)


@no_inline
def callable_one_class() raises:
    var X = features()
    var provider = LinearProvider()
    var onecall = OneClassSVM(kernel="callable")
    _ = onecall.fit_with_kernel(X, provider)
    _ = onecall.predict(X, provider)
    _ = onecall.score_samples(X, provider)
    assert_equal(onecall.fit_predict(X, provider).shape[0], 6)


def main() raises:
    classifier_lifecycle()
    classifier_labels_probability()
    probability_and_coef_gating()
    regression_lifecycle()
    one_class_lifecycle()
    callable_classifiers()
    callable_regressors()
    callable_one_class()
    print("kernel native contract checks passed")
