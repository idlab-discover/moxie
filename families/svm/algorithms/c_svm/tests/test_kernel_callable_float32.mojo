"""Native callable kernels accept Float32 matrices on every public path."""
import numojo as nm
from std.math import abs
from std.testing import assert_equal, assert_true
from moxie.svm import SVC, SVR, NuSVC, NuSVR, OneClassSVM
from moxie.svm._contracts import Dense, KernelProvider


struct LinearProvider(KernelProvider):
    def __init__(out self):
        pass

    def evaluate(self, x: Dense, train: Dense) raises -> Dense:
        return x @ train.T()


def features() raises -> Dense:
    var result = nm.zeros[DType.float64](nm.Shape(6, 2))
    var values: List[Float64] = [-2,-1,-1,-1,-1,-2,1,1,1,2,2,1]
    for i in range(12):
        result.unsafe_set(i, values[i])
    return result^


def targets() raises -> Dense:
    var result = nm.zeros[DType.float64](nm.Shape(6))
    for i in range(6):
        result.unsafe_set(i, Float64(1 if i < 3 else 2))
    return result^


def assert_same[dtype: DType](actual: nm.NDArray[dtype], expected: nm.NDArray[dtype]) raises:
    assert_equal(actual.ndim, expected.ndim)
    assert_equal(actual.size, expected.size)
    for i in range(actual.ndim):
        assert_equal(actual.shape[i], expected.shape[i])
    for i in range(actual.size):
        assert_true(abs(Float64(actual.unsafe_get(i)) - Float64(expected.unsafe_get(i))) < 1e-12)


@no_inline
def test_SVC() raises:
    var X = features()
    var x32 = X.astype[DType.float32]()
    var weights = nm.ones[DType.float64](nm.Shape(6))
    var provider = LinearProvider()
    var y: List[Float64] = [1,1,1,2,2,2]
    var model = SVC(kernel="callable", probability=True, random_state=0)
    _ = model.fit_with_kernel(x32, y, provider, sample_weight=weights.copy())
    assert_same(model.decision_function(x32, provider), model.decision_function(X, provider))
    var prediction = model.predict(x32, provider)
    var expected = model.predict(X, provider)
    for i in range(len(expected)):
        assert_equal(prediction[i], expected[i])
        assert_equal(prediction[i], y[i])
    assert_same(model.predict_proba(x32, provider), model.predict_proba(X, provider))
    assert_same(model.predict_log_proba(x32, provider), model.predict_log_proba(X, provider))
    assert_equal(model.score(x32, y, provider, weights.copy()), model.score(X, y, provider, weights.copy()))


@no_inline
def test_NuSVC() raises:
    var X = features()
    var x32 = X.astype[DType.float32]()
    var weights = nm.ones[DType.float64](nm.Shape(6))
    var provider = LinearProvider()
    var y: List[Float64] = [1,1,1,2,2,2]
    var model = NuSVC(kernel="callable", probability=True, random_state=0)
    _ = model.fit_with_kernel(x32, y, provider, sample_weight=weights.copy())
    assert_same(model.decision_function(x32, provider), model.decision_function(X, provider))
    var prediction = model.predict(x32, provider)
    var expected = model.predict(X, provider)
    for i in range(len(expected)):
        assert_equal(prediction[i], expected[i])
        assert_equal(prediction[i], y[i])
    assert_same(model.predict_proba(x32, provider), model.predict_proba(X, provider))
    assert_same(model.predict_log_proba(x32, provider), model.predict_log_proba(X, provider))
    assert_equal(model.score(x32, y, provider, weights.copy()), model.score(X, y, provider, weights.copy()))


@no_inline
def test_SVR() raises:
    var X = features()
    var x32 = X.astype[DType.float32]()
    var weights = nm.ones[DType.float64](nm.Shape(6))
    var provider = LinearProvider()
    var y = targets()
    var model = SVR(kernel="callable")
    _ = model.fit_with_kernel(x32, y, provider, sample_weight=weights.copy())
    assert_same(model.decision_function(x32, provider), model.decision_function(X, provider))
    assert_same(model.predict(x32, provider), model.predict(X, provider))
    assert_equal(model.score(x32, y, provider, weights.copy()), model.score(X, y, provider, weights.copy()))


@no_inline
def test_NuSVR() raises:
    var X = features()
    var x32 = X.astype[DType.float32]()
    var weights = nm.ones[DType.float64](nm.Shape(6))
    var provider = LinearProvider()
    var y = targets()
    var model = NuSVR(kernel="callable")
    _ = model.fit_with_kernel(x32, y, provider, sample_weight=weights.copy())
    assert_same(model.decision_function(x32, provider), model.decision_function(X, provider))
    assert_same(model.predict(x32, provider), model.predict(X, provider))
    assert_equal(model.score(x32, y, provider, weights.copy()), model.score(X, y, provider, weights.copy()))


@no_inline
def test_OneClassSVM() raises:
    var X = features()
    var x32 = X.astype[DType.float32]()
    var weights = nm.ones[DType.float64](nm.Shape(6))
    var provider = LinearProvider()
    var model = OneClassSVM(kernel="callable")
    _ = model.fit_with_kernel(x32, provider, sample_weight=weights.copy())
    assert_same(model.decision_function(x32, provider), model.decision_function(X, provider))
    assert_same(model.predict(x32, provider), model.predict(X, provider))
    assert_same(model.score_samples(x32, provider), model.score_samples(X, provider))
    var joint = model.fit_predict(x32, provider, sample_weight=weights.copy())
    assert_same(joint, model.predict(x32, provider))


def main() raises:
    test_SVC()
    test_NuSVC()
    test_SVR()
    test_NuSVR()
    test_OneClassSVM()
    print("Float32 callable kernel contracts passed")
