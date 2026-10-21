"""Behavioral contracts using an exact native one-nearest-neighbor component."""
import numojo as nm
from numojo import NDArray, Shape
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import SupervisedEstimator, Predictor, ProbabilityPredictor, SupervisedScorer, WeightedSupervisedEstimator, IncrementalClassifier, IncrementalRegressor, clone
from moxie._multioutput import ChainResponses
from moxie.multioutput import MultiOutputRegressor, MultiOutputClassifier, RegressorChain, ClassifierChain

@fieldwise_init
struct Params(Copyable, Movable):
    var dummy: Int

struct Nearest[label: DType, dtype: DType = DType.float64](Copyable, SupervisedEstimator, Predictor, ProbabilityPredictor, SupervisedScorer, IncrementalClassifier where label == DType.int64, IncrementalRegressor where label == DType.float64):
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.label]
    comptime Prediction = Self.Target
    comptime Probabilities = NDArray[Self.dtype]
    comptime Classes = Self.Target
    comptime SampleWeight = NDArray[Self.dtype]
    comptime Params = Params
    var x: Self.Input
    var y: Self.Target
    var fitted: Bool

    def __init__(out self) raises:
        self.x = nm.zeros[Self.dtype](Shape(0, 0))
        self.y = nm.zeros[Self.label](Shape(0))
        self.fitted = False

    def get_params(self, deep: Bool = True) -> Self.Params:
        return Self.Params(0)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        pass
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self()

    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self.x = X.copy()
        self.y = y.copy()
        self.fitted = True
        return self

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        var result = nm.zeros[Self.label](Shape(X.shape[0]))
        for i in range(X.shape[0]):
            var best = Float64(1e300)
            var winner = 0
            for k in range(self.x.shape[0]):
                var distance = Float64(0)
                for j in range(X.shape[1]):
                    var delta = X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1])) - self.x.unsafe_get(k * Int(self.x.strides[0]) + j * Int(self.x.strides[1]))
                    distance += Float64(delta) * Float64(delta)
                if distance < best:
                    best = distance
                    winner = k
            result.unsafe_set(i, self.y.unsafe_get(winner * Int(self.y.strides[0])))
        return result^

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        var pred = self.predict(X)
        var result = nm.zeros[Self.dtype](Shape(X.shape[0], 2))
        for i in range(X.shape[0]):
            result.unsafe_set(i * 2 + Int(pred.unsafe_get(i)), 1)
        return result^

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        raise Error("Test component does not implement standalone scoring")

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        if not self.fitted:
            _ = self.fit(X, y)
            return self
        var rows = self.x.shape[0] + X.shape[0]
        var joined = nm.zeros[Self.dtype](Shape(rows, X.shape[1]))
        var target = nm.zeros[Self.label](Shape(rows))
        for i in range(rows):
            if i < self.x.shape[0]:
                target.unsafe_set(i, self.y.unsafe_get(i))
                for j in range(X.shape[1]):
                    joined.unsafe_set(i * X.shape[1] + j, self.x.unsafe_get(i * X.shape[1] + j))
            else:
                var k = i - self.x.shape[0]
                target.unsafe_set(i, y.unsafe_get(k))
                for j in range(X.shape[1]):
                    joined.unsafe_set(i * X.shape[1] + j, X.unsafe_get(k * X.shape[1] + j))
        _ = self.fit(joined, target)
        return self

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, classes: Optional[Self.Classes] = None) raises -> ref[self] Self:
        _ = self.partial_fit(X, y)
        return self

@fieldwise_init
struct TwoFold[label: DType](ChainResponses):
    comptime Estimator = Nearest[Self.label]
    comptime Input = NDArray[DType.float64]
    comptime Target = NDArray[Self.label]
    comptime Response = NDArray[DType.float64]

    def cross_val_response(self, estimator: Self.Estimator, X: Self.Input, y: Self.Target, method: String) raises -> Self.Response:
        if method != "predict" or X.shape[0] != 4:
            raise Error("Test provider supports two folds of four samples with predict only")
        var result = nm.zeros[DType.float64](Shape(4))
        for fold in range(2):
            var train = nm.zeros[DType.float64](Shape(2, X.shape[1]))
            var test = nm.zeros[DType.float64](Shape(2, X.shape[1]))
            var target = nm.zeros[Self.label](Shape(2))
            for i in range(2):
                var source = (1 - fold) * 2 + i
                target.unsafe_set(i, y.unsafe_get(source))
                for j in range(X.shape[1]):
                    train.unsafe_set(i * X.shape[1] + j, X.unsafe_get(source * X.shape[1] + j))
                    test.unsafe_set(i * X.shape[1] + j, X.unsafe_get((fold * 2 + i) * X.shape[1] + j))
            var model = clone(estimator)
            _ = model.fit(train, target)
            var prediction = model.predict(test)
            for i in range(2):
                result.unsafe_set(fold * 2 + i, Float64(prediction.unsafe_get(i)))
        return result^


struct WeightedMean(WeightedSupervisedEstimator, Predictor):
    comptime Input = NDArray[DType.float64]
    comptime Target = NDArray[DType.float64]
    comptime Prediction = Self.Target
    comptime SampleWeight = Self.Target
    comptime Params = Params
    var mean: Float64
    var fitted: Bool

    def __init__(out self):
        self.mean = 0
        self.fitted = False
    def get_params(self, deep: Bool = True) -> Self.Params:
        return Params(0)
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        pass
        return self

    def __sklearn_clone__(self) -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted
    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        var weights = nm.ones[DType.float64](Shape(y.shape[0]))
        _ = self.fit(X, y, sample_weight=weights)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Self.SampleWeight) raises -> ref[self] Self:
        var terms = nm.zeros[DType.float64](Shape(y.shape[0]))
        for i in range(y.shape[0]):
            terms.unsafe_set(i, y.unsafe_get(i) * sample_weight.unsafe_get(i))
        self.mean = Float64(nm.sum(terms)) / Float64(nm.sum(sample_weight))
        self.fitted = True
        return self

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        var result = nm.zeros[DType.float64](Shape(X.shape[0]))
        for i in range(X.shape[0]):
            result.unsafe_set(i, self.mean)
        return result^


def parity_matrix[dtype: DType](name: String, values: NDArray[dtype]) raises:
    var line = "PARITY " + name
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            line += " " + String(values.unsafe_get(i * Int(values.strides[0]) + j * Int(values.strides[1])))
    print(line)


def main() raises:
    var X = nm.zeros[DType.float64](Shape(4, 1))
    var y = nm.zeros[DType.float64](Shape(4, 2))
    var labels = nm.zeros[DType.int64](Shape(4, 2))
    for i in range(4):
        X.unsafe_set(i, Float64(i))
        y.unsafe_set(2 * i, Float64(i * 10))
        y.unsafe_set(2 * i + 1, Float64(3 - i))
        labels.unsafe_set(2 * i, Int64(i % 2))
        labels.unsafe_set(2 * i + 1, Int64((i // 2) % 2))
    var reg = MultiOutputRegressor(Nearest[DType.float64]())
    with assert_raises():
        var unused = reg.predict(X)
    _ = reg.fit(X, y)
    assert_equal(reg.score(X, y), Float64(1))
    assert_equal(len(reg.estimators_), 2)
    var copy = clone(reg)
    assert_false(copy.__sklearn_is_fitted__())
    assert_true(reg.__sklearn_is_fitted__())
    var rc = RegressorChain(Nearest[DType.float64](), order=List[Int]( [1, 0] ))
    _ = rc.fit(X, y)
    assert_equal(rc.score(X, y), Float64(1))
    assert_equal(rc.order_[0], 1)
    var clf = MultiOutputClassifier(Nearest[DType.int64]())
    _ = clf.fit(X, labels)
    assert_equal(clf.score(X, labels), Float64(1))
    var probabilities = clf.predict_proba(X)
    assert_equal(len(probabilities), 2)
    assert_equal(probabilities[0].unsafe_get(0), Float64(1))
    var cc = ClassifierChain(Nearest[DType.int64](), order=List[Int]( [1, 0] ))
    _ = cc.fit(X, labels)
    assert_equal(cc.score(X, labels), Float64(1))
    var proba = cc.predict_proba(X)
    for i in range(8):
        assert_equal(proba.unsafe_get(i), Float64(labels.unsafe_get(i)))
    var logs = cc.predict_log_proba(X)
    assert_equal(logs.unsafe_get(2), Float64(0))
    var bad = ClassifierChain(Nearest[DType.int64](), order=List[Int]( [0, 0] ))
    with assert_raises():
        _ = bad.fit(X, labels)
    var wrong = nm.zeros[DType.float64](Shape(4, 3))
    with assert_raises():
        var unused = reg.predict(wrong)
    _ = reg.fit(X, y)
    assert_equal(len(reg.estimators_), 2)
    var query = nm.zeros[DType.float64](Shape(3, 1))
    query.unsafe_set(0, 0.4)
    query.unsafe_set(1, 1.7)
    query.unsafe_set(2, 2.6)
    var a = reg.predict(query)
    var b = rc.predict(query)
    var c = clf.predict(query)
    var d = cc.predict(query)
    var query_probabilities = clf.predict_proba(query)
    for j in range(len(query_probabilities)):
        parity_matrix("MultiOutputClassifierProba" + String(j), query_probabilities[j])
    parity_matrix("ClassifierChainProba", cc.predict_proba(query))
    parity_matrix("ClassifierChainLogProba", cc.predict_log_proba(query))
    print("PARITY MultiOutputRegressor", a.unsafe_get(0), a.unsafe_get(1), a.unsafe_get(2), a.unsafe_get(3), a.unsafe_get(4), a.unsafe_get(5))
    print("PARITY RegressorChain", b.unsafe_get(0), b.unsafe_get(1), b.unsafe_get(2), b.unsafe_get(3), b.unsafe_get(4), b.unsafe_get(5))
    print("PARITY MultiOutputClassifier", c.unsafe_get(0), c.unsafe_get(1), c.unsafe_get(2), c.unsafe_get(3), c.unsafe_get(4), c.unsafe_get(5))
    print("PARITY ClassifierChain", d.unsafe_get(0), d.unsafe_get(1), d.unsafe_get(2), d.unsafe_get(3), d.unsafe_get(4), d.unsafe_get(5))
    var cv_reg = RegressorChain(Nearest[DType.float64](), order=List[Int]([1, 0]))
    _ = cv_reg.fit(X, y, cv=TwoFold[DType.float64]())
    # One output never requests CV responses, so rejection must come from the wrapper.
    var single_target = nm.zeros[DType.float64](Shape(4, 1))
    var invalid_response = RegressorChain(Nearest[DType.float64](), chain_method="predict_proba")
    with assert_raises():
        _ = invalid_response.fit(X, single_target)
    with assert_raises():
        _ = invalid_response.fit(X, single_target, cv=TwoFold[DType.float64]())
    assert_false(invalid_response.__sklearn_is_fitted__())
    var single_chain = RegressorChain(Nearest[DType.float64]())
    _ = single_chain.fit(X, single_target, cv=TwoFold[DType.float64]())
    assert_equal(single_chain.predict(X).shape[1], 1)
    var invalid_params = single_chain.get_params()
    invalid_params.chain_method = "predict_proba"
    _ = single_chain.set_params(invalid_params)
    with assert_raises():
        _ = single_chain.fit(X, y, cv=TwoFold[DType.float64]())
    assert_equal(len(single_chain.estimators_), 1)
    assert_equal(single_chain.chain_method_, "predict")
    assert_equal(single_chain.predict(X).shape[1], 1)
    var cv_clf = ClassifierChain(Nearest[DType.int64](), order=List[Int]([1, 0]))
    _ = cv_clf.fit(X, labels, cv=TwoFold[DType.int64]())
    var e = cv_reg.predict(query)
    var f = cv_clf.predict(query)
    print("PARITY RegressorChainCV", e.unsafe_get(0), e.unsafe_get(1), e.unsafe_get(2), e.unsafe_get(3), e.unsafe_get(4), e.unsafe_get(5))
    print("PARITY ClassifierChainCV", f.unsafe_get(0), f.unsafe_get(1), f.unsafe_get(2), f.unsafe_get(3), f.unsafe_get(4), f.unsafe_get(5))
    var inc = MultiOutputRegressor(Nearest[DType.float64]())
    _ = inc.partial_fit(X, y)
    _ = inc.partial_fit(X, y)
    assert_equal(inc.estimators_[0].x.shape[0], 8)
    assert_equal(inc.score(X, y), Float64(1))
    var inc_clf = MultiOutputClassifier(Nearest[DType.int64]())
    with assert_raises():
        _ = inc_clf.partial_fit(X, labels)
    _ = inc_clf.partial_fit(X, labels, clf.classes_.copy())
    _ = inc_clf.partial_fit(X, labels)
    assert_equal(inc_clf.estimators_[0].x.shape[0], 8)
    assert_equal(inc_clf.score(X, labels), Float64(1))
    var probability_chain = ClassifierChain(Nearest[DType.int64](), chain_method="predict_proba")
    _ = probability_chain.fit(X, labels)
    assert_equal(probability_chain.score(X, labels), Float64(1))
    parity_matrix("ClassifierChainProbabilityResponse", probability_chain.predict(query))
    parity_matrix("ClassifierChainProbabilityResponseProba", probability_chain.predict_proba(query))
    var random_a = ClassifierChain(Nearest[DType.int64](), random_order=True, random_state=7)
    var random_b = ClassifierChain(Nearest[DType.int64](), random_order=True, random_state=7)
    _ = random_a.fit(X, labels)
    _ = random_b.fit(X, labels)
    assert_equal(random_a.order_[0], random_b.order_[0])
    assert_equal(random_a.order_[1], random_b.order_[1])
    print("PARITY ClassifierChainRandomOrder", random_a.order_[0], random_a.order_[1])
    parity_matrix("ClassifierChainRandom", random_a.predict(query))
    var changed = y.copy()
    for i in range(4):
        changed.unsafe_set(2 * i, changed.unsafe_get(2 * i) + 1)
    assert_true(abs(reg.score(X, changed) - 0.996) < 1e-12)
    var weights = nm.ones[DType.float64](Shape(4))
    assert_true(abs(reg.score(X, changed, weights.copy()) - 0.996) < 1e-12)
    with assert_raises():
        var unused = clf.score(X, labels, weights.copy())
    var first_x = nm.zeros[DType.float64](Shape(1, 1))
    var first_y = nm.zeros[DType.float64](Shape(1, 2))
    assert_true(isnan(reg.score(first_x, first_y)))
    var constant = nm.ones[DType.float64](Shape(4, 2))
    var const_reg = MultiOutputRegressor(Nearest[DType.float64]())
    _ = const_reg.fit(X, constant)
    assert_equal(const_reg.score(X, constant), Float64(1))
    var zero = nm.zeros[DType.float64](Shape(4, 2))
    assert_equal(const_reg.score(X, zero), Float64(0))
    var X32 = nm.zeros[DType.float32](Shape(4, 1))
    var y32 = nm.zeros[DType.float32](Shape(4, 2))
    for i in range(4):
        X32.unsafe_set(i, Float32(i))
        y32.unsafe_set(2 * i, Float32(i))
        y32.unsafe_set(2 * i + 1, Float32(i * 2))
    var reg32 = MultiOutputRegressor[Nearest[DType.float32, DType.float32], DType.float32](Nearest[DType.float32, DType.float32]())
    _ = reg32.fit(X32, y32)
    assert_equal(reg32.score(X32, y32), Float64(1))
    var wide_labels = labels.copy()
    for i in range(8):
        wide_labels.unsafe_set(i, wide_labels.unsafe_get(i) + 9007199254740992)
    var wide = MultiOutputClassifier(Nearest[DType.int64]())
    _ = wide.fit(X, wide_labels)
    var wide_pred = wide.predict(X)
    for i in range(8):
        assert_equal(wide_pred.unsafe_get(i), wide_labels.unsafe_get(i))
    # joblib rejects n_jobs=0; every other value yields the serial result.
    var invalid_jobs = MultiOutputRegressor(Nearest[DType.float64](), n_jobs=0)
    with assert_raises(contains="n_jobs == 0 in Parallel has no meaning"):
        _ = invalid_jobs.fit(X, y)
    assert_false(invalid_jobs.__sklearn_is_fitted__())
    for jobs in [-1, 2]:
        var parallel = MultiOutputRegressor(Nearest[DType.float64](), n_jobs=jobs)
        var expected = reg.predict(query)
        var observed = parallel.fit(X, y).predict(query)
        for i in range(6):
            assert_equal(observed.unsafe_get(i), expected.unsafe_get(i))
        var parallel_clf = MultiOutputClassifier(Nearest[DType.int64](), n_jobs=jobs)
        _ = parallel_clf.partial_fit(X, labels, clf.classes_.copy())
        assert_equal(parallel_clf.score(X, labels), Float64(1))
    var weighted = MultiOutputRegressor(WeightedMean())
    for i in range(4):
        weights.unsafe_set(i, Float64(i + 1))
    _ = weighted.fit(X, y, sample_weight=weights)
    var weighted_prediction = weighted.predict(X)
    assert_equal(weighted_prediction.unsafe_get(0), Float64(20))
    assert_equal(weighted_prediction.unsafe_get(1), Float64(1))
    print("PARITY MultiOutputRegressorScore", reg.score(X, changed))
    print("PARITY MultiOutputRegressorWeightedScore", reg.score(X, changed, weights.copy()))
    var changed_labels = labels.copy()
    changed_labels.unsafe_set(0, 1)
    changed_labels.unsafe_set(1, 1)
    print("PARITY MultiOutputClassifierScore", clf.score(X, changed_labels))
    print("PARITY ClassifierChainWeightedScore", cc.score(X, changed_labels, weights.copy()))
    parity_matrix("MultiOutputRegressorWeightedFit", weighted.predict(query))
    print("PASS: four estimators, clone, ordering, probabilities, validation, repeat fit")
