from std.math import isnan
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from numojo import zeros, Shape, NDArray, NDArrayStrides
from moxie.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier, GraphClassifierAdapter
from moxie.semi_supervised._label_propagation import GraphClassifier
from moxie.semi_supervised._affinity import RBFAffinity
from moxie.semi_supervised._probability import RowNormalizer
from moxie.semi_supervised._contracts import ZeroMassPolicy, DenseAffinity

@fieldwise_init
struct Asymmetric(DenseAffinity, Copyable, Movable):
    comptime Input = NDArray[DType.float64]
    def affinity(self, train: Self.Input, query: Self.Input) raises -> NDArray[DType.float64]:
        var result = zeros[DType.float64](Shape(train.shape[0], query.shape[0]))
        for i in range(train.shape[0]):
            for j in range(query.shape[0]):
                result.unsafe_set(i * query.shape[0] + j, Float64(1 + i + 2*j))
        return result^

def graph[dtype: DType, spreading: Bool]() raises:
    var X = zeros[dtype](Shape(4, 2))
    X.unsafe_set(2, 1)
    X.unsafe_set(4, 0.9)
    X.unsafe_set(6, 0.1)
    var model = GraphClassifier[spreading, dtype](gamma=2)
    with assert_raises():
        _ = model.predict(X)
    model.fit(X, List[Int]([7, 3, -1, -1]))
    assert_equal(model.classes_, List[Int]([3, 7]))
    assert_equal(model.transduction_, List[Int]([7, 3, 3, 7]))
    assert_equal(model.predict(X), List[Int]([7, 3, 3, 7]))
    assert_equal(model.score(X, [7, 3, 3, 7]), 1.0)
    for invalid_weight in List[Float64]([Float64("nan"), Float64("inf"), -Float64("inf")]):
        with assert_raises():
            _ = model.score(X, [7, 3, 3, 7], List[Float64]([1.0, 1.0, 1.0, invalid_weight]))
    with assert_raises():
        _ = model.score(X, [7, 3, 3, 7], List[Float64]([1.0]))
    with assert_raises():
        _ = model.score(X, [7, 3, 3, 7], List[Float64]([1.0, -1.0, 0.0, 0.0]))
    assert_equal(model.score(X, [7, 3, 3, 3], List[Float64]([1.0, 2.0, 3.0, -1.0])), 1.2)
    assert_true(model.n_iter_ > 0)
    var cloned = model.__sklearn_clone__()
    assert_false(cloned.__sklearn_is_fitted__())
    X.unsafe_set(0, 99)
    assert_equal(Float64(model.X_.unsafe_get(0)), 0.0)
    X.unsafe_set(0, 0)
    var params = model.get_params()
    params.gamma = 0
    model.set_params(params)
    model.set_params(gamma=0.0)
    model.fit(X, List[Int]([7, 3, -1, -1]))
    assert_equal(model.n_features_in_, 2)
    with assert_raises():
        model.fit(X, List[Int]([-1, -1, -1, -1]))
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises():
        model.fit(X, List[Int]([1]))
    var invalid = GraphClassifier[spreading, dtype](kernel="knn")
    with assert_raises():
        invalid.fit(X, List[Int]([7, 3, -1, -1]))
    var view = X.view_with_layout(Shape(4, 1), NDArrayStrides(2, 1), 0)
    model.fit(view, List[Int]([7, 3, -1, -1]))
    assert_equal(model.n_features_in_, 1)
    _ = model.predict_proba(view)

def main() raises:
    graph[DType.float64, False]()
    graph[DType.float64, True]()
    graph[DType.float32, False]()
    graph[DType.float32, True]()
    var X = zeros[DType.float64](Shape(4, 1))
    X.unsafe_set(1, 1)
    X.unsafe_set(2, 0.9)
    X.unsafe_set(3, 0.1)
    var custom = LabelSpreading[DType.float64, Asymmetric]("callable", affinity=Asymmetric())
    custom.fit(X, List[Int]([0, 1, -1, -1]))
    _ = custom.predict_proba(X)
    var custom_copy = custom.__sklearn_clone__()
    custom_copy.set_params(custom.get_params())
    custom_copy.fit(X, List[Int]([0, 1, -1, -1]))
    assert_equal(custom_copy.transduction_, custom.transduction_)
    var strings = LabelPropagation[DType.float64, RBFAffinity[DType.float64], String](gamma=2)
    var labels: List[Optional[String]] = [String("z"), String("a"), None, None]
    strings.fit(X, labels)
    assert_equal(strings.classes_, List[String](["a", "z"]))
    assert_equal(strings.predict(X), List[String](["z", "a", "a", "z"]))
    var string_base = GraphClassifierAdapter[False, DType.float64, String](strings.__sklearn_clone__())
    var string_model = SelfTrainingClassifier[type_of(string_base)](string_base.copy(), threshold=0.5)
    string_model.fit(X, labels)
    assert_equal(string_model.predict(X), List[String](["z", "a", "a", "z"]))
    var base = GraphClassifierAdapter(LabelPropagation(gamma=2))
    var model = SelfTrainingClassifier[type_of(base)](base.copy(), threshold=0.5)
    with assert_raises():
        _ = model.predict_proba(X)
    model.fit(X, List[Int]([0, 1, -1, -1]))
    assert_equal(model.n_iter_, 1)
    assert_equal(model.termination_condition_, "all_labeled")
    assert_equal(model.labeled_iter_, List[Int]([0, 0, 1, 1]))
    assert_equal(model.predict(X), List[Int]([0, 1, 1, 0]))
    assert_equal(model.score(X, [0, 1, 1, 0]), 1.0)
    assert_false(base.estimator.__sklearn_is_fitted__())
    var cloned = model.__sklearn_clone__()
    assert_false(cloned.__sklearn_is_fitted__())
    model.set_params(max_iter=Optional[Optional[Int]](Optional(0)))
    model.fit(X, List[Int]([0, 1, -1, -1]))
    assert_equal(model.n_iter_, 0)
    assert_equal(model.termination_condition_, "max_iter")
    assert_equal(model.labeled_iter_, List[Int]([0, 0, -1, -1]))
    model.max_iter = None
    model.criterion = "k_best"
    model.k_best = 1
    model.fit(X, List[Int]([0, 1, -1, -1]))
    assert_equal(model.n_iter_, 2)
    model.max_iter = 10
    model.criterion = "threshold"
    model.threshold = 0.9999
    model.fit(X, List[Int]([0, 1, -1, -1]))
    assert_equal(model.termination_condition_, "no_change")
    model.max_iter = 1
    model.fit(X, List[Int]([0, 1, -1, -1]))
    assert_equal(model.termination_condition_, "max_iter")
    model.fit(X, List[Int]([0, 1, 1, 0]))
    assert_equal(model.n_iter_, 0)
    assert_equal(model.termination_condition_, "all_labeled")
    var tied_base = GraphClassifierAdapter(LabelPropagation(gamma=0))
    var tied = SelfTrainingClassifier[type_of(tied_base)](tied_base.copy(), criterion="k_best", k_best=1, max_iter=1)
    tied.fit(X, List[Int]([0, 1, -1, -1]))
    assert_equal(tied.labeled_iter_, List[Int]([0, 0, 1, -1]))
    assert_equal(tied.transduction_[2].value(), 0)
    var missing = SelfTrainingClassifier()
    with assert_raises():
        missing.fit(X, List[Int]([0, 1, -1, -1]))
    var weights = List[Float64]([1.0, 2.0, 3.0, 4.0])
    var weighted = LabelPropagation(gamma=2)
    weighted.fit(X, List[Int]([0, 1, -1, -1]))
    assert_equal(weighted.score(X, [0, 1, 1, 1], weights.copy()), 0.6)
    var empty = zeros[DType.float64](Shape(0, 1))
    with assert_raises():
        _ = weighted.predict(empty)
    var wrong_features = zeros[DType.float64](Shape(1, 2))
    with assert_raises():
        _ = weighted.predict(wrong_features)
    X.unsafe_set(0, Float64("nan"))
    with assert_raises():
        weighted.fit(X, List[Int]([0, 1, -1, -1]))
    X.unsafe_set(0, 0)
    var zero = zeros[DType.float64](Shape(1, 2))
    var normalized = RowNormalizer().normalize(zero, ZeroMassPolicy.PRESERVE)
    assert_equal(normalized.unsafe_get(0), 0.0)
    var ieee = RowNormalizer().normalize(zero, ZeroMassPolicy.IEEE_DIVIDE)
    assert_true(isnan(ieee.unsafe_get(0)))
    zero.unsafe_set(0, 1)
    zero.unsafe_set(1, -1)
    normalized = RowNormalizer().normalize(zero, ZeroMassPolicy.PRESERVE)
    assert_equal(normalized.unsafe_get(0), 1.0)
    assert_equal(normalized.unsafe_get(1), -1.0)
    with assert_raises():
        _ = RowNormalizer().normalize(zero, 9)
    print("PASS dense graph and self-training contracts")
