from std.testing import assert_equal, assert_false
from numojo import NDArray, Shape, zeros
from moxie.semi_supervised import SemiSupervisedClassifier, LabelPropagation, LabelSpreading, SelfTrainingClassifier, GaussianNBAdapter
from moxie._nb_gaussian import GaussianNB
from moxie._naive_bayes_dense import IntListDomain


def generic_chain[T: SemiSupervisedClassifier](mut model: T, X: T.Input, y: T.Target, params: T.Params) raises -> T.Prediction:
    return model.set_params(params).fit(X, y).predict(X)


def graph[dtype: DType]() raises:
    var X = zeros[dtype](Shape(4, 1))
    X.unsafe_set(1, 1)
    X.unsafe_set(2, 0.9)
    X.unsafe_set(3, 0.1)
    var labels = List[Int]([0, 1, -1, -1])
    var expected = List[Int]([0, 1, 1, 0])
    var propagation = LabelPropagation[dtype]()
    assert_equal(propagation.set_params(gamma=2.0).fit(X, labels).predict(X), expected)
    var params = propagation.get_params()
    assert_equal(generic_chain(propagation, X, labels, params), expected)
    var cloned = propagation.__sklearn_clone__()
    assert_false(cloned.__sklearn_is_fitted__())
    var spreading = LabelSpreading[dtype]()
    assert_equal(spreading.set_params(gamma=2.0).fit(X, labels).predict(X), expected)
    var tagged: List[Optional[Int]] = [Optional[Int](0), Optional[Int](1), Optional[Int](), Optional[Int]()]
    assert_equal(spreading.fit(X, tagged).predict(X), expected)
    var spread_params = spreading.get_params()
    assert_equal(generic_chain(spreading, X, labels, spread_params), expected)
    var adapter = GaussianNBAdapter[dtype](GaussianNB[IntListDomain, dtype]())
    var training = SelfTrainingClassifier[type_of(adapter), dtype](adapter.copy())
    var fully_labeled = List[Int]([0, 1, 1, 0])
    assert_equal(training.set_params(threshold=0.8).fit(X, fully_labeled).predict(X), expected)
    var training_params = training.get_params()
    assert_equal(training.set_params(training_params).fit(X, tagged).predict(X), expected)
    var training_clone = training.__sklearn_clone__()
    assert_false(training_clone.__sklearn_is_fitted__())


def main() raises:
    graph[DType.float32]()
    graph[DType.float64]()
    print("PASS concrete and generic fluent mutations, both dtypes and tagged targets")
