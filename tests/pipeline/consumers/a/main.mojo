from moxie import ProbeA, ProbeB
from moxie._pipeline import NamedStep, StepPair
from moxie.base import clone
from moxie.pipeline import FeatureUnion, UnionPair, UnionSentinel
from numojo import NDArray, Shape
from std.testing import assert_equal, assert_false
from test_plumbing import contracts

def check[dtype: DType]() raises:
    var X = NDArray[dtype](Shape(2, 2))
    for i in range(2):
        for j in range(2):
            X.itemset(i * 2 + j, 2)
    var shared = X.view()
    var snapshot = X.copy()
    shared.itemset(0, 3)
    assert_equal(Float64(X.item(0)), 3.0)
    assert_equal(Float64(snapshot.item(0)), 2.0)
    var tree = StepPair(NamedStep("a", ProbeA()), NamedStep("b", ProbeB()))
    var a = tree.first.estimator.consume(X)
    var b = tree.second.estimator.consume(shared)
    assert_equal(Float64(a.item(0)), 13.0)
    assert_equal(Float64(b.item(0)), 13.0)
    assert_equal(Float64(b.item(1)), 8.0)
    var union = FeatureUnion(UnionPair(UnionSentinel[dtype]("left"), UnionSentinel[dtype]("right")))
    var joined = union.fit_transform(X)
    assert_equal(joined.shape[0], 2)
    assert_equal(joined.shape[1], 4)
    assert_equal(Float64(joined.item(0, 2)), 3.0)
    joined.itemset(0, 99)
    assert_equal(Float64(X.item(0)), 3.0)
    var independent = clone(tree)
    independent.first.set_named_params("a", 7)
    assert_equal(tree.first.estimator.config, 1)
    assert_false(independent.__sklearn_is_fitted__())

def main() raises:
    contracts()
    check[DType.float32]()
    check[DType.float64]()
    print("Isolated dense composition consumer passed")
