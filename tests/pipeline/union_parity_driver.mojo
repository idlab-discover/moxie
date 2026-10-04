from numojo import NDArray, Shape, NDArrayStrides, zeros
from std.math import nan
from moxie.base import clone
from moxie.pipeline import FeatureUnion, UnionPair, UnionSentinel, union_step
from moxie._pipeline import StepMode
from union_fixtures import TestTransformer


def emit[dtype: DType](name: String, X: NDArray[dtype]) raises:
    print("matrix", name, 32 if dtype == DType.float32 else 64, X.shape[0], X.shape[1])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            print(Float64(X.item(i, j)))


def check[dtype: DType]() raises:
    for shape in range(5):
        var rows = 0 if shape == 1 else (1 if shape == 2 else 3)
        var cols = 0 if shape == 2 else shape + 1
        var X = zeros[dtype](Shape(rows, cols))
        var y = zeros[dtype](Shape(rows))
        for i in range(rows):
            y.itemset(i, Scalar[dtype](i + 1))
            for j in range(cols):
                X.itemset(i * cols + j, Scalar[dtype](i * cols + j - 3))
        var weights: Dict[String, Float64] = {"a": 0.1, "b": -2.0, "raw": 0.0}
        var union = FeatureUnion(UnionPair(union_step[dtype]("a", TestTransformer[dtype](2)),
            UnionPair(union_step[dtype]("b", TestTransformer[dtype, True](3, 1)),
            UnionPair(UnionSentinel[dtype]("raw"), UnionSentinel[dtype]("drop", StepMode.drop)))),
            transformer_weights=weights.copy())
        var label = String(shape)
        emit("fused-" + label, union.fit_transform(X, y.copy()))
        emit("transform-" + label, union.transform(X))
        _ = union.fit(X, y.copy())
        emit("fit-" + label, union.transform(X))
        var cloned = clone(union)
        var params = cloned.get_params()
        params.transformer_list.first.estimator.factor = -4
        _ = cloned.set_params(params)
        emit("clone-" + label, cloned.fit_transform(X, y.copy()))
        print("state", 32 if dtype == DType.float32 else 64, shape,
              union.transformer_list.first.estimator.fit_calls,
              union.transformer_list.second.first.estimator.fit_calls,
              union.transformer_list.second.first.estimator.fused_calls,
              union.transformer_list.first.estimator.target_sum,
              Int(clone(union).__sklearn_is_fitted__()))
    var backing = zeros[dtype](Shape(3, 4))
    for i in range(12):
        backing.itemset(i, Scalar[dtype](i + 1))
    var view = backing.view_with_layout(Shape(2, 2), NDArrayStrides(-4, 2), 8)
    var passes = FeatureUnion(UnionPair(UnionSentinel[dtype]("raw"), UnionSentinel[dtype]("again")))
    emit("strided", passes.transform(view))
    emit("single", FeatureUnion(UnionSentinel[dtype]("raw")).transform(view))
    var dropped = FeatureUnion(UnionPair(UnionSentinel[dtype]("a", StepMode.drop), UnionSentinel[dtype]("b", StepMode.drop)))
    emit("dropped", dropped.fit_transform(view))
    backing.itemset(8, nan[dtype]())
    backing.itemset(10, Scalar[dtype](Float64.MAX * 2.0))
    var weights: Dict[String, Float64] = {"raw": 0.0, "again": -1.5}
    passes.transformer_weights = weights.copy()
    emit("nonfinite", passes.transform(view))
    weights["raw"] = Float64.MAX * 2.0
    weights["again"] = nan[DType.float64]()
    passes.transformer_weights = weights.copy()
    emit("nonfinite-weights", passes.transform(view))
    _ = passes.fit(view)
    var names = passes.get_feature_names_out()
    for name in names:
        print("name", 32 if dtype == DType.float32 else 64, name)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
