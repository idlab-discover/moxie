from moxie.dummy import DummyRegressor
from moxie.model_selection import DenseSupervisedAdapter, IndexSplit, cross_val_predict
from moxie.pipeline import Pipeline
from moxie.preprocessing import StandardScaler
from moxie.svm import SVC
from numojo import Item, NDArray, Shape
from std.testing import assert_equal, assert_false, assert_raises

comptime FloatArray = NDArray[DType.float32]
comptime Regression = DummyRegressor[DType.float32]
comptime RegressionAdapter = DenseSupervisedAdapter[Regression, FloatArray, DType.float32]


def check_regression() raises:
    var X = FloatArray(Shape(12, 1))
    var y = FloatArray(Shape(12, 2))
    var vector = FloatArray(Shape(12))
    var weights = FloatArray(Shape(12))
    for row in range(12):
        X[Item(row, 0)] = Float32(row)
        vector[Item(row)] = Float32(row % 5)
        y[Item(row, 0)] = vector[Item(row)]
        y[Item(row, 1)] = Float32((row * 3) % 7)
        weights[Item(row)] = Float32(row + 1)
    var model = Regression()
    var adapter = RegressionAdapter()
    var plain: FloatArray = cross_val_predict[RegressionAdapter](model, X, vector, adapter)
    assert_equal(plain.ndim, 1)
    assert_equal(plain.shape[0], 12)
    assert_false(model.__sklearn_is_fitted__())
    _ = model.fit(X, y)
    var original = model.predict(X)

    var folds = adapter.default_cv(X, y)
    var reversed_folds = List[IndexSplit]()
    while len(folds) > 0:
        var fold = folds.pop()
        reversed_folds.append(fold^)
    var weighted: FloatArray = cross_val_predict[RegressionAdapter](
        model, X, y, adapter, reversed_folds, Optional(weights.copy()),
    )
    var default_weighted = cross_val_predict[RegressionAdapter](
        model, X, y, adapter, fit_metadata=weights.copy(),
    )
    assert_equal(weighted.ndim, 2)
    assert_equal(weighted.shape[0], 12)
    assert_equal(weighted.shape[1], 2)
    var after = model.predict(X)
    for row in range(12):
        print("regression", plain[Item(row)])
        for column in range(2):
            assert_equal(after[Item(row, column)], original[Item(row, column)])
            assert_equal(weighted[Item(row, column)], default_weighted[Item(row, column)])
            print("weighted", weighted[Item(row, column)])

    with assert_raises(contains="requires test indices to partition samples"):
        _ = reversed_folds.pop()
        _ = cross_val_predict[RegressionAdapter](model, X, y, adapter, cv=reversed_folds^)
    with assert_raises(contains="one value per sample"):
        _ = cross_val_predict[RegressionAdapter](
            model, X, y, adapter, fit_metadata=FloatArray(Shape(13)),
        )
    with assert_raises(contains="inconsistent sample counts"):
        _ = cross_val_predict[RegressionAdapter](model, X, FloatArray(Shape(13)), adapter)
    with assert_raises(contains="method='predict' only"):
        _ = cross_val_predict[RegressionAdapter](model, X, y, adapter, method="predict_proba")


def check_pipeline() raises:
    comptime Labels = List[String]
    comptime Classifier = Pipeline[StandardScaler[], SVC[String]]
    comptime Adapter = DenseSupervisedAdapter[Classifier, Labels]
    var X = NDArray[DType.float64](Shape(30, 2))
    var y = Labels()
    var classes: Labels = ["red", "green", "blue"]
    for row in range(30):
        var label = (row * 11 % 31) // 11
        X[Item(row, 0)] = 1.5 * Float64(label) + Float64(row) / 20.0
        X[Item(row, 1)] = Float64((row * 7) % 5)
        y.append(classes[label].copy())
    var model = Pipeline(StandardScaler(), SVC[String]())
    var adapter = Adapter()
    var result: Labels = cross_val_predict[Adapter](model, X, y, adapter)
    assert_equal(len(result), 30)
    assert_false(model.__sklearn_is_fitted__())
    for label in result:
        print("classification", label)


def main() raises:
    check_regression()
    check_pipeline()
