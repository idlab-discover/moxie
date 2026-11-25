"""Fitted metadata exposed by both dense searches and their composition traits."""
from moxie._feature_metadata import InputFeatureCount
from moxie.base import NumericClassifierClasses, clone
from moxie.linear_model import LogisticRegression
from moxie.model_selection import DenseSupervisedAdapter, GridSearchCV, ParameterGrid, RandomizedSearchCV
from numojo import Item, NDArray, Shape
from std.testing import assert_equal, assert_raises

comptime Model = LogisticRegression[]
comptime Labels = NDArray[DType.int64]
comptime Dense = NDArray[DType.float64]
comptime Adapter = DenseSupervisedAdapter[Model, Labels]


def data(width: Int, first_label: Int) raises -> Tuple[Dense, Labels]:
    var X = Dense(Shape(12, width))
    var y = Labels(Shape(12))
    for i in range(12):
        y[Item(i)] = Int64(first_label + (1 - i % 2) * 4)
        for j in range(width):
            X[Item(i, j)] = Float64(i % 2) * 3.0 + Float64((i + j) % 5) / 10.0
    return (X^, y^)


def configure(mut estimator: Model, params: Dict[String, Float64]) raises:
    var config = estimator.get_params()
    config.C = params["C"]
    _ = estimator.set_params(config)


def assert_metadata[T: InputFeatureCount & NumericClassifierClasses](
    search: T, width: Int, first_label: Int,
) raises where T.Target == Labels:
    assert_equal(search.get_n_features_in(), width)
    var classes = rebind_var[Labels](search.get_classes())
    assert_equal(classes.size, 2)
    assert_equal(classes[Item(0)], Int64(first_label))
    assert_equal(classes[Item(1)], Int64(first_label + 4))


def main() raises:
    var first = data(2, -3)
    var next = data(3, 5)
    ref X = first[0]
    ref y = first[1]
    ref X_next = next[0]
    ref y_next = next[1]
    var candidates = Dict[String, List[Float64]]()
    candidates["C"] = [0.5, 1.0]
    var grid = ParameterGrid(candidates)
    var adapter = Adapter(configure, n_splits=3)
    var exhaustive = GridSearchCV[Adapter](Model(), grid, adapter)
    with assert_raises(contains="AttributeError: classes_"):
        _ = exhaustive.classes_()
    with assert_raises(contains="AttributeError: n_features_in_"):
        _ = exhaustive.n_features_in_()
    _ = exhaustive.fit(X, y)
    var original = exhaustive.classes_()
    print("grid", exhaustive.n_features_in_(), original[Item(0)], original[Item(1)])
    original[Item(0)] = 100
    assert_metadata(exhaustive, 2, -3)

    var snapshot = exhaustive.copy()
    _ = exhaustive.fit(X_next, y_next)
    assert_metadata(exhaustive, 3, 5)
    assert_metadata(snapshot, 2, -3)
    var fresh = clone(exhaustive)
    with assert_raises(contains="AttributeError: classes_"):
        _ = fresh.classes_()
    with assert_raises(contains="AttributeError: n_features_in_"):
        _ = fresh.n_features_in_()
    with assert_raises():
        _ = exhaustive.fit(X_next, Labels(Shape(1)))
    with assert_raises(contains="AttributeError: classes_"):
        _ = exhaustive.classes_()
    with assert_raises(contains="AttributeError: n_features_in_"):
        _ = exhaustive.n_features_in_()

    var randomized = RandomizedSearchCV[Adapter](
        Model(), grid, adapter, n_iter=2, random_state=42,
    )
    _ = randomized.fit(X_next, y_next)
    assert_metadata(randomized, 3, 5)
    var classes = randomized.classes_()
    print("random", randomized.n_features_in_(), classes[Item(0)], classes[Item(1)])

    var no_refit = RandomizedSearchCV[Adapter](
        Model(), grid, adapter, n_iter=2, random_state=42, refit=False,
    )
    _ = no_refit.fit(X, y)
    with assert_raises(contains="AttributeError: classes_"):
        _ = no_refit.classes_()
    with assert_raises(contains="AttributeError: n_features_in_"):
        _ = no_refit.n_features_in_()
    print("unavailable")
