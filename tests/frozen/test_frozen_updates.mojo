"""Synthetic stateful probes for incremental delegation and cluster identity."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import (
    clone, IncrementalRegressor, IncrementalClassifier, IncrementalTransformer,
    ClusterMixin, FitTransformer,
)
from moxie.frozen import FrozenEstimator, is_clusterer
from moxie.utils.validation import check_is_fitted


struct RegressorProbe(IncrementalRegressor):
    comptime Params = Int
    comptime Input = Int
    comptime Target = Int
    comptime Prediction = Int
    comptime SampleWeight = Int
    var state: Int

    def __init__(out self):
        self.state = 0

    def get_params(self, deep: Bool = True) raises -> Int:
        return 0

    def set_params(mut self, params: Int) raises:
        pass

    def __sklearn_clone__(self) raises -> Self:
        return Self()

    def __sklearn_is_fitted__(self) -> Bool:
        return self.state != 0

    def fit(mut self, X: Int, y: Int) raises:
        raise Error("inner fit must not run")

    def partial_fit(mut self, X: Int, y: Int) raises:
        if y < 0:
            raise Error("invalid target")
        self.state += X + y

    def predict(self, X: Int) raises -> Int:
        check_is_fitted(self)
        return self.state

    def score(self, X: Int, y: Int, sample_weight: Optional[Int] = None) raises -> Float64:
        raise Error("scoring outside probe scope")


struct ClassifierProbe(IncrementalClassifier):
    comptime Params = Int
    comptime Input = Int
    comptime Target = Int
    comptime Prediction = Int
    comptime SampleWeight = Int
    comptime Classes = Int
    var state: Int
    var classes_: Int

    def __init__(out self):
        self.state = 0
        self.classes_ = 0

    def get_params(self, deep: Bool = True) raises -> Int:
        return 0

    def set_params(mut self, params: Int) raises:
        pass

    def __sklearn_clone__(self) raises -> Self:
        return Self()

    def __sklearn_is_fitted__(self) -> Bool:
        return self.classes_ != 0

    def fit(mut self, X: Int, y: Int) raises:
        raise Error("inner fit must not run")

    def partial_fit(mut self, X: Int, y: Int, classes: Optional[Int] = None) raises:
        if not self.classes_ and not classes:
            raise Error("classes required on first call")
        if classes:
            if self.classes_ and classes.value() != self.classes_:
                raise Error("classes changed")
            self.classes_ = classes.value()
        self.state += X + y

    def predict(self, X: Int) raises -> Int:
        check_is_fitted(self)
        return self.state

    def score(self, X: Int, y: Int, sample_weight: Optional[Int] = None) raises -> Float64:
        raise Error("scoring outside probe scope")


struct ClusterProbe(IncrementalTransformer, ClusterMixin):
    comptime Params = Int
    comptime Input = Int
    comptime Target = Int
    comptime Transformed = Int
    comptime Labels = Int
    var state: Int

    def __init__(out self):
        self.state = 0

    def get_params(self, deep: Bool = True) raises -> Int:
        return 0

    def set_params(mut self, params: Int) raises:
        pass

    def __sklearn_clone__(self) raises -> Self:
        return Self()

    def __sklearn_is_fitted__(self) -> Bool:
        return self.state != 0

    def fit(mut self, X: Int, y: Optional[Int] = None) raises:
        raise Error("inner fit must not run")

    def partial_fit(mut self, X: Int, y: Optional[Int] = None) raises:
        if X < 0:
            raise Error("invalid input")
        self.state += X + (y.value() if y else 0)

    def transform(self, X: Int) raises -> Int:
        check_is_fitted(self)
        return self.state + X

    def fit_predict(mut self, X: Int, y: Optional[Int] = None) raises -> Int:
        raise Error("composite fit must not run")


def update[E: IncrementalRegressor](
    mut estimator: E, X: E.Input, y: E.Target
) raises:
    estimator.partial_fit(X, y)


def main() raises:
    comptime assert conforms_to(FrozenEstimator[RegressorProbe], IncrementalRegressor)
    comptime assert conforms_to(FrozenEstimator[ClassifierProbe], IncrementalClassifier)
    comptime assert conforms_to(FrozenEstimator[ClusterProbe], IncrementalTransformer)
    comptime assert not conforms_to(FrozenEstimator[ClusterProbe], ClusterMixin)
    comptime assert not conforms_to(FrozenEstimator[ClusterProbe], FitTransformer)

    var regression = FrozenEstimator(RegressorProbe())
    var shared = clone(regression)
    var copied = regression.copy()
    var params = regression.get_params()
    with assert_raises(contains="NotFittedError"):
        regression.fit(1, 2)
    update(regression, 2, 3)  # partial_fit may initialize an unfitted estimator
    check_is_fitted(shared)
    assert_equal(shared.predict(0), 5)
    assert_equal(copied.predict(0), 5)
    assert_equal(params.estimator().state, 5)
    print("regression_first", shared.predict(0))
    regression.fit(999, 999)
    var nested = FrozenEstimator(shared.copy())
    nested.partial_fit(3, 4)
    assert_equal(regression.predict(0), 12)
    print("regression_nested", regression.predict(0))
    with assert_raises(contains="invalid target"):
        shared.partial_fit(1, -1)
    assert_equal(regression.predict(0), 12)
    var restored = FrozenEstimator[RegressorProbe](params)
    regression.set_params(estimator=RegressorProbe())
    regression.partial_fit(1, 1)
    assert_equal(restored.predict(0), 12)
    assert_equal(regression.predict(0), 2)
    print("regression_replaced", regression.predict(0))
    print("regression_old", restored.predict(0))
    regression.set_params(params)
    assert_equal(regression.predict(0), 12)

    var classification = FrozenEstimator(ClassifierProbe())
    with assert_raises(contains="classes required"):
        classification.partial_fit(1, 2)
    assert_false(classification.__sklearn_is_fitted__())
    classification.partial_fit(1, 2, classes=Optional[Int](7))
    var classifier_clone = clone(classification)
    classification.partial_fit(3, 4)
    assert_equal(classifier_clone.predict(0), 10)
    assert_equal(classifier_clone.estimator().classes_, 7)
    with assert_raises(contains="classes changed"):
        classification.partial_fit(1, 2, Optional[Int](8))
    assert_equal(classification.predict(0), 10)
    assert_false(is_clusterer(classification))
    print("classifier_state", classifier_clone.predict(0))
    print("classifier_classes", classifier_clone.estimator().classes_)

    var cluster = FrozenEstimator(ClusterProbe())
    assert_true(is_clusterer(ClusterProbe()))
    assert_true(is_clusterer(cluster))
    assert_false(is_clusterer(RegressorProbe()))
    assert_false(is_clusterer(regression))
    var nested_cluster = FrozenEstimator(cluster.copy())
    assert_true(is_clusterer(nested_cluster))
    assert_false(is_clusterer(nested))
    cluster.partial_fit(3)
    nested_cluster.partial_fit(4, Optional[Int](5))
    cluster.fit(999)
    assert_equal(cluster.transform(0), 12)
    with assert_raises(contains="invalid input"):
        cluster.partial_fit(-1)
    assert_equal(cluster.transform(0), 12)
    print("cluster_state", cluster.transform(0))
    print("cluster_identity", Int(is_clusterer(cluster)))
    print("nested_cluster_identity", Int(is_clusterer(nested_cluster)))
    print("regressor_identity", Int(is_clusterer(regression)))
