"""Unsupervised prefixes before estimators with another target type (sklearn passes y; they ignore it)."""
from numojo import NDArray, Shape
from std.testing import assert_equal, assert_false, assert_true
from moxie.base import TransformerMixin, FitTransformer, UnsupervisedEstimator, clone
from moxie.pipeline import Pipeline
from moxie.preprocessing import StandardScaler
from moxie.svm import SVC


struct FitOnlyCopy(TransformerMixin, UnsupervisedEstimator):
    """Test fixture: fit/transform without fit_transform, recording the target it received."""
    comptime Input = NDArray[DType.float64]
    comptime Target = NDArray[DType.float64]
    comptime Transformed = NDArray[DType.float64]
    comptime Params = Int
    var fitted: Bool
    var saw_target: Bool
    def __init__(out self):
        self.fitted = False
        self.saw_target = True
    def get_params(self, deep: Bool = True) raises -> Int:
        return 0
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Int,
    ) raises -> ref[self] Self:
        return self
    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted
    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        self.saw_target = Bool(y)
        self.fitted = True
        return self
    def transform(self, X: Self.Input) raises -> Self.Transformed:
        return X.copy()


struct FitTransformCopy(TransformerMixin, FitTransformer, UnsupervisedEstimator):
    """Test fixture: fit_transform path, recording the target it received."""
    comptime Input = NDArray[DType.float64]
    comptime Target = NDArray[DType.float64]
    comptime Transformed = NDArray[DType.float64]
    comptime Params = Int
    var fitted: Bool
    var saw_target: Bool
    def __init__(out self):
        self.fitted = False
        self.saw_target = True
    def get_params(self, deep: Bool = True) raises -> Int:
        return 0
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Int,
    ) raises -> ref[self] Self:
        return self
    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted
    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        raise Error("fit_transform must be preferred")
    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self.saw_target = Bool(y)
        self.fitted = True
        return X.copy()
    def transform(self, X: Self.Input) raises -> Self.Transformed:
        return X.copy()


def data() raises -> Tuple[NDArray[DType.float64], List[Float64]]:
    """Must match `prefix_data()` in parity.py."""
    var X = NDArray[DType.float64](Shape(12, 3))
    var y = List[Float64]()
    for i in range(12):
        var label = Float64(i % 2)
        y.append(label)
        X.itemset(i * 3 + 0, Float64(i) * 10.0 + label * 25.0)
        X.itemset(i * 3 + 1, Float64((i * 7) % 5) * 0.01 - label * 0.02)
        X.itemset(i * 3 + 2, 1000.0 + Float64((i * 3) % 4))
    return (X^, y^)


def main() raises:
    var d = data()
    var X = d[0].copy()
    var y = d[1].copy()

    var fit_only = Pipeline(FitOnlyCopy(), SVC())
    _ = fit_only.fit(X, y)
    assert_true(fit_only.transformer.fitted)
    assert_false(fit_only.transformer.saw_target)
    var fused = Pipeline(FitTransformCopy(), SVC())
    _ = fused.fit(X, y)
    assert_true(fused.transformer.fitted)
    assert_false(fused.transformer.saw_target)

    var pipe = Pipeline(StandardScaler(), SVC(), transformer_name="standardscaler", estimator_name="svc")
    var prediction = pipe.fit(X, y).predict(X)
    var decision = pipe.decision_function(X)
    var accuracy = pipe.score(X, y)
    # Same scenario composed by hand: the pipeline adds no numerical work.
    var scaler = StandardScaler()
    var model = SVC()
    var scaled = scaler.fit_transform(X)
    _ = model.fit(scaled, y)
    var manual = model.decision_function(scaler.transform(X))
    assert_equal(len(prediction), 12)
    for i in range(12):
        assert_equal(Float64(decision.item(i)), Float64(manual.item(i)))
    var other = clone(pipe)
    assert_false(other.__sklearn_is_fitted__())
    _ = other.fit(X, y)
    var again = other.decision_function(X)
    for i in range(12):
        assert_equal(Float64(again.item(i)), Float64(decision.item(i)))
    var line = String("prefix_svc ") + String(accuracy)
    for i in range(12):
        line += " " + String(prediction[i]) + " " + String(Float64(decision.item(i)))
    print(line)
