"""Feature-name composition probes, not statistical estimators."""
from std.testing import assert_equal, assert_false, assert_raises
from moxie.base import SupervisedEstimator, TransformerMixin, Predictor, clone
from moxie.pipeline import Pipeline, UnionFeatureNames
from test_sequential import Shift, Recorder


struct NameStep(SupervisedEstimator, TransformerMixin, Predictor, UnionFeatureNames):
    comptime Input = Int
    comptime Target = Int
    comptime Transformed = Int
    comptime Prediction = Int
    comptime Params = String
    var prefix: String
    var fitted: Bool

    def __init__(out self, prefix: String):
        self.prefix = prefix
        self.fitted = False
    def get_params(self, deep: Bool = True) raises -> String:
        return self.prefix
    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: String,
    ) raises -> ref[self] Self:
        self.prefix = params
        return self
    def __sklearn_clone__(self) raises -> Self:
        return Self(self.prefix)
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted
    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Int,
        ref[y_origin] y: Int,
    ) raises -> ref[self] Self:
        self.fitted = True
        return self
    def transform(self, X: Int) raises -> Int:
        return X
    def predict(self, X: Int) raises -> Int:
        return X
    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        var names: List[String]
        if input_features:
            names = input_features.value().copy()
        else:
            if not self.fitted:
                raise Error("NotFittedError: default names require fit")
            names = ["x0", "x1"]
        for i in range(len(names)):
            if names[i] == "invalid":
                raise Error("ValueError: invalid feature name")
            names[i] = self.prefix + names[i]
        return names^


def emit(label: String, names: List[String]):
    print(label, end="")
    for name in names:
        print(" " + name, end="")
    print()


def main() raises:
    var pipe = Pipeline(NameStep("a_"), NameStep("b_"))
    with assert_raises():
        _ = pipe.get_feature_names_out()
    # There is no pipeline-wide fitted check: explicit names can be stateless.
    var inputs: List[String] = ["height", "width"]
    var names = pipe.get_feature_names_out(inputs.copy())
    assert_equal(names[0], "b_a_height")
    assert_equal(names[1], "b_a_width")
    assert_equal(inputs[0], "height")
    emit("names_explicit", names)
    names[0] = "changed"
    assert_equal(pipe.get_feature_names_out(inputs.copy())[0], "b_a_height")
    _ = pipe.fit(1, 1)
    emit("names_default", pipe.get_feature_names_out())
    assert_equal(pipe.get_feature_names_out()[1], "b_a_x1")
    emit("names_empty", pipe.get_feature_names_out(List[String]()))
    var duplicates: List[String] = ["same", "same"]
    emit("names_duplicates", pipe.get_feature_names_out(duplicates^))
    var invalid: List[String] = ["invalid"]
    var valid: List[String] = ["valid"]
    with assert_raises():
        _ = pipe.get_feature_names_out(invalid^)
    # Final-step validation errors propagate as well.
    var final_error = Pipeline(NameStep("in"), NameStep("b_"))
    with assert_raises():
        _ = final_error.get_feature_names_out(valid^)
    var fresh = clone(pipe)
    assert_false(fresh.__sklearn_is_fitted__())
    with assert_raises():
        _ = fresh.get_feature_names_out()
    var params = fresh.get_params()
    params.transformer = "new_"
    _ = fresh.set_params(params)
    _ = fresh.fit(2, 2)
    emit("names_clone", fresh.get_feature_names_out())
    assert_equal(pipe.get_feature_names_out()[0], "b_a_x0")
    _ = pipe.fit(2, 2)
    emit("names_refit", pipe.get_feature_names_out())
    var nested = Pipeline(NameStep("outer_"), pipe^)
    emit("names_nested", nested.get_feature_names_out(inputs.copy()))
    assert_equal(nested.get_feature_names_out(inputs.copy())[0], "b_a_outer_height")
    # Missing capabilities must identify the first offending step.
    var missing_first = Pipeline(Shift(), NameStep("b_"), transformer_name="missing_first")
    try:
        _ = missing_first.get_feature_names_out()
        raise Error("expected missing capability")
    except error:
        assert_equal(String(error), "AttributeError: Estimator missing_first does not provide get_feature_names_out")
    var missing_last = Pipeline(NameStep("a_"), Recorder(), estimator_name="missing_last")
    # The prefix's failure takes precedence over the missing final capability.
    try:
        _ = missing_last.get_feature_names_out()
        raise Error("expected prefix failure")
    except error:
        assert_equal(String(error), "NotFittedError: default names require fit")
    try:
        _ = missing_last.get_feature_names_out(inputs.copy())
        raise Error("expected missing capability")
    except error:
        assert_equal(String(error), "AttributeError: Estimator missing_last does not provide get_feature_names_out")
    print("Pipeline feature-name contracts passed")
