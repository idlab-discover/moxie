from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie._pipeline import CompositionStep
from moxie.base import BaseEstimator, clone
from moxie._pipeline import NamedStep, StepPair, PairParams, validate_names, resolve_step, StepMode, validate_mode, require_nonempty_pipeline, check_union_rows
from moxie.utils.validation import FittedState

struct Probe(CompositionStep):
    comptime Params = List[Int]
    var config: List[Int]
    var state: FittedState

    def __init__(out self, value: Int):
        self.config = [value]
        self.state = FittedState()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.config.copy()

    def set_params(mut self, params: Self.Params) raises:
        self.config = params.copy()

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.config[0])

    def __sklearn_is_fitted__(self) -> Bool:
        return self.state.is_fitted()


def contracts() raises:
    var a = NamedStep("first", Probe(3))
    a.estimator.state.record_fit(2, 4)
    var b = NamedStep("second", Probe(5))
    b.estimator.state.record_fit(2, 4)
    var c = NamedStep("third", Probe(7))
    c.estimator.state.record_fit(2, 4)
    var tree = StepPair(StepPair(a^, b^), c^)
    assert_true(tree.__sklearn_is_fitted__())
    var independent = clone(tree)
    assert_false(independent.__sklearn_is_fitted__())
    assert_equal(independent.first.first.name, "first")
    assert_equal(independent.first.second.name, "second")
    assert_equal(independent.second.name, "third")
    assert_false(independent.first.first.__sklearn_is_fitted__())
    assert_false(independent.first.second.__sklearn_is_fitted__())
    assert_false(independent.second.__sklearn_is_fitted__())
    var params = independent.get_params()
    params.first.first[0] = 19
    assert_equal(independent.first.first.estimator.config[0], 3)
    independent.set_params(params)
    params.first.first[0] = 99
    assert_equal(independent.first.first.estimator.config[0], 19)
    assert_equal(tree.first.first.estimator.config[0], 3)
    var shallow = tree.get_params(False)
    assert_equal(shallow.first.second[0], 5)
    independent.second.set_named_params("third", [11])
    assert_equal(independent.second.estimator.config[0], 11)
    with assert_raises():
        independent.second.set_named_params("absent", [0])
    assert_equal(independent.second.estimator.config[0], 11)
    independent.second.replace(Probe(21))
    assert_equal(independent.second.estimator.config[0], 21)
    assert_false(independent.second.__sklearn_is_fitted__())
    var names: List[String] = ["first", "second", "third"]
    var reserved: List[String] = ["steps", "memory", "verbose", "transform_input"]
    validate_names(names, reserved)
    validate_names([""], reserved)
    assert_equal(resolve_step(names, "second"), 1)
    with assert_raises():
        validate_names(["a", "a"], reserved)
    with assert_raises():
        validate_names(["a__b"], reserved)
    with assert_raises():
        validate_names(["memory"], reserved)
    with assert_raises():
        _ = resolve_step(names, "missing")
    with assert_raises():
        _ = NamedStep("a__b", Probe(0))
    require_nonempty_pipeline(1)
    with assert_raises():
        require_nonempty_pipeline(0)
    validate_mode(StepMode.estimator)
    validate_mode(StepMode.passthrough)
    validate_mode(StepMode.drop, union=True)
    with assert_raises():
        validate_mode(StepMode.drop)
    with assert_raises():
        validate_mode(42, union=True)
    check_union_rows(0, 0)
    check_union_rows(2, 2)
    with assert_raises():
        check_union_rows(2, 3)
    with assert_raises():
        check_union_rows(-1, -1)
    print("Composition ownership and routing contracts passed")


def main() raises:
    contracts()
