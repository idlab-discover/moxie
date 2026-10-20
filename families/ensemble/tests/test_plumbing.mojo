from std.testing import assert_equal, assert_raises
from numojo import Shape, Item, full
from moxie._random import RandomState
from moxie.ensemble._records import SamplingPlan, ClassAlignment
from moxie.ensemble._plumbing import (
    snapshot, validate_sampling, additional_estimators, validate_oob,
    next_estimator_seed, validate_alignment, validate_names,
)

def ownership[dtype: DType]() raises:
    var X = full[dtype](Shape(2, 2), 2)
    var independent = snapshot(X)
    var shared = X.view()
    X[Item(0, 0)] = 9
    assert_equal(independent.item(0, 0), 2)
    assert_equal(shared.item(0, 0), 9)

def main() raises:
    ownership[DType.float32]()
    ownership[DType.float64]()
    validate_sampling(SamplingPlan(5, 3, 5, 2, True, False))
    with assert_raises():
        validate_sampling(SamplingPlan(0, 3, 1, 1, True, False))
    validate_sampling(SamplingPlan(5, 3, 6, 1, True, False))
    with assert_raises():
        validate_sampling(SamplingPlan(5, 3, 6, 1, False, False))
    with assert_raises():
        validate_sampling(SamplingPlan(5, 3, 1, 0, True, False))
    assert_equal(additional_estimators(3, 5, False), 3)
    assert_equal(additional_estimators(5, 3, True), 2)
    assert_equal(additional_estimators(3, 3, True), 0)
    with assert_raises():
        _ = additional_estimators(2, 3, True)
    with assert_raises():
        _ = additional_estimators(0, 0, False)
    validate_oob(True, True, False)
    validate_oob(True, False, True)
    with assert_raises():
        validate_oob(False, False, False)
    with assert_raises():
        validate_oob(True, True, True)
    var rng = RandomState(42)
    assert_equal(next_estimator_seed(rng), 1608637542)
    assert_equal(next_estimator_seed(rng), 1273642419)
    var columns = full[DType.int64](Shape(2), 0)
    columns[Item(1)] = 2
    var mapping = ClassAlignment(columns^, 3)
    validate_alignment(mapping, 2)
    with assert_raises():
        validate_alignment(mapping, 3)
    mapping.columns[Item(1)] = 0
    with assert_raises():
        validate_alignment(mapping, 2)
    mapping.columns[Item(1)] = 3
    with assert_raises():
        validate_alignment(mapping, 2)
    var names: List[String] = ["first", "second"]
    var active: List[Bool] = [True, False]
    var reserved: List[String] = ["estimators", "weights"]
    validate_names(names, active, reserved)
    active[0] = False
    with assert_raises():
        validate_names(names, active, reserved)
    active[0] = True
    names[1] = "first"
    with assert_raises():
        validate_names(names, active, reserved)
    names[1] = "a__b"
    with assert_raises():
        validate_names(names, active, reserved)
    names[1] = "weights"
    with assert_raises():
        validate_names(names, active, reserved)
    print("PASS ensemble validation, ownership and seed contracts")
