"""Lifecycle contracts for TargetEncoder and its KFold/StratifiedKFold fold stand-ins."""
from numojo import NDArray, Shape, Item, zeros
from std.math import isnan, abs
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie._categorical import categorical_column, CategoricalColumn, StringCategories, NumericCategories
from moxie._random import RandomState, seed_global_random_state
from moxie.preprocessing import TargetEncoder, TargetPreprocessor
from moxie.preprocessing._label import StringLabels, NumericLabels
from moxie.preprocessing._vocabulary import StringKind, NumericKind
from moxie.preprocessing._target_encoder import kfold_test_folds, stratified_kfold_test_folds


def generic_fit[P: TargetPreprocessor](mut model: P, X: P.Input, y: P.Target) raises -> P.Transformed:
    # Compile-time trait conformance plus generic dispatch.
    model.fit(X, y)
    return model.transform(X)


def table() raises -> List[CategoricalColumn[StringCategories]]:
    var color: List[Optional[String]] = []
    var size: List[Optional[String]] = []
    var colors: List[String] = ["red", "blue", "red", "green", "blue", "red", "blue", "red", "green", "red"]
    var sizes: List[String] = ["s", "m", "m", "l", "s", "s", "m", "l", "s", "m"]
    for i in range(10):
        color.append(Optional[String](colors[i]))
        size.append(None if i == 3 else Optional[String](sizes[i]))
    var result: List[CategoricalColumn[StringCategories]] = [categorical_column(color), categorical_column(size)]
    return result^


def floats(values: List[Float64]) raises -> NDArray[DType.float64]:
    var result = zeros[DType.float64](Shape(len(values)))
    for i in range(len(values)):
        result.unsafe_set(i, values[i])
    return result^


def ints(values: List[Int]) raises -> NDArray[DType.int64]:
    var result = zeros[DType.int64](Shape(len(values)))
    for i in range(len(values)):
        result.unsafe_set(i, Int64(values[i]))
    return result^


def target() raises -> NDArray[DType.float64]:
    return floats([1.0, 4.0, 2.0, 8.0, 5.0, 0.5, 3.0, 1.5, 9.0, 2.5])


def assert_same(a: NDArray[DType.float64], b: NDArray[DType.float64]) raises:
    assert_equal(a.size, b.size)
    for k in range(a.size):
        assert_equal(a.unsafe_get(k), b.unsafe_get(k))


def test_lifecycle() raises:
    var X = table()
    var y = target()
    var model = TargetEncoder(random_state=0)
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(X)
    with assert_raises(contains="NotFittedError"):
        _ = model.get_feature_names_out()
    var Z = model.fit_transform(X, y)
    assert_equal(Z.shape[0], 10)
    assert_equal(Z.shape[1], 2)
    assert_equal(model.target_type_, "continuous")
    assert_false(Bool(model.classes_))
    assert_equal(model.n_features_in_, 2)
    assert_equal(len(model.encodings_), 2)
    assert_equal(model.encodings_[1].size, 4)  # l, m, s, None
    assert_false(Bool(model.categories_[1][3]))
    assert_equal(model.target_mean_.size, 1)
    assert_true(abs(model.target_mean_.item(0) - 3.65) < 1e-12)
    # Cross fitting differs from fit().transform() by design.
    var plain = model.transform(X)
    var differs = False
    for k in range(Z.size):
        differs = differs or Z.unsafe_get(k) != plain.unsafe_get(k)
    assert_true(differs)
    # A seeded fit_transform is deterministic; fit learns the same state.
    var again = TargetEncoder(random_state=0)
    assert_same(again.fit_transform(X, y), Z)
    var fitted = TargetEncoder(random_state=0)
    assert_same(generic_fit(fitted, X, y), plain)
    # Unknown categories encode as the target mean.
    var other: List[Optional[String]] = [Optional[String]("violet"), Optional[String]("red")]
    var sizes: List[Optional[String]] = [Optional[String]("m"), Optional[String]("xl")]
    var unseen: List[CategoricalColumn[StringCategories]] = [categorical_column(other), categorical_column(sizes)]
    var U = model.transform(unseen)
    assert_equal(U.item(0, 0), model.target_mean_.item(0))
    assert_equal(U.item(1, 1), model.target_mean_.item(0))
    assert_equal(U.item(1, 0), model.encodings_[0].item(2))  # red
    with assert_raises(contains="is expecting 2 features"):
        var narrow: List[CategoricalColumn[StringCategories]] = [X[0].copy()]
        _ = model.transform(narrow)
    # Names, clone and repeat fits.
    var names = model.get_feature_names_out()
    assert_equal(names[1], "x1")
    var named: List[String] = ["color", "size"]
    model.fit_named(X, named, y)
    assert_equal(model.get_feature_names_out()[0], "color")
    model.fit(X, y)
    assert_false(Bool(model.feature_names_in_))
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    assert_equal(copy.random_state.value(), 0)
    model.set_output(transform="default")
    with assert_raises(contains="set_output"):
        model.set_output(transform="pandas")


def test_params() raises:
    var X = table()
    var y = target()
    var model = TargetEncoder(smooth=2.0, cv=3, shuffle=False)
    var params = model.get_params()
    assert_equal(params.smooth.value(), 2.0)
    assert_equal(params.cv, 3)
    assert_false(params.shuffle.value())
    model.set_params(smooth=Optional[Float64](None), cv=4, random_state=Optional[Int](7), shuffle=Optional[Bool](True))
    assert_false(Bool(model.smooth))
    assert_equal(model.cv, 4)
    assert_equal(model.random_state.value(), 7)
    var record = model.get_params()
    record.target_type = "binary"
    model.set_params(record)
    assert_equal(model.target_type, "binary")
    # Invalid parameters raise at fit, as sklearn validates there.
    var bad = TargetEncoder(target_type="ordinal")
    with assert_raises(contains="'target_type' parameter"):
        bad.fit(X, y)
    var negative = TargetEncoder(smooth=Optional[Float64](-1.0))
    with assert_raises(contains="'smooth' parameter"):
        negative.fit(X, y)
    var one = TargetEncoder(cv=1)
    with assert_raises(contains="'cv' parameter"):
        _ = one.fit_transform(X, y)
    var seed = TargetEncoder(random_state=-3)
    with assert_raises(contains="'random_state' parameter"):
        seed.fit(X, y)
    var fixed = TargetEncoder(shuffle=False, random_state=0)
    fixed.fit(X, y)  # only the cross-fitting splitter rejects this pair
    with assert_raises(contains="no effect since shuffle is False"):
        _ = fixed.fit_transform(X, y)
    var many = TargetEncoder(cv=11, random_state=0)
    with assert_raises(contains="n_splits=11 greater than the number of samples"):
        _ = many.fit_transform(X, y)
    var short = floats([1.0, 2.0])
    var default = TargetEncoder()
    with assert_raises(contains="inconsistent numbers of samples: [10, 2]"):
        default.fit(X, short)
    var with_nan = target()
    with_nan.unsafe_set(4, Float64("nan"))
    with assert_raises(contains="Input y contains NaN"):
        default.fit(X, with_nan)


def test_targets() raises:
    var X = table()
    # Binary: LabelEncoder codes, stratified folds.
    var labels: List[String] = ["no", "yes", "no", "yes", "yes", "no", "no", "no", "yes", "no"]
    var binary = TargetEncoder[StringKind, StringLabels](cv=2, random_state=1)
    var B = binary.fit_transform(X, labels)
    assert_equal(B.shape[1], 2)
    assert_equal(binary.target_type_, "binary")
    assert_equal(binary.classes_.value()[1], "yes")
    assert_true(abs(binary.target_mean_.item(0) - 0.4) < 1e-15)
    # Multiclass: one output per feature and class, names <feature>_<class>.
    var classes = ints([2, 0, 1, 2, 0, 1, 2, 0, 1, 1])
    var multi = TargetEncoder[StringKind, NumericLabels[DType.int64]](cv=3, random_state=0)
    var M = multi.fit_transform(X, classes)
    assert_equal(M.shape[1], 6)
    assert_equal(len(multi.encodings_), 6)
    assert_equal(multi.target_mean_.size, 3)
    var names = multi.get_feature_names_out()
    assert_equal(names[0], "x0_0")
    assert_equal(names[5], "x1_2")
    var two = TargetEncoder[StringKind, NumericLabels[DType.int64]](target_type="multiclass")
    with assert_raises(contains="exactly 2 classes"):
        two.fit(X, ints([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))
    # Float class labels format as Python floats.
    var float_classes = floats([2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0])
    var named = TargetEncoder[StringKind, NumericLabels[DType.float64]](target_type="multiclass")
    named.fit(X, float_classes)
    assert_equal(named.get_feature_names_out()[1], "x0_1.0")
    # Float32 continuous targets average in Float32, like NumPy.
    var y32 = zeros[DType.float32](Shape(10))
    for i in range(10):
        y32.unsafe_set(i, Float32(i) * Float32(0.1))
    var f32 = TargetEncoder[StringKind, NumericLabels[DType.float32]](random_state=0)
    _ = f32.fit_transform(X, y32)
    var m32 = f32.target_mean_.item(0)
    assert_equal(Float64(Float32(m32)), m32)
    assert_true(abs(m32 - 0.45) < 1e-6)
    # String targets cannot be continuous.
    var strings = TargetEncoder[StringKind, StringLabels](target_type="continuous")
    with assert_raises(contains="TypeError"):
        strings.fit(X, labels)
    # Numeric features and explicit categories (an unseen value is unknown).
    var values = zeros[DType.int64](Shape(10))
    for i in range(10):
        values.unsafe_set(i, Int64(i % 3))
    var numeric: List[CategoricalColumn[NumericCategories[DType.int64]]] = [categorical_column(values)]
    var cats: List[List[Optional[Int64]]] = [[Optional[Int64](0), Optional[Int64](1)]]
    var explicit = TargetEncoder[NumericKind[DType.int64]](categories=Optional[List[List[Optional[Int64]]]](cats^), random_state=2, cv=2)
    var E = explicit.fit_transform(numeric, target())
    assert_equal(explicit.encodings_[0].size, 2)
    # Unknown rows (value 2) all carry the last fold's training mean.
    assert_equal(E.item(2, 0), E.item(5, 0))


def test_folds_and_rng() raises:
    var plain = kfold_test_folds(7, 3, False, None)
    var expected: List[Int] = [0, 0, 0, 1, 1, 2, 2]
    assert_true(plain == expected)
    var labels: List[Int] = [5, 5, 5, 9, 9, 9]
    var strat = stratified_kfold_test_folds(labels, 3, False, None)
    var by_class: List[Int] = [0, 1, 2, 0, 1, 2]
    assert_true(strat == by_class)
    with assert_raises(contains="number of members in each class"):
        _ = stratified_kfold_test_folds(labels, 4, False, None)
    # A borrowed RandomState advances: two draws differ, a fresh copy repeats.
    var rng = RandomState(3)
    var first = kfold_test_folds(20, 4, True, rng)
    var second = kfold_test_folds(20, 4, True, rng)
    assert_false(first == second)
    var fresh = RandomState(3)
    assert_true(kfold_test_folds(20, 4, True, fresh) == first)
    assert_true(kfold_test_folds(20, 4, True, Optional[Int](3)) == first)
    # None draws from the process stream, reproducible after reseeding.
    seed_global_random_state(5)
    var a = kfold_test_folds(20, 4, True, None)
    seed_global_random_state(5)
    assert_true(kfold_test_folds(20, 4, True, None) == a)
    # Estimator overload with a caller-owned state ignores random_state.
    var X = table()
    var y = target()
    var model = TargetEncoder(random_state=99)
    var state = RandomState(0)
    var Z = model.fit_transform(X, y, state)
    var seeded = TargetEncoder(random_state=0)
    assert_same(Z, seeded.fit_transform(X, y))


def main() raises:
    test_lifecycle()
    test_params()
    test_targets()
    test_folds_and_rng()
    print("PASS target encoder lifecycle contracts")
