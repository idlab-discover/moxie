"""All classifiers preserve typed target identity and lifecycle contracts."""
import numojo as nm
from numojo import Shape, Item
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie._labels import StringTarget
from moxie.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from moxie._naive_bayes_dense import ClassDomain, NumericDomain, StringDomain, dense_from_lists

def check_MultinomialNB[D: ClassDomain, dtype: DType](y: D.Target, other: D.Target, bad: D.Target) raises:
    var data: List[List[Float64]] = [[0.0, 2.0], [0.0, 1.0], [1.0, 0.0], [2.0, 0.0]]
    var X = dense_from_lists(data).astype[dtype]()
    var model = MultinomialNB[D]()
    with assert_raises():
        _ = model.predict(X)
    with assert_raises():
        _ = model.partial_fit(X, y)
    _ = model.partial_fit(X, y, classes=y.copy())
    assert_equal(model.score(X, y), 1.0)
    var predicted = model.predict(X)
    for i in range(4):
        assert_true(D.equal_at(y, predicted, i))
    var domain = D.from_classes(y)
    var expected_classes = domain.class_values()
    for i in range(2):
        assert_true(D.equal_at(model.classes_, expected_classes, i))
    # An unfitted clone retains only configuration; a fitted copy owns its domain.
    var cloned = model.__sklearn_clone__()
    assert_false(cloned.__sklearn_is_fitted__())
    var copied = model.copy()
    _ = model.fit(data, other)
    assert_equal(model.score(data, other), 1.0)
    assert_equal(copied.score(X, y), 1.0)
    with assert_raises():
        _ = model.partial_fit(X, y)
    with assert_raises():
        _ = model.partial_fit(X, other, classes=y.copy())
    with assert_raises():
        _ = model.fit(X, bad)
    assert_equal(model.score(X, other), 1.0)
    assert_equal(model.score(X, y), 0.0)
    # Weighted List and array updates preserve the selected target domain.
    _ = model.fit(X, y, sample_weight=List[Float64]([0.5, 2.0, 1.0, 3.0]))
    _ = model.partial_fit(data, y, classes=y.copy(), sample_weight=List[Float64]([0.5, 2.0, 1.0, 3.0]))
    assert_equal(model.score(X, y, sample_weight=List[Float64]([1.0, 2.0, 3.0, 4.0])), 1.0)

def check_ComplementNB[D: ClassDomain, dtype: DType](y: D.Target, other: D.Target, bad: D.Target) raises:
    var data: List[List[Float64]] = [[0.0, 2.0], [0.0, 1.0], [1.0, 0.0], [2.0, 0.0]]
    var X = dense_from_lists(data).astype[dtype]()
    var model = ComplementNB[D]()
    with assert_raises():
        _ = model.predict(X)
    with assert_raises():
        _ = model.partial_fit(X, y)
    _ = model.partial_fit(X, y, classes=y.copy())
    assert_equal(model.score(X, y), 1.0)
    var predicted = model.predict(X)
    for i in range(4):
        assert_true(D.equal_at(y, predicted, i))
    var domain = D.from_classes(y)
    var expected_classes = domain.class_values()
    for i in range(2):
        assert_true(D.equal_at(model.classes_, expected_classes, i))
    # An unfitted clone retains only configuration; a fitted copy owns its domain.
    var cloned = model.__sklearn_clone__()
    assert_false(cloned.__sklearn_is_fitted__())
    var copied = model.copy()
    _ = model.fit(data, other)
    assert_equal(model.score(data, other), 1.0)
    assert_equal(copied.score(X, y), 1.0)
    with assert_raises():
        _ = model.partial_fit(X, y)
    with assert_raises():
        _ = model.partial_fit(X, other, classes=y.copy())
    with assert_raises():
        _ = model.fit(X, bad)
    assert_equal(model.score(X, other), 1.0)
    assert_equal(model.score(X, y), 0.0)
    # Weighted List and array updates preserve the selected target domain.
    _ = model.fit(X, y, sample_weight=List[Float64]([0.5, 2.0, 1.0, 3.0]))
    _ = model.partial_fit(data, y, classes=y.copy(), sample_weight=List[Float64]([0.5, 2.0, 1.0, 3.0]))
    assert_equal(model.score(X, y, sample_weight=List[Float64]([1.0, 2.0, 3.0, 4.0])), 1.0)

def check_BernoulliNB[D: ClassDomain, dtype: DType](y: D.Target, other: D.Target, bad: D.Target) raises:
    var data: List[List[Float64]] = [[0.0, 2.0], [0.0, 1.0], [1.0, 0.0], [2.0, 0.0]]
    var X = dense_from_lists(data).astype[dtype]()
    var model = BernoulliNB[D]()
    with assert_raises():
        _ = model.predict(X)
    with assert_raises():
        _ = model.partial_fit(X, y)
    _ = model.partial_fit(X, y, classes=y.copy())
    assert_equal(model.score(X, y), 1.0)
    var predicted = model.predict(X)
    for i in range(4):
        assert_true(D.equal_at(y, predicted, i))
    var domain = D.from_classes(y)
    var expected_classes = domain.class_values()
    for i in range(2):
        assert_true(D.equal_at(model.classes_, expected_classes, i))
    # An unfitted clone retains only configuration; a fitted copy owns its domain.
    var cloned = model.__sklearn_clone__()
    assert_false(cloned.__sklearn_is_fitted__())
    var copied = model.copy()
    _ = model.fit(data, other)
    assert_equal(model.score(data, other), 1.0)
    assert_equal(copied.score(X, y), 1.0)
    with assert_raises():
        _ = model.partial_fit(X, y)
    with assert_raises():
        _ = model.partial_fit(X, other, classes=y.copy())
    with assert_raises():
        _ = model.fit(X, bad)
    assert_equal(model.score(X, other), 1.0)
    assert_equal(model.score(X, y), 0.0)
    # Weighted List and array updates preserve the selected target domain.
    _ = model.fit(X, y, sample_weight=List[Float64]([0.5, 2.0, 1.0, 3.0]))
    _ = model.partial_fit(data, y, classes=y.copy(), sample_weight=List[Float64]([0.5, 2.0, 1.0, 3.0]))
    assert_equal(model.score(X, y, sample_weight=List[Float64]([1.0, 2.0, 3.0, 4.0])), 1.0)

def check_CategoricalNB[D: ClassDomain, dtype: DType](y: D.Target, other: D.Target, bad: D.Target) raises:
    var data: List[List[Float64]] = [[0.0, 2.0], [0.0, 1.0], [1.0, 0.0], [2.0, 0.0]]
    var X = dense_from_lists(data).astype[dtype]()
    var model = CategoricalNB[D]()
    with assert_raises():
        _ = model.predict(X)
    with assert_raises():
        _ = model.partial_fit(X, y)
    _ = model.partial_fit(X, y, classes=y.copy())
    assert_equal(model.score(X, y), 1.0)
    var predicted = model.predict(X)
    for i in range(4):
        assert_true(D.equal_at(y, predicted, i))
    var domain = D.from_classes(y)
    var expected_classes = domain.class_values()
    for i in range(2):
        assert_true(D.equal_at(model.classes_, expected_classes, i))
    # An unfitted clone retains only configuration; a fitted copy owns its domain.
    var cloned = model.__sklearn_clone__()
    assert_false(cloned.__sklearn_is_fitted__())
    var copied = model.copy()
    _ = model.fit(data, other)
    assert_equal(model.score(data, other), 1.0)
    assert_equal(copied.score(X, y), 1.0)
    with assert_raises():
        _ = model.partial_fit(X, y)
    with assert_raises():
        _ = model.partial_fit(X, other, classes=y.copy())
    with assert_raises():
        _ = model.fit(X, bad)
    assert_equal(model.score(X, other), 1.0)
    assert_equal(model.score(X, y), 0.0)
    # Weighted List and array updates preserve the selected target domain.
    _ = model.fit(X, y, sample_weight=List[Float64]([0.5, 2.0, 1.0, 3.0]))
    _ = model.partial_fit(data, y, classes=y.copy(), sample_weight=List[Float64]([0.5, 2.0, 1.0, 3.0]))
    assert_equal(model.score(X, y, sample_weight=List[Float64]([1.0, 2.0, 3.0, 4.0])), 1.0)

def numeric[label: DType, dtype: DType]() raises:
    var y = nm.zeros[label](Shape(4))
    var other = nm.zeros[label](Shape(4))
    for i in range(4):
        comptime if label == DType.int64:
            y[Item(i)] = (Int64(1152921504606846979) if i < 2 else Int64(-1152921504606846969)).cast[label]()
        else:
            y[Item(i)] = 9 if i < 2 else -2
        other[Item(i)] = 4 if i < 2 else 5
    var reversed = y.view_with_layout(Shape(4), nm.NDArrayStrides(-1), 3)
    var invalid = nm.zeros[label](Shape(4, 1))
    check_MultinomialNB[NumericDomain[label], dtype](reversed, other, invalid)
    check_ComplementNB[NumericDomain[label], dtype](reversed, other, invalid)
    check_BernoulliNB[NumericDomain[label], dtype](reversed, other, invalid)
    check_CategoricalNB[NumericDomain[label], dtype](reversed, other, invalid)
    var features = dense_from_lists([[0.0, 2.0], [0.0, 1.0], [1.0, 0.0], [2.0, 0.0]])
    var owned_MultinomialNB = MultinomialNB[NumericDomain[label]]()
    var input_MultinomialNB = reversed.copy()
    _ = owned_MultinomialNB.fit(features, input_MultinomialNB)
    input_MultinomialNB[Item(0)] = 77
    assert_equal(owned_MultinomialNB.score(features, reversed), 1.0)
    var copy_MultinomialNB = owned_MultinomialNB.copy()
    copy_MultinomialNB.classes_[Item(0)] = 77
    assert_true(copy_MultinomialNB.classes_[Item(0)] != owned_MultinomialNB.classes_[Item(0)])
    var owned_ComplementNB = ComplementNB[NumericDomain[label]]()
    var input_ComplementNB = reversed.copy()
    _ = owned_ComplementNB.fit(features, input_ComplementNB)
    input_ComplementNB[Item(0)] = 77
    assert_equal(owned_ComplementNB.score(features, reversed), 1.0)
    var copy_ComplementNB = owned_ComplementNB.copy()
    copy_ComplementNB.classes_[Item(0)] = 77
    assert_true(copy_ComplementNB.classes_[Item(0)] != owned_ComplementNB.classes_[Item(0)])
    var owned_BernoulliNB = BernoulliNB[NumericDomain[label]]()
    var input_BernoulliNB = reversed.copy()
    _ = owned_BernoulliNB.fit(features, input_BernoulliNB)
    input_BernoulliNB[Item(0)] = 77
    assert_equal(owned_BernoulliNB.score(features, reversed), 1.0)
    var copy_BernoulliNB = owned_BernoulliNB.copy()
    copy_BernoulliNB.classes_[Item(0)] = 77
    assert_true(copy_BernoulliNB.classes_[Item(0)] != owned_BernoulliNB.classes_[Item(0)])
    var owned_CategoricalNB = CategoricalNB[NumericDomain[label]]()
    var input_CategoricalNB = reversed.copy()
    _ = owned_CategoricalNB.fit(features, input_CategoricalNB)
    input_CategoricalNB[Item(0)] = 77
    assert_equal(owned_CategoricalNB.score(features, reversed), 1.0)
    var copy_CategoricalNB = owned_CategoricalNB.copy()
    copy_CategoricalNB.classes_[Item(0)] = 77
    assert_true(copy_CategoricalNB.classes_[Item(0)] != owned_CategoricalNB.classes_[Item(0)])
    comptime if label != DType.int64:
        var X = dense_from_lists([[0.0], [1.0], [0.0], [1.0]])
        var model_MultinomialNB = MultinomialNB[NumericDomain[label]]()
        for value in List[Float64]([0.5, Float64("nan"), Float64("inf")]):
            y[Item(0)] = SIMD[label, 1](value)
            with assert_raises():
                _ = model_MultinomialNB.fit(X, y)
        var model_ComplementNB = ComplementNB[NumericDomain[label]]()
        for value in List[Float64]([0.5, Float64("nan"), Float64("inf")]):
            y[Item(0)] = SIMD[label, 1](value)
            with assert_raises():
                _ = model_ComplementNB.fit(X, y)
        var model_BernoulliNB = BernoulliNB[NumericDomain[label]]()
        for value in List[Float64]([0.5, Float64("nan"), Float64("inf")]):
            y[Item(0)] = SIMD[label, 1](value)
            with assert_raises():
                _ = model_BernoulliNB.fit(X, y)
        var model_CategoricalNB = CategoricalNB[NumericDomain[label]]()
        for value in List[Float64]([0.5, Float64("nan"), Float64("inf")]):
            y[Item(0)] = SIMD[label, 1](value)
            with assert_raises():
                _ = model_CategoricalNB.fit(X, y)

def main() raises:
    var y = StringTarget(["éclair", "éclair", "apple", "apple"])
    var other = StringTarget(["red", "red", "blue", "blue"])
    var bad = StringTarget(["short"])
    numeric[DType.int64, DType.float32]()
    numeric[DType.float32, DType.float32]()
    numeric[DType.float64, DType.float32]()
    check_MultinomialNB[StringDomain, DType.float32](y, other, bad)
    check_ComplementNB[StringDomain, DType.float32](y, other, bad)
    check_BernoulliNB[StringDomain, DType.float32](y, other, bad)
    check_CategoricalNB[StringDomain, DType.float32](y, other, bad)
    numeric[DType.int64, DType.float64]()
    numeric[DType.float32, DType.float64]()
    numeric[DType.float64, DType.float64]()
    check_MultinomialNB[StringDomain, DType.float64](y, other, bad)
    check_ComplementNB[StringDomain, DType.float64](y, other, bad)
    check_BernoulliNB[StringDomain, DType.float64](y, other, bad)
    check_CategoricalNB[StringDomain, DType.float64](y, other, bad)
    var features = dense_from_lists([[0.0, 2.0], [0.0, 1.0], [1.0, 0.0], [2.0, 0.0]])
    var owned_MultinomialNB = MultinomialNB[StringDomain]()
    var input_MultinomialNB = y.copy()
    _ = owned_MultinomialNB.fit(features, input_MultinomialNB)
    input_MultinomialNB.set(0, "changed")
    assert_equal(owned_MultinomialNB.score(features, y), 1.0)
    var copy_MultinomialNB = owned_MultinomialNB.copy()
    copy_MultinomialNB.classes_.set(0, "changed")
    assert_true(copy_MultinomialNB.classes_.get(0) != owned_MultinomialNB.classes_.get(0))
    var owned_ComplementNB = ComplementNB[StringDomain]()
    var input_ComplementNB = y.copy()
    _ = owned_ComplementNB.fit(features, input_ComplementNB)
    input_ComplementNB.set(0, "changed")
    assert_equal(owned_ComplementNB.score(features, y), 1.0)
    var copy_ComplementNB = owned_ComplementNB.copy()
    copy_ComplementNB.classes_.set(0, "changed")
    assert_true(copy_ComplementNB.classes_.get(0) != owned_ComplementNB.classes_.get(0))
    var owned_BernoulliNB = BernoulliNB[StringDomain]()
    var input_BernoulliNB = y.copy()
    _ = owned_BernoulliNB.fit(features, input_BernoulliNB)
    input_BernoulliNB.set(0, "changed")
    assert_equal(owned_BernoulliNB.score(features, y), 1.0)
    var copy_BernoulliNB = owned_BernoulliNB.copy()
    copy_BernoulliNB.classes_.set(0, "changed")
    assert_true(copy_BernoulliNB.classes_.get(0) != owned_BernoulliNB.classes_.get(0))
    var owned_CategoricalNB = CategoricalNB[StringDomain]()
    var input_CategoricalNB = y.copy()
    _ = owned_CategoricalNB.fit(features, input_CategoricalNB)
    input_CategoricalNB.set(0, "changed")
    assert_equal(owned_CategoricalNB.score(features, y), 1.0)
    var copy_CategoricalNB = owned_CategoricalNB.copy()
    copy_CategoricalNB.classes_.set(0, "changed")
    assert_true(copy_CategoricalNB.classes_.get(0) != owned_CategoricalNB.classes_.get(0))
    print("PASS: typed labels and lifecycle")
