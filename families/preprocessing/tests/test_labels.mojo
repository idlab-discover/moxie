"""Lifecycle contracts for LabelEncoder, LabelBinarizer, MultiLabelBinarizer and label_binarize."""
from numojo import NDArray, Shape, Item, zeros
import numojo as nm
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer, label_binarize
from moxie.preprocessing._contracts import LabelTransformer
from moxie.preprocessing._label import StringLabels, NumericLabels


def generic_fit_transform[T: LabelTransformer](mut model: T, y: T.Labels) raises -> T.Encoded:
    # Compile-time trait conformance plus generic dispatch. (The shared trait's
    # Encoded type is not Deinitable, so generic code cannot drop a temporary.)
    return model.fit_transform(y)


def ints(values: List[Int]) raises -> NDArray[DType.int64]:
    var result = zeros[DType.int64](Shape(len(values)))
    for i in range(len(values)):
        result.unsafe_set(i, Int64(values[i]))
    return result^


def floats(values: List[Float64]) raises -> NDArray[DType.float64]:
    var result = zeros[DType.float64](Shape(len(values)))
    for i in range(len(values)):
        result.unsafe_set(i, values[i])
    return result^


def int_matrix(rows: Int, cols: Int, values: List[Int]) raises -> NDArray[DType.int64]:
    var result = zeros[DType.int64](Shape(rows, cols))
    for i in range(len(values)):
        result.unsafe_set(i, Int64(values[i]))
    return result^


def test_label_encoder_strings() raises:
    var y: List[String] = ["paris", "tokyo", "amsterdam", "paris"]
    var model = LabelEncoder()
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(y)
    var codes = model.fit_transform(y)
    assert_equal(len(model.classes_), 3)
    assert_equal(model.classes_[0], "amsterdam")
    assert_equal(Int(codes.item(0)), 1)
    assert_equal(Int(codes.item(2)), 0)
    var back = model.inverse_transform(generic_fit_transform(model, y))
    assert_equal(back[1], "tokyo")
    var unseen: List[String] = ["rome", "paris", "oslo"]
    with assert_raises(contains="previously unseen labels: ['oslo' 'rome']"):
        _ = model.transform(unseen)
    with assert_raises(contains="previously unseen labels: [-1  3]"):
        _ = model.inverse_transform(ints([3, 0, -1]))
    # Empty inputs give empty outputs (sklearn returns float64 empties).
    var none = List[String]()
    assert_equal(model.transform(none).size, 0)
    assert_equal(len(model.inverse_transform(ints(List[Int]()))), 0)
    # Adaptation: labels are compared in full. sklearn casts input to the
    # classes' '<U9' dtype, so it would encode "amsterdamX" as "amsterdam".
    var longer: List[String] = ["amsterdamX"]
    with assert_raises(contains="previously unseen labels: ['amsterdamX']"):
        _ = model.transform(longer)
    # Clone is unfitted; refit replaces classes.
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    var other: List[String] = ["b", "a"]
    _ = model.fit(other)
    assert_equal(len(model.classes_), 2)
    # Fitting on no labels is a valid fit with no classes.
    _ = model.fit(none)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(len(model.classes_), 0)


def test_label_encoder_numeric() raises:
    var nan = Float64("nan")
    var inf = Float64("inf")
    var y = floats([2.0, nan, -inf, 1.5, inf, nan, 2.0])
    var model = LabelEncoder[NumericLabels[DType.float64]]()
    var codes = model.fit_transform(y)
    # Sorted, infinities ordered, a single NaN last (sklearn _unique_np).
    assert_equal(model.classes_.size, 5)
    assert_equal(model.classes_.item(0), -inf)
    assert_equal(model.classes_.item(3), inf)
    assert_true(isnan(model.classes_.item(4)))
    assert_equal(Int(codes.item(1)), 4)
    assert_equal(Int(codes.item(0)), 2)
    var back = model.inverse_transform(codes)
    assert_true(isnan(back.item(5)))
    # NaN is unseen when no NaN was fitted.
    var finite = LabelEncoder[NumericLabels[DType.float64]]()
    _ = finite.fit(floats([1.0, 2.0]))
    with assert_raises(contains="unseen labels: [nan]"):
        _ = finite.transform(floats([nan]))
    # Integer labels and a one-column matrix (sklearn column_or_1d).
    var int_model = LabelEncoder[NumericLabels[DType.int64]]()
    var column = int_matrix(3, 1, [7, -3, 7])
    var int_codes = int_model.fit_transform(column)
    assert_equal(Int(int_codes.item(1)), 0)
    with assert_raises(contains="y should be a 1d array, got an array of shape (2, 2) instead."):
        _ = int_model.fit(int_matrix(2, 2, [1, 2, 3, 4]))
    # Strided (negative-stride) views are read in place.
    var base = ints([5, 1, 9])
    var reversed = base.view_with_layout(Shape(3), nm.NDArrayStrides(-1), 2)
    var reversed_codes = int_model.fit_transform(reversed)
    assert_equal(Int(reversed_codes.item(0)), 2)
    assert_equal(Int(reversed_codes.item(2)), 1)


def test_label_binarizer() raises:
    var model = LabelBinarizer[NumericLabels[DType.int64]](neg_label=-1, pos_label=2)
    var params = model.get_params()
    assert_equal(params.neg_label, -1)
    assert_equal(params.pos_label, 2)
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(ints([1]))
    var y = ints([1, 4, 6, 4])
    var Y = model.fit_transform(y)
    assert_equal(model.y_type_, "multiclass")
    assert_false(model.sparse_input_)
    assert_equal(Y.shape[1], 3)
    assert_equal(Int(Y.item(0, 0)), 2)
    assert_equal(Int(Y.item(0, 1)), -1)
    var back = model.inverse_transform(generic_fit_transform(model, y))
    assert_equal(Int(back.item(2)), 6)
    # Float decision scores with an explicit or default threshold.
    var scores = zeros[DType.float64](Shape(2, 3))
    scores.unsafe_set(1, 0.7)
    scores.unsafe_set(5, 0.9)
    var picked = model.inverse_transform(scores, None)
    assert_equal(Int(picked.item(0)), 4)
    assert_equal(Int(picked.item(1)), 6)
    # Binary targets give one column; pos_label=0 needs a negative neg_label.
    var binary = LabelBinarizer[NumericLabels[DType.int64]](neg_label=-2, pos_label=0)
    var B = binary.fit_transform(ints([3, 5, 3]))
    assert_equal(B.shape[1], 1)
    assert_equal(Int(B.item(0, 0)), -2)
    assert_equal(Int(B.item(1, 0)), 0)
    # Single class: all neg_label, inverse repeats the class.
    var single = LabelBinarizer[NumericLabels[DType.int64]]()
    var S = single.fit_transform(ints([8, 8]))
    assert_equal(Int(S.item(1, 0)), 0)
    assert_equal(Int(single.inverse_transform(S).item(0)), 8)
    # Strings and multilabel indicators.
    var words: List[String] = ["no", "yes", "maybe"]
    var text = LabelBinarizer()
    var T = text.fit_transform(words)
    assert_equal(text.classes_[0], "maybe")
    assert_equal(Int(T.item(1, 2)), 1)
    var indicator = int_matrix(3, 2, [1, 0, 1, 1, 0, 0])
    var multi = LabelBinarizer[NumericLabels[DType.int64]]()
    var M = multi.fit_transform(indicator)
    assert_equal(multi.y_type_, "multilabel-indicator")
    assert_equal(Int(multi.classes_.item(1)), 1)
    assert_equal(Int(M.item(1, 1)), 1)
    with assert_raises(contains="was not fitted with multilabel input"):
        _ = model.transform(indicator)
    # Parameter errors are raised at fit; sparse output is deferred.
    _ = model.set_params(neg_label=3)
    with assert_raises(contains="neg_label=3 must be strictly less than pos_label=2."):
        _ = model.fit(y)
    _ = model.set_params(LabelBinarizer[NumericLabels[DType.int64]].Params(0, 1, True))
    with assert_raises(contains="sparse output is deferred"):
        _ = model.fit(y)
    var copy = clone(model)
    assert_true(copy.sparse_output)
    assert_false(copy.__sklearn_is_fitted__())
    with assert_raises(contains="Multioutput target data is not supported"):
        _ = multi.fit(int_matrix(2, 2, [1, 2, 3, 4]))
    var scores_model = LabelBinarizer[NumericLabels[DType.float64]]()
    with assert_raises(contains="Unknown label type: continuous"):
        _ = scores_model.fit(floats([0.5, 1.0]))
    with assert_raises(contains="Input y contains NaN."):
        _ = scores_model.fit(floats([Float64("nan"), 1.0]))
    var empty_model = LabelBinarizer[NumericLabels[DType.int64]]()
    with assert_raises(contains="y has 0 samples"):
        _ = empty_model.fit(ints(List[Int]()))


def test_label_binarize() raises:
    var Y = label_binarize(ints([1, 6]), classes=ints([1, 2, 4, 6]))
    assert_equal(Int(Y.item(1, 3)), 1)
    # Class order controls column order: the positive class is classes[-1].
    var R = label_binarize(ints([1, 2, 1]), classes=ints([2, 1]))
    assert_equal(R.shape[1], 1)
    assert_equal(Int(R.item(0, 0)), 1)
    assert_equal(Int(R.item(1, 0)), 0)
    var words: List[String] = ["cat", "dog", "emu"]
    var classes: List[String] = ["dog", "cat", "bird"]
    var W = label_binarize(words, classes=classes)
    assert_equal(Int(W.item(0, 1)), 1)
    assert_equal(Int(W.item(2, 0)) + Int(W.item(2, 1)) + Int(W.item(2, 2)), 0)
    with assert_raises(contains="continuous target data is not supported"):
        _ = label_binarize(floats([1.2, 2.7]), classes=floats([0.0, 1.0]))
    with assert_raises(contains="mismatch with the labels [0 1] found in the data"):
        _ = label_binarize(int_matrix(1, 2, [1, 0]), classes=ints([0, 1, 2]))
    with assert_raises(contains="Found array with 0 sample(s)"):
        _ = label_binarize(ints(List[Int]()), classes=ints([1, 2]))
    with assert_raises(contains="sparse output is deferred"):
        _ = label_binarize(ints([1, 2]), classes=ints([1, 2]), sparse_output=True)
    # Adaptations: NaN classes and empty binary classes raise ValueError.
    with assert_raises(contains="classes must not contain NaN"):
        _ = label_binarize(floats([1.0, 2.0]), classes=floats([1.0, Float64("nan")]))
    with assert_raises(contains="at least one label"):
        _ = label_binarize(ints([1, 2]), classes=ints(List[Int]()))
    # Float multilabel classes_ and inverses keep the label dtype.
    var indicator = zeros[DType.float64](Shape(2, 2))
    indicator.unsafe_set(0, 1.0)
    indicator.unsafe_set(3, 1.0)
    var multi = LabelBinarizer[NumericLabels[DType.float64]]()
    var M = multi.fit_transform(indicator)
    assert_equal(multi.classes_.item(1), 1.0)
    assert_equal(multi.inverse_transform(M).item(1, 1), 1.0)


def test_multilabel_binarizer() raises:
    var y: List[List[String]] = [["sci-fi", "thriller"], ["comedy"], []]
    var model = MultiLabelBinarizer()
    var Y = model.fit_transform(y)
    assert_equal(len(model.classes_), 3)
    assert_equal(model.classes_[0], "comedy")
    assert_equal(Int(Y.item(0, 2)), 1)
    assert_equal(Y.shape[0], 3)
    var back = model.inverse_transform(generic_fit_transform(model, y))
    assert_equal(len(back[0]), 2)
    assert_equal(len(back[2]), 0)
    # Explicit classes keep their order; unknown labels are ignored.
    var order: List[String] = ["thriller", "comedy"]
    var given = MultiLabelBinarizer(classes=order^)
    var G = given.fit_transform(y)
    assert_equal(G.shape[1], 2)
    assert_equal(Int(G.item(0, 0)), 1)
    assert_equal(Int(G.item(0, 1)), 0)
    with assert_raises(contains="Expected only 0s and 1s in label indicator. Also got [2]"):
        _ = given.inverse_transform(int_matrix(1, 2, [2, 0]))
    with assert_raises(contains="Expected indicator for 2 classes, but got 3"):
        _ = given.inverse_transform(int_matrix(1, 3, [1, 0, 0]))
    var twice: List[String] = ["a", "a"]
    var duplicate = MultiLabelBinarizer(classes=twice^)
    with assert_raises(contains="duplicate classes"):
        _ = duplicate.fit(y)
    var copy = clone(given)
    assert_false(copy.__sklearn_is_fitted__())
    assert_equal(len(copy.get_params().classes.value()), 2)
    # Integer labels.
    var numbers: List[List[Int64]] = [[3, 1], [2], [1, 1]]
    var numeric = MultiLabelBinarizer[NumericLabels[DType.int64]]()
    var N = numeric.fit_transform(numbers)
    assert_equal(Int(numeric.classes_.item(2)), 3)
    assert_equal(Int(N.item(2, 0)), 1)
    var nan_rows: List[List[Float64]] = [[Float64("nan")]]
    var floats_model = MultiLabelBinarizer[NumericLabels[DType.float64]]()
    with assert_raises(contains="does not support NaN labels"):
        _ = floats_model.fit(nan_rows)
    _ = numeric.set_params(sparse_output=True)
    with assert_raises(contains="sparse output is deferred"):
        _ = numeric.transform(numbers)


def main() raises:
    test_label_encoder_strings()
    test_label_encoder_numeric()
    test_label_binarizer()
    test_label_binarize()
    test_multilabel_binarizer()
    print("PASS label lifecycle contracts")
