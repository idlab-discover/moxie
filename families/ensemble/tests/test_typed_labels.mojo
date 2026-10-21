"""Forest classifiers with Int64, Float32 and string labels.

Members train on class codes, so a typed-label forest must reproduce the
Float64-label forest exactly (same codes, probabilities and leaves) and map
predictions back through its own sorted `classes_`, as sklearn does for any
label dtype.
"""
import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.base import clone
from moxie._labels import StringTarget
from moxie.ensemble import RandomForestClassifier, ExtraTreesClassifier
from moxie.tree._classes import TypedClassWeight

comptime F = nm.NDArray[DType.float32]
comptime V = nm.NDArray[DType.float64]
comptime I = nm.NDArray[DType.int64]


def data() raises -> Tuple[F, V]:
    var X = nm.zeros[DType.float32](nm.Shape(40, 3))
    var y = nm.zeros[DType.float64](nm.Shape(40))
    for i in range(40):
        var label = (i * 7) % 3
        X.unsafe_set(3 * i, Float32(label) + Float32((i * 5) % 7) / 4)
        X.unsafe_set(3 * i + 1, Float32((i * 3) % 11) / 5)
        X.unsafe_set(3 * i + 2, Float32(i % 4))
        y.unsafe_set(i, Float64(label))
    return (X^, y^)


def assert_same(a: V, b: V) raises:
    assert_equal(a.size, b.size)
    for i in range(a.size):
        assert_equal(a.unsafe_get(i), b.unsafe_get(i))


def main() raises:
    var d = data()
    var X = d[0].copy()
    var y = d[1].copy()
    # Labels 0, 1, 2 become 20, -5, 7 (Int64) and "pear", "apple", "fig" (strings).
    var ints = nm.zeros[DType.int64](nm.Shape(40))
    var codes: List[Int64] = [20, -5, 7]
    var names: List[String] = ["pear", "apple", "fig"]
    var words = List[String]()
    for i in range(40):
        ints.unsafe_set(i, codes[Int(y.unsafe_get(i))])
        words.append(names[Int(y.unsafe_get(i))])
    var strings = StringTarget(words)
    # Same codes as the typed labels: sorted order is -5 < 7 < 20 and apple < fig < pear.
    var recoded = y.copy()
    for i in range(40):
        var v = Int(y.unsafe_get(i))
        recoded.unsafe_set(i, Float64(2 if v == 0 else (0 if v == 1 else 1)))

    var reference = RandomForestClassifier(12, random_state=5, oob_score=True)
    _ = reference.fit(X, recoded)

    var typed = RandomForestClassifier[label_dtype=DType.int64](12, random_state=5, oob_score=True)
    _ = typed.fit(X, ints)
    assert_equal(typed.n_classes_, 3)
    assert_equal(typed.classes_.unsafe_get(0), Int64(-5))
    assert_equal(typed.classes_.unsafe_get(1), Int64(7))
    assert_equal(typed.classes_.unsafe_get(2), Int64(20))
    assert_same(typed.predict_proba(X), reference.predict_proba(X))
    assert_same(typed.oob_decision_function_, reference.oob_decision_function_)
    assert_equal(typed.oob_score_, reference.oob_score_)
    var predicted = typed.predict(X)
    var expected = reference.predict(X)
    for i in range(40):
        assert_equal(predicted.unsafe_get(i), typed.classes_.unsafe_get(Int(expected.unsafe_get(i))))
    assert_equal(typed.score(X, ints), reference.score(X, recoded))
    var unfitted = clone(typed)
    assert_true(not unfitted.__sklearn_is_fitted__())
    assert_equal(unfitted.classes_.size, 0)

    var text = ExtraTreesClassifier[string_labels=True](9, random_state=1)
    var text_reference = ExtraTreesClassifier(9, random_state=1)
    _ = text.fit(X, strings)
    _ = text_reference.fit(X, recoded)
    assert_equal(text.classes_[0], "apple")
    assert_equal(text.classes_[1], "fig")
    assert_equal(text.classes_[2], "pear")
    assert_same(text.predict_proba(X), text_reference.predict_proba(X))
    var labels = text.predict(X)
    var codes_predicted = text_reference.predict(X)
    assert_equal(labels.rows(), 40)
    for i in range(40):
        assert_equal(labels.get(i), text.classes_[Int(codes_predicted.unsafe_get(i))])
    assert_equal(text.score(X, strings), text_reference.score(X, recoded))

    # Class weights are keyed by the original labels in their own type.
    var weights = Dict[Int64, Float64]()
    weights[20] = 3.0
    weights[-5] = 0.5
    var float_weights = Dict[Float64, Float64]()
    float_weights[2] = 3.0
    float_weights[0] = 0.5
    var weighted = RandomForestClassifier[label_dtype=DType.int64](6, random_state=2, class_weight=weights.copy())
    var weighted_reference = RandomForestClassifier(6, random_state=2, class_weight=float_weights.copy())
    _ = weighted.fit(X, ints)
    _ = weighted_reference.fit(X, recoded)
    assert_same(weighted.predict_proba(X), weighted_reference.predict_proba(X))
    var unknown = Dict[Int64, Float64]()
    unknown[20] = 3.0
    unknown[99] = 1.0
    _ = weighted.set_params(class_weight=TypedClassWeight[Int64](unknown.copy()))
    with assert_raises(contains="are not in class_weight"):
        _ = weighted.fit(X, ints)

    # Multiple Int64 outputs: one sorted domain per output.
    var multi_y = nm.zeros[DType.int64](nm.Shape(40, 2))
    var multi_reference_y = nm.zeros[DType.float64](nm.Shape(40, 2))
    for i in range(40):
        multi_y.unsafe_set(2 * i, ints.unsafe_get(i))
        multi_y.unsafe_set(2 * i + 1, Int64(100 * (i % 2)))
        multi_reference_y.unsafe_set(2 * i, recoded.unsafe_get(i))
        multi_reference_y.unsafe_set(2 * i + 1, Float64(i % 2))
    var multi = RandomForestClassifier[multi_output=True, label_dtype=DType.int64](7, random_state=4)
    var multi_reference = RandomForestClassifier[multi_output=True](7, random_state=4)
    _ = multi.fit(X, multi_y)
    _ = multi_reference.fit(X, multi_reference_y)
    assert_equal(multi.classes_[1].unsafe_get(1), Int64(100))
    var multi_predicted = multi.predict(X)
    var multi_expected = multi_reference.predict(X)
    assert_equal(multi_predicted.shape[1], 2)
    for i in range(40):
        for k in range(2):
            var code = Int(multi_expected.unsafe_get(2 * i + k))
            assert_equal(multi_predicted.unsafe_get(2 * i + k), multi.classes_[k].unsafe_get(code))

    # Float32 labels, and continuous labels are still rejected.
    var small = nm.zeros[DType.float32](nm.Shape(40))
    for i in range(40):
        small.unsafe_set(i, Float32(recoded.unsafe_get(i)) / 4)
    var f32 = RandomForestClassifier[label_dtype=DType.float32](5, random_state=0)
    with assert_raises(contains="unsupported classification target"):
        _ = f32.fit(X, small)
    for i in range(40):
        small.unsafe_set(i, Float32(recoded.unsafe_get(i)) * 4)
    _ = f32.fit(X, small)
    assert_equal(f32.classes_.unsafe_get(2), Float32(8))
    print("PASS: typed Int64, Float32 and string forest labels")
