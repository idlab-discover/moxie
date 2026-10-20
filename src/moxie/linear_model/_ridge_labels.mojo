# Adapted from scikit-learn sklearn/preprocessing/_label.py (LabelBinarizer) and
# sklearn/utils/class_weight.py; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Ridge-classifier target encoding: LabelBinarizer(pos_label=1, neg_label=-1).

Minimal native stand-in for `preprocessing.labels` (a deferred session) covering
exactly what RidgeClassifier/RidgeClassifierCV use: integer class labels
(binary/multiclass, rank-1 y) and 0/1 multilabel indicators (rank-2 y), sorted
`classes_`, the +/-1 target matrix, decision-to-label mapping and
`class_weight` ('balanced' or per-class weights) sample reweighting.
"""
from numojo import NDArray, Shape, zeros
from std.math import isfinite
from moxie.linear_model._contracts import validate_weights


@fieldwise_init
struct ClassWeight(Copyable, Movable):
    """`class_weight`: `balanced=True`, or explicit `weights[label]` (default 1)."""

    var balanced: Bool
    var weights: Dict[Int, Float64]

    @staticmethod
    def balanced_weights() -> ClassWeight:
        return ClassWeight(True, Dict[Int, Float64]())

    @staticmethod
    def explicit(var weights: Dict[Int, Float64]) -> ClassWeight:
        return ClassWeight(False, weights^)


struct LabelEncoding[label_dtype: DType](Copyable, Movable):
    """Fitted label binarization; `multilabel` marks an indicator target."""

    var classes: List[Int]
    var multilabel: Bool

    def __init__(out self, var classes: List[Int], multilabel: Bool):
        self.classes = classes^
        self.multilabel = multilabel

    def n_classes(self) -> Int:
        return len(self.classes)

    def classes_array(self) raises -> NDArray[Self.label_dtype]:
        var result = zeros[Self.label_dtype](Shape(len(self.classes)))
        for i in range(len(self.classes)):
            result.unsafe_set(i, Scalar[Self.label_dtype](self.classes[i]))
        return result^

    def index(self, label: Int) raises -> Int:
        for i in range(len(self.classes)):
            if self.classes[i] == label:
                return i
        raise Error(String("ValueError: y contains previously unseen label ", label))

    def labels_from_decision[dtype: DType](self, decision: NDArray[dtype]) raises -> NDArray[Self.label_dtype]:
        """LinearClassifierMixin.predict / multilabel inverse_transform.

        decision is (samples, columns). One column: classes_[decision > 0];
        several columns: argmax (first maximum); multilabel: indicator of > 0.
        """
        var n = decision.shape[0]
        var k = decision.shape[1]
        if self.multilabel:
            var result = zeros[Self.label_dtype](Shape(n, k))
            for i in range(n):
                for j in range(k):
                    result.unsafe_set(i * k + j, Scalar[Self.label_dtype](1 if decision.unsafe_get(i * k + j) > 0 else 0))
            return result^
        var result = zeros[Self.label_dtype](Shape(n))
        for i in range(n):
            var chosen = 0
            if k == 1:
                if decision.unsafe_get(i) > 0:
                    chosen = 1 if len(self.classes) > 1 else 0
            else:
                var best = decision.unsafe_get(i * k)
                for j in range(1, k):
                    if decision.unsafe_get(i * k + j) > best:
                        best = decision.unsafe_get(i * k + j)
                        chosen = j
            result.unsafe_set(i, Scalar[Self.label_dtype](self.classes[chosen]))
        return result^


def _label[dt: DType](y: NDArray[dt], i: Int, j: Int) raises -> Int:
    var index = i * Int(y.strides[0]) + (j * Int(y.strides[1]) if y.ndim == 2 else 0)
    var value = y.unsafe_get(index)
    comptime if dt.is_floating_point():
        if not isfinite(value) or Float64(Int(value)) != Float64(value):
            raise Error("ValueError: RidgeClassifier labels must be finite integers")
    return Int(value)


def fit_labels[label_dtype: DType](y: NDArray[label_dtype], owner: String) raises -> LabelEncoding[label_dtype]:
    """sklearn LabelBinarizer.fit: sorted unique labels or indicator columns."""
    if y.ndim != 1 and y.ndim != 2:
        raise Error("ValueError: y must have rank one or two")
    if y.shape[0] == 0 or (y.ndim == 2 and y.shape[1] == 0):
        raise Error("ValueError: y must be nonempty")
    if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
        var classes = List[Int]()
        for i in range(y.shape[0]):
            var label = _label(y, i, 0)
            var found = False
            for c in classes:
                if c == label:
                    found = True
                    break
            if not found:
                classes.append(label)
        sort(classes)
        return LabelEncoding[label_dtype](classes^, False)
    if y.ndim == 2:
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                var label = _label(y, i, j)
                if label != 0 and label != 1:
                    raise Error(String("ValueError: ", owner, ": a rank-2 y must be a 0/1 multilabel indicator (multiclass-multioutput is not supported)"))
        var classes = List[Int]()
        for j in range(y.shape[1]):
            classes.append(j)
        return LabelEncoding[label_dtype](classes^, True)
    raise Error(String("ValueError: ", owner, ": y must have rank one or two"))


def binarize[dtype: DType, label_dtype: DType](encoding: LabelEncoding[label_dtype], y: NDArray[label_dtype]) raises -> NDArray[dtype]:
    """LabelBinarizer(pos_label=1, neg_label=-1).transform: (samples, 1) for two
    classes, (samples, n_classes) otherwise (one column per class or label)."""
    var n = y.shape[0]
    var c = encoding.n_classes()
    if encoding.multilabel:
        var Y = zeros[dtype](Shape(n, c))
        for i in range(n):
            for j in range(c):
                Y.unsafe_set(i * c + j, Scalar[dtype](1 if _label(y, i, j) == 1 else -1))
        return Y^
    var columns = 1 if c <= 2 else c
    var Y = zeros[dtype](Shape(n, columns))
    for i in range(n):
        var k = encoding.index(_label(y, i, 0))
        if columns == 1:
            Y.unsafe_set(i, Scalar[dtype](1 if k == 1 else -1))
        else:
            for j in range(columns):
                Y.unsafe_set(i * columns + j, Scalar[dtype](1 if j == k else -1))
    return Y^


def class_sample_weight[
    dtype: DType, label_dtype: DType
](weight: ClassWeight, encoding: LabelEncoding[label_dtype], y: NDArray[label_dtype]) raises -> NDArray[dtype]:
    """sklearn compute_sample_weight(class_weight, y) for 1-D labels or indicators.

    'balanced' uses n_samples / (n_classes * count) per class (per indicator
    column, multiplied across columns for multilabel y). Explicit weights map a
    label to its weight; unlisted labels keep weight 1 (rank-1 y only).
    """
    var n = y.shape[0]
    var result = zeros[dtype](Shape(n))
    for i in range(n):
        result.unsafe_set(i, 1)
    if encoding.multilabel:
        if not weight.balanced:
            raise Error("ValueError: RidgeClassifier: explicit class_weight is not supported for multilabel y; use 'balanced'")
        for j in range(y.shape[1]):
            var counts = List[Int](length=2, fill=0)
            for i in range(n):
                counts[_label(y, i, j)] += 1
            var present = (1 if counts[0] > 0 else 0) + (1 if counts[1] > 0 else 0)
            for i in range(n):
                var label = _label(y, i, j)
                var w = Float64(n) / (Float64(present) * Float64(counts[label]))
                result.unsafe_set(i, (Float64(result.unsafe_get(i)) * w).cast[dtype]())
        return result^
    var c = encoding.n_classes()
    var counts = List[Int](length=c, fill=0)
    for i in range(n):
        counts[encoding.index(_label(y, i, 0))] += 1
    for i in range(n):
        var label = _label(y, i, 0)
        var w: Float64
        if weight.balanced:
            w = Float64(n) / (Float64(c) * Float64(counts[encoding.index(label)]))
        else:
            w = weight.weights.get(label, 1.0)
        result.unsafe_set(i, w.cast[dtype]())
    if not weight.balanced:
        for entry in weight.weights.items():
            var known = False
            for cl in encoding.classes:
                if cl == entry.key:
                    known = True
            if not known:
                raise Error(String("ValueError: The classes, [", entry.key, "], are not in class_weight"))
    return result^


def accuracy[dtype: DType, label_dtype: DType](y: NDArray[label_dtype], predicted: NDArray[label_dtype], sample_weight: Optional[NDArray[dtype]]) raises -> Float64:
    """ClassifierMixin.score: (weighted) accuracy; subset accuracy for indicators."""
    if (y.ndim != 1 and y.ndim != 2) or (predicted.ndim != 1 and predicted.ndim != 2):
        raise Error("ValueError: y and predictions must be vectors or matrices")
    if y.shape[0] != predicted.shape[0]:
        raise Error("ValueError: y and predictions have different shapes")
    var n = y.shape[0]
    var cols = y.shape[1] if y.ndim == 2 else 1
    var pred_cols = predicted.shape[1] if predicted.ndim == 2 else 1
    if pred_cols != cols:
        raise Error("ValueError: y and predictions have different shapes")
    if n == 0 or cols == 0:
        raise Error("ValueError: score requires nonempty targets")
    if sample_weight:
        validate_weights(sample_weight.value(), n, nonnegative=False)
    var correct = Float64(0)
    var total = Float64(0)
    for i in range(n):
        var w = Float64(1)
        if sample_weight:
            w = Float64(sample_weight.value().unsafe_get(i * Int(sample_weight.value().strides[0])))
        var same = True
        for j in range(cols):
            if _label(y, i, j) != _label(predicted, i, j):
                same = False
        if same:
            correct += w
        total += w
    if total == 0:
        raise Error("ValueError: sample weights sum to zero")
    return correct / total
