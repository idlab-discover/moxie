# Adapted from scikit-learn multiclass.py; BSD-3-Clause, see LICENSES/multiclass-scikit-learn.txt.
"""Family label dispatch; class domains and mappings belong to shared moxie._labels."""
from numojo import NDArray, Shape, Item
import numojo as nm
from std.math import isfinite, isnan, isinf
from moxie._labels import StringTarget, StringEncoding, NumericEncoding, encode_column, type_of_target, check_classification_targets
from moxie.preprocessing._label import label_binarize
from moxie._multiclass_binary import mean_accuracy


def _shape_repr[dtype: DType](a: NDArray[dtype]) raises -> String:
    """Python tuple spelling of a shape, e.g. `(6, 2)`, `(6,)`, `()`."""
    var text = String("(")
    for d in range(a.ndim):
        text += (", " if d else "") + String(a.shape[d])
    if a.ndim == 1:
        text += ","
    return text + ")"


trait MulticlassLabels:
    comptime Target: Copyable & Deinitable
    comptime Classes: Copyable & Deinitable
    comptime Encoding: Copyable & Deinitable

    @staticmethod
    def empty() raises -> Self.Classes:
        ...

    @staticmethod
    def flat(y: Self.Target, mode: Int) raises -> Self.Target:
        """Flatten with validation for OvR (0), OvO (1), or output codes (2)."""
        ...

    @staticmethod
    def size(y: Self.Target) raises -> Int:
        ...

    @staticmethod
    def rank(y: Self.Target) raises -> Int:
        ...

    @staticmethod
    def kind(y: Self.Target) raises -> String:
        ...

    @staticmethod
    def make(y: Self.Target) raises -> Self.Encoding:
        ...

    @staticmethod
    def classes(encoding: Self.Encoding) raises -> Self.Classes:
        ...

    @staticmethod
    def domain_count(encoding: Self.Encoding) raises -> Int:
        ...

    @staticmethod
    def codes(encoding: Self.Encoding) raises -> NDArray[DType.int64]:
        ...

    @staticmethod
    def encode(encoding: Self.Encoding, y: Self.Target) raises -> NDArray[DType.int64]:
        ...

    @staticmethod
    def decode(encoding: Self.Encoding, codes: NDArray[DType.int64]) raises -> Self.Target:
        ...

    @staticmethod
    def count(classes: Self.Classes) raises -> Int:
        ...

    @staticmethod
    def same(a: Self.Classes, b: Self.Classes) raises -> Bool:
        ...

    @staticmethod
    def from_classes(classes: Self.Classes) raises -> Self.Encoding:
        ...

    @staticmethod
    def binarize(y: Self.Target, classes: Self.Classes) raises -> NDArray[DType.int64]:
        ...

    @staticmethod
    def accuracy(y: Self.Target, predicted: Self.Target, weights: Optional[NDArray[DType.float64]]) raises -> Float64:
        ...


struct NumericLabels[dtype: DType](MulticlassLabels):
    comptime Target = NDArray[Self.dtype]
    comptime Classes = NDArray[Self.dtype]
    comptime Encoding = NumericEncoding[Self.dtype]

    @staticmethod
    def empty() raises -> Self.Classes:
        return nm.zeros[Self.dtype](Shape(0))

    @staticmethod
    def flat(y: Self.Target, mode: Int) raises -> Self.Target:
        if mode == 0:
            var kind = type_of_target(y)
            if kind.startswith("multilabel"):
                raise Error("ValueError: expected single-label targets, got multilabel-indicator")
            if "multioutput" in kind:
                raise Error("ValueError: Multioutput target data is not supported with label binarization")
            check_classification_targets(y)
        if not (y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1)):
            if mode == 1:
                raise Error(String("ValueError: y should be a 1d array, got an array of shape ", _shape_repr(y), " instead."))
            raise Error("ValueError: y should be a 1d array or a single-column array")
        var result = nm.zeros[Self.dtype](Shape(y.shape[0]))
        for i in range(result.size):
            var value = y[Item(i)] if y.ndim == 1 else y[Item(i, 0)]
            if mode == 1:
                if isnan(value):
                    raise Error("ValueError: Input y contains NaN.")
                if isinf(value):
                    raise Error(String("ValueError: Input y contains infinity or a value too large for dtype('", Self.dtype, "')."))
            if not isfinite(value):
                raise Error("ValueError: Input y contains NaN or infinity")
            result[Item(i)] = value
        if mode == 2:
            check_classification_targets(result)
        return result^

    @staticmethod
    def size(y: Self.Target) raises -> Int:
        return y.size

    @staticmethod
    def rank(y: Self.Target) raises -> Int:
        return y.ndim

    @staticmethod
    def kind(y: Self.Target) raises -> String:
        return type_of_target(y)

    @staticmethod
    def make(y: Self.Target) raises -> Self.Encoding:
        return encode_column(y)

    @staticmethod
    def classes(encoding: Self.Encoding) raises -> Self.Classes:
        return encoding.classes.copy()

    @staticmethod
    def domain_count(encoding: Self.Encoding) raises -> Int:
        return encoding.classes.size

    @staticmethod
    def codes(encoding: Self.Encoding) raises -> NDArray[DType.int64]:
        return encoding.index.inverse.copy()

    @staticmethod
    def encode(encoding: Self.Encoding, y: Self.Target) raises -> NDArray[DType.int64]:
        return encoding.encode(y)

    @staticmethod
    def decode(encoding: Self.Encoding, codes: NDArray[DType.int64]) raises -> Self.Target:
        return encoding.decode(codes)

    @staticmethod
    def count(classes: Self.Classes) raises -> Int:
        return classes.size

    @staticmethod
    def same(a: Self.Classes, b: Self.Classes) raises -> Bool:
        if a.size != b.size:
            return False
        for i in range(a.size):
            if a[Item(i)] != b[Item(i)]:
                return False
        return True

    @staticmethod
    def from_classes(classes: Self.Classes) raises -> Self.Encoding:
        return encode_column(classes)

    @staticmethod
    def binarize(y: Self.Target, classes: Self.Classes) raises -> NDArray[DType.int64]:
        return label_binarize(y, classes=classes)

    @staticmethod
    def accuracy(y: Self.Target, predicted: Self.Target, weights: Optional[NDArray[DType.float64]]) raises -> Float64:
        return mean_accuracy(y, predicted, weights)


struct StringLabels(MulticlassLabels):
    comptime Target = StringTarget
    comptime Classes = List[String]
    comptime Encoding = StringEncoding

    @staticmethod
    def empty() raises -> Self.Classes:
        return List[String]()

    @staticmethod
    def flat(y: Self.Target, mode: Int) raises -> Self.Target:
        if y.columns() != 1:
            raise Error("ValueError: y should be a 1d array or a single-column array; multioutput strings are unsupported")
        return StringTarget(y.values())

    @staticmethod
    def size(y: Self.Target) raises -> Int:
        return y.rows() * y.columns()

    @staticmethod
    def rank(y: Self.Target) raises -> Int:
        return y.rank()

    @staticmethod
    def kind(y: Self.Target) raises -> String:
        return type_of_target(y)

    @staticmethod
    def make(y: Self.Target) raises -> Self.Encoding:
        return encode_column(y)

    @staticmethod
    def classes(encoding: Self.Encoding) raises -> Self.Classes:
        return encoding.classes.copy()

    @staticmethod
    def domain_count(encoding: Self.Encoding) raises -> Int:
        return len(encoding.classes)

    @staticmethod
    def codes(encoding: Self.Encoding) raises -> NDArray[DType.int64]:
        return encoding.index.inverse.copy()

    @staticmethod
    def encode(encoding: Self.Encoding, y: Self.Target) raises -> NDArray[DType.int64]:
        return encoding.encode(y)

    @staticmethod
    def decode(encoding: Self.Encoding, codes: NDArray[DType.int64]) raises -> Self.Target:
        return encoding.decode(codes)

    @staticmethod
    def count(classes: Self.Classes) raises -> Int:
        return len(classes)

    @staticmethod
    def same(a: Self.Classes, b: Self.Classes) raises -> Bool:
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if a[i] != b[i]:
                return False
        return True

    @staticmethod
    def from_classes(classes: Self.Classes) raises -> Self.Encoding:
        return encode_column(StringTarget(classes))

    @staticmethod
    def binarize(y: Self.Target, classes: Self.Classes) raises -> NDArray[DType.int64]:
        return label_binarize(y.values(), classes=classes)

    @staticmethod
    def accuracy(y: Self.Target, predicted: Self.Target, weights: Optional[NDArray[DType.float64]]) raises -> Float64:
        return mean_accuracy(y, predicted, weights)
