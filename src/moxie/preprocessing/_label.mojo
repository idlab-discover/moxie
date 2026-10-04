# Adapted from scikit-learn sklearn/preprocessing/_label.py and sklearn/utils/_encode.py;
# authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Target label encoders: LabelEncoder, LabelBinarizer, MultiLabelBinarizer, label_binarize.

Labels are target-only (`fit(y)`), typed by a label kind: `StringLabels`
(`List[String]`) or `NumericLabels[dtype]` (`NDArray[int64/float32/float64]`).
Sorted class domains and target kinds come from the shared engine
`moxie._labels`; this module adds sklearn's NaN/infinity rules for
LabelEncoder, class lookup by value and the binarization semantics.
See docs/api/labels.md.
"""
from std.math import isnan, isfinite
from std.sys import size_of
from numojo import NDArray
from moxie._labels import StringTarget, encode_column, type_of_target as _engine_type_of_target
from moxie.preprocessing._contracts import LabelTransformer
from moxie.preprocessing._common import at, at1, matrix, vector, check_fitted, dtype_name
from moxie.preprocessing._validation import require_dense_output


# ---------------------------------------------------------------------------
# Label kinds


trait LabelKind:
    """Typed label container operations used by the generic label estimators.

    `Labels` is the container users pass and receive (rank-1 targets, or a
    rank-2 indicator matrix for numeric multilabel data); `Value` is one label.
    """
    comptime Value: Copyable & Deinitable
    comptime Labels: Copyable & Deinitable

    @staticmethod
    def rows(y: Self.Labels) raises -> Int:
        """Number of samples (first dimension)."""
        ...

    @staticmethod
    def column_length(y: Self.Labels) raises -> Int:
        """As sklearn `column_or_1d`: a vector or one-column matrix; returns its length."""
        ...

    @staticmethod
    def get(y: Self.Labels, i: Int) raises -> Self.Value:
        """Label `i` of a vector or one-column matrix."""
        ...

    @staticmethod
    def make(values: List[Self.Value]) raises -> Self.Labels:
        ...

    @staticmethod
    def take(classes: Self.Labels, indices: List[Int]) raises -> Self.Labels:
        ...

    @staticmethod
    def before(a: Self.Value, b: Self.Value) -> Bool:
        """Strict total order of sorted classes (NumPy order: NaN last)."""
        ...

    @staticmethod
    def same(a: Self.Value, b: Self.Value) -> Bool:
        """Label identity; NaN matches NaN as in sklearn `_encode`."""
        ...

    @staticmethod
    def is_nan(value: Self.Value) -> Bool:
        ...

    @staticmethod
    def text(value: Self.Value) -> String:
        """NumPy array-element form, used in error messages."""
        ...

    @staticmethod
    def padded() -> Bool:
        """True when NumPy right-aligns elements of printed arrays."""
        ...

    @staticmethod
    def unique(y: Self.Labels) raises -> Self.Labels:
        """As sklearn `_unique`: sorted distinct labels of a vector, NaN last."""
        ...

    @staticmethod
    def target_type(y: Self.Labels) raises -> String:
        """As sklearn `type_of_target` (rejects NaN/infinity)."""
        ...

    @staticmethod
    def check_array(y: Self.Labels) raises:
        """As check_array in `label_binarize`: rank <= 2, finite, at least one sample."""
        ...

    @staticmethod
    def indicator_columns(y: Self.Labels) raises -> Int:
        ...

    @staticmethod
    def binarize_indicator(y: Self.Labels, columns: List[Int], neg_label: Int, pos_label: Int, pos_switch: Bool) raises -> NDArray[DType.int64]:
        """Multilabel branch of label_binarize; output column j reads input column columns[j]."""
        ...

    @staticmethod
    def arange(n: Int) raises -> Self.Labels:
        """Classes 0..n-1 of an n-column multilabel indicator."""
        ...

    @staticmethod
    def from_indicator(Y: NDArray[DType.int64]) raises -> Self.Labels:
        """A thresholded multilabel indicator in this kind's container."""
        ...

    # Numeric-target access, used by TargetEncoder's continuous targets.
    comptime number_dtype: DType
    """NumPy's mean/var dtype for these labels: Float32 for Float32, else Float64."""

    @staticmethod
    def number(value: Self.Value) raises -> Float64:
        """The label as a number (exact for Float32/Float64; strings raise TypeError)."""
        ...

    @staticmethod
    def python_str(value: Self.Value) -> String:
        """Python `str(label)`, as used in generated feature names."""
        ...


def _shape_text(rows: Int, cols: Int, ndim: Int) -> String:
    if ndim == 1:
        return "(" + String(rows) + ",)"
    return "(" + String(rows) + ", " + String(cols) + ")"


struct StringLabels(LabelKind):
    """String labels as `List[String]`, ordered by UTF-8 bytes (= code points)."""
    comptime Value = String
    comptime Labels = List[String]

    @staticmethod
    def rows(y: List[String]) raises -> Int:
        return len(y)

    @staticmethod
    def column_length(y: List[String]) raises -> Int:
        return len(y)

    @staticmethod
    def get(y: List[String], i: Int) raises -> String:
        return y[i].copy()

    @staticmethod
    def make(values: List[String]) raises -> List[String]:
        return values.copy()

    @staticmethod
    def take(classes: List[String], indices: List[Int]) raises -> List[String]:
        var result = List[String](capacity=len(indices))
        for index in indices:
            result.append(classes[index].copy())
        return result^

    @staticmethod
    def before(a: String, b: String) -> Bool:
        return a < b

    @staticmethod
    def same(a: String, b: String) -> Bool:
        return a == b

    @staticmethod
    def is_nan(value: String) -> Bool:
        return False

    @staticmethod
    def text(value: String) -> String:
        return "'" + value + "'"

    @staticmethod
    def padded() -> Bool:
        return False

    @staticmethod
    def unique(y: List[String]) raises -> List[String]:
        var encoding = encode_column(StringTarget(y), classification=False)
        return encoding.classes.copy()

    @staticmethod
    def target_type(y: List[String]) raises -> String:
        return _engine_type_of_target(StringTarget(y))

    @staticmethod
    def check_array(y: List[String]) raises:
        if len(y) == 0:
            raise Error("ValueError: Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required.")

    @staticmethod
    def indicator_columns(y: List[String]) raises -> Int:
        raise Error("ValueError: string labels cannot form a multilabel indicator")

    @staticmethod
    def binarize_indicator(y: List[String], columns: List[Int], neg_label: Int, pos_label: Int, pos_switch: Bool) raises -> NDArray[DType.int64]:
        raise Error("ValueError: string labels cannot form a multilabel indicator")

    @staticmethod
    def arange(n: Int) raises -> List[String]:
        raise Error("ValueError: string labels cannot form a multilabel indicator")

    @staticmethod
    def from_indicator(Y: NDArray[DType.int64]) raises -> List[String]:
        raise Error("ValueError: string labels cannot form a multilabel indicator")

    comptime number_dtype = DType.float64

    @staticmethod
    def number(value: String) raises -> Float64:
        raise Error("TypeError: string labels are not numeric targets; got '" + value + "'")

    @staticmethod
    def python_str(value: String) -> String:
        return value.copy()


def _check_label_dtype[d: DType]():
    comptime assert d == DType.int64 or d == DType.float32 or d == DType.float64, "numeric labels must be Int64, Float32 or Float64"


struct NumericLabels[dtype: DType](LabelKind):
    """Numeric labels as NuMojo `NDArray[dtype]` (Int64, Float32 or Float64).

    Targets are rank 1 (or one-column rank 2); LabelBinarizer and
    label_binarize also accept a rank-2 multilabel indicator. Strided views are
    read in place.
    """
    comptime Value = Scalar[Self.dtype]
    comptime Labels = NDArray[Self.dtype]

    @staticmethod
    def rows(y: NDArray[Self.dtype]) raises -> Int:
        return y.shape[0] if y.ndim >= 1 else 0

    @staticmethod
    def column_length(y: NDArray[Self.dtype]) raises -> Int:
        _check_label_dtype[Self.dtype]()
        if y.ndim == 1:
            return y.shape[0]
        if y.ndim == 2 and y.shape[1] == 1:
            return y.shape[0]
        if y.ndim == 2:
            raise Error("ValueError: y should be a 1d array, got an array of shape " + _shape_text(y.shape[0], y.shape[1], 2) + " instead.")
        raise Error("ValueError: y should be a 1d array, got an array of rank " + String(y.ndim) + " instead.")

    @staticmethod
    def get(y: NDArray[Self.dtype], i: Int) raises -> Scalar[Self.dtype]:
        if y.ndim == 1:
            return at1(y, i)
        return at(y, i, 0)

    @staticmethod
    def make(values: List[Scalar[Self.dtype]]) raises -> NDArray[Self.dtype]:
        var result = vector[Self.dtype](len(values))
        for i in range(len(values)):
            result.unsafe_set(i, values[i])
        return result^

    @staticmethod
    def take(classes: NDArray[Self.dtype], indices: List[Int]) raises -> NDArray[Self.dtype]:
        var result = vector[Self.dtype](len(indices))
        for i in range(len(indices)):
            result.unsafe_set(i, at1(classes, indices[i]))
        return result^

    @staticmethod
    def before(a: Scalar[Self.dtype], b: Scalar[Self.dtype]) -> Bool:
        comptime if Self.dtype != DType.int64:
            if isnan(a):
                return False
            if isnan(b):
                return True
        return Bool(a < b)

    @staticmethod
    def same(a: Scalar[Self.dtype], b: Scalar[Self.dtype]) -> Bool:
        comptime if Self.dtype != DType.int64:
            if isnan(a) and isnan(b):
                return True
        return Bool(a == b)

    @staticmethod
    def is_nan(value: Scalar[Self.dtype]) -> Bool:
        comptime if Self.dtype != DType.int64:
            return isnan(value)
        else:
            return False

    @staticmethod
    def text(value: Scalar[Self.dtype]) -> String:
        comptime if Self.dtype == DType.int64:
            return String(value)
        else:
            if isnan(value):
                return "nan"
            if not isfinite(value):
                return "inf" if value > 0 else "-inf"
            var x = Float64(value)
            if abs(x) < 1e16 and x == Float64(Int(x)):
                return String(Int(x)) + "."
            return String(value)

    @staticmethod
    def padded() -> Bool:
        return True

    @staticmethod
    def unique(y: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        var n = Self.column_length(y)
        comptime if Self.dtype == DType.int64:
            return encode_column(y, classification=False).classes.copy()
        else:
            var count = 0
            var has_nan = False
            var has_neg = False
            var has_pos = False
            for i in range(n):
                var value = Self.get(y, i)
                if isnan(value):
                    has_nan = True
                elif not isfinite(value):
                    if value > 0:
                        has_pos = True
                    else:
                        has_neg = True
                else:
                    count += 1
            if count == n:
                return encode_column(y, classification=False).classes.copy()
            # sklearn `_unique_np` keeps -inf/inf as ordinary values and one
            # NaN last. The shared engine accepts finite labels only, so the
            # finite labels are compacted and the others placed around them.
            var compact = vector[Self.dtype](count)
            var k = 0
            for i in range(n):
                var value = Self.get(y, i)
                if isfinite(value):
                    compact.unsafe_set(k, value)
                    k += 1
            var middle = encode_column(compact, classification=False).classes.copy()
            var m = middle.shape[0]
            var result = vector[Self.dtype](m + Int(has_neg) + Int(has_pos) + Int(has_nan))
            var j = 0
            var infinity = Scalar[Self.dtype](Float64("inf"))
            if has_neg:
                result.unsafe_set(j, -infinity)
                j += 1
            for i in range(m):
                result.unsafe_set(j, at1(middle, i))
                j += 1
            if has_pos:
                result.unsafe_set(j, infinity)
                j += 1
            if has_nan:
                result.unsafe_set(j, Scalar[Self.dtype](Float64("nan")))
            return result^

    @staticmethod
    def _check_finite(y: NDArray[Self.dtype]) raises:
        comptime if Self.dtype != DType.int64:
            var rows = y.shape[0]
            var cols = y.shape[1] if y.ndim == 2 else 1
            for i in range(rows):
                for j in range(cols):
                    var value = at1(y, i) if y.ndim == 1 else at(y, i, j)
                    if isnan(value):
                        raise Error("ValueError: Input y contains NaN.")
                    if not isfinite(value):
                        raise Error("ValueError: Input y contains infinity or a value too large for dtype('" + dtype_name[Self.dtype]() + "').")

    @staticmethod
    def target_type(y: NDArray[Self.dtype]) raises -> String:
        _check_label_dtype[Self.dtype]()
        if y.ndim == 1 or y.ndim == 2:
            Self._check_finite(y)
        return _engine_type_of_target(y)

    @staticmethod
    def check_array(y: NDArray[Self.dtype]) raises:
        _check_label_dtype[Self.dtype]()
        if y.ndim != 1 and y.ndim != 2:
            raise Error("ValueError: Found array with dim " + String(y.ndim) + ". None expected <= 2.")
        Self._check_finite(y)
        var cols = y.shape[1] if y.ndim == 2 else 0
        if y.shape[0] < 1:
            raise Error("ValueError: Found array with 0 sample(s) (shape=" + _shape_text(0, cols, y.ndim) + ") while a minimum of 1 is required.")
        if y.ndim == 2 and cols < 1:
            raise Error("ValueError: Found array with 0 feature(s) (shape=" + _shape_text(y.shape[0], 0, 2) + ") while a minimum of 1 is required.")

    @staticmethod
    def indicator_columns(y: NDArray[Self.dtype]) raises -> Int:
        if y.ndim != 2:
            raise Error("ValueError: a multilabel indicator must be a matrix")
        return y.shape[1]

    @staticmethod
    def binarize_indicator(y: NDArray[Self.dtype], columns: List[Int], neg_label: Int, pos_label: Int, pos_switch: Bool) raises -> NDArray[DType.int64]:
        # sklearn: Y = y.copy(); Y[Y != 0] = pos_label (if pos_label != 1);
        # Y[Y == 0] = neg_label; Y[Y == pos_label] = 0 (pos_switch); astype(int).
        # Applied per output cell in the input dtype, then truncated to Int64.
        var n = y.shape[0]
        var k = len(columns)
        var result = matrix[DType.int64](n, k)
        var pos = Scalar[Self.dtype](pos_label)
        var neg = Scalar[Self.dtype](neg_label)
        for i in range(n):
            for j in range(k):
                var value = at(y, i, columns[j])
                if pos_label != 1 and value != 0:
                    value = pos
                if neg_label != 0 and value == 0:
                    value = neg
                if pos_switch and value == pos:
                    value = 0
                result.unsafe_set(i * k + j, value.cast[DType.int64]())
        return result^

    @staticmethod
    def arange(n: Int) raises -> NDArray[Self.dtype]:
        var result = vector[Self.dtype](n)
        for i in range(n):
            result.unsafe_set(i, Scalar[Self.dtype](i))
        return result^

    @staticmethod
    def from_indicator(Y: NDArray[DType.int64]) raises -> NDArray[Self.dtype]:
        var n = Y.shape[0]
        var k = Y.shape[1]
        var result = matrix[Self.dtype](n, k)
        for i in range(n * k):
            result.unsafe_set(i, Y.unsafe_get(i).cast[Self.dtype]())
        return result^

    comptime number_dtype = DType.float32 if Self.dtype == DType.float32 else DType.float64

    @staticmethod
    def number(value: Scalar[Self.dtype]) raises -> Float64:
        return Float64(value)

    @staticmethod
    def python_str(value: Scalar[Self.dtype]) -> String:
        comptime if Self.dtype == DType.int64:
            return String(value)
        else:
            if isnan(value):
                return "nan"
            if not isfinite(value):
                return "inf" if value > 0 else "-inf"
            var x = Float64(value)
            if abs(x) < 1e16 and x == Float64(Int(x)):
                return String(Int(x)) + ".0"
            return String(value)


comptime _Codes = NumericLabels[DType.int64]


# ---------------------------------------------------------------------------
# Generic class lookup


def _argsort[K: LabelKind](values: K.Labels) raises -> List[Int]:
    """Stable merge-sort permutation of a label vector (O(k log k))."""
    var n = K.rows(values)
    var a = List[Int](capacity=n)
    for i in range(n):
        a.append(i)
    var b = a.copy()
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if K.before(K.get(values, a[j]), K.get(values, a[i])):
                    b[k] = a[j]
                    j += 1
                else:
                    b[k] = a[i]
                    i += 1
                k += 1
            while i < mid:
                b[k] = a[i]
                i += 1
                k += 1
            while j < hi:
                b[k] = a[j]
                j += 1
                k += 1
            lo += 2 * width
        var swap = a^
        a = b^
        b = swap^
        width *= 2
    return a^


def _identity(n: Int) -> List[Int]:
    var result = List[Int](capacity=n)
    for i in range(n):
        result.append(i)
    return result^


def _lower_bound[K: LabelKind](values: K.Labels, order: List[Int], value: K.Value) raises -> Int:
    """First sorted position whose label is not before `value` (searchsorted left)."""
    var lo = 0
    var hi = len(order)
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if K.before(K.get(values, order[mid]), value):
            lo = mid + 1
        else:
            hi = mid
    return lo


def _lookup[K: LabelKind](values: K.Labels, order: List[Int], value: K.Value) raises -> Int:
    """Sorted position of `value`, or -1 when absent."""
    var position = _lower_bound[K](values, order, value)
    if position < len(order) and K.same(K.get(values, order[position]), value):
        return position
    return -1


def _labels_text[K: LabelKind](labels: K.Labels) raises -> String:
    """NumPy `str(array)` form of a label vector, for error messages."""
    var texts = List[String]()
    var width = 0
    for i in range(K.rows(labels)):
        var text = K.text(K.get(labels, i))
        width = max(width, text.byte_length())
        texts.append(text^)
    var result = String("[")
    for i in range(len(texts)):
        if i:
            result += " "
        if K.padded():
            for _ in range(width - texts[i].byte_length()):
                result += " "
        result += texts[i]
    return result + "]"


# ---------------------------------------------------------------------------
# label_binarize


def _label_binarize[K: LabelKind](
    y: K.Labels, classes: K.Labels, neg_label: Int, pos_label: Int, sparse_output: Bool,
    known_type: Optional[String] = None,
) raises -> NDArray[DType.int64]:
    """Dense sklearn `label_binarize`; `known_type` skips re-deriving the target kind."""
    K.check_array(y)
    if neg_label >= pos_label:
        raise Error("ValueError: neg_label=" + String(neg_label) + " must be strictly less than pos_label=" + String(pos_label) + ".")
    if sparse_output and (pos_label == 0 or neg_label != 0):
        raise Error("ValueError: Sparse binarization is only supported with non zero pos_label and zero neg_label, got pos_label=" + String(pos_label) + " and neg_label=" + String(neg_label))
    require_dense_output(sparse_output)
    var pos_switch = pos_label == 0
    var pos = -neg_label if pos_switch else pos_label
    var y_type = known_type.value() if known_type else K.target_type(y)
    if "multioutput" in y_type:
        raise Error("ValueError: Multioutput target data is not supported with label binarization")
    if y_type == "unknown":
        raise Error("ValueError: The type of target data is not known")
    var k = K.rows(classes)
    for j in range(k):
        if K.is_nan(K.get(classes, j)):
            raise Error("ValueError: classes must not contain NaN")
    var n = K.rows(y)
    if y_type == "binary":
        if k == 1:
            var single = matrix[DType.int64](n, 1)
            for i in range(n):
                single.unsafe_set(i, Int64(neg_label))
            return single^
        elif k >= 3:
            y_type = "multiclass"
        elif k == 0:
            raise Error("ValueError: classes must contain at least one label for binary targets")
    var order = _argsort[K](classes)
    if y_type == "multilabel-indicator":
        var columns = K.indicator_columns(y)
        if k != columns:
            raise Error("ValueError: classes " + _labels_text[K](classes) + " mismatch with the labels " + _labels_text[_Codes](_Codes.arange(columns)) + " found in the data")
    # Output column j holds sorted-class position source[j]: j itself when
    # classes are sorted, else searchsorted(sorted_classes, classes[j]).
    var reorder = False
    for j in range(k):
        if not K.same(K.get(classes, j), K.get(classes, order[j])):
            reorder = True
    var source = List[Int](capacity=k)
    for j in range(k):
        source.append(_lower_bound[K](classes, order, K.get(classes, j)) if reorder else j)
    if y_type == "binary" or y_type == "multiclass":
        _ = K.column_length(y)
        var binary = y_type == "binary"
        var width = 1 if binary else k
        var first = k - 1 if binary else 0
        var result = matrix[DType.int64](n, width)
        # Cells are pos_label or 0, then 0 -> neg_label, then (pos_switch) pos -> 0.
        var off = Int64(neg_label)
        var on = Int64(0) if pos_switch else Int64(pos)
        for i in range(n):
            var position = _lookup[K](classes, order, K.get(y, i))
            for c in range(width):
                var hit = position >= 0 and position == source[first + c]
                result.unsafe_set(i * width + c, on if hit else off)
        return result^
    if y_type == "multilabel-indicator":
        return K.binarize_indicator(y, source, neg_label, pos, pos_switch)
    raise Error("ValueError: " + y_type + " target data is not supported with label binarization")


def label_binarize[dtype: DType](y: NDArray[dtype], *, classes: NDArray[dtype], neg_label: Int = 0, pos_label: Int = 1, sparse_output: Bool = False) raises -> NDArray[DType.int64]:
    """Binarize numeric labels one-vs-all; `y` is a vector or a multilabel indicator.

    `classes` shares y's dtype. The result is a dense Int64 matrix; sparse
    output is deferred (`sparse_output=True` raises).
    """
    if classes.ndim != 1:
        raise Error("ValueError: classes must be a vector")
    return _label_binarize[NumericLabels[dtype]](y, classes, neg_label, pos_label, sparse_output)


def label_binarize(y: List[String], *, classes: List[String], neg_label: Int = 0, pos_label: Int = 1, sparse_output: Bool = False) raises -> NDArray[DType.int64]:
    """Binarize string labels one-vs-all into a dense Int64 matrix."""
    return _label_binarize[StringLabels](y, classes, neg_label, pos_label, sparse_output)


# ---------------------------------------------------------------------------
# LabelEncoder


def _check_kind_container[K: LabelKind, L: Copyable & Deinitable]():
    comptime assert size_of[L]() == size_of[K.Labels](), "the container parameter L must be K.Labels (leave it at its default)"


@fieldwise_init
struct LabelEncoderParams(Copyable, Movable):
    pass


struct LabelEncoder[K: LabelKind = StringLabels, L: Copyable & Deinitable = K.Labels](LabelTransformer):
    """Encode target labels as Int64 codes 0..n_classes-1.

    `K` selects the label kind (`StringLabels` or `NumericLabels[dtype]`).
    `L` is K's container type; it only works around a Mojo 1.0 issue with
    fields of associated types and must be left at its default.
    """
    comptime Params = LabelEncoderParams
    comptime Labels = Self.L
    comptime Encoded = NDArray[DType.int64]
    var classes_: Self.L
    var _fitted: Bool

    def __init__(out self) raises:
        _check_kind_container[Self.K, Self.L]()
        self.classes_ = rebind_var[Self.L](Self.K.make(List[Self.K.Value]()))
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params()

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self()

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def fit[
        origin: MutOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[y_origin] y: Self.L,
    ) raises -> ref[self] Self:
        """Learn the sorted distinct labels (NaN last for floating labels)."""
        ref labels = rebind[Self.K.Labels](y)
        self.classes_ = rebind_var[Self.L](Self.K.unique(labels))
        self._fitted = True
        return self

    def fit_transform(mut self, y: Self.L) raises -> NDArray[DType.int64]:
        _ = self.fit(y)
        return self.transform(y)

    def transform(self, y: Self.L) raises -> NDArray[DType.int64]:
        """Codes of known labels; raises for labels absent from classes_."""
        check_fitted(self._fitted, "LabelEncoder")
        ref labels = rebind[Self.K.Labels](y)
        ref classes = rebind[Self.K.Labels](self.classes_)
        var n = Self.K.column_length(labels)
        var result = vector[DType.int64](n)
        var order = _identity(Self.K.rows(classes))
        var unknown = List[Self.K.Value]()
        for i in range(n):
            var value = Self.K.get(labels, i)
            var code = _lookup[Self.K](classes, order, value)
            if code < 0:
                unknown.append(value^)
            else:
                result.unsafe_set(i, Int64(code))
        if len(unknown):
            raise Error("ValueError: y contains previously unseen labels: " + _labels_text[Self.K](Self.K.unique(Self.K.make(unknown))))
        return result^

    def inverse_transform(self, y: NDArray[DType.int64]) raises -> Self.L:
        """Labels for Int64 codes; raises for codes outside 0..n_classes-1."""
        check_fitted(self._fitted, "LabelEncoder")
        ref classes = rebind[Self.K.Labels](self.classes_)
        var n = _Codes.column_length(y)
        var k = Self.K.rows(classes)
        var indices = List[Int](capacity=n)
        var bad = List[Int64]()
        for i in range(n):
            var code = _Codes.get(y, i)
            if code < 0 or code >= Int64(k):
                bad.append(code)
            else:
                indices.append(Int(code))
        if len(bad):
            raise Error("ValueError: y contains previously unseen labels: " + _labels_text[_Codes](_Codes.unique(_Codes.make(bad))))
        return rebind_var[Self.L](Self.K.take(classes, indices))


# ---------------------------------------------------------------------------
# LabelBinarizer


@fieldwise_init
struct LabelBinarizerParams(Copyable, Movable):
    var neg_label: Int
    var pos_label: Int
    var sparse_output: Bool


def _above[d: DType](value: Scalar[d], threshold: Float64) -> Bool:
    """Compute `Y > threshold` in sklearn's matching floating dtype."""
    comptime if d == DType.float32:
        return value.cast[DType.float32]() > Float32(threshold)
    else:
        return Float64(value) > threshold


struct LabelBinarizer[K: LabelKind = StringLabels, L: Copyable & Deinitable = K.Labels](LabelTransformer):
    """Binarize labels one-vs-all into a dense Int64 matrix.

    Numeric kinds also accept a rank-2 multilabel indicator. `sparse_output`
    must stay False (sparse output is deferred). `L` must stay at its default.
    """
    comptime Params = LabelBinarizerParams
    comptime Labels = Self.L
    comptime Encoded = NDArray[DType.int64]
    var neg_label: Int
    var pos_label: Int
    var sparse_output: Bool
    var classes_: Self.L
    var y_type_: String
    var sparse_input_: Bool
    var _fitted: Bool

    def __init__(out self, *, neg_label: Int = 0, pos_label: Int = 1, sparse_output: Bool = False) raises:
        _check_kind_container[Self.K, Self.L]()
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output
        self.classes_ = rebind_var[Self.L](Self.K.make(List[Self.K.Value]()))
        self.y_type_ = ""
        self.sparse_input_ = False
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.neg_label, self.pos_label, self.sparse_output)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.neg_label = params.neg_label
        self.pos_label = params.pos_label
        self.sparse_output = params.sparse_output
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        neg_label: Optional[Int] = None,
        pos_label: Optional[Int] = None,
        sparse_output: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        if neg_label:
            self.neg_label = neg_label.value()
        if pos_label:
            self.pos_label = pos_label.value()
        if sparse_output:
            self.sparse_output = sparse_output.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(neg_label=self.neg_label, pos_label=self.pos_label, sparse_output=self.sparse_output)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def fit[
        origin: MutOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[y_origin] y: Self.L,
    ) raises -> ref[self] Self:
        if self.neg_label >= self.pos_label:
            raise Error("ValueError: neg_label=" + String(self.neg_label) + " must be strictly less than pos_label=" + String(self.pos_label) + ".")
        if self.sparse_output and (self.pos_label == 0 or self.neg_label != 0):
            raise Error("ValueError: Sparse binarization is only supported with non zero pos_label and zero neg_label, got pos_label=" + String(self.pos_label) + " and neg_label=" + String(self.neg_label))
        require_dense_output(self.sparse_output)
        ref labels = rebind[Self.K.Labels](y)
        var y_type = Self.K.target_type(labels)
        if "multioutput" in y_type:
            raise Error("ValueError: Multioutput target data is not supported with label binarization")
        if Self.K.rows(labels) == 0:
            raise Error("ValueError: y has 0 samples")
        if y_type == "multilabel-indicator":
            self.classes_ = rebind_var[Self.L](Self.K.arange(Self.K.indicator_columns(labels)))
        elif y_type == "binary" or y_type == "multiclass":
            self.classes_ = rebind_var[Self.L](Self.K.unique(labels))
        else:
            raise Error("ValueError: Unknown label type: " + y_type)
        self.y_type_ = y_type
        self.sparse_input_ = False
        self._fitted = True
        return self

    def fit_transform(mut self, y: Self.L) raises -> NDArray[DType.int64]:
        _ = self.fit(y)
        return self.transform(y)

    def transform(self, y: Self.L) raises -> NDArray[DType.int64]:
        check_fitted(self._fitted, "LabelBinarizer")
        ref labels = rebind[Self.K.Labels](y)
        var y_type = Self.K.target_type(labels)
        if y_type.startswith("multilabel") and not self.y_type_.startswith("multilabel"):
            raise Error("ValueError: The object was not fitted with multilabel input.")
        return _label_binarize[Self.K](labels, rebind[Self.K.Labels](self.classes_), self.neg_label, self.pos_label, self.sparse_output, y_type)

    def inverse_transform(self, y: NDArray[DType.int64]) raises -> Self.L:
        """Labels from a binarized matrix (threshold=None)."""
        return self.inverse_transform(y, None)

    def inverse_transform[d: DType](self, Y: NDArray[d], threshold: Optional[Float64]) raises -> Self.L:
        """Labels from a binarized matrix or decision scores.

        Multiclass rows take the class of their first maximum. Binary and
        multilabel entries are positive when greater than `threshold`
        (None: midway between neg_label and pos_label), compared in Y's
        floating dtype (Float32 for Float32 scores, else Float64).
        """
        comptime assert d == DType.int64 or d == DType.float32 or d == DType.float64, "Y must be Int64, Float32 or Float64"
        check_fitted(self._fitted, "LabelBinarizer")
        ref classes = rebind[Self.K.Labels](self.classes_)
        var k = Self.K.rows(classes)
        var cut = threshold.value() if threshold else Float64(self.pos_label + self.neg_label) / 2.0
        if Y.ndim != 1 and Y.ndim != 2:
            raise Error("ValueError: Y must be a vector or a matrix")
        if self.y_type_ == "multiclass":
            if Y.ndim != 2 or Y.shape[1] == 0:
                raise Error("ValueError: multiclass inverse_transform requires a matrix with at least one column")
            var picks = List[Int](capacity=Y.shape[0])
            for i in range(Y.shape[0]):
                var best = 0
                var top = at(Y, i, 0)
                for j in range(1, Y.shape[1]):
                    var value = at(Y, i, j)
                    comptime if d != DType.int64:
                        if isnan(top):
                            break
                        if isnan(value):
                            best = j
                            break
                    if value > top:
                        top = value
                        best = j
                picks.append(min(best, k - 1))
            return rebind_var[Self.L](Self.K.take(classes, picks))
        if self.y_type_ == "binary" and Y.ndim == 2 and Y.shape[1] > 2:
            raise Error("ValueError: output_type='binary', but y.shape = " + _shape_text(Y.shape[0], Y.shape[1], 2))
        if self.y_type_ != "binary" and (Y.ndim != 2 or Y.shape[1] != k):
            raise Error("ValueError: The number of class is not equal to the number of dimension of y.")
        var rows = Y.shape[0]
        var cols = Y.shape[1] if Y.ndim == 2 else 1
        if self.y_type_ == "binary":
            var picks = List[Int]()
            if Y.ndim == 2 and cols == 2:
                for i in range(rows):
                    var p = Int(_above(at(Y, i, 1), cut))
                    if p >= k:
                        raise Error("ValueError: a positive second column has no class when only one class was fitted")
                    picks.append(p)
            elif k == 1:
                for _ in range(rows):
                    picks.append(0)
            else:
                for i in range(rows):
                    for j in range(cols):
                        var value = at1(Y, i) if Y.ndim == 1 else at(Y, i, j)
                        picks.append(Int(_above(value, cut)))
            return rebind_var[Self.L](Self.K.take(classes, picks))
        if self.y_type_ == "multilabel-indicator":
            var indicator = matrix[DType.int64](rows, cols)
            for i in range(rows):
                for j in range(cols):
                    indicator.unsafe_set(i * cols + j, Int64(Int(_above(at(Y, i, j), cut))))
            return rebind_var[Self.L](Self.K.from_indicator(indicator))
        raise Error("ValueError: " + self.y_type_ + " format is not supported")


# ---------------------------------------------------------------------------
# MultiLabelBinarizer


@fieldwise_init
struct MultiLabelBinarizerParams[K: LabelKind](Copyable):
    var classes: Optional[List[Self.K.Value]]
    var sparse_output: Bool


def _reject_nan[K: LabelKind](values: List[K.Value]) raises:
    for value in values:
        if K.is_nan(value):
            raise Error("ValueError: MultiLabelBinarizer does not support NaN labels")


struct MultiLabelBinarizer[K: LabelKind = StringLabels, L: Copyable & Deinitable = K.Labels](LabelTransformer):
    """Convert per-sample label collections to a dense Int64 indicator matrix.

    Input is `List[List[K.Value]]` (one label list per sample). `classes_`
    is a `K.Labels` vector. Unknown labels at transform are ignored silently
    (sklearn also warns). `L` must stay at its default.
    """
    comptime Params = MultiLabelBinarizerParams[Self.K]
    comptime Labels = List[List[Self.K.Value]]
    comptime Encoded = NDArray[DType.int64]
    var classes: Optional[List[Self.K.Value]]
    var sparse_output: Bool
    var classes_: Self.L
    var _order: List[Int]  # sorted permutation of classes_, for lookup
    var _fitted: Bool

    def __init__(out self, *, classes: Optional[List[Self.K.Value]] = None, sparse_output: Bool = False) raises:
        _check_kind_container[Self.K, Self.L]()
        self.classes = classes.copy()
        self.sparse_output = sparse_output
        self.classes_ = rebind_var[Self.L](Self.K.make(List[Self.K.Value]()))
        self._order = List[Int]()
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.classes.copy(), self.sparse_output)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.classes = params.classes.copy()
        self.sparse_output = params.sparse_output
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        classes: Optional[List[Self.K.Value]] = None,
        sparse_output: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        """Keyword updates; resetting `classes` to None needs the record overload."""
        if classes:
            self.classes = classes.copy()
        if sparse_output:
            self.sparse_output = sparse_output.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(classes=self.classes.copy(), sparse_output=self.sparse_output)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def fit[
        origin: MutOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[y_origin] y: List[List[Self.K.Value]],
    ) raises -> ref[self] Self:
        """Classes are the sorted distinct labels, or `classes` in its given order."""
        require_dense_output(self.sparse_output)
        var learned: Self.K.Labels
        var order: List[Int]
        if not self.classes:
            var flat = List[Self.K.Value]()
            for labels in y:
                _reject_nan[Self.K](labels)
                for label in labels:
                    flat.append(label.copy())
            learned = Self.K.unique(Self.K.make(flat))
            order = _identity(Self.K.rows(learned))
        else:
            _reject_nan[Self.K](self.classes.value())
            learned = Self.K.make(self.classes.value())
            order = _argsort[Self.K](learned)
            for s in range(1, len(order)):
                if Self.K.same(Self.K.get(learned, order[s - 1]), Self.K.get(learned, order[s])):
                    raise Error("ValueError: The classes argument contains duplicate classes. Remove these duplicates before passing them to MultiLabelBinarizer.")
        self.classes_ = rebind_var[Self.L](learned^)
        self._order = order^
        self._fitted = True
        return self

    def fit_transform(mut self, y: List[List[Self.K.Value]]) raises -> NDArray[DType.int64]:
        _ = self.fit(y)
        return self.transform(y)

    def transform(self, y: List[List[Self.K.Value]]) raises -> NDArray[DType.int64]:
        """Row i has a 1 in each column whose class appears in y[i]; unknown labels are ignored."""
        check_fitted(self._fitted, "MultiLabelBinarizer")
        require_dense_output(self.sparse_output)
        ref classes = rebind[Self.K.Labels](self.classes_)
        var k = Self.K.rows(classes)
        var result = matrix[DType.int64](len(y), k)
        for i in range(len(y)):
            for label in y[i]:
                var position = _lookup[Self.K](classes, self._order, label)
                if position >= 0:
                    result.unsafe_set(i * k + self._order[position], 1)
        return result^

    def inverse_transform(self, y: NDArray[DType.int64]) raises -> List[List[Self.K.Value]]:
        """Per row, the classes whose indicator is 1 (sklearn returns tuples)."""
        check_fitted(self._fitted, "MultiLabelBinarizer")
        ref classes = rebind[Self.K.Labels](self.classes_)
        var k = Self.K.rows(classes)
        if y.ndim != 2:
            raise Error("ValueError: Expected 2D array, got " + String(y.ndim) + "D array instead")
        if y.shape[1] != k:
            raise Error("ValueError: Expected indicator for " + String(k) + " classes, but got " + String(y.shape[1]))
        var unexpected = List[Int64]()
        for i in range(y.shape[0]):
            for j in range(k):
                var value = at(y, i, j)
                if value != 0 and value != 1:
                    unexpected.append(value)
        if len(unexpected):
            raise Error("ValueError: Expected only 0s and 1s in label indicator. Also got " + _labels_text[_Codes](_Codes.unique(_Codes.make(unexpected))))
        var result = List[List[Self.K.Value]](capacity=y.shape[0])
        for i in range(y.shape[0]):
            var row = List[Self.K.Value]()
            for j in range(k):
                if at(y, i, j) != 0:
                    row.append(Self.K.get(classes, j))
            result.append(row^)
        return result^
