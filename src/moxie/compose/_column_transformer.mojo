# Adapted from scikit-learn sklearn/compose/_column_transformer.py
# (ColumnTransformer, make_column_transformer), sklearn/pipeline.py
# (_name_estimators, _fit_transform_one, _transform_one),
# sklearn/utils/metaestimators.py (_validate_names) and
# sklearn/utils/_user_interface.py (_message_with_time); authors:
# scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Apply transformers to column subsets of a dense array (`ColumnTransformer`)."""
from numojo import NDArray, Shape, zeros
from moxie.compose._table import ColumnSchema, ColumnTable
from std.reflection.type_info import _unqualified_type_name
from std.time import perf_counter_ns
from std.sys import num_logical_cores
from std.memory import Pointer
from max.algorithm import parallelize
from moxie.base import BaseEstimator, FitTransformer, TransformerMixin, UnsupervisedEstimator, clone
from moxie.utils.validation import check_is_fitted
from moxie.compose._base import ColumnComposer
from moxie.compose._plumbing import (
    BlockLayout,
    ColumnSelection,
    DenseBlockAssembler,
    OutputSpan,
    feature_name,
)
from moxie.compose._column_keys import ColumnKey, KeyKind, KeyType, make_column_selector


comptime ColumnComponent = TransformerMixin & UnsupervisedEstimator & Copyable & Deinitable
"""Column transformers: `Input`/`Transformed` are `NDArray[dtype]`.

A component that also conforms to `FitTransformer` is fitted with
`fit_transform`, otherwise with `fit` then `transform` (sklearn's rule).
"""

comptime ColumnCallable[dtype: DType] = def(NDArray[dtype]) thin raises -> ColumnKey
"""A `columns` callable: receives X at fit and returns a column key."""

comptime FeatureNameFormatter = def(String, String) thin raises -> String
"""`verbose_feature_names_out` callable: (transformer_name, feature_name) -> name."""

comptime _1D_MESSAGE = (
    "ValueError: 1D data passed to a transformer that expects 2D data. Try to specify the column "
    "selection as a list of one item instead of a scalar."
)


trait FeatureNamesOut:
    """Components that report output feature names (sklearn `get_feature_names_out`).

    An empty `input_features` list means "not given", following the moxie
    convention used by the feature_selection family.
    """

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        ...


struct Mode:
    comptime ESTIMATOR = 0
    comptime PASSTHROUGH = 1
    comptime DROP = 2


# ---------------------------------------------------------------------------
# Placeholder component type for entries that only hold 'drop'/'passthrough'.


@fieldwise_init
struct NoTransformerParams(Copyable, Movable):
    pass


trait _ModeOnly:
    """Marker for NoTransformer: its entries never reach a component call."""

    pass


struct NoTransformer[dtype: DType = DType.float64](ColumnComponent, _ModeOnly):
    """Type slot of an entry written with "drop" or "passthrough" only.

    It holds no estimator and cannot be fitted; give the entry a concrete
    transformer type (`ColumnEntry[MyScaler]("a", "drop", [0])`) to replace
    the mode with an estimator later through `set_transformer`.
    """

    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    comptime Params = NoTransformerParams

    def __init__(out self):
        pass

    def get_params(self, deep: Bool = True) raises -> NoTransformerParams:
        return NoTransformerParams()

    def set_params(mut self, params: NoTransformerParams) raises:
        pass

    def __sklearn_clone__(self) raises -> Self:
        return Self()

    def __sklearn_is_fitted__(self) -> Bool:
        return False

    def fit(mut self, X: NDArray[Self.dtype], y: Optional[NDArray[Self.dtype]] = None) raises:
        raise Error("TypeError: NoTransformer marks 'drop'/'passthrough' entries and cannot be fitted")

    def transform(self, X: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        raise Error("TypeError: NoTransformer marks 'drop'/'passthrough' entries and cannot transform")


# ---------------------------------------------------------------------------
# Constructor values.


struct TransformerSpec[E: ColumnComponent](Copyable, Movable):
    """A transformer value: an estimator, "passthrough" or "drop".

    Strings are validated at fit, like sklearn's parameter validation.
    """

    var estimator: Optional[Self.E]
    var text: String
    var _unset: Bool

    @staticmethod
    def unset() -> Self:
        """Keyword default meaning "leave unchanged" in `set_params`."""
        var spec = Self(String())
        spec._unset = True
        return spec^

    def is_set(self) -> Bool:
        return not self._unset

    @implicit
    def __init__(out self, var estimator: Self.E):
        self.estimator = estimator^
        self.text = String()
        self._unset = False

    @implicit
    def __init__(out self, text: StringLiteral):
        self.estimator = None
        self.text = String(text)
        self._unset = False

    @implicit
    def __init__(out self, text: String):
        self.estimator = None
        self.text = text
        self._unset = False

    def mode(self) -> Int:
        if self.estimator:
            return Mode.ESTIMATOR
        return Mode.DROP if self.text == "drop" else Mode.PASSTHROUGH

    def is_valid(self) -> Bool:
        return Bool(self.estimator) or self.text == "drop" or self.text == "passthrough"

    def description(self) -> String:
        if self.estimator:
            return _type_label[Self.E]()
        return self.text

    def cloned(self) raises -> Self:
        if self.estimator:
            return Self(clone(self.estimator.value()))
        return Self(self.text)


struct Columns[dtype: DType = DType.float64](Copyable, Movable, Writable):
    """A `columns` value: any `ColumnKey` form or a `ColumnCallable[dtype]`."""

    var key: ColumnKey
    var function: Optional[ColumnCallable[Self.dtype]]

    def __init__(out self, var key: ColumnKey):
        self.key = key^
        self.function = None

    @implicit
    def __init__(out self, index: Int):
        self = Self(ColumnKey(index))

    @implicit
    def __init__(out self, var indices: List[Int]):
        self = Self(ColumnKey(indices^))

    def __init__(out self, *indices: Int, __list_literal__: NoneType = None):
        """List literal of positions: `[0, 2]`."""
        var values = List[Int]()
        for index in indices:
            values.append(index)
        self = Self(ColumnKey(values^))

    def __init__(out self, *mask: Bool, __list_literal__: NoneType = None):
        """List literal mask: `[True, False]`."""
        var values = List[Bool]()
        for flag in mask:
            values.append(flag)
        self = Self(ColumnKey(values^))

    @implicit
    def __init__(out self, positions: Slice):
        self = Self(ColumnKey(positions))

    @implicit
    def __init__(out self, var mask: List[Bool]):
        self = Self(ColumnKey(mask^))

    @implicit
    def __init__(out self, name: StringLiteral):
        self = Self(ColumnKey(name))

    @implicit
    def __init__(out self, name: String):
        self = Self(ColumnKey(name))

    @implicit
    def __init__(out self, var names: List[String]):
        self = Self(ColumnKey(names^))

    @implicit
    def __init__(out self, var selector: make_column_selector):
        self = Self(ColumnKey(selector^))

    @implicit
    def __init__(out self, function: ColumnCallable[Self.dtype]):
        self.key = ColumnKey(kind=KeyKind.INTS)
        self.function = function

    def resolve(self, X: NDArray[Self.dtype], schema: Optional[ColumnSchema] = None) raises -> ColumnKey:
        """Equivalent of sklearn `_validate_column_callables`: call callables with X."""
        if self.function:
            return self.function.value()(X)
        if self.key.is_callable():
            if schema:
                return self.key.selector.value()(schema.value())
            return self.key.selector.value()(X)
        return self.key.copy()

    def key_type(self) raises -> Int:
        if self.function or self.key.is_callable():
            raise Error("ValueError: callable column specification")
        return self.key.key_type()

    def write_to(self, mut writer: Some[Writer]):
        if self.function:
            writer.write("<callable>")
        else:
            self.key.write_to(writer)


struct FeatureNamesOutOption(Copyable, Movable):
    """`verbose_feature_names_out`: Bool, a format string or a callable."""

    var enabled: Bool
    var format: Optional[String]
    var function: Optional[FeatureNameFormatter]
    var _unset: Bool

    @staticmethod
    def unset() -> Self:
        """Keyword default meaning "leave unchanged" in `set_params`."""
        var option = Self(True)
        option._unset = True
        return option^

    def is_set(self) -> Bool:
        return not self._unset

    @implicit
    def __init__(out self, enabled: Bool):
        self.enabled = enabled
        self.format = None
        self.function = None
        self._unset = False

    @implicit
    def __init__(out self, format: StringLiteral):
        self.enabled = True
        self.format = String(format)
        self.function = None
        self._unset = False

    @implicit
    def __init__(out self, format: String):
        self.enabled = True
        self.format = format
        self.function = None
        self._unset = False

    @implicit
    def __init__(out self, function: FeatureNameFormatter):
        self.enabled = True
        self.format = None
        self.function = function
        self._unset = False

    def name(self, transformer: String, feature: String) raises -> String:
        if self.function:
            return self.function.value()(transformer, feature)
        if self.format:
            return _format_name(self.format.value(), transformer, feature)
        return feature_name(transformer, feature)


def _format_name(format: String, transformer: String, feature: String) raises -> String:
    """Python `str.format(transformer_name=..., feature_name=...)` for plain fields."""
    var out = String()
    var bytes = format.as_bytes()
    var i = 0
    var n = len(bytes)
    while i < n:
        var c = Int(bytes[i])
        if c == ord("{"):
            if i + 1 < n and Int(bytes[i + 1]) == ord("{"):
                out += "{"
                i += 2
                continue
            var j = i + 1
            while j < n and Int(bytes[j]) != ord("}"):
                j += 1
            if j >= n:
                raise Error("ValueError: Single '{' encountered in format string")
            var field = String(format[byte=i + 1 : j])
            if field == "transformer_name":
                out += transformer
            elif field == "feature_name":
                out += feature
            else:
                raise Error(String("KeyError: '", field, "'"))
            i = j + 1
        elif c == ord("}"):
            if i + 1 < n and Int(bytes[i + 1]) == ord("}"):
                out += "}"
                i += 2
                continue
            raise Error("ValueError: Single '}' encountered in format string")
        else:
            var j = i
            while j < n and Int(bytes[j]) != ord("{") and Int(bytes[j]) != ord("}"):
                j += 1
            out += String(format[byte=i:j])
            i = j
    return out^


def _type_label[E: AnyType]() -> String:
    """Struct name without module path or parameters (Python `type(e).__name__`)."""
    var full = String(_unqualified_type_name[E]())
    var bracket = full.find("[")
    return String(full[byte=0:bracket]) if bracket >= 0 else full


def _py_list(items: List[String]) -> String:
    """Python repr of a list of strings."""
    var out = String("[")
    for i in range(len(items)):
        out += String(", " if i > 0 else "", "'", items[i], "'")
    return out + "]"


# ---------------------------------------------------------------------------
# Dense helpers.


def _select[dtype: DType](X: NDArray[dtype], indices: List[Int], scalar: Bool) raises -> NDArray[dtype]:
    """`X[:, key]`: a vector for a scalar key, else an (n, k) matrix (owned)."""
    var n = X.shape[0]
    var rs = Int(X.strides[0])
    var cs = Int(X.strides[1])
    if scalar:
        var out = zeros[dtype](Shape(n))
        for i in range(n):
            out.unsafe_set(i, X.unsafe_get(i * rs + indices[0] * cs))
        return out^
    var k = len(indices)
    var out = zeros[dtype](Shape(n, k))
    for i in range(n):
        for j in range(k):
            out.unsafe_set(i * k + j, X.unsafe_get(i * rs + indices[j] * cs))
    return out^


def _scaled[dtype: DType](block: NDArray[dtype], weight: Float64) raises -> NDArray[dtype]:
    """`block * weight` in the block dtype (NumPy keeps Float32 against a Python float)."""
    var out = block.copy()
    var w = Scalar[dtype](weight)
    for i in range(out.size):
        out.unsafe_set(i, out.unsafe_get(i) * w)
    return out^


struct DenseHStack(DenseBlockAssembler):
    """`numpy.hstack` of rank-2 blocks into the layout's owned (n_rows, width) array."""

    def __init__(out self):
        pass

    def concatenate[dtype: DType](self, blocks: List[NDArray[dtype]], layout: BlockLayout) raises -> NDArray[dtype]:
        if len(blocks) != len(layout.spans):
            raise Error("ValueError: blocks and layout disagree")
        var out = zeros[dtype](Shape(layout.n_rows, layout.width))
        for b in range(len(blocks)):
            ref block = blocks[b]
            var start = layout.spans[b].start
            var width = layout.spans[b].stop - start
            if block.ndim != 2 or block.shape[0] != layout.n_rows or block.shape[1] != width:
                raise Error("ValueError: block does not match its layout span")
            var rs = Int(block.strides[0])
            var cs = Int(block.strides[1])
            for i in range(layout.n_rows):
                for j in range(width):
                    out.unsafe_set(i * layout.width + start + j, block.unsafe_get(i * rs + j * cs))
        return out^


def _hstack[dtype: DType](blocks: List[NDArray[dtype]]) raises -> NDArray[dtype]:
    """NumPy `hstack` errors for mismatched rows, then dense assembly."""
    var rows = blocks[0].shape[0]
    for b in range(1, len(blocks)):
        if blocks[b].shape[0] != rows:
            raise Error(String("ValueError: all the input array dimensions except for the concatenation axis must "
                               "match exactly, but along dimension 0, the array at index 0 has size ", rows,
                               " and the array at index ", b, " has size ", blocks[b].shape[0]))
    var layout = BlockLayout(rows)
    for b in range(len(blocks)):
        layout.append(rows, blocks[b].shape[1])
    return DenseHStack().concatenate(blocks, layout)


def _remap_1d(error: Error) -> Error:
    """Equivalent of sklearn rewrites the 2D-array error of scalar selections."""
    if "Expected 2D array, got 1D array instead" in String(error):
        return Error(_1D_MESSAGE)
    return error.copy()


def _elapsed_message(name: String, index: Int, total: Int, seconds: Float64) -> String:
    """Equivalent of sklearn `_message_with_time('ColumnTransformer', '(i of n) Processing name', t)`."""
    var start = String("[ColumnTransformer] ")
    var time_text: String
    if seconds > 60:
        time_text = _fixed(seconds / 60, 4) + "min"
    else:
        time_text = " " + _fixed(seconds, 5) + "s"
    var end = String(" (", index, " of ", total, ") Processing ", name, ", total=", time_text)
    var dots = 70 - start.byte_length() - end.byte_length()
    var out = start
    for _ in range(dots):
        out += "."
    return out + end


def _fixed(value: Float64, width: Int) -> String:
    """`'%{width}.1f' % value` for nonnegative values."""
    var tenths = Int(value * 10 + 0.5)
    var text = String(tenths // 10, ".", tenths % 10)
    while text.byte_length() < width:
        text = " " + text
    return text^


# ---------------------------------------------------------------------------
# Component calls (destructibility evidence for foundation associated types).


def _fit_transform_impl[dtype: DType, E: ColumnComponent](
    mut estimator: E, X: NDArray[dtype], y: Optional[NDArray[dtype]]
) raises -> NDArray[dtype] where conforms_to(E.Input, Deinitable) and conforms_to(E.Target, Deinitable) and conforms_to(
    E.Transformed, Deinitable
):
    var arg = rebind_var[E.Input](X.copy())
    comptime assert E.Target == NDArray[dtype], "column components take NDArray[dtype] targets"
    var target: Optional[E.Target] = None
    if y:
        target = rebind_var[E.Target](y.value().copy())
    estimator.fit(arg, target)
    var out = estimator.transform(arg)
    return rebind_var[NDArray[dtype]](out^)


def _fit_transform_combined[dtype: DType, E: ColumnComponent](
    mut estimator: E, X: NDArray[dtype], y: Optional[NDArray[dtype]]
) raises -> NDArray[dtype] where conforms_to(E, FitTransformer) and conforms_to(E.Input, Deinitable) and conforms_to(
    E.Target, Deinitable
) and conforms_to(E.Transformed, Deinitable):
    comptime assert E.Target == NDArray[dtype], "column components take NDArray[dtype] targets"
    var arg = rebind_var[E.Input](X.copy())
    # The Optional is moved back to a destructible type after the call (on
    # every path), so no destructibility evidence for FitTransformer.Target
    # is needed.
    var target = rebind_var[Optional[E.Target]](y.copy())
    try:
        var out = estimator.fit_transform(arg, target)
        _ = rebind_var[Optional[NDArray[dtype]]](target^)
        return rebind_var[NDArray[dtype]](out^)
    except error:
        _ = rebind_var[Optional[NDArray[dtype]]](target^)
        raise error^


def _transform_impl[dtype: DType, E: ColumnComponent](
    estimator: E, X: NDArray[dtype]
) raises -> NDArray[dtype] where conforms_to(E.Input, Deinitable) and conforms_to(E.Transformed, Deinitable):
    var arg = rebind_var[E.Input](X.copy())
    var out = estimator.transform(arg)
    return rebind_var[NDArray[dtype]](out^)


def _fit_transform_one[dtype: DType, E: ColumnComponent](
    mut estimator: E, X: NDArray[dtype], y: Optional[NDArray[dtype]]
) raises -> NDArray[dtype]:
    comptime assert E.Input == NDArray[dtype], "column components take NDArray[dtype] input"
    comptime assert E.Transformed == NDArray[dtype], "column components return NDArray[dtype]"
    comptime assert E.Target == NDArray[dtype], "column components take NDArray[dtype] targets"
    comptime if conforms_to(E.Input, Deinitable) and conforms_to(E.Target, Deinitable) and conforms_to(
        E.Transformed, Deinitable
    ):
        comptime if conforms_to(E, FitTransformer):
            return _fit_transform_combined[dtype](estimator, X, y)
        else:
            return _fit_transform_impl[dtype](estimator, X, y)
    else:
        comptime assert False, "column components must use NDArray[dtype] arrays"


def _transform_one[dtype: DType, E: ColumnComponent](estimator: E, X: NDArray[dtype]) raises -> NDArray[dtype]:
    comptime if conforms_to(E.Input, Deinitable) and conforms_to(E.Transformed, Deinitable):
        return _transform_impl[dtype](estimator, X)
    else:
        comptime assert False, "column components must use NDArray[dtype] arrays"


def _names_out[E: ColumnComponent](estimator: E, selected: List[String]) raises -> List[String] where conforms_to(
    E, FeatureNamesOut
):
    return estimator.get_feature_names_out(selected)


# ---------------------------------------------------------------------------
# Entries.


@fieldwise_init
struct FitContext(Copyable, Movable):
    """Per-fit settings shared by all entries."""

    var weights: Optional[Dict[String, Float64]]
    var verbose: Bool
    var total: Int

    def weight(self, name: String) raises -> Optional[Float64]:
        if self.weights and name in self.weights.value():
            return self.weights.value()[name]
        return None


@fieldwise_init
struct TransformerRecord(Copyable, Movable, Writable):
    """One `transformers_` row: name, fitted kind and the resolved columns.

    `kind` is "estimator" (fitted), "unfitted" (an empty selection keeps the
    configured estimator unfitted), "passthrough" or "drop"; `transformer`
    names the component type.
    """

    var name: String
    var kind: String
    var transformer: String
    var columns: ColumnKey

    def write_to(self, mut writer: Some[Writer]):
        writer.write("('", self.name, "', ", self.transformer if self.kind == "estimator" or self.kind == "unfitted"
                     else String("'", self.kind, "'"), ", ", self.columns, ")")


struct ColumnEntry[E: ColumnComponent = NoTransformer[DType.float64], dtype: DType = DType.float64](Copyable, Movable):
    """One `(name, transformer, columns)` triple.

    Construction: `ColumnEntry("num", MyScaler(), [0, 1])`,
    `ColumnEntry("raw", "passthrough", 2)`. ColumnTransformer fits a fresh
    copy of its entries and keeps that copy (with the learned fields set) as
    its fitted state; the configured entries are never mutated by fit.
    """

    var name: String
    var transformer: TransformerSpec[Self.E]
    var columns: Columns[Self.dtype]
    # learned
    var fitted_: Optional[Self.E]
    var mode_: Int
    var columns_: Optional[ColumnKey]
    var indices_: List[Int]
    var active_: Bool

    def __init__(out self, name: String, var transformer: TransformerSpec[Self.E], var columns: Columns[Self.dtype]):
        self.name = name
        self.transformer = transformer^
        self.columns = columns^
        self.fitted_ = None
        self.mode_ = Mode.DROP
        self.columns_ = None
        self.indices_ = List[Int]()
        self.active_ = False

    def __init__(out self, var transformer: TransformerSpec[Self.E], var columns: Columns[Self.dtype]):
        """An unnamed entry for `make_column_transformer`, which assigns its name."""
        self = Self(String(), transformer^, columns^)

    def unfitted(self) raises -> Self:
        return Self(self.name, self.transformer.cloned(), self.columns.copy())

    # -- fit phases ---------------------------------------------------------

    def resolve(mut self, X: NDArray[Self.dtype], schema: Optional[ColumnSchema] = None) raises:
        var key = self.columns.resolve(X, schema)
        self.indices_ = key.indices(schema.value()) if schema else key.indices(X.shape[1])
        self.columns_ = key^

    def set_resolved(mut self, var key: ColumnKey, var indices: List[Int]):
        self.columns_ = key^
        self.indices_ = indices^

    def is_skipped(self) -> Bool:
        """Drop entries and empty selections are not fitted (sklearn `_iter`)."""
        return self.transformer.mode() == Mode.DROP or self.columns_.value().is_empty()

    def fit_transform(
        mut self, X: NDArray[Self.dtype], y: Optional[NDArray[Self.dtype]], context: FitContext, mut counter: Int,
        mut blocks: List[NDArray[Self.dtype]], mut names: List[String]
    ) raises:
        self.mode_ = self.transformer.mode()
        self.fitted_ = None
        self.active_ = not self.is_skipped()
        if not self.active_:
            if self.mode_ == Mode.ESTIMATOR:
                self.fitted_ = self.transformer.estimator.value().copy()  # sklearn keeps the unfitted original
            return
        counter += 1
        var started = perf_counter_ns()
        var selection = _select(X, self.indices_, self.columns_.value().is_scalar())
        var block: NDArray[Self.dtype]
        if self.mode_ == Mode.PASSTHROUGH:
            block = selection^
        else:
            comptime if conforms_to(Self.E, _ModeOnly):
                raise Error("TypeError: NoTransformer entries hold no estimator")
            else:
                var estimator = clone(self.transformer.estimator.value())
                try:
                    block = _fit_transform_one[Self.dtype](estimator, selection, y)
                except error:
                    raise _remap_1d(error)
                self.fitted_ = estimator^
        var weight = context.weight(self.name)
        if weight:
            block = _scaled(block, weight.value())
        if context.verbose:
            print(_elapsed_message(self.name, counter, context.total, Float64(perf_counter_ns() - started) / 1e9))
        blocks.append(block^)
        names.append(self.name)

    # -- fitted use ---------------------------------------------------------

    def transform(self, X: NDArray[Self.dtype], weights: Optional[Dict[String, Float64]], mut blocks: List[NDArray[Self.dtype]],
                  mut names: List[String], mapping: List[Int] = List[Int]()) raises:
        if not self.active_:
            return
        var indices = self.indices_.copy()
        if len(mapping) > 0:
            for i in range(len(indices)):
                indices[i] = mapping[indices[i]]
                if indices[i] < 0:
                    raise Error("ValueError: columns are missing from the named input")
        var selection = _select(X, indices, self.columns_.value().is_scalar())
        var block: NDArray[Self.dtype]
        if self.mode_ == Mode.PASSTHROUGH:
            block = selection^
        else:
            comptime if conforms_to(Self.E, _ModeOnly):
                raise Error("TypeError: NoTransformer entries hold no estimator")
            else:
                try:
                    block = _transform_one[Self.dtype](self.fitted_.value(), selection)
                except error:
                    raise _remap_1d(error)
        if weights and self.name in weights.value():
            block = _scaled(block, weights.value()[self.name])
        blocks.append(block^)
        names.append(self.name)

    def feature_names(self, input_names: List[String], mut groups: List[Tuple[String, List[String]]]) raises:
        if not self.active_:
            return
        var selected = List[String]()
        for index in self.indices_:
            selected.append(input_names[index])
        if self.mode_ == Mode.PASSTHROUGH:
            groups.append((self.name, selected^))
            return
        comptime if conforms_to(Self.E, FeatureNamesOut):
            groups.append((self.name, _names_out(self.fitted_.value(), selected)))
        else:
            raise Error(String("AttributeError: Transformer ", self.name, " (type ", _type_label[Self.E](),
                               ") does not provide get_feature_names_out."))

    def record(self) -> TransformerRecord:
        var kind: String
        if self.mode_ == Mode.DROP:
            kind = "drop"
        elif self.mode_ == Mode.PASSTHROUGH:
            kind = "passthrough"
        else:
            kind = "estimator" if self.active_ else "unfitted"
        return TransformerRecord(self.name, kind, _type_label[Self.E](), self.columns_.value().copy())

    def find[T: ColumnComponent](self, name: String) raises -> Optional[T]:
        comptime if Self.E == T:
            if self.name == name:
                if not self.fitted_:
                    raise Error(String("TypeError: transformer '", name, "' is '",
                                       "drop" if self.mode_ == Mode.DROP else "passthrough", "', not an estimator"))
                return rebind_var[T](self.fitted_.value().copy())
        return None


struct _TaskResult[dtype: DType](Copyable, Movable):
    var blocks: List[NDArray[Self.dtype]]
    var names: List[String]
    var error: Optional[String]

    def __init__(out self):
        self.blocks = List[NDArray[Self.dtype]]()
        self.names = List[String]()
        self.error = None


def _workers(n_jobs: Optional[Int], tasks: Int) raises -> Int:
    var jobs = n_jobs.value() if n_jobs else 1
    if jobs == 0:
        raise Error("ValueError: n_jobs == 0 in Parallel has no meaning")
    if jobs < 0:
        jobs = max(1, num_logical_cores() + 1 + jobs)
    return min(max(1, tasks), jobs)


def _drain[dtype: DType](mut results: List[_TaskResult[dtype]], mut blocks: List[NDArray[dtype]], mut names: List[String]) raises:
    # Join precedes this call. Check errors in declaration order before moving
    # any outputs; no partial learned state is published by the caller.
    for result in results:
        if result.error:
            raise Error(result.error.value())
    for i in range(len(results)):
        for name in results[i].names:
            names.append(name)
        while len(results[i].blocks) > 0:
            blocks.append(results[i].blocks.pop(0))


trait ColumnList(Copyable, Movable, Deinitable):
    """Typed heterogeneous sequence of `ColumnEntry` values (internal)."""

    comptime size: Int
    comptime dtype: DType

    def active_before(self, index: Int) -> Int:
        ...

    def fit_at(mut self, index: Int, X: NDArray[Self.dtype], y: Optional[NDArray[Self.dtype]], context: FitContext, mut counter: Int, mut result: _TaskResult[Self.dtype]) raises:
        ...

    def transform_at(self, index: Int, X: NDArray[Self.dtype], weights: Optional[Dict[String, Float64]], mapping: List[Int], mut result: _TaskResult[Self.dtype]) raises:
        ...

    def unfitted(self) raises -> Self:
        ...

    def append_names(self, mut names: List[String]):
        ...

    def append_specs(self, mut specs: List[TransformerRecord]) raises:
        ...

    def all_valid(self, mut invalid: List[String]):
        ...

    def resolve(mut self, X: NDArray[Self.dtype], mut used: List[Int], schema: Optional[ColumnSchema] = None) raises:
        ...

    def key_types(self, mut types: List[Int]) raises:
        ...

    def count_active(self) -> Int:
        ...

    def fit_transform(mut self, X: NDArray[Self.dtype], y: Optional[NDArray[Self.dtype]], context: FitContext,
                      mut counter: Int, mut blocks: List[NDArray[Self.dtype]], mut names: List[String]) raises:
        ...

    def transform(self, X: NDArray[Self.dtype], weights: Optional[Dict[String, Float64]], mut blocks: List[NDArray[Self.dtype]],
                  mut names: List[String], mapping: List[Int] = List[Int]()) raises:
        ...

    def feature_names(self, input_names: List[String], mut groups: List[Tuple[String, List[String]]]) raises:
        ...

    def records(self, mut out: List[TransformerRecord]):
        ...

    def find[T: ColumnComponent](self, name: String) raises -> Optional[T]:
        ...

    def replace[T: ColumnComponent](mut self, name: String, spec: TransformerSpec[T]) raises -> Bool:
        ...

    def configured[T: ColumnComponent](self, name: String) raises -> Optional[TransformerSpec[T]]:
        ...


@fieldwise_init
struct EndColumns[d: DType = DType.float64](ColumnList):
    """Empty tail of a `ColumnList`."""

    comptime size = 0
    comptime dtype = Self.d

    def active_before(self, index: Int) -> Int:
        return 0

    def fit_at(mut self, index: Int, X: NDArray[Self.dtype], y: Optional[NDArray[Self.dtype]], context: FitContext, mut counter: Int, mut result: _TaskResult[Self.dtype]) raises:
        pass

    def transform_at(self, index: Int, X: NDArray[Self.dtype], weights: Optional[Dict[String, Float64]], mapping: List[Int], mut result: _TaskResult[Self.dtype]) raises:
        pass

    def unfitted(self) raises -> Self:
        return Self()

    def append_names(self, mut names: List[String]):
        pass

    def append_specs(self, mut specs: List[TransformerRecord]) raises:
        pass

    def all_valid(self, mut invalid: List[String]):
        pass

    def resolve(mut self, X: NDArray[Self.dtype], mut used: List[Int], schema: Optional[ColumnSchema] = None) raises:
        pass

    def key_types(self, mut types: List[Int]) raises:
        pass

    def count_active(self) -> Int:
        return 0

    def fit_transform(mut self, X: NDArray[Self.dtype], y: Optional[NDArray[Self.dtype]], context: FitContext,
                      mut counter: Int, mut blocks: List[NDArray[Self.dtype]], mut names: List[String]) raises:
        pass

    def transform(self, X: NDArray[Self.dtype], weights: Optional[Dict[String, Float64]], mut blocks: List[NDArray[Self.dtype]],
                  mut names: List[String], mapping: List[Int] = List[Int]()) raises:
        pass

    def feature_names(self, input_names: List[String], mut groups: List[Tuple[String, List[String]]]) raises:
        pass

    def records(self, mut out: List[TransformerRecord]):
        pass

    def find[T: ColumnComponent](self, name: String) raises -> Optional[T]:
        return None

    def replace[T: ColumnComponent](mut self, name: String, spec: TransformerSpec[T]) raises -> Bool:
        return False

    def configured[T: ColumnComponent](self, name: String) raises -> Optional[TransformerSpec[T]]:
        return None


@fieldwise_init
struct ColumnCons[E: ColumnComponent, Rest: ColumnList](ColumnList):
    """`ColumnEntry[E]` followed by the remaining entries."""

    comptime size = 1 + Self.Rest.size
    comptime dtype = Self.Rest.dtype
    var head: ColumnEntry[Self.E, Self.Rest.dtype]
    var tail: Self.Rest

    def active_before(self, index: Int) -> Int:
        if index <= 0:
            return 0
        return (0 if self.head.is_skipped() else 1) + self.tail.active_before(index - 1)

    def fit_at(mut self, index: Int, X: NDArray[Self.dtype], y: Optional[NDArray[Self.dtype]], context: FitContext, mut counter: Int, mut result: _TaskResult[Self.dtype]) raises:
        if index == 0:
            self.head.fit_transform(X, y, context, counter, result.blocks, result.names)
        else:
            self.tail.fit_at(index - 1, X, y, context, counter, result)

    def transform_at(self, index: Int, X: NDArray[Self.dtype], weights: Optional[Dict[String, Float64]], mapping: List[Int], mut result: _TaskResult[Self.dtype]) raises:
        if index == 0:
            self.head.transform(X, weights, result.blocks, result.names, mapping)
        else:
            self.tail.transform_at(index - 1, X, weights, mapping, result)

    def unfitted(self) raises -> Self:
        return Self(self.head.unfitted(), self.tail.unfitted())

    def append_names(self, mut names: List[String]):
        names.append(self.head.name)
        self.tail.append_names(names)

    def append_specs(self, mut specs: List[TransformerRecord]) raises:
        specs.append(TransformerRecord(self.head.name, "", self.head.transformer.description(), self.head.columns.key.copy()))
        self.tail.append_specs(specs)

    def all_valid(self, mut invalid: List[String]):
        if not self.head.transformer.is_valid():
            invalid.append(self.head.transformer.text)
        self.tail.all_valid(invalid)

    def resolve(mut self, X: NDArray[Self.dtype], mut used: List[Int], schema: Optional[ColumnSchema] = None) raises:
        self.head.resolve(X, schema)
        for index in self.head.indices_:
            used.append(index)
        self.tail.resolve(X, used, schema)

    def key_types(self, mut types: List[Int]) raises:
        types.append(self.head.columns.key_type())
        self.tail.key_types(types)

    def count_active(self) -> Int:
        return (0 if self.head.is_skipped() else 1) + self.tail.count_active()

    def fit_transform(mut self, X: NDArray[Self.dtype], y: Optional[NDArray[Self.dtype]], context: FitContext,
                      mut counter: Int, mut blocks: List[NDArray[Self.dtype]], mut names: List[String]) raises:
        self.head.fit_transform(X, y, context, counter, blocks, names)
        self.tail.fit_transform(X, y, context, counter, blocks, names)

    def transform(self, X: NDArray[Self.dtype], weights: Optional[Dict[String, Float64]], mut blocks: List[NDArray[Self.dtype]],
                  mut names: List[String], mapping: List[Int] = List[Int]()) raises:
        self.head.transform(X, weights, blocks, names, mapping)
        self.tail.transform(X, weights, blocks, names, mapping)

    def feature_names(self, input_names: List[String], mut groups: List[Tuple[String, List[String]]]) raises:
        self.head.feature_names(input_names, groups)
        self.tail.feature_names(input_names, groups)

    def records(self, mut out: List[TransformerRecord]):
        out.append(self.head.record())
        self.tail.records(out)

    def find[T: ColumnComponent](self, name: String) raises -> Optional[T]:
        var found = self.head.find[T](name)
        if found:
            return found^
        return self.tail.find[T](name)

    def replace[T: ColumnComponent](mut self, name: String, spec: TransformerSpec[T]) raises -> Bool:
        comptime if Self.E == T:
            if self.head.name == name:
                self.head.transformer = rebind_var[TransformerSpec[Self.E]](spec.copy())
                return True
        return self.tail.replace[T](name, spec)

    def configured[T: ColumnComponent](self, name: String) raises -> Optional[TransformerSpec[T]]:
        comptime if Self.E == T:
            if self.head.name == name:
                return rebind_var[TransformerSpec[T]](self.head.transformer.copy())
        return self.tail.configured[T](name)


# ---------------------------------------------------------------------------
# ColumnTransformer.


@fieldwise_init
struct ColumnTransformerParams[L: ColumnList, R: ColumnComponent](Copyable, Movable):
    """Typed `get_params`/`set_params` record; field names follow sklearn."""

    var transformers: Self.L
    var remainder: TransformerSpec[Self.R]
    var sparse_threshold: Float64
    var n_jobs: Optional[Int]
    var transformer_weights: Optional[Dict[String, Float64]]
    var verbose: Bool
    var verbose_feature_names_out: FeatureNamesOutOption


comptime _CONSTRUCTOR_PARAMS: List[String] = [
    "n_jobs", "remainder", "sparse_threshold", "transformer_weights", "transformers", "verbose",
    "verbose_feature_names_out",
]


struct ColumnTransformer[L: ColumnList, R: ColumnComponent = NoTransformer[L.dtype]](ColumnComposer, Copyable):
    """Applies transformers to columns of a dense array and concatenates the results.

    `ColumnTransformer(ColumnEntry("num", Scaler(), [0, 1]), ColumnEntry("raw", "passthrough", [3]),
    remainder="drop")` mirrors sklearn's `ColumnTransformer([(name, transformer, columns), ...])`.
    Each fit clones every non-dropped transformer with a non-empty selection,
    fits it on its columns (`fit_transform`, forwarding y), scales the output
    by `transformer_weights[name]`, and concatenates the blocks in declaration
    order with the remainder last. `transform` reuses the fitted column indices.

    Mojo adaptations (see docs/api.md): the entry list is a static type, so
    replacing a transformer requires its declared type; outputs are always
    dense (`sparse_output_` is False, `sparse_threshold` is validated only);
    `n_jobs` dispatches independent components to MAX CPU workers. Native named
    tables support column names, `feature_names_in_` and dtype selectors.
    Unnamed arrays retain the sklearn NumPy-input selection contract.
    """

    comptime dtype = Self.L.dtype
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    comptime Params = ColumnTransformerParams[Self.L, Self.R]

    var transformers: Self.L
    var remainder: TransformerSpec[Self.R]
    var sparse_threshold: Float64
    var n_jobs: Optional[Int]
    var transformer_weights: Optional[Dict[String, Float64]]
    var verbose: Bool
    var verbose_feature_names_out: FeatureNamesOutOption
    # learned
    var n_features_in_: Optional[Int]
    var feature_names_in_: Optional[List[String]]
    var sparse_output_: Bool
    var output_indices_: Optional[Dict[String, OutputSpan]]
    var _fitted_transformers: Optional[Self.L]
    var _remainder_entry: Optional[ColumnEntry[Self.R, Self.dtype]]

    def __init__(
        out self,
        var transformers: Self.L,
        *,
        var remainder: TransformerSpec[Self.R] = "drop",
        sparse_threshold: Float64 = 0.3,
        n_jobs: Optional[Int] = None,
        var transformer_weights: Optional[Dict[String, Float64]] = None,
        verbose: Bool = False,
        var verbose_feature_names_out: FeatureNamesOutOption = True,
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32 and Float64 inputs only"
        self.transformers = transformers^
        self.remainder = remainder^
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights^
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out^
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.sparse_output_ = False
        self.output_indices_ = None
        self._fitted_transformers = None
        self._remainder_entry = None

    def __init__(out self, params: Self.Params) raises:
        self = Self(params.transformers.unfitted(), remainder=params.remainder.cloned(),
                    sparse_threshold=params.sparse_threshold, n_jobs=params.n_jobs,
                    transformer_weights=params.transformer_weights.copy(), verbose=params.verbose,
                    verbose_feature_names_out=params.verbose_feature_names_out.copy())

    # GENERATED-ARITY-BEGIN (tools/gen_arity.py)
    def __init__[d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
        out self: ColumnTransformer[EndColumns[d], RR],
        *,
        var remainder: TransformerSpec[RR] = "drop",
        sparse_threshold: Float64 = 0.3,
        n_jobs: Optional[Int] = None,
        var transformer_weights: Optional[Dict[String, Float64]] = None,
        verbose: Bool = False,
        var verbose_feature_names_out: FeatureNamesOutOption = True,
    ):
        """Equivalent of sklearn `ColumnTransformer([...0 entries...], ...)`."""
        self = ColumnTransformer[EndColumns[d], RR](
            EndColumns[d](), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
            transformer_weights=transformer_weights^, verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out^)

    def __init__[E1: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
        out self: ColumnTransformer[ColumnCons[E1, EndColumns[d]], RR],
        var t1: ColumnEntry[E1, d], *,
        var remainder: TransformerSpec[RR] = "drop",
        sparse_threshold: Float64 = 0.3,
        n_jobs: Optional[Int] = None,
        var transformer_weights: Optional[Dict[String, Float64]] = None,
        verbose: Bool = False,
        var verbose_feature_names_out: FeatureNamesOutOption = True,
    ):
        """Equivalent of sklearn `ColumnTransformer([...1 entries...], ...)`."""
        self = ColumnTransformer[ColumnCons[E1, EndColumns[d]], RR](
            ColumnCons[E1, EndColumns[d]](t1^, EndColumns[d]()), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
            transformer_weights=transformer_weights^, verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out^)

    def __init__[E1: ColumnComponent, E2: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
        out self: ColumnTransformer[ColumnCons[E1, ColumnCons[E2, EndColumns[d]]], RR],
        var t1: ColumnEntry[E1, d], var t2: ColumnEntry[E2, d], *,
        var remainder: TransformerSpec[RR] = "drop",
        sparse_threshold: Float64 = 0.3,
        n_jobs: Optional[Int] = None,
        var transformer_weights: Optional[Dict[String, Float64]] = None,
        verbose: Bool = False,
        var verbose_feature_names_out: FeatureNamesOutOption = True,
    ):
        """Equivalent of sklearn `ColumnTransformer([...2 entries...], ...)`."""
        self = ColumnTransformer[ColumnCons[E1, ColumnCons[E2, EndColumns[d]]], RR](
            ColumnCons[E1, ColumnCons[E2, EndColumns[d]]](t1^, ColumnCons[E2, EndColumns[d]](t2^, EndColumns[d]())), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
            transformer_weights=transformer_weights^, verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out^)

    def __init__[E1: ColumnComponent, E2: ColumnComponent, E3: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
        out self: ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, EndColumns[d]]]], RR],
        var t1: ColumnEntry[E1, d], var t2: ColumnEntry[E2, d], var t3: ColumnEntry[E3, d], *,
        var remainder: TransformerSpec[RR] = "drop",
        sparse_threshold: Float64 = 0.3,
        n_jobs: Optional[Int] = None,
        var transformer_weights: Optional[Dict[String, Float64]] = None,
        verbose: Bool = False,
        var verbose_feature_names_out: FeatureNamesOutOption = True,
    ):
        """Equivalent of sklearn `ColumnTransformer([...3 entries...], ...)`."""
        self = ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, EndColumns[d]]]], RR](
            ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, EndColumns[d]]]](t1^, ColumnCons[E2, ColumnCons[E3, EndColumns[d]]](t2^, ColumnCons[E3, EndColumns[d]](t3^, EndColumns[d]()))), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
            transformer_weights=transformer_weights^, verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out^)

    def __init__[E1: ColumnComponent, E2: ColumnComponent, E3: ColumnComponent, E4: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
        out self: ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, EndColumns[d]]]]], RR],
        var t1: ColumnEntry[E1, d], var t2: ColumnEntry[E2, d], var t3: ColumnEntry[E3, d], var t4: ColumnEntry[E4, d], *,
        var remainder: TransformerSpec[RR] = "drop",
        sparse_threshold: Float64 = 0.3,
        n_jobs: Optional[Int] = None,
        var transformer_weights: Optional[Dict[String, Float64]] = None,
        verbose: Bool = False,
        var verbose_feature_names_out: FeatureNamesOutOption = True,
    ):
        """Equivalent of sklearn `ColumnTransformer([...4 entries...], ...)`."""
        self = ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, EndColumns[d]]]]], RR](
            ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, EndColumns[d]]]]](t1^, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, EndColumns[d]]]](t2^, ColumnCons[E3, ColumnCons[E4, EndColumns[d]]](t3^, ColumnCons[E4, EndColumns[d]](t4^, EndColumns[d]())))), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
            transformer_weights=transformer_weights^, verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out^)

    def __init__[E1: ColumnComponent, E2: ColumnComponent, E3: ColumnComponent, E4: ColumnComponent, E5: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
        out self: ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, EndColumns[d]]]]]], RR],
        var t1: ColumnEntry[E1, d], var t2: ColumnEntry[E2, d], var t3: ColumnEntry[E3, d], var t4: ColumnEntry[E4, d], var t5: ColumnEntry[E5, d], *,
        var remainder: TransformerSpec[RR] = "drop",
        sparse_threshold: Float64 = 0.3,
        n_jobs: Optional[Int] = None,
        var transformer_weights: Optional[Dict[String, Float64]] = None,
        verbose: Bool = False,
        var verbose_feature_names_out: FeatureNamesOutOption = True,
    ):
        """Equivalent of sklearn `ColumnTransformer([...5 entries...], ...)`."""
        self = ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, EndColumns[d]]]]]], RR](
            ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, EndColumns[d]]]]]](t1^, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, EndColumns[d]]]]](t2^, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, EndColumns[d]]]](t3^, ColumnCons[E4, ColumnCons[E5, EndColumns[d]]](t4^, ColumnCons[E5, EndColumns[d]](t5^, EndColumns[d]()))))), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
            transformer_weights=transformer_weights^, verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out^)

    def __init__[E1: ColumnComponent, E2: ColumnComponent, E3: ColumnComponent, E4: ColumnComponent, E5: ColumnComponent, E6: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
        out self: ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, EndColumns[d]]]]]]], RR],
        var t1: ColumnEntry[E1, d], var t2: ColumnEntry[E2, d], var t3: ColumnEntry[E3, d], var t4: ColumnEntry[E4, d], var t5: ColumnEntry[E5, d], var t6: ColumnEntry[E6, d], *,
        var remainder: TransformerSpec[RR] = "drop",
        sparse_threshold: Float64 = 0.3,
        n_jobs: Optional[Int] = None,
        var transformer_weights: Optional[Dict[String, Float64]] = None,
        verbose: Bool = False,
        var verbose_feature_names_out: FeatureNamesOutOption = True,
    ):
        """Equivalent of sklearn `ColumnTransformer([...6 entries...], ...)`."""
        self = ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, EndColumns[d]]]]]]], RR](
            ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, EndColumns[d]]]]]]](t1^, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, EndColumns[d]]]]]](t2^, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, EndColumns[d]]]]](t3^, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, EndColumns[d]]]](t4^, ColumnCons[E5, ColumnCons[E6, EndColumns[d]]](t5^, ColumnCons[E6, EndColumns[d]](t6^, EndColumns[d]())))))), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
            transformer_weights=transformer_weights^, verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out^)

    def __init__[E1: ColumnComponent, E2: ColumnComponent, E3: ColumnComponent, E4: ColumnComponent, E5: ColumnComponent, E6: ColumnComponent, E7: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
        out self: ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]]]]]]], RR],
        var t1: ColumnEntry[E1, d], var t2: ColumnEntry[E2, d], var t3: ColumnEntry[E3, d], var t4: ColumnEntry[E4, d], var t5: ColumnEntry[E5, d], var t6: ColumnEntry[E6, d], var t7: ColumnEntry[E7, d], *,
        var remainder: TransformerSpec[RR] = "drop",
        sparse_threshold: Float64 = 0.3,
        n_jobs: Optional[Int] = None,
        var transformer_weights: Optional[Dict[String, Float64]] = None,
        verbose: Bool = False,
        var verbose_feature_names_out: FeatureNamesOutOption = True,
    ):
        """Equivalent of sklearn `ColumnTransformer([...7 entries...], ...)`."""
        self = ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]]]]]]], RR](
            ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]]]]]]](t1^, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]]]]]](t2^, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]]]]](t3^, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]]]](t4^, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]]](t5^, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]](t6^, ColumnCons[E7, EndColumns[d]](t7^, EndColumns[d]()))))))), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
            transformer_weights=transformer_weights^, verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out^)

    def __init__[E1: ColumnComponent, E2: ColumnComponent, E3: ColumnComponent, E4: ColumnComponent, E5: ColumnComponent, E6: ColumnComponent, E7: ColumnComponent, E8: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
        out self: ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]]]]]]], RR],
        var t1: ColumnEntry[E1, d], var t2: ColumnEntry[E2, d], var t3: ColumnEntry[E3, d], var t4: ColumnEntry[E4, d], var t5: ColumnEntry[E5, d], var t6: ColumnEntry[E6, d], var t7: ColumnEntry[E7, d], var t8: ColumnEntry[E8, d], *,
        var remainder: TransformerSpec[RR] = "drop",
        sparse_threshold: Float64 = 0.3,
        n_jobs: Optional[Int] = None,
        var transformer_weights: Optional[Dict[String, Float64]] = None,
        verbose: Bool = False,
        var verbose_feature_names_out: FeatureNamesOutOption = True,
    ):
        """Equivalent of sklearn `ColumnTransformer([...8 entries...], ...)`."""
        self = ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]]]]]]], RR](
            ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]]]]]]](t1^, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]]]]]](t2^, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]]]]](t3^, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]]]](t4^, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]]](t5^, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]](t6^, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]](t7^, ColumnCons[E8, EndColumns[d]](t8^, EndColumns[d]())))))))), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
            transformer_weights=transformer_weights^, verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out^)

    # GENERATED-ARITY-END

    # -- parameters ---------------------------------------------------------

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.transformers.unfitted(), self.remainder.cloned(), self.sparse_threshold, self.n_jobs,
                           self.transformer_weights.copy(), self.verbose, self.verbose_feature_names_out.copy())

    def set_params(mut self, params: Self.Params) raises:
        self.transformers = params.transformers.unfitted()
        self.remainder = params.remainder.cloned()
        self.sparse_threshold = params.sparse_threshold
        self.n_jobs = params.n_jobs
        self.transformer_weights = params.transformer_weights.copy()
        self.verbose = params.verbose
        self.verbose_feature_names_out = params.verbose_feature_names_out.copy()

    def set_params(
        mut self,
        *,
        var remainder: TransformerSpec[Self.R] = TransformerSpec[Self.R].unset(),
        sparse_threshold: Optional[Float64] = None,
        n_jobs: Optional[Int] = None,
        var transformer_weights: Optional[Dict[String, Float64]] = None,
        verbose: Optional[Bool] = None,
        var verbose_feature_names_out: FeatureNamesOutOption = FeatureNamesOutOption.unset(),
        remainder_params: Optional[Self.R.Params] = None,
    ) raises:
        """Update only the supplied values (use the Params record to reset one to None).

        `remainder_params` (sklearn `remainder__<param>`) is applied after a
        remainder replacement. Named transformers: `set_transformer`,
        `set_transformer_params`.
        """
        if remainder_params and not (remainder.is_set() and remainder.estimator) and not self.remainder.estimator:
            raise Error("ValueError: Invalid parameter 'remainder_params' for estimator ColumnTransformer: "
                        "remainder is not an estimator.")
        if remainder.is_set():
            self.remainder = remainder^
        if sparse_threshold:
            self.sparse_threshold = sparse_threshold.value()
        if n_jobs:
            self.n_jobs = n_jobs
        if transformer_weights:
            self.transformer_weights = transformer_weights^
        if verbose:
            self.verbose = verbose.value()
        if verbose_feature_names_out.is_set():
            self.verbose_feature_names_out = verbose_feature_names_out^
        if remainder_params:
            self.remainder.estimator.value().set_params(remainder_params.value())

    def set_transformer[T: ColumnComponent](mut self, name: String, var transformer: TransformerSpec[T]) raises:
        """Equivalent of sklearn `set_params(name=transformer)`: replace a named entry's transformer (same type T)."""
        if not self.transformers.replace[T](name, transformer):
            raise self._invalid_parameter(name)

    def set_transformer_params[T: ColumnComponent](mut self, name: String, params: T.Params) raises:
        """Equivalent of sklearn `set_params(name__param=value)` through the entry's typed parameter record."""
        var spec = self.transformers.configured[T](name)
        if not spec:
            raise self._invalid_parameter(name)
        var updated = spec.value().copy()
        if not updated.estimator:
            raise self._invalid_parameter(name + "_params")
        updated.estimator.value().set_params(params)
        _ = self.transformers.replace[T](name, updated)

    def _invalid_parameter(self, name: String) -> Error:
        var valid = materialize[_CONSTRUCTOR_PARAMS]()
        self.transformers.append_names(valid)
        return Error(String("ValueError: Invalid parameter '", name, "' for estimator ColumnTransformer. ",
                            "Valid parameters are: ", _py_list(valid), "."))

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        """Only the default (dense array) container is available natively."""
        if not transform or transform.value() == "default":
            return
        if transform.value() == "pandas" or transform.value() == "polars":
            raise Error(String("ValueError: set_output(transform='", transform.value(),
                               "') needs dataframe containers, which native moxie does not provide"))
        raise Error(String("ValueError: output config must be in ['default', 'pandas', 'polars'], got ",
                           transform.value()))

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.get_params())

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._fitted_transformers)

    # -- validation -----------------------------------------------------------

    def _validate_params(self) raises:
        if not self.remainder.is_valid():
            raise Error(String("InvalidParameterError: The 'remainder' parameter of ColumnTransformer must be a str among ",
                               "{'drop', 'passthrough'}, an object implementing 'fit' and 'transform' or an object ",
                               "implementing 'fit_transform' and 'transform'. Got '", self.remainder.text,
                               "' instead."))
        if not (self.sparse_threshold >= 0.0 and self.sparse_threshold <= 1.0):
            raise Error(String("InvalidParameterError: The 'sparse_threshold' parameter of ColumnTransformer must be a float in ",
                               "the range [0.0, 1.0]. Got ", self.sparse_threshold, " instead."))

    def _validate_transformers(self) raises:
        var names = List[String]()
        self.transformers.append_names(names)
        if len(names) == 0:
            return
        for i in range(len(names)):
            for j in range(i):
                if names[i] == names[j]:
                    raise Error(String("ValueError: Names provided are not unique: ", _py_list(names)))
        var conflicts = List[String]()
        for name in materialize[_CONSTRUCTOR_PARAMS]():
            for candidate in names:
                if candidate == name:
                    conflicts.append(name)
        if len(conflicts) > 0:
            raise Error(String("ValueError: Estimator names conflict with constructor arguments: ", _py_list(conflicts)))
        var dunder = List[String]()
        for name in names:
            if "__" in name:
                dunder.append(name)
        if len(dunder) > 0:
            raise Error(String("ValueError: Estimator names must not contain __: got ", _py_list(dunder)))
        var invalid = List[String]()
        self.transformers.all_valid(invalid)
        if len(invalid) > 0:
            raise Error(String("TypeError: All estimators should implement fit and transform, or can be 'drop' or ",
                               "'passthrough' specifiers. '", invalid[0], "' (type <class 'str'>) doesn't."))

    def _check_input(self, X: NDArray[Self.dtype]) raises:
        if X.ndim != 2:
            raise Error(String("ValueError: Expected a 2D array, got ", X.ndim, "D array instead. ColumnTransformer ",
                               "needs a (samples, features) matrix."))

    def _remainder_key(self, remaining: List[Int], n_features: Int) raises -> ColumnKey:
        """Equivalent of sklearn `_get_remainder_cols`: a mask when every key is a mask, else positions."""
        var types = List[Int]()
        try:
            self.transformers.key_types(types)
        except:
            return ColumnKey(remaining.copy())
        var first = types[0] if len(types) > 0 else -1
        var uniform = len(types) > 0
        for t in types:
            uniform = uniform and t == first
        if uniform and first == KeyType.BOOL:
            var mask = List[Bool]()
            for i in range(n_features):
                var used = False
                for index in remaining:
                    used = used or index == i
                mask.append(used)
            return ColumnKey(mask^)
        return ColumnKey(remaining.copy())

    # -- fitting --------------------------------------------------------------

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        return self._fit_transform(X, y)

    def fit_transform(mut self, X: ColumnTable[Self.dtype], y: Optional[Self.Target] = None) raises -> Self.Transformed:
        return self._fit_transform(X.values(), y, X.schema())

    def fit(mut self, X: ColumnTable[Self.dtype], y: Optional[Self.Target] = None) raises:
        _ = self.fit_transform(X, y)

    def _fit_transform(mut self, X: Self.Input, y: Optional[Self.Target], schema: Optional[ColumnSchema] = None) raises -> Self.Transformed:
        """Fit all transformers and return the horizontally stacked outputs.

        Learned state (including `n_features_in_`) is replaced only on success.
        """
        self._validate_params()
        self._check_input(X)
        var n_features = X.shape[1]
        var n_samples = X.shape[0]
        self._validate_transformers()
        var used = List[Int]()
        var blocks = List[NDArray[Self.dtype]]()
        var names = List[String]()
        var fitted = self.transformers.unfitted()
        var remainder = ColumnEntry[Self.R, Self.dtype](
            "remainder", self.remainder.cloned(), Columns[Self.dtype](ColumnKey(List[Int]())))
        fitted.resolve(X, used, schema)
        var remaining = ColumnSelection(used, n_features).remainder(n_features).indices.copy()
        var key = self._remainder_key(remaining, n_features)
        if schema:
            var types = List[Int]()
            var all_names = True
            try:
                self.transformers.key_types(types)
                for kind in types:
                    all_names = all_names and kind == KeyType.STR
            except:
                all_names = False
            if all_names and len(types) > 0:
                var remainder_names = List[String]()
                for index in remaining:
                    remainder_names.append(schema.value().names[index])
                key = ColumnKey(remainder_names^)
        var included = len(remaining) > 0 or (key.kind == KeyKind.MASK and n_features > 0)
        remainder.set_resolved(key^, remaining^)
        var total = fitted.count_active()
        if included and not remainder.is_skipped():
            total += 1
        var context = FitContext(self.transformer_weights.copy(), self.verbose, total)
        var counter = 0
        var tasks = Self.L.size + Int(included)
        var workers = _workers(self.n_jobs, tasks)
        if workers == 1:
            fitted.fit_transform(X, y, context, counter, blocks, names)
            if included:
                remainder.fit_transform(X, y, context, counter, blocks, names)
        else:
            var results = List[_TaskResult[Self.dtype]](length=tasks, fill=_TaskResult[Self.dtype]())
            var offsets = List[Int]()
            for i in range(tasks):
                offsets.append(fitted.active_before(i))
            var fitted_ptr = Pointer(to=fitted)
            var remainder_ptr = Pointer(to=remainder)
            var slots = results.unsafe_ptr()
            def work(i: Int) {fitted_ptr, remainder_ptr, slots, X, y, context, offsets}:
                var ordinal = offsets[i]
                try:
                    if i < Self.L.size:
                        fitted_ptr[].fit_at(i, X, y, context, ordinal, slots[unsafe_offset=i])
                    else:
                        remainder_ptr[].fit_transform(X, y, context, ordinal, slots[unsafe_offset=i].blocks, slots[unsafe_offset=i].names)
                except error:
                    slots[unsafe_offset=i].error = String(error)
            parallelize(work, tasks, workers)
            _drain(results, blocks, names)
        for b in range(len(blocks)):
            if blocks[b].ndim != 2:
                raise Error(String("ValueError: The output of the '", names[b], "' transformer should be 2D ",
                                   "(numpy array, scipy sparse array, dataframe)."))
        var output = zeros[Self.dtype](Shape(n_samples, 0)) if len(blocks) == 0 else _hstack(blocks)
        # Every step succeeded: publish the fitted state.
        self._fitted_transformers = fitted^
        if included:
            self._remainder_entry = remainder^
        else:
            self._remainder_entry = None
        self.n_features_in_ = n_features
        self.feature_names_in_ = schema.value().names.copy() if schema else Optional[List[String]](None)
        self.sparse_output_ = False
        if len(blocks) > 0:
            self.output_indices_ = self._output_indices(blocks, names)
        else:
            self.output_indices_ = None
        return output^

    def _output_indices(self, blocks: List[NDArray[Self.dtype]], names: List[String]) raises -> Dict[String, OutputSpan]:
        var layout = BlockLayout(blocks[0].shape[0])
        var spans = Dict[String, OutputSpan]()
        for b in range(len(blocks)):
            layout.append(blocks[b].shape[0], blocks[b].shape[1])
            spans[names[b]] = layout.spans[b].copy()
        var all_names = List[String]()
        self.transformers.append_names(all_names)  # the configuration that was just fitted
        all_names.append("remainder")
        for name in all_names:
            if name not in spans:
                spans[name] = OutputSpan(0, 0)
        return spans^

    def fit(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises:
        _ = self.fit_transform(X, y)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """Transform with the fitted transformers and their fitted column indices."""
        check_is_fitted(self)
        self._check_input(X)
        if X.shape[1] != self.n_features_in_.value():
            raise Error(String("ValueError: X has ", X.shape[1], " features, but ColumnTransformer is expecting ",
                               self.n_features_in_.value(), " features as input."))
        return self._transform(X)

    def transform(self, X: ColumnTable[Self.dtype]) raises -> Self.Transformed:
        check_is_fitted(self)
        if not self.feature_names_in_:
            return self.transform(X.values())
        var schema = X.schema()
        var positions = Dict[String, Int]()
        for i in range(len(schema.names)):
            positions[schema.names[i]] = i
        var mapping = List[Int]()
        for name in self.feature_names_in_.value():
            mapping.append(positions[name] if name in positions else -1)
        return self._transform(X.values(), mapping)

    def _transform(self, X: Self.Input, mapping: List[Int] = List[Int]()) raises -> Self.Transformed:
        var blocks = List[NDArray[Self.dtype]]()
        var names = List[String]()
        var tasks = Self.L.size + Int(Bool(self._remainder_entry))
        var workers = _workers(self.n_jobs, tasks)
        if workers == 1:
            self._fitted_transformers.value().transform(X, self.transformer_weights, blocks, names, mapping)
            if self._remainder_entry:
                self._remainder_entry.value().transform(X, self.transformer_weights, blocks, names, mapping)
        else:
            var results = List[_TaskResult[Self.dtype]](length=tasks, fill=_TaskResult[Self.dtype]())
            var slots = results.unsafe_ptr()
            def work(i: Int) {self, slots, X, mapping}:
                try:
                    if i < Self.L.size:
                        self._fitted_transformers.value().transform_at(i, X, self.transformer_weights, mapping, slots[unsafe_offset=i])
                    else:
                        self._remainder_entry.value().transform(X, self.transformer_weights, slots[unsafe_offset=i].blocks, slots[unsafe_offset=i].names, mapping)
                except error:
                    slots[unsafe_offset=i].error = String(error)
            parallelize(work, tasks, workers)
            _drain(results, blocks, names)
        for b in range(len(blocks)):
            if blocks[b].ndim != 2:
                raise Error(String("ValueError: The output of the '", names[b], "' transformer should be 2D ",
                                   "(numpy array, scipy sparse array, dataframe)."))
        if len(blocks) == 0:
            return zeros[Self.dtype](Shape(X.shape[0], 0))
        return _hstack(blocks)

    # -- fitted attributes ----------------------------------------------------

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        """Output feature names; `x0, x1, ...` input names when none are given."""
        check_is_fitted(self)
        var n = self.n_features_in_.value()
        var names = List[String]()
        if len(input_features) > 0:
            if len(input_features) != n:
                raise Error(String("ValueError: input_features should have length equal to number of features (",
                                   n, "), got ", len(input_features)))
            if self.feature_names_in_:
                for i in range(n):
                    if input_features[i] != self.feature_names_in_.value()[i]:
                        raise Error("ValueError: input_features is not equal to feature_names_in_")
            names = input_features.copy()
        elif self.feature_names_in_:
            names = self.feature_names_in_.value().copy()
        else:
            for i in range(n):
                names.append(String("x", i))
        var groups = List[Tuple[String, List[String]]]()
        self._fitted_transformers.value().feature_names(names, groups)
        if self._remainder_entry:
            self._remainder_entry.value().feature_names(names, groups)
        var out = List[String]()
        if self.verbose_feature_names_out.enabled:
            for group in groups:
                for feature in group[1]:
                    out.append(self.verbose_feature_names_out.name(group[0], feature))
            return out^
        for group in groups:
            for feature in group[1]:
                out.append(feature)
        _check_unique(out)
        return out^

    def transformers_(self) raises -> List[TransformerRecord]:
        """Equivalent of sklearn `transformers_`: one record per entry, then the remainder when it has columns."""
        check_is_fitted(self)
        var out = List[TransformerRecord]()
        self._fitted_transformers.value().records(out)
        if self._remainder_entry:
            out.append(self._remainder_entry.value().record())
        return out^

    def named_transformers_[T: ColumnComponent](self, name: String) raises -> T:
        """Equivalent of sklearn `named_transformers_[name]` for an entry (or "remainder") of type T.

        Returns a copy of the fitted transformer, or of the configured one when
        its selection was empty. 'drop'/'passthrough' entries raise TypeError.
        """
        check_is_fitted(self)
        var found = self._fitted_transformers.value().find[T](name)
        if found:
            return found.value().copy()
        if self._remainder_entry:
            var rest = self._remainder_entry.value().find[T](name)
            if rest:
                return rest.value().copy()
        raise Error(String("KeyError: '", name, "' is not a valid transformer name"))


def _check_unique(names: List[String]) raises:
    """sklearn's duplicate-name error for `verbose_feature_names_out=False`."""
    var counts = Dict[String, Int]()
    var order = List[String]()
    for name in names:
        if name in counts:
            counts[name] += 1
        else:
            counts[name] = 1
            order.append(name)
    # Counter.most_common(6) keeps first-seen order among equal counts.
    var ranked = List[String]()
    var taken = List[Bool]()
    for _ in order:
        taken.append(False)
    for _ in range(min(6, len(order))):
        var best = -1
        for i in range(len(order)):
            if not taken[i] and (best < 0 or counts[order[i]] > counts[order[best]]):
                best = i
        taken[best] = True
        ranked.append(order[best])
    var overlap = List[String]()
    for name in ranked:
        if counts[name] > 1:
            overlap.append(name)
    if len(overlap) == 0:
        return
    sort(overlap)
    var text: String
    if len(overlap) == 6:
        var first = List[String]()
        for i in range(5):
            first.append(overlap[i])
        var shown = _py_list(first)
        text = String(shown[byte=0 : shown.byte_length() - 1]) + ", ...]"
    else:
        text = _py_list(overlap)
    raise Error(String("ValueError: Output feature names: ", text, " are not unique. Please set ",
                       "verbose_feature_names_out=True to add prefixes to feature names"))


# ---------------------------------------------------------------------------
# make_column_transformer.


def _auto_names(var labels: List[String]) -> List[String]:
    """Equivalent of sklearn `_name_estimators`: duplicate labels get -1, -2, ... suffixes."""
    var counts = Dict[String, Int]()
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    var remaining = counts.copy()
    for i in range(len(labels) - 1, -1, -1):
        var label = labels[i]
        if counts.get(label, 0) > 1:
            labels[i] = String(label, "-", remaining.get(label, 0))
            remaining[label] = remaining.get(label, 0) - 1
    return labels^


def _label[E: ColumnComponent](spec: TransformerSpec[E]) -> String:
    """`estimator if isinstance(estimator, str) else type(estimator).__name__.lower()`."""
    if spec.estimator:
        return _type_label[E]().lower()
    return spec.text


# GENERATED-MAKE-BEGIN (tools/gen_arity.py)
def make_column_transformer[E1: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
    var t1: ColumnEntry[E1, d], *,
    var remainder: TransformerSpec[RR] = "drop",
    sparse_threshold: Float64 = 0.3,
    n_jobs: Optional[Int] = None,
    verbose: Bool = False,
    var verbose_feature_names_out: FeatureNamesOutOption = True,
) raises -> ColumnTransformer[ColumnCons[E1, EndColumns[d]], RR]:
    """Equivalent of sklearn `make_column_transformer` with 1 unnamed `ColumnEntry(transformer, columns)` values."""
    var names = _auto_names([_label(t1.transformer)])
    t1.name = names[0]
    return ColumnTransformer[ColumnCons[E1, EndColumns[d]], RR](
        ColumnCons[E1, EndColumns[d]](t1^, EndColumns[d]()), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs, verbose=verbose,
        verbose_feature_names_out=verbose_feature_names_out^)


def make_column_transformer[E1: ColumnComponent, E2: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
    var t1: ColumnEntry[E1, d], var t2: ColumnEntry[E2, d], *,
    var remainder: TransformerSpec[RR] = "drop",
    sparse_threshold: Float64 = 0.3,
    n_jobs: Optional[Int] = None,
    verbose: Bool = False,
    var verbose_feature_names_out: FeatureNamesOutOption = True,
) raises -> ColumnTransformer[ColumnCons[E1, ColumnCons[E2, EndColumns[d]]], RR]:
    """Equivalent of sklearn `make_column_transformer` with 2 unnamed `ColumnEntry(transformer, columns)` values."""
    var names = _auto_names([_label(t1.transformer), _label(t2.transformer)])
    t1.name = names[0]
    t2.name = names[1]
    return ColumnTransformer[ColumnCons[E1, ColumnCons[E2, EndColumns[d]]], RR](
        ColumnCons[E1, ColumnCons[E2, EndColumns[d]]](t1^, ColumnCons[E2, EndColumns[d]](t2^, EndColumns[d]())), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs, verbose=verbose,
        verbose_feature_names_out=verbose_feature_names_out^)


def make_column_transformer[E1: ColumnComponent, E2: ColumnComponent, E3: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
    var t1: ColumnEntry[E1, d], var t2: ColumnEntry[E2, d], var t3: ColumnEntry[E3, d], *,
    var remainder: TransformerSpec[RR] = "drop",
    sparse_threshold: Float64 = 0.3,
    n_jobs: Optional[Int] = None,
    verbose: Bool = False,
    var verbose_feature_names_out: FeatureNamesOutOption = True,
) raises -> ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, EndColumns[d]]]], RR]:
    """Equivalent of sklearn `make_column_transformer` with 3 unnamed `ColumnEntry(transformer, columns)` values."""
    var names = _auto_names([_label(t1.transformer), _label(t2.transformer), _label(t3.transformer)])
    t1.name = names[0]
    t2.name = names[1]
    t3.name = names[2]
    return ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, EndColumns[d]]]], RR](
        ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, EndColumns[d]]]](t1^, ColumnCons[E2, ColumnCons[E3, EndColumns[d]]](t2^, ColumnCons[E3, EndColumns[d]](t3^, EndColumns[d]()))), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs, verbose=verbose,
        verbose_feature_names_out=verbose_feature_names_out^)


def make_column_transformer[E1: ColumnComponent, E2: ColumnComponent, E3: ColumnComponent, E4: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
    var t1: ColumnEntry[E1, d], var t2: ColumnEntry[E2, d], var t3: ColumnEntry[E3, d], var t4: ColumnEntry[E4, d], *,
    var remainder: TransformerSpec[RR] = "drop",
    sparse_threshold: Float64 = 0.3,
    n_jobs: Optional[Int] = None,
    verbose: Bool = False,
    var verbose_feature_names_out: FeatureNamesOutOption = True,
) raises -> ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, EndColumns[d]]]]], RR]:
    """Equivalent of sklearn `make_column_transformer` with 4 unnamed `ColumnEntry(transformer, columns)` values."""
    var names = _auto_names([_label(t1.transformer), _label(t2.transformer), _label(t3.transformer), _label(t4.transformer)])
    t1.name = names[0]
    t2.name = names[1]
    t3.name = names[2]
    t4.name = names[3]
    return ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, EndColumns[d]]]]], RR](
        ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, EndColumns[d]]]]](t1^, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, EndColumns[d]]]](t2^, ColumnCons[E3, ColumnCons[E4, EndColumns[d]]](t3^, ColumnCons[E4, EndColumns[d]](t4^, EndColumns[d]())))), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs, verbose=verbose,
        verbose_feature_names_out=verbose_feature_names_out^)


def make_column_transformer[E1: ColumnComponent, E2: ColumnComponent, E3: ColumnComponent, E4: ColumnComponent, E5: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
    var t1: ColumnEntry[E1, d], var t2: ColumnEntry[E2, d], var t3: ColumnEntry[E3, d], var t4: ColumnEntry[E4, d], var t5: ColumnEntry[E5, d], *,
    var remainder: TransformerSpec[RR] = "drop",
    sparse_threshold: Float64 = 0.3,
    n_jobs: Optional[Int] = None,
    verbose: Bool = False,
    var verbose_feature_names_out: FeatureNamesOutOption = True,
) raises -> ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, EndColumns[d]]]]]], RR]:
    """Equivalent of sklearn `make_column_transformer` with 5 unnamed `ColumnEntry(transformer, columns)` values."""
    var names = _auto_names([_label(t1.transformer), _label(t2.transformer), _label(t3.transformer), _label(t4.transformer), _label(t5.transformer)])
    t1.name = names[0]
    t2.name = names[1]
    t3.name = names[2]
    t4.name = names[3]
    t5.name = names[4]
    return ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, EndColumns[d]]]]]], RR](
        ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, EndColumns[d]]]]]](t1^, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, EndColumns[d]]]]](t2^, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, EndColumns[d]]]](t3^, ColumnCons[E4, ColumnCons[E5, EndColumns[d]]](t4^, ColumnCons[E5, EndColumns[d]](t5^, EndColumns[d]()))))), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs, verbose=verbose,
        verbose_feature_names_out=verbose_feature_names_out^)


def make_column_transformer[E1: ColumnComponent, E2: ColumnComponent, E3: ColumnComponent, E4: ColumnComponent, E5: ColumnComponent, E6: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
    var t1: ColumnEntry[E1, d], var t2: ColumnEntry[E2, d], var t3: ColumnEntry[E3, d], var t4: ColumnEntry[E4, d], var t5: ColumnEntry[E5, d], var t6: ColumnEntry[E6, d], *,
    var remainder: TransformerSpec[RR] = "drop",
    sparse_threshold: Float64 = 0.3,
    n_jobs: Optional[Int] = None,
    verbose: Bool = False,
    var verbose_feature_names_out: FeatureNamesOutOption = True,
) raises -> ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, EndColumns[d]]]]]]], RR]:
    """Equivalent of sklearn `make_column_transformer` with 6 unnamed `ColumnEntry(transformer, columns)` values."""
    var names = _auto_names([_label(t1.transformer), _label(t2.transformer), _label(t3.transformer), _label(t4.transformer), _label(t5.transformer), _label(t6.transformer)])
    t1.name = names[0]
    t2.name = names[1]
    t3.name = names[2]
    t4.name = names[3]
    t5.name = names[4]
    t6.name = names[5]
    return ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, EndColumns[d]]]]]]], RR](
        ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, EndColumns[d]]]]]]](t1^, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, EndColumns[d]]]]]](t2^, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, EndColumns[d]]]]](t3^, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, EndColumns[d]]]](t4^, ColumnCons[E5, ColumnCons[E6, EndColumns[d]]](t5^, ColumnCons[E6, EndColumns[d]](t6^, EndColumns[d]())))))), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs, verbose=verbose,
        verbose_feature_names_out=verbose_feature_names_out^)


def make_column_transformer[E1: ColumnComponent, E2: ColumnComponent, E3: ColumnComponent, E4: ColumnComponent, E5: ColumnComponent, E6: ColumnComponent, E7: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
    var t1: ColumnEntry[E1, d], var t2: ColumnEntry[E2, d], var t3: ColumnEntry[E3, d], var t4: ColumnEntry[E4, d], var t5: ColumnEntry[E5, d], var t6: ColumnEntry[E6, d], var t7: ColumnEntry[E7, d], *,
    var remainder: TransformerSpec[RR] = "drop",
    sparse_threshold: Float64 = 0.3,
    n_jobs: Optional[Int] = None,
    verbose: Bool = False,
    var verbose_feature_names_out: FeatureNamesOutOption = True,
) raises -> ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]]]]]]], RR]:
    """Equivalent of sklearn `make_column_transformer` with 7 unnamed `ColumnEntry(transformer, columns)` values."""
    var names = _auto_names([_label(t1.transformer), _label(t2.transformer), _label(t3.transformer), _label(t4.transformer), _label(t5.transformer), _label(t6.transformer), _label(t7.transformer)])
    t1.name = names[0]
    t2.name = names[1]
    t3.name = names[2]
    t4.name = names[3]
    t5.name = names[4]
    t6.name = names[5]
    t7.name = names[6]
    return ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]]]]]]], RR](
        ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]]]]]]](t1^, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]]]]]](t2^, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]]]]](t3^, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]]]](t4^, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]]](t5^, ColumnCons[E6, ColumnCons[E7, EndColumns[d]]](t6^, ColumnCons[E7, EndColumns[d]](t7^, EndColumns[d]()))))))), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs, verbose=verbose,
        verbose_feature_names_out=verbose_feature_names_out^)


def make_column_transformer[E1: ColumnComponent, E2: ColumnComponent, E3: ColumnComponent, E4: ColumnComponent, E5: ColumnComponent, E6: ColumnComponent, E7: ColumnComponent, E8: ColumnComponent, d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
    var t1: ColumnEntry[E1, d], var t2: ColumnEntry[E2, d], var t3: ColumnEntry[E3, d], var t4: ColumnEntry[E4, d], var t5: ColumnEntry[E5, d], var t6: ColumnEntry[E6, d], var t7: ColumnEntry[E7, d], var t8: ColumnEntry[E8, d], *,
    var remainder: TransformerSpec[RR] = "drop",
    sparse_threshold: Float64 = 0.3,
    n_jobs: Optional[Int] = None,
    verbose: Bool = False,
    var verbose_feature_names_out: FeatureNamesOutOption = True,
) raises -> ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]]]]]]], RR]:
    """Equivalent of sklearn `make_column_transformer` with 8 unnamed `ColumnEntry(transformer, columns)` values."""
    var names = _auto_names([_label(t1.transformer), _label(t2.transformer), _label(t3.transformer), _label(t4.transformer), _label(t5.transformer), _label(t6.transformer), _label(t7.transformer), _label(t8.transformer)])
    t1.name = names[0]
    t2.name = names[1]
    t3.name = names[2]
    t4.name = names[3]
    t5.name = names[4]
    t6.name = names[5]
    t7.name = names[6]
    t8.name = names[7]
    return ColumnTransformer[ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]]]]]]], RR](
        ColumnCons[E1, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]]]]]]](t1^, ColumnCons[E2, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]]]]]](t2^, ColumnCons[E3, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]]]]](t3^, ColumnCons[E4, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]]]](t4^, ColumnCons[E5, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]]](t5^, ColumnCons[E6, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]]](t6^, ColumnCons[E7, ColumnCons[E8, EndColumns[d]]](t7^, ColumnCons[E8, EndColumns[d]](t8^, EndColumns[d]())))))))), remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs, verbose=verbose,
        verbose_feature_names_out=verbose_feature_names_out^)

# GENERATED-MAKE-END
