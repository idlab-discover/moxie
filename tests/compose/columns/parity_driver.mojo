"""Parity driver: build the scenario named in each fixture, fit, transform and report.

Each scenario mirrors one sklearn construction in tests/parity.py. Test doubles
(not library code): Scale(f) ~ FunctionTransformer(x * f, one-to-one names),
Center ~ StandardScaler(with_std=False), Expand ~ FunctionTransformer
interleaving [x, x**2] with names [f, f^2], Short ~ FunctionTransformer(X[:-1]).
Output lines: result (out/tz arrays), names|id|..., spans|id|..., records|id|...,
or error|id|message.
"""
from numojo import NDArray, Shape, zeros
from moxie.base import FitTransformer, TransformerMixin, UnsupervisedEstimator
from moxie.compose import (
    ColumnTransformer,
    ColumnEntry,
    ColumnKey,
    ColumnList,
    ColumnComponent,
    FeatureNamesOut,
    make_column_transformer,
)
from parity_io import Fixture, ResultLine, fixture_lines, report_error


@fieldwise_init
struct DoubleParams(Copyable, Movable):
    var factor: Float64


struct Scale[d: DType](Copyable, FeatureNamesOut, TransformerMixin, UnsupervisedEstimator):
    comptime Input = NDArray[Self.d]
    comptime Target = NDArray[Self.d]
    comptime Transformed = NDArray[Self.d]
    comptime Params = DoubleParams
    var factor: Float64

    def __init__(out self, factor: Float64 = 1.0):
        self.factor = factor

    def get_params(self, deep: Bool = True) raises -> DoubleParams:
        return DoubleParams(self.factor)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: DoubleParams,
    ) raises -> ref[self] Self:
        self.factor = params.factor
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.factor)

    def __sklearn_is_fitted__(self) -> Bool:
        return True

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: NDArray[Self.d],
        y: Optional[NDArray[Self.d]] = None,
    ) raises -> ref[self] Self:
        return self

    def transform(self, X: NDArray[Self.d]) raises -> NDArray[Self.d]:
        var out = X.copy()
        for i in range(out.size):
            out.unsafe_set(i, out.unsafe_get(i) * Scalar[Self.d](self.factor))
        return out^

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        return input_features.copy()


struct Center[d: DType](Copyable, FeatureNamesOut, FitTransformer, TransformerMixin, UnsupervisedEstimator):
    """Mean removal with Float64 statistics, results rounded to d (as StandardScaler)."""

    comptime Input = NDArray[Self.d]
    comptime Target = NDArray[Self.d]
    comptime Transformed = NDArray[Self.d]
    comptime Params = DoubleParams
    var means_: List[Float64]

    def __init__(out self):
        self.means_ = List[Float64]()

    def get_params(self, deep: Bool = True) raises -> DoubleParams:
        return DoubleParams(0.0)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: DoubleParams,
    ) raises -> ref[self] Self:
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self()

    def __sklearn_is_fitted__(self) -> Bool:
        return len(self.means_) > 0

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: NDArray[Self.d],
        y: Optional[NDArray[Self.d]] = None,
    ) raises -> ref[self] Self:
        if X.ndim != 2:
            raise Error("ValueError: Expected 2D array, got 1D array instead:")
        self.means_ = List[Float64]()
        for j in range(X.shape[1]):
            var total = 0.0
            for i in range(X.shape[0]):
                total += Float64(X.unsafe_get(i * X.shape[1] + j))
            self.means_.append(total / Float64(X.shape[0]))
        return self

    def fit_transform(mut self, X: NDArray[Self.d], y: Optional[NDArray[Self.d]] = None) raises -> NDArray[Self.d]:
        _ = self.fit(X, y)
        return self.transform(X)

    def transform(self, X: NDArray[Self.d]) raises -> NDArray[Self.d]:
        if X.ndim != 2:
            raise Error("ValueError: Expected 2D array, got 1D array instead:")
        var out = X.copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                var v = Float64(X.unsafe_get(i * X.shape[1] + j)) - self.means_[j]
                out.unsafe_set(i * X.shape[1] + j, v.cast[Self.d]())
        return out^

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        return input_features.copy()


struct Expand[d: DType](Copyable, FeatureNamesOut, TransformerMixin, UnsupervisedEstimator):
    comptime Input = NDArray[Self.d]
    comptime Target = NDArray[Self.d]
    comptime Transformed = NDArray[Self.d]
    comptime Params = DoubleParams
    var short: Bool

    def __init__(out self, short: Bool = False):
        self.short = short

    def get_params(self, deep: Bool = True) raises -> DoubleParams:
        return DoubleParams(1.0 if self.short else 0.0)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: DoubleParams,
    ) raises -> ref[self] Self:
        self.short = params.factor != 0
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.short)

    def __sklearn_is_fitted__(self) -> Bool:
        return True

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: NDArray[Self.d],
        y: Optional[NDArray[Self.d]] = None,
    ) raises -> ref[self] Self:
        return self

    def transform(self, X: NDArray[Self.d]) raises -> NDArray[Self.d]:
        var n = X.shape[0]
        var k = X.shape[1]
        if self.short:
            var out = zeros[Self.d](Shape(n - 1, k))
            for i in range((n - 1) * k):
                out.unsafe_set(i, X.unsafe_get(i))
            return out^
        var out = zeros[Self.d](Shape(n, 2 * k))
        for i in range(n):
            for j in range(k):
                var v = X.unsafe_get(i * k + j)
                out.unsafe_set(i * 2 * k + 2 * j, v)
                out.unsafe_set(i * 2 * k + 2 * j + 1, v * v)
        return out^

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        var out = List[String]()
        for name in input_features:
            out.append(name)
            out.append(name + "^2")
        return out^


def last_two[d: DType](X: NDArray[d]) raises -> ColumnKey:
    return ColumnKey([X.shape[1] - 2, X.shape[1] - 1])


def join(items: List[String]) -> String:
    var out = String()
    for i in range(len(items)):
        out += String("," if i > 0 else "", items[i])
    return out^


def report[L: ColumnList, R: ColumnComponent](
    mut ct: ColumnTransformer[L, R], fx: Fixture, names: Bool = True
) raises:
    var X = fx.array[L.dtype]("X")
    var out = ct.fit_transform(X)
    var tz = ct.transform(fx.array[L.dtype]("Z"))
    var line = ResultLine(fx.id)
    line.add("out", out)
    line.add("tz", tz)
    line.emit()
    if names:
        print(String("names|", fx.id, "|", join(ct.get_feature_names_out())))
    var spans = String("spans|", fx.id, "|")
    if ct.output_indices_:
        for entry in ct.output_indices_.value().items():
            spans += String(entry.key, ":", entry.value.start, ":", entry.value.stop, ";")
    print(spans)
    var records = String("records|", fx.id, "|")
    for record in ct.transformers_():
        records += String(record.name, "=", record.kind, "=", record.columns, ";")
    print(records)


def run[d: DType](fx: Fixture) raises:
    var s = fx.text("scenario")
    if s == "lists":
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("a", Scale[d](2.0), [0, 1]),
                                   ColumnEntry[Scale[d], d]("b", Scale[d](-1.0), [3]))
        report(ct, fx)
    elif s == "passthrough_weights":
        var weights = Dict[String, Float64]()
        weights["a"] = 0.5
        weights["remainder"] = 10.0
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("a", Scale[d](2.0), [1]), remainder="passthrough",
                                   transformer_weights=weights^)
        report(ct, fx)
    elif s == "remainder_estimator":
        var ct = ColumnTransformer(ColumnEntry[dtype=d]("a", "drop", [0, 1]), remainder=Center[d]())
        report(ct, fx)
    elif s == "forms":
        var ct = ColumnTransformer(ColumnEntry[dtype=d]("neg", "passthrough", [-1, 0, 0]),
                                   ColumnEntry[dtype=d]("sl", "passthrough", Slice(4, 0, -2)),
                                   ColumnEntry[dtype=d]("mask", "passthrough", [False, True, False, False, True]))
        report(ct, fx)
    elif s == "callable":
        var ct = ColumnTransformer(ColumnEntry[dtype=d]("f", "passthrough", last_two[d]),
                                   ColumnEntry[Center[d], d]("c", Center[d](), [0]), remainder="passthrough")
        report(ct, fx)
    elif s == "empty":
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("e", Scale[d](), List[Int]()),
                                   ColumnEntry[Scale[d], d]("m", Scale[d](), [False, False, False, False, False]),
                                   ColumnEntry[dtype=d]("d", "drop", [2]),
                                   ColumnEntry[Expand[d], d]("a", Expand[d](), [0, 1]), remainder="passthrough")
        report(ct, fx)
    elif s == "masks":
        var ct = ColumnTransformer(ColumnEntry[dtype=d]("m", "passthrough", [True, False, False, False, False]),
                                   ColumnEntry[Center[d], d]("n", Center[d](), [False, False, True, False, False]))
        report(ct, fx)
    elif s == "names_false":
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("a", Scale[d](), [0]),
                                   ColumnEntry[Expand[d], d]("b", Expand[d](), [2]), remainder="passthrough",
                                   verbose_feature_names_out=False)
        report(ct, fx)
    elif s == "names_format":
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("a", Scale[d](), [0]),
                                   ColumnEntry[Expand[d], d]("b", Expand[d](), [2]), remainder="passthrough",
                                   verbose_feature_names_out="{feature_name}@{transformer_name}")
        report(ct, fx)
    elif s == "make":
        var ct = make_column_transformer(ColumnEntry[Scale[d], d](Scale[d](2.0), [0]),
                                         ColumnEntry[Scale[d], d](Scale[d](3.0), [1]),
                                         ColumnEntry[dtype=d]("passthrough", [2]), ColumnEntry[Center[d], d](Center[d](), [3]))
        report(ct, fx)
    elif s == "dense_center":
        var ct = ColumnTransformer(ColumnEntry[Center[d], d]("a", Center[d](), [0, 1, 2, 3, 4]))
        report(ct, fx)
    elif s == "err_duplicate":
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("a", Scale[d](), [0]), ColumnEntry[Scale[d], d]("a", Scale[d](), [1]))
        report(ct, fx)
    elif s == "err_reserved":
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("remainder", Scale[d](), [0]))
        report(ct, fx)
    elif s == "err_dunder":
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("a__b", Scale[d](), [0]))
        report(ct, fx)
    elif s == "err_mode":
        var ct = ColumnTransformer(ColumnEntry[dtype=d]("a", "dorp", [0]))
        report(ct, fx)
    elif s == "err_remainder":
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("a", Scale[d](), [0]), remainder="keep")
        report(ct, fx)
    elif s == "err_threshold":
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("a", Scale[d](), [0]), sparse_threshold=1.5)
        report(ct, fx)
    elif s == "err_range":
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("a", Scale[d](), [5]))
        report(ct, fx)
    elif s == "err_mask":
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("a", Scale[d](), [True, False]))
        report(ct, fx)
    elif s == "err_full_slice":
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("a", Scale[d](), Slice(None, None, 2)))
        report(ct, fx)
    elif s == "err_string":
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("a", Scale[d](), "age"))
        report(ct, fx)
    elif s == "err_scalar_center":
        var ct = ColumnTransformer(ColumnEntry[Center[d], d]("c", Center[d](), 2))
        report(ct, fx)
    elif s == "err_scalar_passthrough":
        var ct = ColumnTransformer(ColumnEntry[dtype=d]("p", "passthrough", 2))
        report(ct, fx)
    elif s == "err_rows":
        var ct = ColumnTransformer(ColumnEntry[dtype=d]("a", "passthrough", [0]), ColumnEntry[Expand[d], d]("r", Expand[d](True), [1]))
        report(ct, fx)
    elif s == "err_collision":
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("a", Scale[d](), [0, 1]),
                                   ColumnEntry[dtype=d]("b", "passthrough", [1, 0]), verbose_feature_names_out=False)
        report(ct, fx)
    elif s == "err_features":
        var ct = ColumnTransformer(ColumnEntry[Scale[d], d]("a", Scale[d](), [0]))
        report(ct, fx)
    else:
        raise Error("unknown scenario " + s)


def main() raises:
    for line in fixture_lines():
        var fx = Fixture(line)
        try:
            if fx.bits == 32:
                run[DType.float32](fx)
            else:
                run[DType.float64](fx)
        except error:
            report_error(fx.id, error)
