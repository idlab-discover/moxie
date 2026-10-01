"""Native contracts for ColumnTransformer, make_column_transformer and make_column_selector."""
from numojo import NDArray, Shape, zeros
from std.testing import assert_equal, assert_true, assert_false, assert_almost_equal, assert_raises, TestSuite
from moxie.base import FitTransformer, TransformerMixin, UnsupervisedEstimator, clone
from moxie.compose import (
    ColumnTransformer,
    ColumnEntry,
    ColumnKey,
    Columns,
    FeatureNamesOut,
    NoTransformer,
    make_column_transformer,
    make_column_selector,
)
from moxie.compose._column_transformer import _elapsed_message, _format_name, _auto_names

comptime F64 = DType.float64
comptime F32 = DType.float32


# -- test doubles (fixtures, not library transformers) -----------------------


@fieldwise_init
struct FactorParams(Copyable, Movable):
    var factor: Float64


struct Scale[d: DType = F64](Copyable, FeatureNamesOut, TransformerMixin, UnsupervisedEstimator):
    """Multiplies by a factor (any rank); fitted by fit + transform; one-to-one names."""

    comptime Input = NDArray[Self.d]
    comptime Target = NDArray[Self.d]
    comptime Transformed = NDArray[Self.d]
    comptime Params = FactorParams
    var factor: Float64
    var fits_: Int

    def __init__(out self, factor: Float64 = 2.0):
        self.factor = factor
        self.fits_ = 0

    def get_params(self, deep: Bool = True) raises -> FactorParams:
        return FactorParams(self.factor)

    def set_params(mut self, params: FactorParams) raises:
        self.factor = params.factor

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.factor)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.fits_ > 0

    def fit(mut self, X: NDArray[Self.d], y: Optional[NDArray[Self.d]] = None) raises:
        self.fits_ += 1

    def transform(self, X: NDArray[Self.d]) raises -> NDArray[Self.d]:
        var out = X.copy()
        for i in range(out.size):
            out.unsafe_set(i, out.unsafe_get(i) * Scalar[Self.d](self.factor))
        return out^

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        return input_features.copy()


@fieldwise_init
struct CenterParams(Copyable, Movable):
    pass


struct Center(Copyable, FitTransformer, TransformerMixin, UnsupervisedEstimator):
    """Subtracts fitted column means; implements fit_transform; records y."""

    comptime Input = NDArray[F64]
    comptime Target = NDArray[F64]
    comptime Transformed = NDArray[F64]
    comptime Params = CenterParams
    var means_: List[Float64]
    var fit_transform_calls_: Int
    var saw_y_: Bool

    def __init__(out self):
        self.means_ = List[Float64]()
        self.fit_transform_calls_ = 0
        self.saw_y_ = False

    def get_params(self, deep: Bool = True) raises -> CenterParams:
        return CenterParams()

    def set_params(mut self, params: CenterParams) raises:
        pass

    def __sklearn_clone__(self) raises -> Self:
        return Self()

    def __sklearn_is_fitted__(self) -> Bool:
        return len(self.means_) > 0

    def fit(mut self, X: NDArray[F64], y: Optional[NDArray[F64]] = None) raises:
        if X.ndim != 2:
            raise Error("ValueError: Expected 2D array, got 1D array instead: array=[...]")
        self.means_ = List[Float64]()
        for j in range(X.shape[1]):
            var total = 0.0
            for i in range(X.shape[0]):
                total += X.unsafe_get(i * X.shape[1] + j)
            self.means_.append(total / Float64(X.shape[0]))
        self.saw_y_ = Bool(y)

    def fit_transform(mut self, X: NDArray[F64], y: Optional[NDArray[F64]] = None) raises -> NDArray[F64]:
        self.fit(X, y)
        self.fit_transform_calls_ += 1
        return self.transform(X)

    def transform(self, X: NDArray[F64]) raises -> NDArray[F64]:
        if X.ndim != 2:
            raise Error("ValueError: Expected 2D array, got 1D array instead: array=[...]")
        var out = X.copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                out.unsafe_set(i * X.shape[1] + j, X.unsafe_get(i * X.shape[1] + j) - self.means_[j])
        return out^


struct Expand(Copyable, FeatureNamesOut, TransformerMixin, UnsupervisedEstimator):
    """Outputs [x, x^2] per input column, or a vector / short block when configured."""

    comptime Input = NDArray[F64]
    comptime Target = NDArray[F64]
    comptime Transformed = NDArray[F64]
    comptime Params = FactorParams
    var factor: Float64  # 0: [x, x^2]; 1: return a vector; 2: drop the last row
    var fitted_: Bool

    def __init__(out self, factor: Float64 = 0):
        self.factor = factor
        self.fitted_ = False

    def get_params(self, deep: Bool = True) raises -> FactorParams:
        return FactorParams(self.factor)

    def set_params(mut self, params: FactorParams) raises:
        self.factor = params.factor

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.factor)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted_

    def fit(mut self, X: NDArray[F64], y: Optional[NDArray[F64]] = None) raises:
        self.fitted_ = True

    def transform(self, X: NDArray[F64]) raises -> NDArray[F64]:
        var n = X.shape[0]
        var k = X.shape[1]
        if self.factor == 1:
            return zeros[F64](Shape(n))
        if self.factor == 2:
            return zeros[F64](Shape(n - 1, k))
        var out = zeros[F64](Shape(n, 2 * k))
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


struct Anonymous(Copyable, TransformerMixin, UnsupervisedEstimator):
    """No get_feature_names_out capability."""

    comptime Input = NDArray[F64]
    comptime Target = NDArray[F64]
    comptime Transformed = NDArray[F64]
    comptime Params = CenterParams
    var fitted_: Bool

    def __init__(out self):
        self.fitted_ = False

    def get_params(self, deep: Bool = True) raises -> CenterParams:
        return CenterParams()

    def set_params(mut self, params: CenterParams) raises:
        pass

    def __sklearn_clone__(self) raises -> Self:
        return Self()

    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted_

    def fit(mut self, X: NDArray[F64], y: Optional[NDArray[F64]] = None) raises:
        self.fitted_ = True

    def transform(self, X: NDArray[F64]) raises -> NDArray[F64]:
        return X.copy()


# -- helpers ------------------------------------------------------------------


def data[d: DType = F64]() raises -> NDArray[d]:
    """4 x 5 matrix with X[i, j] = 10 * i + j."""
    var X = zeros[d](Shape(4, 5))
    for i in range(4):
        for j in range(5):
            X.unsafe_set(i * 5 + j, Scalar[d](10 * i + j))
    return X^


def column_of[d: DType](X: NDArray[d], j: Int, i: Int) raises -> Float64:
    return Float64(X.unsafe_get(i * X.shape[1] + j))


def assert_columns[d: DType](got: NDArray[d], expected: List[Float64], rows: Int = 4) raises:
    """Each output column equals scale * (10 * i + source) given as pairs (source, scale)."""
    assert_equal(got.ndim, 2)
    assert_equal(got.shape[0], rows)
    assert_equal(got.shape[1], len(expected) // 2)
    for c in range(got.shape[1]):
        for i in range(rows):
            var want = expected[2 * c + 1] * (10.0 * Float64(i) + expected[2 * c])
            assert_almost_equal(column_of(got, c, i), want, atol=1e-12)


def expect_error(message: String, err: Error) raises:
    assert_true(String(err).startswith(message), String("expected '", message, "', got '", err, "'"))


# -- tests --------------------------------------------------------------------


def test_basic_fit_transform() raises:
    var X = data()
    var ct = ColumnTransformer(ColumnEntry("a", Scale(2.0), [0, 1]), ColumnEntry("b", Scale(-1.0), [3]))
    assert_false(ct.__sklearn_is_fitted__())
    var out = ct.fit_transform(X)
    assert_columns(out, [0.0, 2.0, 1.0, 2.0, 3.0, -1.0])
    assert_true(ct.__sklearn_is_fitted__())
    assert_equal(ct.n_features_in_.value(), 5)
    assert_false(ct.sparse_output_)
    assert_columns(ct.transform(X), [0.0, 2.0, 1.0, 2.0, 3.0, -1.0])
    var spans = ct.output_indices_.value().copy()
    assert_equal(spans["a"].start, 0)
    assert_equal(spans["a"].stop, 2)
    assert_equal(spans["b"].start, 2)
    assert_equal(spans["b"].stop, 3)
    assert_equal(spans["remainder"].stop, 0)
    var records = ct.transformers_()
    assert_equal(len(records), 3)
    assert_equal(String(records[0]), "('a', Scale, [0, 1])")
    assert_equal(String(records[2]), "('remainder', 'drop', [2, 4])")
    # The configured transformer stays unfitted; the fitted clone is reachable by type.
    assert_equal(ct.named_transformers_[Scale[]]("a").fits_, 1)
    with assert_raises(contains="KeyError: 'zzz' is not a valid transformer name"):
        _ = ct.named_transformers_[Scale[]]("zzz")


def test_remainder_modes_and_weights() raises:
    var X = data()
    var passthrough = ColumnTransformer(ColumnEntry("a", Scale(2.0), [3]), remainder="passthrough")
    assert_columns(passthrough.fit_transform(X), [3.0, 2.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 4.0, 1.0])
    assert_equal(passthrough.output_indices_.value()["remainder"].start, 1)
    assert_equal(String(passthrough.transformers_()[1]), "('remainder', 'passthrough', [0, 1, 2, 4])")
    var est = ColumnTransformer(ColumnEntry("a", "drop", [0, 1]), remainder=Scale(3.0))
    assert_columns(est.fit_transform(X), [2.0, 3.0, 3.0, 3.0, 4.0, 3.0])
    assert_equal(est.named_transformers_[Scale[]]("remainder").fits_, 1)
    var weights = Dict[String, Float64]()
    weights["a"] = 0.5
    weights["remainder"] = 10.0
    weights["unknown"] = 7.0  # ignored, as in sklearn
    var weighted = ColumnTransformer(ColumnEntry("a", Scale(2.0), [1]), remainder="passthrough",
                                     transformer_weights=weights^)
    var out = weighted.fit_transform(X)
    assert_columns(out, [1.0, 1.0, 0.0, 10.0, 2.0, 10.0, 3.0, 10.0, 4.0, 10.0])
    assert_columns(weighted.transform(X), [1.0, 1.0, 0.0, 10.0, 2.0, 10.0, 3.0, 10.0, 4.0, 10.0])


def test_selection_forms() raises:
    var X = data()
    var ct = ColumnTransformer(
        ColumnEntry("neg", "passthrough", [-1, 0, 0]),
        ColumnEntry("sl", "passthrough", Slice(4, 0, -2)),
        ColumnEntry("mask", "passthrough", [False, True, False, False, True]),
    )
    assert_columns(ct.fit_transform(X), [4.0, 1.0, 0.0, 1.0, 0.0, 1.0, 4.0, 1.0, 2.0, 1.0, 1.0, 1.0, 4.0, 1.0])
    assert_equal(String(ct.transformers_()[3]), "('remainder', 'drop', [3])")

    def pick_last_two(X: NDArray[F64]) raises -> ColumnKey:
        return ColumnKey([X.shape[1] - 2, X.shape[1] - 1])

    var by_callable = ColumnTransformer(ColumnEntry("f", "passthrough", pick_last_two))
    assert_columns(by_callable.fit_transform(X), [3.0, 1.0, 4.0, 1.0])
    # transformers_ keeps the callable's result.
    assert_equal(String(by_callable.transformers_()[0]), "('f', 'passthrough', [3, 4])")
    # Only masks: the remainder is recorded as a mask.
    var masks = ColumnTransformer(ColumnEntry("m", "passthrough", [True, False, False, False, False]))
    _ = masks.fit_transform(X)
    assert_equal(String(masks.transformers_()[1]), "('remainder', 'drop', [False, True, True, True, True])")


def test_scalar_and_empty_selections() raises:
    var X = data()
    # A scalar key hands a vector to the transformer; a 1-D result is rejected.
    var scalar = ColumnTransformer(ColumnEntry("s", Scale(), 2))
    with assert_raises(contains="ValueError: The output of the 's' transformer should be 2D"):
        _ = scalar.fit_transform(X)
    var center = ColumnTransformer(ColumnEntry("c", Center(), 2))
    with assert_raises(contains="ValueError: 1D data passed to a transformer that expects 2D data."):
        _ = center.fit_transform(X)
    # Empty selections are not fitted and keep the configured transformer.
    var empty = ColumnTransformer(ColumnEntry("e", Scale(), List[Int]()), ColumnEntry("m", Scale(), [False, False, False, False, False]),
                                  ColumnEntry("a", Scale(1.0), [0]))
    var out = empty.fit_transform(X)
    assert_columns(out, [0.0, 1.0])
    var records = empty.transformers_()
    assert_equal(records[0].kind, "unfitted")
    assert_equal(records[1].kind, "unfitted")
    assert_equal(empty.named_transformers_[Scale[]]("e").fits_, 0)
    assert_equal(empty.output_indices_.value()["e"].stop, 0)
    # A slice is never empty: an empty slice reaches the transformer.
    var empty_slice = ColumnTransformer(ColumnEntry("z", Expand(), Slice(2, 2)))
    assert_equal(empty_slice.fit_transform(X).shape[1], 0)
    assert_equal(empty_slice.transformers_()[0].kind, "estimator")


def test_drop_and_no_output() raises:
    var X = data()
    var dropped = ColumnTransformer(ColumnEntry("a", "drop", [0, 1, 2, 3, 4]))
    var out = dropped.fit_transform(X)
    assert_equal(out.shape[0], 4)
    assert_equal(out.shape[1], 0)
    assert_false(Bool(dropped.output_indices_))
    assert_equal(len(dropped.transformers_()), 1)  # no remaining columns: no remainder row
    assert_equal(dropped.transform(X).shape[1], 0)
    with assert_raises(contains="TypeError: transformer 'a' is 'drop'"):
        _ = dropped.named_transformers_[NoTransformer[]]("a")
    var nothing = ColumnTransformer(remainder="passthrough")
    assert_columns(nothing.fit_transform(X), [0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0])


def test_y_and_fit_transform_dispatch() raises:
    var X = data()
    var y = zeros[F64](Shape(4))
    var ct = ColumnTransformer(ColumnEntry("c", Center(), [0, 1]))
    var out = ct.fit_transform(X, y.copy())
    assert_almost_equal(column_of(out, 0, 0), -15.0)
    var fitted = ct.named_transformers_[Center]("c")
    assert_equal(fitted.fit_transform_calls_, 1)
    assert_true(fitted.saw_y_)
    ct.fit(X)
    assert_false(ct.named_transformers_[Center]("c").saw_y_)


def test_validation_errors() raises:
    var X = data()
    var duplicate = ColumnTransformer(ColumnEntry("a", Scale(), [0]), ColumnEntry("a", Scale(), [1]))
    with assert_raises(contains="ValueError: Names provided are not unique: ['a', 'a']"):
        _ = duplicate.fit_transform(X)
    var reserved = ColumnTransformer(ColumnEntry("remainder", Scale(), [0]))
    with assert_raises(contains="ValueError: Estimator names conflict with constructor arguments: ['remainder']"):
        _ = reserved.fit_transform(X)
    var dunder = ColumnTransformer(ColumnEntry("a__b", Scale(), [0]))
    with assert_raises(contains="ValueError: Estimator names must not contain __: got ['a__b']"):
        _ = dunder.fit_transform(X)
    var bad_mode = ColumnTransformer(ColumnEntry("a", "dorp", [0]))
    with assert_raises(contains="TypeError: All estimators should implement fit and transform, or can be 'drop' or 'passthrough' specifiers. 'dorp'"):
        _ = bad_mode.fit_transform(X)
    var bad_remainder = ColumnTransformer(ColumnEntry("a", Scale(), [0]), remainder="keep")
    with assert_raises(contains="InvalidParameterError: The 'remainder' parameter of ColumnTransformer must be a str among"):
        _ = bad_remainder.fit_transform(X)
    var threshold = ColumnTransformer(ColumnEntry("a", Scale(), [0]), sparse_threshold=1.5)
    with assert_raises(contains="InvalidParameterError: The 'sparse_threshold' parameter of ColumnTransformer must be a float in the range [0.0, 1.0]."):
        _ = threshold.fit_transform(X)
    var out_of_range = ColumnTransformer(ColumnEntry("a", Scale(), [5]))
    with assert_raises(contains="ValueError: all features must be in [0, 4] or [-5, 0]"):
        _ = out_of_range.fit_transform(X)
    var short_mask = ColumnTransformer(ColumnEntry("a", Scale(), [True, False]))
    with assert_raises(contains="ValueError: all features must be in [0, 4] or [-5, 0]"):
        _ = short_mask.fit_transform(X)
    var named = ColumnTransformer(ColumnEntry("a", Scale(), "age"))
    with assert_raises(contains="ValueError: Specifying the columns using strings is only supported for dataframes."):
        _ = named.fit_transform(X)
    var full_slice = ColumnTransformer(ColumnEntry("a", Scale(), Slice(None, None, 2)))
    with assert_raises(contains="ValueError: Specifying the columns using strings is only supported for dataframes."):
        _ = full_slice.fit_transform(X)
    var selector = ColumnTransformer(ColumnEntry("a", Scale(), make_column_selector(pattern="^a")))
    with assert_raises(contains="ValueError: make_column_selector can only be applied to pandas dataframes"):
        _ = selector.fit_transform(X)
    var vector_out = ColumnTransformer(ColumnEntry("v", Expand(1), [0]))
    with assert_raises(contains="ValueError: The output of the 'v' transformer should be 2D"):
        _ = vector_out.fit_transform(X)
    var short_rows = ColumnTransformer(ColumnEntry("a", "passthrough", [0]), ColumnEntry("r", Expand(2), [1]))
    with assert_raises(contains="ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 4 and the array at index 1 has size 3"):
        _ = short_rows.fit_transform(X)
    assert_false(short_rows.__sklearn_is_fitted__())
    var fitted = ColumnTransformer(ColumnEntry("a", Scale(), [0]))
    with assert_raises(contains="NotFittedError"):
        _ = fitted.transform(X)
    _ = fitted.fit_transform(X)
    with assert_raises(contains="ValueError: X has 3 features, but ColumnTransformer is expecting 5 features as input."):
        _ = fitted.transform(zeros[F64](Shape(2, 3)))
    with assert_raises(contains="ValueError: Expected a 2D array"):
        _ = fitted.fit_transform(zeros[F64](Shape(4)))


def test_feature_names() raises:
    var X = data()
    var ct = ColumnTransformer(ColumnEntry("a", Scale(), [0, 1]), ColumnEntry("sq", Expand(), [3]),
                               remainder="passthrough")
    _ = ct.fit_transform(X)
    var names = ct.get_feature_names_out()
    var expected: List[String] = ["a__x0", "a__x1", "sq__x3", "sq__x3^2", "remainder__x2", "remainder__x4"]
    assert_equal(len(names), len(expected))
    for i in range(len(names)):
        assert_equal(names[i], expected[i])
    var given: List[String] = ["p", "q", "r", "s", "t"]
    assert_equal(ct.get_feature_names_out(given)[4], "remainder__r")
    with assert_raises(contains="ValueError: input_features should have length equal to number of features (5), got 2"):
        _ = ct.get_feature_names_out(["p", "q"])
    ct.set_params(verbose_feature_names_out=False)
    assert_equal(ct.get_feature_names_out()[2], "x3")
    ct.set_params(verbose_feature_names_out="{feature_name}@{transformer_name}")
    assert_equal(ct.get_feature_names_out()[0], "x0@a")

    def upper(transformer: String, feature: String) raises -> String:
        return transformer.upper() + ":" + feature

    ct.set_params(verbose_feature_names_out=upper)
    assert_equal(ct.get_feature_names_out()[3], "SQ:x3^2")
    var clash = ColumnTransformer(ColumnEntry("a", Scale(), [0, 1]), ColumnEntry("b", "passthrough", [1, 0]),
                                  verbose_feature_names_out=False)
    _ = clash.fit_transform(X)
    with assert_raises(contains="ValueError: Output feature names: ['x0', 'x1'] are not unique. Please set verbose_feature_names_out=True to add prefixes to feature names"):
        _ = clash.get_feature_names_out()
    var anonymous = ColumnTransformer(ColumnEntry("n", Anonymous(), [0]))
    _ = anonymous.fit_transform(X)
    with assert_raises(contains="AttributeError: Transformer n (type Anonymous) does not provide get_feature_names_out."):
        _ = anonymous.get_feature_names_out()
    assert_equal(_format_name("{{x}} {transformer_name}", "t", "f"), "{x} t")
    with assert_raises(contains="KeyError: 'other'"):
        _ = _format_name("{other}", "t", "f")


def test_make_column_transformer() raises:
    var X = data()
    var ct = make_column_transformer(ColumnEntry(Scale(2.0), [0]), ColumnEntry(Scale(3.0), [1]),
                                     ColumnEntry("passthrough", [2]), ColumnEntry(Center(), [3]), remainder="drop")
    var names = List[String]()
    ct.transformers.append_names(names)
    var expected: List[String] = ["scale-1", "scale-2", "passthrough", "center"]
    for i in range(4):
        assert_equal(names[i], expected[i])
    _ = ct.fit_transform(X)
    with assert_raises(contains="AttributeError: Transformer center (type Center)"):
        _ = ct.get_feature_names_out()
    var auto = _auto_names(["a", "b", "a", "a", "c", "b"])
    var want: List[String] = ["a-1", "b-1", "a-2", "a-3", "c", "b-2"]
    for i in range(6):
        assert_equal(auto[i], want[i])


def test_params_clone_and_refit() raises:
    var X = data()
    var ct = ColumnTransformer(ColumnEntry("a", Scale(2.0), [0]), ColumnEntry[Scale[]]("b", "drop", [1]),
                               remainder=Scale(5.0))
    _ = ct.fit_transform(X)
    var copy = clone(ct)
    assert_false(copy.__sklearn_is_fitted__())
    copy.set_transformer("b", Scale(4.0))
    copy.set_transformer_params[Scale[]]("a", FactorParams(-1.0))
    copy.set_params(remainder_params=FactorParams(0.5))
    var out = copy.fit_transform(X)
    assert_columns(out, [0.0, -1.0, 1.0, 4.0, 2.0, 0.5, 3.0, 0.5, 4.0, 0.5])
    # The original is unaffected and still fitted.
    assert_columns(ct.transform(X), [0.0, 2.0, 2.0, 5.0, 3.0, 5.0, 4.0, 5.0])
    copy.set_transformer[Scale[]]("a", "drop")
    assert_equal(copy.fit_transform(X).shape[1], 4)
    with assert_raises(contains="ValueError: Invalid parameter 'zzz' for estimator ColumnTransformer."):
        copy.set_transformer("zzz", Scale())
    var params = ct.get_params()
    params.sparse_threshold = 0.0
    var rebuilt = ColumnTransformer(params)
    assert_equal(rebuilt.sparse_threshold, 0.0)
    assert_columns(rebuilt.fit_transform(X), [0.0, 2.0, 2.0, 5.0, 3.0, 5.0, 4.0, 5.0])
    # A failed refit keeps the previous learned state.
    var broken = ct.get_params()
    broken.remainder = "keep"
    ct.set_params(broken)
    with assert_raises(contains="'remainder' parameter"):
        _ = ct.fit_transform(X)
    assert_true(ct.__sklearn_is_fitted__())
    assert_equal(ct.transform(X).shape[1], 4)
    ct.set_params(remainder=Scale(1.0))
    ct.set_params(n_jobs=4, verbose=False)
    assert_columns(ct.fit_transform(X), [0.0, 2.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0])
    with assert_raises(contains="pandas"):
        ct.set_output(transform="pandas")


def test_verbose_messages() raises:
    assert_equal(_elapsed_message("num", 1, 3, 0.04), "[ColumnTransformer] ........... (1 of 3) Processing num, total=   0.0s")
    assert_equal(_elapsed_message("a_much_longer_transformer_name_here_ok", 2, 3, 75.3),
                 "[ColumnTransformer]  (2 of 3) Processing a_much_longer_transformer_name_here_ok, total= 1.3min")
    var ct = ColumnTransformer(ColumnEntry("a", Scale(), [0]), remainder="passthrough", verbose=True)
    _ = ct.fit_transform(data())


def test_float32() raises:
    var X = data[F32]()
    var ct = ColumnTransformer(ColumnEntry[Scale[F32], F32]("a", Scale[F32](0.5), [4, 0]), remainder="passthrough")
    assert_columns(ct.fit_transform(X), [4.0, 0.5, 0.0, 0.5, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0])


def test_parallel_components() raises:
    var X = data[F64]()
    var serial = ColumnTransformer(ColumnEntry("a", Scale(2), [0, 1]), ColumnEntry("b", Center(), [3]), remainder="passthrough")
    var concurrent = ColumnTransformer(ColumnEntry("a", Scale(2), [0, 1]), ColumnEntry("b", Center(), [3]), remainder="passthrough", n_jobs=2)
    var expected = serial.fit_transform(X)
    var actual = concurrent.fit_transform(X)
    for i in range(actual.size):
        assert_equal(Float64(actual.unsafe_get(i)), Float64(expected.unsafe_get(i)))
    var transformed = concurrent.transform(X)
    for i in range(actual.size):
        assert_equal(Float64(transformed.unsafe_get(i)), Float64(expected.unsafe_get(i)))
    assert_equal(concurrent.output_indices_.value()["b"].start, serial.output_indices_.value()["b"].start)
    concurrent.set_params(n_jobs=-1)
    _ = concurrent.fit_transform(X)
    concurrent.set_params(n_jobs=0)
    with assert_raises():
        _ = concurrent.transform(X)
    concurrent.set_params(n_jobs=2)
    _ = concurrent.transform(X)
    # A worker exception must leave the previous fitted estimators usable.
    var failing = ColumnTransformer(ColumnEntry("a", Scale(2), [0]), ColumnEntry("b", Center(), [1]), n_jobs=2)
    var before = failing.fit_transform(X)
    failing.transformers.tail.head.columns = Columns[F64](1)
    with assert_raises():
        failing.fit(X)
    var after = failing.transform(X)
    for i in range(before.size):
        assert_equal(Float64(before.unsafe_get(i)), Float64(after.unsafe_get(i)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
