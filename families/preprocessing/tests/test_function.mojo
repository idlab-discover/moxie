"""Lifecycle contracts for FunctionTransformer and its typed callables."""
from std.math import log1p, expm1, sqrt, round
from numojo import NDArray, Shape, NDArrayStrides, zeros, full
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.preprocessing import Preprocessor, InvertiblePreprocessor, FunctionTransformer
from moxie.preprocessing._function_transformer import (
    ArrayFunction, Identity, Elementwise, ArrayFn, KwArrayFn, NamesFn, NoKwArgs,
    FeatureNamesCallable, FeatureNamesOut,
)


# ---- typed callables used by the contracts --------------------------------

def log1p64(x: Float64) -> Float64:
    return log1p(x)

def expm164(x: Float64) -> Float64:
    return expm1(x)

def sqrt64(x: Float64) -> Float64:
    return sqrt(x)

def round64(x: Float64) -> Float64:
    return round(x)

def log1p32(x: Float32) -> Float32:
    return log1p(x)

def expm132(x: Float32) -> Float32:
    return expm1(x)

def corrupt_negative(x: Float64) -> Float64:
    # Exact inverse of the identity except on negative values.
    return x + 1.0 if x < 0 else x

def negate_bool(x: Scalar[DType.bool]) -> Scalar[DType.bool]:
    return not x

def double_int(x: Int64) -> Int64:
    return 2 * x

def half_int(x: Int64) -> Int64:
    return x // 2

def to_f32(x: Float64) -> Float32:
    return Float32(x)

def to_f64(x: Float32) -> Float64:
    return Float64(x)


@fieldwise_init
struct ScaleKw(Copyable, Movable):
    var factor: Float64
    var shift: Float64


def affine(X: NDArray[DType.float64], kw: Optional[ScaleKw]) raises -> NDArray[DType.float64]:
    var factor = kw.value().factor if kw else 1.0
    var shift = kw.value().shift if kw else 0.0
    var out = zeros[DType.float64](Shape(X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            out.unsafe_set(i * X.shape[1] + j, X.item(i, j) * factor + shift)
    return out^


def drop_last(X: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
    var p = X.shape[1] - 1
    var out = zeros[DType.float64](Shape(X.shape[0], p))
    for i in range(X.shape[0]):
        for j in range(p):
            out.unsafe_set(i * p + j, X.item(i, j))
    return out^


def add_ones(X: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
    var p = X.shape[1] + 1
    var out = zeros[DType.float64](Shape(X.shape[0], p))
    for i in range(X.shape[0]):
        for j in range(p):
            out.unsafe_set(i * p + j, X.item(i, j) if j < p - 1 else 1.0)
    return out^


def suffix_log(input_features: Optional[List[String]]) raises -> List[String]:
    var names = List[String]()
    if input_features:
        for name in input_features.value():
            names.append(name + "_log")
    else:
        names.append("unfitted")
    return names^


@fieldwise_init
struct AppendNames(FeatureNamesCallable, Copyable, Movable):
    """A names callable carrying its own configuration (sklearn reads kw_args)."""
    var count: Int

    def __call__(self, input_features: Optional[List[String]]) raises -> List[String]:
        var names = input_features.value().copy()
        for k in range(self.count):
            names.append("rnd" + String(k))
        return names^


@fieldwise_init
struct AddColumns(ArrayFunction, Copyable, Movable):
    """A user functor whose configuration lives in its fields."""
    comptime InDType = DType.float64
    comptime OutDType = DType.float64
    comptime KwArgs = NoKwArgs
    var count: Int

    def __call__(self, X: NDArray[DType.float64], kw_args: Optional[NoKwArgs]) raises -> NDArray[DType.float64]:
        var p = X.shape[1] + self.count
        var out = zeros[DType.float64](Shape(X.shape[0], p))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                out.unsafe_set(i * p + j, X.item(i, j))
        return out^


def generic_transform[P: InvertiblePreprocessor](model: P, X: P.Input) raises -> P.Transformed:
    # Compile-time Pipeline-step conformance plus generic dispatch.
    return model.transform(X)


def generic_inverse[P: InvertiblePreprocessor](model: P, Xt: P.Transformed) raises -> P.Input:
    return model.inverse_transform(Xt)


def matrix(rows: Int, cols: Int, start: Float64 = 0) raises -> NDArray[DType.float64]:
    var x = zeros[DType.float64](Shape(rows, cols))
    for k in range(rows * cols):
        x.unsafe_set(k, start + Float64(k) * 0.5)
    return x^


def names(*values: String) -> List[String]:
    var out = List[String]()
    for v in values:
        out.append(v)
    return out^


# ---- contracts --------------------------------------------------------------

def test_defaults_and_identity() raises:
    var model = FunctionTransformer()
    var params = model.get_params()
    assert_false(Bool(params.func))
    assert_false(params.validate)
    assert_false(params.accept_sparse)
    assert_true(params.check_inverse)
    assert_true(params.feature_names_out.is_none())
    assert_false(Bool(params.kw_args))
    assert_true(model.__sklearn_is_fitted__())
    assert_false(Bool(model.n_features_in_))
    var X = matrix(3, 2)
    # Stateless: transform before fit; identity output is an independent copy.
    var out = model.transform(X)
    assert_equal(out.shape[0], 3)
    out.unsafe_set(0, 99.0)
    assert_equal(X.item(0, 0), 0.0)
    _ = model.fit(X)
    assert_equal(model.n_features_in_.value(), 2)
    assert_false(Bool(model.inverse_check_warning_))
    var back = model.inverse_transform(model.transform(X))
    for k in range(6):
        assert_equal(back.unsafe_get(k), X.unsafe_get(k))
    var fresh = clone(model)
    assert_false(Bool(fresh.n_features_in_))
    # validate=False: no feature-count check, NaN and rank-1 input accepted.
    _ = model.transform(matrix(2, 5))
    var nan = matrix(2, 2)
    nan.unsafe_set(1, Float64("nan"))
    _ = model.fit_transform(nan)
    var vector = zeros[DType.float64](Shape(4))
    _ = model.fit(vector)
    assert_equal(model.n_features_in_.value(), 2)  # sklearn keeps it: no second axis
    assert_equal(model.transform(vector).ndim, 1)
    # Strided negative-stride view: logical elements only.
    var reversed = X.view_with_layout(Shape(3, 2), NDArrayStrides(-2, 1), X.offset + 4)
    var rev = model.transform(reversed)
    assert_equal(rev.item(0, 0), X.item(2, 0))
    assert_equal(rev.item(2, 1), X.item(0, 1))


def test_validate() raises:
    var model = FunctionTransformer(validate=True, accept_sparse=True)
    var X = matrix(3, 2)
    var nan = X.copy()
    nan.unsafe_set(1, Float64("nan"))
    with assert_raises(contains="Input X contains NaN"):
        _ = model.fit(nan)
    var inf = X.copy()
    inf.unsafe_set(1, Float64("inf"))
    with assert_raises(contains="infinity"):
        _ = model.transform(inf)
    with assert_raises(contains="Expected 2D array"):
        _ = model.fit(zeros[DType.float64](Shape(3)))
    with assert_raises(contains="dim 3"):
        _ = model.fit(zeros[DType.float64](Shape(2, 2, 2)))
    with assert_raises(contains="0 sample(s)"):
        _ = model.fit(zeros[DType.float64](Shape(0, 2)))
    with assert_raises(contains="0 feature(s)"):
        _ = model.fit(zeros[DType.float64](Shape(2, 0)))
    # Unfitted transform skips the count check, as sklearn does.
    _ = model.transform(matrix(2, 4))
    _ = model.fit(X)
    with assert_raises(contains="is expecting 2 features"):
        _ = model.transform(matrix(2, 3))
    # inverse_transform uses a bare check_array (no estimator context, no count).
    with assert_raises(contains="Input contains NaN"):
        _ = model.inverse_transform(nan)
    _ = model.inverse_transform(matrix(2, 3))


def test_check_inverse() raises:
    var X = zeros[DType.float64](Shape(2, 2))
    for k in range(4):
        X.unsafe_set(k, Float64((k + 1) * (k + 1)))
    var bad = FunctionTransformer(Elementwise[sqrt64](), Elementwise[round64](), validate=True)
    _ = bad.fit(X)
    assert_true(Bool(bad.inverse_check_warning_))
    assert_true("not strictly inverse" in bad.inverse_check_warning_.value())
    _ = bad.set_params(check_inverse=False)
    _ = bad.fit(X)
    assert_false(Bool(bad.inverse_check_warning_))   # every fit resets it
    var good = FunctionTransformer(Elementwise[expm164](), Elementwise[log1p64](), validate=True)
    var Xt = good.fit_transform(X)
    assert_false(Bool(good.inverse_check_warning_))
    var back = good.inverse_transform(Xt)
    for k in range(4):
        # Mojo 1.0 std.math.log1p is accurate to ~3e-10 relative in Float64.
        assert_true(abs(back.unsafe_get(k) - X.unsafe_get(k)) <= 1e-9 * X.unsafe_get(k))
    # Only one callable given: no check.
    var only = FunctionTransformer(Elementwise[sqrt64](), validate=True)
    _ = only.fit(X)
    assert_false(Bool(only.inverse_check_warning_))
    # Subsample X[::max(1, n // 100)]: odd rows are negative and corrupted by
    # the inverse; n=250 samples only even rows (no warning), n=150 all rows.
    for n in [250, 150]:
        var Y = zeros[DType.float64](Shape(n, 1))
        for i in range(n):
            Y.unsafe_set(i, Float64(i + 1) if i % 2 == 0 else -Float64(i))
        var sub = FunctionTransformer(Identity[DType.float64](), Elementwise[corrupt_negative]())
        _ = sub.fit(Y)
        assert_equal(Bool(sub.inverse_check_warning_), n == 150)
    # Non-numeric (bool) input cannot be checked, as in sklearn.
    var B = zeros[DType.bool](Shape(2, 2))
    var boolean = FunctionTransformer(Elementwise[negate_bool](), Elementwise[negate_bool]())
    with assert_raises(contains="only supported when all the elements"):
        _ = boolean.fit(B)
    _ = boolean.set_params(check_inverse=False)
    _ = boolean.fit(B)
    # Integer input keeps its dtype; the check compares in Float64.
    var I = zeros[DType.int64](Shape(3, 2))
    for k in range(6):
        I.unsafe_set(k, Int64(k * 3))
    var ints = FunctionTransformer(Elementwise[double_int](), Elementwise[half_int](), validate=True)
    var It = ints.fit_transform(I)
    assert_false(Bool(ints.inverse_check_warning_))
    assert_equal(It.item(1, 1), 18)
    # Float32 round trip.
    var F = zeros[DType.float32](Shape(2, 2))
    F.unsafe_set(3, 0.25)
    var f32 = FunctionTransformer(Elementwise[log1p32](), Elementwise[expm132]())
    _ = f32.fit_transform(F)
    assert_false(Bool(f32.inverse_check_warning_))
    var Ft = generic_transform(f32, F)
    var round_trip = generic_inverse(f32, Ft)
    assert_equal(round_trip.shape[1], 2)


def test_kw_args_and_shape_changes() raises:
    var X = matrix(3, 2)
    var model = FunctionTransformer(KwArrayFn[affine](), kw_args=ScaleKw(2.0, 1.0))
    var out = model.transform(X)
    assert_equal(out.item(2, 1), X.item(2, 1) * 2.0 + 1.0)
    _ = model.set_params(kw_args=ScaleKw(3.0, 0.0))
    assert_equal(model.transform(X).item(2, 1), X.item(2, 1) * 3.0)
    var params = model.get_params()
    params.kw_args = None
    _ = model.set_params(params)
    assert_equal(model.transform(X).item(2, 1), X.item(2, 1))   # kw_args reset: defaults
    var copied = clone(model)
    assert_false(Bool(copied.kw_args))
    # Identity with keyword arguments: sklearn _identity raises TypeError.
    var ident = FunctionTransformer[KwArrayFn[affine]](kw_args=ScaleKw(2.0, 0.0))
    with assert_raises(contains="TypeError"):
        _ = ident.transform(X)
    # Column-changing pair; the inverse does not reset n_features_in_.
    var drop = FunctionTransformer(ArrayFn[add_ones](), ArrayFn[drop_last](), validate=True)
    var wide = drop.fit_transform(X)
    assert_equal(wide.shape[1], 3)
    assert_false(Bool(drop.inverse_check_warning_))
    var narrow = drop.inverse_transform(wide)
    assert_equal(narrow.shape[1], 2)
    assert_equal(drop.n_features_in_.value(), 2)
    # A functor carrying its configuration.
    var cols = FunctionTransformer(AddColumns(3))
    assert_equal(cols.transform(X).shape[1], 5)
    # Setting func back to None goes through the params record.
    var record = cols.get_params()
    record.func = None
    _ = cols.set_params(record)
    assert_equal(cols.transform(X).shape[1], 2)


def test_dtype_change() raises:
    var X = matrix(2, 2)
    var to32 = FunctionTransformer(Elementwise[to_f32]())
    var out = to32.fit_transform(X)
    assert_equal(out.item(1, 1), Float32(X.item(1, 1)))
    # The identity inverse cannot change dtype back: explicit TypeError.
    with assert_raises(contains="TypeError"):
        _ = to32.inverse_transform(out)
    var pair = FunctionTransformer(Elementwise[to_f32](), Elementwise[to_f64]())
    _ = pair.fit(X)
    assert_false(Bool(pair.inverse_check_warning_))
    assert_equal(pair.inverse_transform(out).item(1, 0), X.item(1, 0))


def test_feature_names() raises:
    var X = matrix(4, 3)
    var plain = FunctionTransformer()
    _ = plain.fit(X)
    with assert_raises(contains="AttributeError"):
        _ = plain.get_feature_names_out()
    var one = FunctionTransformer(feature_names_out="one-to-one")
    with assert_raises(contains="n_features_in_"):
        _ = one.get_feature_names_out()   # unfitted, no names (sklearn: array(None))
    assert_equal(one.get_feature_names_out(names("a", "b"))[1], "b")   # unfitted: no count check
    _ = one.fit(X)
    var generated = one.get_feature_names_out()
    assert_equal(len(generated), 3)
    assert_equal(generated[2], "x2")
    assert_equal(one.get_feature_names_out(names("a", "b", "c"))[0], "a")
    with assert_raises(contains="length equal to number of features"):
        _ = one.get_feature_names_out(names("a"))
    _ = one.fit_named(X, names("p", "q", "r"))
    assert_equal(one.get_feature_names_out()[1], "q")
    with assert_raises(contains="not equal to feature_names_in_"):
        _ = one.get_feature_names_out(names("p", "q", "s"))
    _ = one.fit(X)
    assert_false(Bool(one.feature_names_in_))
    with assert_raises(contains="does not match"):
        _ = one.fit_named(X, names("p"))
    # Callable names; N inferred from the keyword overload.
    var named = FunctionTransformer(Elementwise[log1p64](), feature_names_out=NamesFn[suffix_log]())
    assert_equal(named.get_feature_names_out()[0], "unfitted")   # receives None
    _ = named.fit_named(X, names("a", "b", "c"))
    assert_equal(named.get_feature_names_out()[2], "c_log")
    var copied = clone(named)
    assert_true(copied.feature_names_out.is_callable())
    var extra = FunctionTransformer(AddColumns(2), feature_names_out=AppendNames(2), validate=True)
    _ = extra.fit(X)
    var out_names = extra.get_feature_names_out()
    assert_equal(len(out_names), 5)
    assert_equal(out_names[4], "rnd1")
    # set_params switches modes; invalid strings fail at fit and in get_feature_names_out.
    _ = extra.set_params(feature_names_out="one-to-one")
    assert_equal(len(extra.get_feature_names_out()), 3)
    _ = extra.set_params(feature_names_out=None)
    assert_true(extra.feature_names_out.is_none())
    var invalid = FunctionTransformer(feature_names_out="one-to-many")
    with assert_raises(contains="must be a callable, a str among {'one-to-one'} or None"):
        _ = invalid.fit(X)
    with assert_raises(contains="is invalid"):
        _ = invalid.get_feature_names_out()


def test_set_output() raises:
    var model = FunctionTransformer()
    model.set_output(transform=String("default"))
    model.set_output()
    with assert_raises(contains="only 'default'"):
        model.set_output(transform=String("pandas"))


def main() raises:
    test_defaults_and_identity()
    test_validate()
    test_check_inverse()
    test_kw_args_and_shape_changes()
    test_dtype_change()
    test_feature_names()
    test_set_output()
    print("PASS function transformer lifecycle contracts")
