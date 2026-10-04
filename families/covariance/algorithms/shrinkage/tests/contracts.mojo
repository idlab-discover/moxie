"""Native contracts for LedoitWolf, OAS and the shrinkage helpers."""
import numojo as nm
from numojo import NDArray, Shape
from std.math import abs, isnan
from std.testing import assert_equal, assert_true, assert_false
from moxie.base import clone
from moxie.covariance import LedoitWolf, OAS, ledoit_wolf, ledoit_wolf_shrinkage, oas, empirical_covariance
from moxie.covariance._engines import ShrinkageEngine
from moxie.covariance._shrunk_covariance_adaptive import AdaptiveShrinkageEngine


def close(a: Float64, b: Float64, tol: Float64 = 1e-5) -> Bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def has(message: String, error: Error) -> Bool:
    return String(error).find(message) >= 0


def data[dtype: DType](n: Int, p: Int) raises -> NDArray[dtype]:
    var X = nm.zeros[dtype](Shape(n, p))
    for k in range(n * p):
        X.unsafe_set(k, (Float64((k * 7) % 11) - 5.0 + 0.1 * Float64(k % 3)).cast[dtype]())
    return X^


def generic_lw[E: ShrinkageEngine](centered: NDArray[E.dtype]) raises -> Float64:
    return E.ledoit_wolf(E.statistics(centered, 2))


def check[dtype: DType]() raises:
    var X = data[dtype](9, 6)
    # Float32 blocks retain reference rounding; the mathematical formula is unchanged.
    var rounding = 3e-6 if dtype == DType.float32 else 1e-12
    var reference = ledoit_wolf_shrinkage(X)
    for block_size in [1, 2, 4, 6, 1000]:
        assert_true(close(ledoit_wolf_shrinkage(X, block_size=block_size), reference, rounding))
    try:
        _ = ledoit_wolf_shrinkage(X, block_size=0)
        assert_true(False)
    except e:
        assert_true(has("InvalidParameterError: The 'block_size' parameter", e))
    # Estimator lifecycle, shrinkage in [0, 1], covariance formula.
    var lw = LedoitWolf[dtype]()
    assert_false(lw.__sklearn_is_fitted__())
    try:
        _ = lw.shrinkage_()
        assert_true(False)
    except e:
        assert_true(has("NotFittedError: This LedoitWolf instance", e))
    lw.fit(X)
    var s = lw.shrinkage_()
    assert_true(s >= 0 and s <= 1)
    assert_true(close(s, reference, 1e-12))
    var S = empirical_covariance(X)
    var mu: Float64 = 0
    for j in range(6):
        mu += Float64(S.unsafe_get(j * 7)) / 6.0
    var C = lw.covariance_()
    assert_true(close(Float64(C.unsafe_get(1)), (1 - s) * Float64(S.unsafe_get(1)), 1e-5))
    assert_true(close(Float64(C.unsafe_get(0)), (1 - s) * Float64(S.unsafe_get(0)) + s * mu, 1e-5))
    var pair = ledoit_wolf(X)
    assert_true(close(pair[1], s, 1e-12))
    assert_equal(pair[0].shape[0], 6)
    var lazy = LedoitWolf[dtype](store_precision=False, block_size=2)
    lazy.fit(X)
    assert_false(Bool(lazy.precision_()))
    assert_true(close(Float64(lazy.get_precision().unsafe_get(0)), Float64(lw.get_precision().unsafe_get(0)), 1e-4))
    var copy = clone(lazy)
    assert_equal(copy.block_size, 2)
    assert_false(copy.__sklearn_is_fitted__())
    var bad = LedoitWolf[dtype](block_size=0)
    try:
        bad.fit(X)
        assert_true(False)
    except e:
        assert_true(has("block_size", e))
    # Single feature: shrinkage 0 and mean of squares; OAS gives 0 too.
    var one = data[dtype](5, 1)
    var single = OAS[dtype]()
    single.fit(one)
    assert_equal(single.shrinkage_(), 0.0)
    assert_equal(ledoit_wolf_shrinkage(one), 0.0)
    # Single sample: zero covariance; OAS denominator zero gives shrinkage 1.
    var row = data[dtype](1, 3)
    assert_equal(oas(row)[1], 1.0)
    assert_equal(ledoit_wolf(row)[1], 0.0)
    var model = OAS[dtype]()
    model.fit(X)
    assert_true(model.shrinkage_() >= 0 and model.shrinkage_() <= 1)
    try:
        _ = model.score(data[dtype](3, 2))
        assert_true(False)
    except e:
        assert_true(has("X has 2 features, but OAS is expecting 6", e))
    # The widened engine and public arithmetic agree within dtype rounding here.
    var centered = data[dtype](9, 6)
    var location = empirical_covariance(centered)
    _ = location
    assert_true(close(generic_lw[AdaptiveShrinkageEngine[dtype]](centered), ledoit_wolf_shrinkage(centered, assume_centered=True), rounding))
    print("PASS shrinkage contracts", dtype)


def check_overflow_refit() raises:
    var model = OAS()
    model.fit(data[DType.float64](9, 2))
    var covariance = model.covariance_()
    model.set_params(assume_centered=True)
    var huge = nm.zeros[DType.float64](Shape(9, 2))
    for k in range(18):
        huge.unsafe_set(k, 1e200)
    var failed = False
    try:
        model.fit(huge)
    except:
        failed = True
    assert_true(failed)
    assert_equal(model.location_().unsafe_get(0), 0)
    assert_true(isnan(model.shrinkage_()))
    assert_equal(model.covariance_().unsafe_get(0), covariance.unsafe_get(0))


def main() raises:
    check_overflow_refit()
    check[DType.float64]()
    check[DType.float32]()
