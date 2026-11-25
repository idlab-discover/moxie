from std.math import inf
from std.testing import assert_equal, assert_raises, assert_true
from numojo import NDArray, Shape, zeros
from moxie.base import clone
from moxie.random_projection import (
    ComponentCount,
    GaussianRandomProjection,
    ProjectionDensity,
    SparseRandomProjection,
)
from moxie.random_projection_serialization import (
    GaussianRandomProjectionSchema,
    SparseRandomProjectionSchema,
)
from moxie.serialization import dumps, load_model, save_model


def inputs[dtype: DType]() raises -> NDArray[dtype]:
    var x = zeros[dtype](Shape(2, 4))
    for i in range(x.size):
        x.unsafe_set(i, Scalar[dtype](i - 3))
    return x^


def same_array[dtype: DType](actual: NDArray[dtype], expected: NDArray[dtype]) raises:
    assert_equal(actual.ndim, 2)
    assert_equal(actual.shape[0], expected.shape[0])
    assert_equal(actual.shape[1], expected.shape[1])
    for i in range(expected.size):
        assert_equal(actual.unsafe_get(i), expected.unsafe_get(i))


def gaussian_round_trips[dtype: DType]() raises:
    comptime Schema = GaussianRandomProjectionSchema[dtype]
    var path = "builds/projection-gaussian-" + String(dtype) + ".moxie"
    var x = inputs[dtype]()
    var model = GaussianRandomProjection[dtype](
        eps=0.5, compute_inverse_components=True, random_state=0,
    )
    save_model[Schema](model, path)
    var fresh = load_model[Schema](path)
    assert_equal(fresh.n_components[String], "auto")
    assert_equal(fresh.eps, 0.5)
    assert_equal(fresh.random_state.or_else(-1), 0)
    assert_true(fresh.compute_inverse_components)
    assert_true(not fresh.__sklearn_is_fitted__())
    with assert_raises():
        _ = fresh.transform(x)

    _ = model.set_params(n_components=ComponentCount(Int(2))).fit(x)
    var projected = model.transform(x)
    var reconstructed = model.inverse_transform(projected)
    # A load cannot refit with this pending configuration.
    _ = model.set_params(
        n_components=ComponentCount(Int(-1)), compute_inverse_components=False,
    )
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_true(restored.__sklearn_is_fitted__())
    assert_equal(restored.n_components[Int], -1)
    assert_equal(restored.n_components_, 2)
    assert_equal(restored.get_n_features_in(), 4)
    assert_equal(restored.get_feature_names_out(), model.get_feature_names_out())
    assert_true(not restored.compute_inverse_components)
    assert_true(Bool(restored.inverse_components_))
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
    _ = restored.set_params(compute_inverse_components=True)
    same_array(restored.transform(x), projected)
    same_array(restored.inverse_transform(projected), reconstructed)
    assert_true(not clone(restored).__sklearn_is_fitted__())
    restored.components_.data[0] = 99
    same_array(model.transform(x), projected)

    _ = model.set_params(n_components=ComponentCount(Int(2))).fit(x)
    save_model[Schema](model, path)
    var uncached = load_model[Schema](path)
    assert_true(not uncached.inverse_components_)
    same_array(uncached.inverse_transform(projected), reconstructed)
    _ = uncached.set_params(compute_inverse_components=True)
    with assert_raises():
        _ = uncached.inverse_transform(projected)


def sparse_round_trips[dtype: DType]() raises:
    comptime Schema = SparseRandomProjectionSchema[dtype]
    var path = "builds/projection-sparse-" + String(dtype) + ".moxie"
    var x = inputs[dtype]()
    var model = SparseRandomProjection[dtype](2, eps=0.3, dense_output=True)
    save_model[Schema](model, path)
    var fresh = load_model[Schema](path)
    assert_equal(fresh.n_components[Int], 2)
    assert_equal(fresh.density[String], "auto")
    assert_equal(fresh.eps, 0.3)
    assert_true(fresh.dense_output)
    assert_true(not fresh.random_state)
    assert_equal(fresh.density_, 0.0)
    assert_true(not fresh.__sklearn_is_fitted__())
    with assert_raises():
        _ = fresh.transform(x)

    for cached in [False, True]:
        _ = model.set_params(
            n_components=ComponentCount(Int(2)),
            density=ProjectionDensity(Float64(1.0)),
            compute_inverse_components=cached,
            random_state=17,
        ).fit(x)
        _ = model.set_params(
            n_components=ComponentCount(Int(3)),
            density=ProjectionDensity(Float64(0.25)),
            dense_output=False,
        )
        save_model[Schema](model, path)
        var restored = load_model[Schema](path)
        assert_true(restored.__sklearn_is_fitted__())
        assert_equal(restored.n_components[Int], 3)
        assert_equal(restored.n_components_, 2)
        assert_equal(restored.get_n_features_in(), 4)
        assert_equal(restored.density[Float64], 0.25)
        assert_equal(restored.density_, 1.0)
        assert_true(not restored.dense_output)
        assert_true(not restored.components_.is_sparse)
        assert_equal(Bool(restored.inverse_components_), cached)
        assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
        var projected = model.transform(x)
        same_array(restored.transform(x), projected)
        same_array(
            restored.inverse_transform(projected), model.inverse_transform(projected)
        )
        assert_true(not clone(restored).__sklearn_is_fitted__())
        _ = restored.set_params(dense_output=True).fit(x)
        assert_equal(restored.density_, 0.25)
        assert_true(restored.components_.is_sparse)
        with assert_raises(contains="CSR projection state is unsupported"):
            save_model[Schema](restored, path)
        var previous = load_model[Schema](path)
        same_array(previous.transform(x), projected)


def invalid_archives() raises:
    comptime Schema = GaussianRandomProjectionSchema
    var model = GaussianRandomProjection(
        2, compute_inverse_components=True, random_state=42,
    )
    _ = model.fit(inputs[DType.float64]())
    var archive = Schema.encode(model)
    with assert_raises():
        _ = SparseRandomProjectionSchema.decode(archive)
    with assert_raises(contains="dtype mismatch"):
        _ = GaussianRandomProjectionSchema[DType.float32].decode(archive)
    archive.schema_version += 1
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = Schema.decode(archive)
    archive.schema_version -= 1

    archive.payloads[0].shape = [4, 2]
    with assert_raises(contains="invalid matrix shape for components_"):
        _ = Schema.decode(archive)
    archive.payloads[0].shape = [2, 4]
    archive.payloads[1].shape = [2, 4]
    with assert_raises(contains="invalid matrix shape for inverse_components_"):
        _ = Schema.decode(archive)
    archive.payloads[1].shape = [4, 2]
    archive.payloads[1].name = "unknown"
    with assert_raises(contains="missing matrix inverse_components_"):
        _ = Schema.decode(archive)
    _ = archive.payloads.pop()
    with assert_raises(contains="unexpected projection arrays"):
        _ = Schema.decode(archive)

    archive.add_array[DType.float64](
        "inverse_components_",
        List[Int]([4, 2]),
        List[Float64](length=8, fill=inf[DType.float64]()),
    )
    with assert_raises(contains="contains NaN or infinity"):
        _ = Schema.decode(archive)
    archive.manifest.append(0)
    with assert_raises():
        _ = Schema.decode(archive)

    var fresh = Schema.encode(GaussianRandomProjection())
    with assert_raises(contains="dtype mismatch"):
        _ = GaussianRandomProjectionSchema[DType.float32].decode(fresh)
    fresh.payloads.append(archive.payloads[0].copy())
    with assert_raises(contains="unexpected projection arrays"):
        _ = Schema.decode(fresh)


def main() raises:
    gaussian_round_trips[DType.float32]()
    gaussian_round_trips[DType.float64]()
    sparse_round_trips[DType.float32]()
    sparse_round_trips[DType.float64]()
    invalid_archives()
    print("PASS: five projection persistence contracts")
