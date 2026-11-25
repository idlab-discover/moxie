"""Consume numerical oracle fixtures; all computations under test are native."""
from std.sys import argv
from moxie._random import RandomState
from std.math import isfinite
from std.testing import assert_equal, assert_raises
from numojo import NDArray, Shape, zeros
from moxie.base import clone
from moxie.random_projection import (
    ProjectionMatrix,
    GaussianRandomProjection,
    SparseRandomProjection,
    ComponentCount,
    ProjectionDensity,
    johnson_lindenstrauss_min_dim,
    _gaussian_random_matrix,
    _sparse_random_matrix,
)
from moxie._projection.linalg import pseudoinverse
from moxie.random_projection_serialization import (
    GaussianRandomProjectionSchema,
    SparseRandomProjectionSchema,
)
from moxie.serialization import ModelSchema, dumps, loads
from moxie.warnings import FitWarning


def vector[dtype: DType](text: String) raises -> List[Scalar[dtype]]:
    var result = List[Scalar[dtype]]()
    if text == "-":
        return result^
    for item in text.split(","):
        result.append(Scalar[dtype](Float64(String(item))))
    return result^


def compare[
    dtype: DType
](
    actual: ProjectionMatrix[dtype],
    expected: String,
    rows: Int,
    cols: Int,
    tolerance: Float64,
) raises:
    assert_equal(actual.rows, rows)
    assert_equal(actual.cols, cols)
    var dense = actual.to_dense()
    var values = vector[DType.float64](expected)
    assert_equal(len(dense.data), len(values))
    for i in range(len(values)):
        var value = Float64(dense.data[i])
        if (
            not isfinite(value)
            or not isfinite(values[i])
            or abs(value - values[i]) > tolerance * (1.0 + abs(values[i]))
        ):
            raise Error(
                "numerical mismatch at "
                + String(i)
                + ": "
                + String(value)
                + " != "
                + String(values[i])
            )


def matrix_case[dtype: DType](fields: List[String]) raises:
    var rows = Int(fields[2])
    var cols = Int(fields[3])
    var seed = Int(fields[4])
    var actual: ProjectionMatrix[dtype]
    if fields[0] == "gaussian":
        actual = _gaussian_random_matrix[dtype](rows, cols, seed)
    else:
        actual = _sparse_random_matrix[dtype](
            rows, cols, Float64(fields[5]), seed
        )
    var tolerance = 2e-6 if dtype == DType.float32 else 3e-13
    compare(actual, fields[6], rows, cols, tolerance)
    assert_equal(actual.is_sparse, fields[7] == "true")
    if actual.is_sparse and dtype == DType.float64:
        var indices = vector[DType.float64](fields[8])
        var indptr = vector[DType.float64](fields[9])
        assert_equal(len(actual.indices), len(indices))
        assert_equal(len(actual.indptr), len(indptr))
        for i in range(len(indices)):
            assert_equal(actual.indices[i], Int(indices[i]))
        for i in range(len(indptr)):
            assert_equal(actual.indptr[i], Int(indptr[i]))


def as_array[dtype: DType](matrix: ProjectionMatrix[dtype]) raises -> NDArray[dtype]:
    var result = zeros[dtype](Shape(matrix.rows, matrix.cols))
    for i in range(len(matrix.data)):
        result.unsafe_set(i, matrix.data[i])
    return result^


def same_dense[dtype: DType](
    actual: NDArray[dtype], matrix: ProjectionMatrix[dtype]
) raises:
    assert_equal(actual.ndim, 2)
    assert_equal(actual.shape[0], matrix.rows)
    assert_equal(actual.shape[1], matrix.cols)
    for i in range(actual.size):
        assert_equal(actual.unsafe_get(i), matrix.data[i])


def round_trip[Schema: ModelSchema](model: Schema.Model) raises -> Schema.Model:
    var bytes = dumps(Schema.encode(model))
    return Schema.decode(loads(bytes))


def check_warnings(warnings: List[FitWarning], fields: List[String]) raises:
    if fields[19] == "-":
        assert_equal(len(warnings), 0)
        return
    assert_equal(len(warnings), 1)
    assert_equal(warnings[0].category, fields[19])
    assert_equal(warnings[0].message, fields[20])


def model_case[dtype: DType](fields: List[String]) raises:
    var rows = Int(fields[3])
    var cols = Int(fields[4])
    var count: ComponentCount = String("auto")
    if fields[5] != "auto":
        count = Int(fields[5])
    var eps = Float64(fields[6])
    var density: ProjectionDensity = String("auto")
    if fields[7] != "auto":
        density = Float64(fields[7])
    var dense_output = fields[8] == "true"
    var cached = fields[9] == "true"
    var seed = Int(fields[10])
    var X = ProjectionMatrix[dtype](rows, cols, vector[dtype](fields[12]))
    if fields[11] == "true":
        X = X.to_csr()
    var expected_count = Int(fields[13])
    var tolerance = 3e-5 if dtype == DType.float32 else 3e-10
    if fields[2] == "gaussian":
        var model = GaussianRandomProjection[dtype](
            count^,
            eps=eps,
            compute_inverse_components=cached,
            random_state=seed,
        )
        _ = model.fit(X)
        check_warnings(model.fit_warnings, fields)
        model = round_trip[GaussianRandomProjectionSchema[dtype]](model)
        assert_equal(len(model.fit_warnings), 0)
        var transformed = model.transform(X)
        compare(model.components_, fields[14], expected_count, cols, tolerance)
        compare(transformed, fields[15], rows, expected_count, tolerance)
        var reconstructed = model.inverse_transform(transformed)
        compare(
            reconstructed,
            fields[16],
            rows,
            cols,
            tolerance,
        )
        assert_equal(transformed.is_sparse, fields[17] == "true")
        assert_equal(model.n_components_, expected_count)
        assert_equal(model.n_features_in_, cols)
        assert_equal(Bool(model.inverse_components_), cached)
        if not X.is_sparse:
            var dense = clone(model)
            var dense_transformed = dense.fit_transform(as_array(X))
            check_warnings(dense.fit_warnings, fields)
            same_dense(dense_transformed, transformed)
            same_dense(dense.inverse_transform(dense_transformed), reconstructed)
        if cached:
            compare(
                model.inverse_components_.value(),
                fields[18],
                cols,
                expected_count,
                tolerance,
            )
    else:
        var model = SparseRandomProjection[dtype](
            count^,
            density=density^,
            eps=eps,
            dense_output=dense_output,
            compute_inverse_components=cached,
            random_state=seed,
        )
        _ = model.fit(X)
        check_warnings(model.fit_warnings, fields)
        if not model.components_.is_sparse:
            model = round_trip[SparseRandomProjectionSchema[dtype]](model)
            assert_equal(len(model.fit_warnings), 0)
        var transformed = model.transform(X)
        compare(model.components_, fields[14], expected_count, cols, tolerance)
        compare(transformed, fields[15], rows, expected_count, tolerance)
        var reconstructed = model.inverse_transform(transformed)
        compare(
            reconstructed,
            fields[16],
            rows,
            cols,
            tolerance,
        )
        assert_equal(transformed.is_sparse, fields[17] == "true")
        assert_equal(model.n_components_, expected_count)
        assert_equal(model.n_features_in_, cols)
        assert_equal(Bool(model.inverse_components_), cached)
        if not X.is_sparse:
            var dense = clone(model)
            var dense_transformed = dense.fit_transform(as_array(X))
            check_warnings(dense.fit_warnings, fields)
            same_dense(dense_transformed, transformed)
            same_dense(dense.inverse_transform(dense_transformed), reconstructed)
        if cached:
            compare(
                model.inverse_components_.value(),
                fields[18],
                cols,
                expected_count,
                tolerance,
            )


def inverse_case[dtype: DType](fields: List[String]) raises:
    var rows = Int(fields[2])
    var cols = Int(fields[3])
    var matrix = ProjectionMatrix[dtype](rows, cols, vector[dtype](fields[4]))
    var actual = pseudoinverse(matrix)
    compare(actual, fields[5], cols, rows, Float64(fields[6]))


def stream_case[dtype: DType](fields: List[String]) raises:
    var rng = RandomState(42)
    var X = zeros[dtype](Shape(2, 101))
    var g = GaussianRandomProjection[dtype](3)
    var s = SparseRandomProjection[dtype](3, density=Float64(fields[2]))
    var tolerance = 2e-6 if dtype == DType.float32 else 3e-13
    _ = g.fit(X, rng=rng)
    compare(g.components_, fields[3], 3, 101, tolerance)
    _ = s.fit(X, rng=rng)
    compare(s.components_, fields[4], 3, 101, tolerance)
    assert_equal(s.components_.is_sparse, Float64(fields[2]) != 1)
    _ = g.fit(X, rng=rng)
    compare(g.components_, fields[5], 3, 101, tolerance)
    assert_equal(rng.next_uint32(), UInt32(Int(fields[6])))


def dispatch[dtype: DType](fields: List[String]) raises:
    if fields[0] == "stream":
        stream_case[dtype](fields)
    elif fields[0] == "model":
        model_case[dtype](fields)
    elif fields[0] == "inverse":
        inverse_case[dtype](fields)
    else:
        matrix_case[dtype](fields)


def main() raises:
    # Ensure the comparator rejects nonfinite and wrong-length expectations.
    var unit = ProjectionMatrix(1, 1, List[Float64]([1.0]))
    with assert_raises():
        compare(unit, "nan", 1, 1, 1e-10)
    with assert_raises():
        compare(unit, "1,1", 1, 1, 1e-10)
    var args = argv()
    var text: String
    with open(args[1], "r") as source:
        text = source.read()
    var count = 0
    for line in text.split("\n"):
        if line == "":
            continue
        var fields = List[String]()
        for item in line.split("|"):
            fields.append(String(item))
        try:
            if fields[0] == "jl":
                assert_equal(
                    johnson_lindenstrauss_min_dim(
                        Float64(fields[1]), eps=Float64(fields[2])
                    ),
                    Int(fields[3]),
                )
            elif fields[1] == "32":
                dispatch[DType.float32](fields)
            else:
                dispatch[DType.float64](fields)
        except error:
            raise Error(
                "fixture "
                + String(count)
                + " ("
                + fields[0]
                + "): "
                + String(error)
            )
        count += 1
    print("Passed reference fixtures:", count)
