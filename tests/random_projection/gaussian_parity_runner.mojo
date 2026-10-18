"""Consume numerical oracle fixtures; all computations under test are native."""
from std.sys import argv
from moxie._random import RandomState
from std.math import isfinite
from std.testing import assert_equal, assert_raises
from moxie.random_projection import (
    ProjectionMatrix,
    GaussianRandomProjection,
    ComponentCount,
    johnson_lindenstrauss_min_dim,
    _gaussian_random_matrix,
)
from moxie._projection.linalg import pseudoinverse


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
    actual = _gaussian_random_matrix[dtype](rows, cols, seed)
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


def model_case[dtype: DType](fields: List[String]) raises:
    var rows = Int(fields[3])
    var cols = Int(fields[4])
    var count: ComponentCount = String("auto")
    if fields[5] != "auto":
        count = Int(fields[5])
    var eps = Float64(fields[6])
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
        if len(fields) > 19:
            var rng = RandomState(seed)
            for _ in range(Int(fields[19])):
                _ = rng.standard_normal()
            var params = model.get_params()
            params.random_state = None
            _ = model.set_params(params)
            _ = model.fit(X, rng=rng)
        else:
            _ = model.fit(X)
        var transformed = model.transform(X)
        compare(model.components_, fields[14], expected_count, cols, tolerance)
        compare(transformed, fields[15], rows, expected_count, tolerance)
        compare(
            model.inverse_transform(transformed),
            fields[16],
            rows,
            cols,
            tolerance,
        )
        assert_equal(transformed.is_sparse, fields[17] == "true")
        assert_equal(model.n_components_, expected_count)
        assert_equal(model.n_features_in_, cols)
        assert_equal(Bool(model.inverse_components_), cached)
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
    # Keep huge-scale inverses meaningful even below the absolute tolerance.
    var expected = vector[DType.float64](fields[5])
    for i in range(len(expected)):
        if expected[i] != 0 and abs(expected[i]) < 1e-100:
            assert_equal(abs(Float64(actual.data[i]) / expected[i] - 1) < 3e-10, True)


def dispatch[dtype: DType](fields: List[String]) raises:
    if fields[0] == "model":
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
