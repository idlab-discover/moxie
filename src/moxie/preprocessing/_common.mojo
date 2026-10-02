# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Family helpers shared by the public dense preprocessing estimators.

Borrowed rank-2 NuMojo arrays use element strides and offsets; `at` and `put`
honor them. Outputs are compact row-major arrays owned by the caller.
"""
from numojo import NDArray, Shape, zeros
from std.math import isnan, isfinite
from moxie.preprocessing._validation import validate_dense


comptime _FLOAT_ONLY = "preprocessing estimators accept Float32 or Float64 arrays"


@always_inline
def at[d: DType](x: NDArray[d], i: Int, j: Int) -> Scalar[d]:
    """Strided rank-2 read; callers validate rank and bounds."""
    return x.unsafe_get(i * Int(x.strides.unsafe_load(0)) + j * Int(x.strides.unsafe_load(1)))


@always_inline
def at1[d: DType](x: NDArray[d], i: Int) -> Scalar[d]:
    """Strided rank-1 read; callers validate rank and bounds."""
    return x.unsafe_get(i * Int(x.strides.unsafe_load(0)))


@always_inline
def put[d: DType](mut x: NDArray[d], i: Int, j: Int, value: Scalar[d]):
    """Strided rank-2 write; callers validate rank, bounds and writability."""
    x.unsafe_set(i * Int(x.strides.unsafe_load(0)) + j * Int(x.strides.unsafe_load(1)), value)


def matrix[d: DType](rows: Int, cols: Int) raises -> NDArray[d]:
    if rows < 0 or cols < 0:
        raise Error("ValueError: negative output dimensions")
    return zeros[d](Shape(rows, cols))


def vector[d: DType](size: Int) raises -> NDArray[d]:
    if size < 0:
        raise Error("ValueError: negative vector length")
    return zeros[d](Shape(size))


def check_matrix[d: DType](
    x: NDArray[d], *, allow_nan: Bool, min_samples: Int = 1, min_features: Int = 1,
) raises:
    """sklearn check_array for dense Float32/64 input: rank 2, finite (or NaN)."""
    comptime assert d == DType.float32 or d == DType.float64, _FLOAT_ONLY
    if x.ndim != 2:
        raise Error("ValueError: Expected 2D array, got " + String(x.ndim) + "D array instead")
    if x.shape[0] < min_samples:
        raise Error("ValueError: Found array with " + String(x.shape[0]) + " sample(s) (shape=(" + String(x.shape[0]) + ", " + String(x.shape[1]) + ")) while a minimum of " + String(min_samples) + " is required.")
    if x.shape[1] < min_features:
        raise Error("ValueError: Found array with " + String(x.shape[1]) + " feature(s) (shape=(" + String(x.shape[0]) + ", " + String(x.shape[1]) + ")) while a minimum of " + String(min_features) + " is required.")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            var value = at(x, i, j)
            if not isfinite(value):
                if isnan(value):
                    if not allow_nan:
                        raise Error("ValueError: Input X contains NaN.")
                else:
                    raise Error("ValueError: Input X contains infinity or a value too large for dtype('" + dtype_name[d]() + "').")


def dtype_name[d: DType]() -> String:
    comptime if d == DType.float32:
        return "float32"
    elif d == DType.float64:
        return "float64"
    elif d == DType.int64:
        return "int64"
    elif d == DType.bool:
        return "bool"
    else:
        return String(d)


def check_fitted(fitted: Bool, name: String) raises:
    if not fitted:
        raise Error("NotFittedError: This " + name + " instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")


def check_n_features(name: String, expected: Int, actual: Int) raises:
    if expected != actual:
        raise Error("ValueError: X has " + String(actual) + " features, but " + name + " is expecting " + String(expected) + " features as input.")


def check_set_output(transform: Optional[String]) raises:
    """Only the default NumPy-like container exists natively."""
    if transform:
        var value = transform.value()
        if value != "default":
            raise Error("ValueError: set_output(transform='" + value + "') is unsupported; only 'default' dense NuMojo output is available")


def check_names_length(names: List[String], n_features: Int) raises:
    if len(names) != n_features:
        raise Error("ValueError: feature_names length " + String(len(names)) + " does not match n_features " + String(n_features))


def feature_names_in(
    n_features_in: Int, feature_names_in_: Optional[List[String]],
    input_features: Optional[List[String]], *, generate_names: Bool = True,
) raises -> List[String]:
    """sklearn `_check_feature_names_in`: validate or generate input names."""
    if input_features:
        var supplied = input_features.value().copy()
        if feature_names_in_:
            if supplied != feature_names_in_.value():
                raise Error("ValueError: input_features is not equal to feature_names_in_")
        if len(supplied) != n_features_in:
            raise Error("ValueError: input_features should have length equal to number of features (" + String(n_features_in) + "), got " + String(len(supplied)))
        return supplied^
    if feature_names_in_:
        return feature_names_in_.value().copy()
    var generated = List[String]()
    if generate_names:
        for i in range(n_features_in):
            generated.append("x" + String(i))
    return generated^


def prefixed_feature_names(
    prefix: String, n_features_out: Int, n_features_in: Int,
    feature_names_in_: Optional[List[String]], input_features: Optional[List[String]],
) raises -> List[String]:
    """sklearn ClassNamePrefixFeaturesOutMixin: `<lowercase class name><i>`."""
    _ = feature_names_in(n_features_in, feature_names_in_, input_features, generate_names=False)
    var names = List[String]()
    for i in range(n_features_out):
        names.append(prefix + String(i))
    return names^


def copy_matrix[d: DType](x: NDArray[d]) raises -> NDArray[d]:
    """Compact row-major copy of a possibly strided matrix (not a backing copy)."""
    var result = matrix[d](x.shape[0], x.shape[1])
    var p = x.shape[1]
    for i in range(x.shape[0]):
        for j in range(p):
            result.unsafe_set(i * p + j, at(x, i, j))
    return result^


def copy_vector[d: DType](x: NDArray[d]) raises -> NDArray[d]:
    """Compact copy of a possibly strided rank-1 array."""
    if x.ndim != 1:
        raise Error("ValueError: expected a rank-1 array")
    var result = vector[d](x.shape[0])
    for i in range(x.shape[0]):
        result.unsafe_set(i, at1(x, i))
    return result^


def check_writable[d: DType](x: NDArray[d]) raises:
    if not x.flags.WRITEABLE:
        raise Error("ValueError: in-place operation requires a writable array; no fallback copy is made")
