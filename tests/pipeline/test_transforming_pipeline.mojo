"""Inverse transforms on transformer-only pipelines."""
from moxie.base import InverseTransformer, clone
from moxie.impute import SimpleImputer
from moxie.pipeline import TransformerPipeline, make_pipeline
from moxie.preprocessing import MinMaxScaler, StandardScaler
from numojo import NDArray, Shape, zeros
from parity_io import ResultLine
from std.testing import assert_raises


def inverse[T: InverseTransformer](pipe: T, X: T.Transformed) raises -> T.Input:
    return pipe.inverse_transform(X)


def check_scalers[dtype: DType]() raises:
    var X = zeros[dtype](Shape(3, 2))
    var values: List[Int] = [2, 3, 4, 9, 6, 3]
    for i in range(X.size):
        X.unsafe_set(i, Scalar[dtype](values[i]))
    var pipe = make_pipeline(StandardScaler[dtype](), MinMaxScaler[dtype]())
    with assert_raises(contains="NotFittedError"):
        _ = pipe.inverse_transform(X)

    var line = ResultLine(String(dtype))
    var transformed = pipe.fit_transform(X)
    var restored: NDArray[dtype] = inverse(pipe, transformed)
    line.add("restored", restored)
    line.add("new_values", inverse(pipe, zeros[dtype](Shape(2, 2))))
    with assert_raises():
        _ = pipe.inverse_transform(zeros[dtype](Shape(2, 1)))

    var fresh = clone(pipe)
    with assert_raises(contains="NotFittedError"):
        _ = fresh.inverse_transform(transformed)
    var params = fresh.get_params()
    params.estimator.feature_range = (-1.0, 1.0)
    _ = fresh.set_params(params).fit(X)
    line.add("refit", fresh.inverse_transform(transformed))
    line.add("original", pipe.inverse_transform(transformed))
    line.emit()


def main() raises:
    comptime assert not conforms_to(
        TransformerPipeline[SimpleImputer[], StandardScaler[]], InverseTransformer
    )
    comptime assert not conforms_to(
        TransformerPipeline[StandardScaler[], SimpleImputer[]], InverseTransformer
    )
    check_scalers[DType.float32]()
    check_scalers[DType.float64]()
