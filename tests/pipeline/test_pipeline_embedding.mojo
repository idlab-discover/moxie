"""Training-only transformation through a dense manifold final step."""
from moxie.base import FitTransformer, InverseTransformer, TransformerMixin, clone
from moxie.manifold import ClassicalMDS
from moxie.pipeline import TransformingPipeline, make_pipeline
from moxie.preprocessing import StandardScaler
from numojo import NDArray, Shape, zeros
from parity_io import ResultLine
from std.testing import assert_equal, assert_false, assert_raises, assert_true


def fit_embedding[E: FitTransformer](
    mut pipe: E, X: E.Input, y: Optional[E.Target],
) raises -> E.Transformed:
    return pipe.fit_transform(X, y)


def check_embedding[dtype: DType]() raises:
    var X = zeros[dtype](Shape(5, 3))
    var values: List[Int] = [0, 3, 1, 1, 5, 5, 2, 4, 2, 3, 0, 6, 5, 1, 4]
    for i in range(X.size):
        X.unsafe_set(i, Scalar[dtype](values[i]))

    var pipe = make_pipeline(StandardScaler[dtype](), ClassicalMDS[dtype]())
    comptime Pipe = type_of(pipe)
    comptime assert conforms_to(Pipe, FitTransformer)
    comptime assert not conforms_to(Pipe, TransformerMixin)
    comptime assert not conforms_to(Pipe, TransformingPipeline)
    comptime assert not conforms_to(Pipe, InverseTransformer)

    assert_false(pipe.__sklearn_is_fitted__())
    with assert_raises():
        _ = pipe.fit_transform(zeros[dtype](Shape(0, 3)))

    var fitted: NDArray[DType.float64] = fit_embedding(pipe, X, None)
    assert_true(pipe.__sklearn_is_fitted__())
    assert_equal(fitted.shape[0], 5)
    assert_equal(fitted.shape[1], 2)
    var result = ResultLine(String(dtype))
    result.add("fitted", fitted)

    var fresh = clone(pipe)
    assert_false(fresh.__sklearn_is_fitted__())
    _ = fresh.fit(X)
    result.add("fit", fresh.estimator.embedding_.value())

    var params = pipe.get_params()
    params.estimator.n_components = 1
    _ = pipe.set_params(params)
    var target = Optional[NDArray[dtype]](zeros[dtype](Shape(5)))
    result.add("refit", pipe.fit_transform(X, target))
    assert_equal(fresh.estimator.embedding_.value().shape[1], 2)
    var nested = make_pipeline(StandardScaler[dtype](), clone(fresh))
    result.add("nested", nested.fit_transform(X))
    result.emit()


def main() raises:
    check_embedding[DType.float32]()
    check_embedding[DType.float64]()
