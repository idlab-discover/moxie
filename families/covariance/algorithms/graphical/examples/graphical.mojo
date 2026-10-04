"""Sparse precision estimation with GraphicalLasso."""
import numojo as nm
from numojo import Shape
from moxie._random import RandomState
from moxie.covariance import GraphicalLasso, graphical_lasso, empirical_covariance


def main() raises:
    var rng = RandomState(0)
    var X = nm.zeros[DType.float64](Shape(60, 4))
    for i in range(60):
        var shared = rng.standard_normal()
        X.unsafe_set(4 * i, shared + 0.3 * rng.standard_normal())
        X.unsafe_set(4 * i + 1, shared + 0.3 * rng.standard_normal())
        X.unsafe_set(4 * i + 2, rng.standard_normal())
        X.unsafe_set(4 * i + 3, rng.standard_normal())
    var model = GraphicalLasso(alpha=0.1)
    model.fit(X)
    var precision = model.get_precision()
    print("precision_[0, 1] (strong link):", precision.unsafe_get(1))
    print("precision_[2, 3] (no link):", precision.unsafe_get(11))
    print("n_iter_:", model.n_iter_())
    var result = graphical_lasso(empirical_covariance(X), 0.1)
    print("graphical_lasso precision[0, 1]:", result.precision.unsafe_get(1))
