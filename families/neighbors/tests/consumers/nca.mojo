import numojo as nm
from moxie.neighbors import NeighborhoodComponentsAnalysis
from moxie.base import clone
from std.testing import assert_equal, assert_true


def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(4, 2))
    for i in range(x.size):
        x.unsafe_set(i, Float64(i))
    var y = nm.zeros[DType.float64](nm.Shape(4))
    y.unsafe_set(2, 1)
    y.unsafe_set(3, 1)
    var estimator = NeighborhoodComponentsAnalysis(
        1, init="pca", random_state=42
    )
    var result = estimator.fit_transform(x, y)
    assert_equal(result.shape[0], 4)
    assert_equal(result.shape[1], 1)
    assert_equal(len(estimator.get_feature_names_out()), 1)
    assert_true(not clone(estimator).__sklearn_is_fitted__())
    print("PASS: independently assembled NCA consumer")
