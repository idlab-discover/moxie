"""Project paired blocks with PLSSVD."""
import numojo as nm
from moxie.cross_decomposition import PLSSVD


def block(rows: Int, cols: Int, values: List[Float64]) raises -> nm.NDArray[DType.float64]:
    var a = nm.zeros[DType.float64](nm.Shape(rows, cols))
    for i in range(len(values)):
        a.unsafe_set(i, values[i])
    return a^


def main() raises:
    # The scikit-learn docstring example.
    var X = block(4, 3, [0.0, 0, 1, 1, 0, 0, 2, 2, 2, 2, 5, 4])
    var y = block(4, 2, [0.1, -0.2, 0.9, 1.1, 6.2, 5.9, 11.9, 12.3])
    var pls = PLSSVD(n_components=2)
    pls.fit(X, y)
    var scores = pls.transform(X, y)
    print("X_c shape:", scores.x_scores.shape[0], "x", scores.x_scores.shape[1])
    print("y_c shape:", scores.y_scores.shape[0], "x", scores.y_scores.shape[1])
    print("first x weight column:", pls.x_weights_.item(0, 0), pls.x_weights_.item(1, 0), pls.x_weights_.item(2, 0))
