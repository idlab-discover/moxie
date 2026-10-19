"""Fit PLSRegression, PLSCanonical and CCA on a small two-block dataset."""
import numojo as nm
from moxie.cross_decomposition import CCA, PLSCanonical, PLSRegression


def block(rows: Int, cols: Int, values: List[Float64]) raises -> nm.NDArray[DType.float64]:
    var a = nm.zeros[DType.float64](nm.Shape(rows, cols))
    for i in range(len(values)):
        a.unsafe_set(i, values[i])
    return a^


def main() raises:
    # The first rows of scikit-learn's Linnerud dataset (exercise vs physiology).
    var X = block(6, 3, [5.0, 162, 60, 2, 110, 60, 12, 101, 101, 12, 105, 37, 13, 155, 58, 4, 101, 42])
    var y = block(6, 3, [191.0, 36, 50, 189, 37, 52, 193, 38, 58, 162, 35, 62, 189, 35, 46, 182, 36, 56])

    var pls = PLSRegression(n_components=2)
    pls.fit(X, y)
    print("PLSRegression coef_ shape:", pls.coef_.shape[0], "x", pls.coef_.shape[1])
    print("n_iter_ per component:", pls.n_iter_[0], pls.n_iter_[1])
    var prediction = pls.predict(X)
    print("first prediction:", prediction.item(0, 0), prediction.item(0, 1), prediction.item(0, 2))
    print("R^2 on training data:", pls.score(X, y))

    var canonical = PLSCanonical(n_components=2, algorithm="svd")
    var scores = canonical.fit_transform(X, y)
    print("PLSCanonical(svd) x_scores shape:", scores.x_scores.shape[0], "x", scores.x_scores.shape[1])

    var cca = CCA(n_components=1)
    cca.fit(X, y)
    var names = cca.get_feature_names_out()
    print("CCA output feature:", names[0], "diagnostics:", len(cca.diagnostics_))
