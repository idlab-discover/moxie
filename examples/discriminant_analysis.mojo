"""Fit LDA and QDA on a small dataset, as in the sklearn docstring examples."""
from numojo import NDArray, Shape, zeros
from moxie.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


def samples() raises -> NDArray[DType.float64]:
    var values: List[Float64] = [-1, -1, -2, -1, -3, -2, 1, 1, 2, 1, 3, 2]
    var X = zeros[DType.float64](Shape(6, 2))
    for i in range(len(values)):
        X.unsafe_set(i, values[i])
    return X^


def main() raises:
    var X = samples()
    var y: List[Int] = [1, 1, 1, 2, 2, 2]
    var query = zeros[DType.float64](Shape(1, 2))
    query.unsafe_set(0, -0.8)
    query.unsafe_set(1, -1.0)

    # Default 'svd' solver; transform projects onto n_classes - 1 = 1 axis.
    var lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    print("LDA predict([[-0.8, -1]]):", lda.predict(query)[0])
    print("LDA transform shape:", lda.transform(X).shape)

    # Shrinkage needs the 'lsqr' or 'eigen' solver.
    var shrunk = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    shrunk.fit(X, y)
    print("LDA(lsqr, auto) P(class 2):", shrunk.predict_proba(query).unsafe_get(1))

    # String labels work too; classes_ are sorted.
    var names: List[String] = ["left", "left", "left", "right", "right", "right"]
    var qda = QuadraticDiscriminantAnalysis[DType.float64, String]()
    qda.fit(X, names)
    if qda.predict(query)[0] != "left":
        raise Error("unexpected QDA prediction")
    print("QDA predict([[-0.8, -1]]):", qda.predict(query)[0])
