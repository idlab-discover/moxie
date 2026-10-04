"""Fit mixed precision targets and predict from integer feature inputs."""
import numojo as nm
from moxie.kernel_ridge import KernelRidge


def main() raises:
    var X = nm.arange[DType.float32](4).reshape(nm.Shape(4, 1))
    var y = nm.arange[DType.float64](4)
    var model = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5)
    _ = model.fit(X, y)
    var query = nm.arange[DType.int64](4).reshape(nm.Shape(4, 1))
    var prediction = model.predict(query)
    print("Predictions:", prediction[nm.NDArray[DType.float64]])
    print("Training R2:", model.score(X, y))
