"""Approximate a degree-2 polynomial kernel with a seeded Tensor Sketch."""
from numojo import NDArray, Shape
from moxie.kernel_approximation import PolynomialCountSketch


def main() raises:
    var x = NDArray[DType.float32](Shape(4, 2))
    var values: List[Float32] = [0, 0, 1, 1, 1, 0, 0, 1]
    for k in range(len(values)):
        x.unsafe_set(k, values[k])
    var sketch = PolynomialCountSketch[DType.float32](degree=3, coef0=1.0, n_components=64, random_state=1)
    var features = sketch.fit_transform(x)  # Float64, like sklearn
    print("PolynomialCountSketch output:", features.shape[0], features.shape[1])
    print("Hash shape:", sketch.indexHash_.shape[0], sketch.indexHash_.shape[1])
