"""Fit weighted monotonic observations and interpolate unseen inputs."""
from moxie.base import clone
from moxie.isotonic import IsotonicRegression, isotonic_regression


def main() raises:
    var X: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var y: List[Float64] = [0.0, 3.0, 1.0, 4.0]
    var weights: List[Float64] = [1.0, 1.0, 3.0, 1.0]
    var model = IsotonicRegression(out_of_bounds="clip")
    _ = model.fit(X, y, sample_weight=weights)
    print("X thresholds:", model.X_thresholds_)
    print("y thresholds:", model.y_thresholds_)
    print("interpolation:", model.predict([0.5, 1.5, 2.5]))
    print("clipped endpoints:", model.predict([-1.0, 4.0]))
    print("weighted R2:", model.score(X, y, weights.copy()))
    print(
        "ordered helper:", isotonic_regression(y, sample_weight=weights.copy())
    )
    var fresh = clone(model)
    print("clone is fitted:", fresh.__sklearn_is_fitted__())

    # Select Float32 explicitly; predictions and thresholds keep that dtype.
    var x32: List[Float32] = [0, 1, 2]
    var y32: List[Float32] = [3, 1, 2]
    var model32 = IsotonicRegression[DType.float32](increasing="auto")
    _ = model32.fit(x32, y32)
    print("Float32 automatic direction:", model32.increasing_)
    print("Float32 predictions:", model32.predict(x32))
