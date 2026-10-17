from moxie.isotonic import IsotonicRegression

def main() raises:
    var model = IsotonicRegression[DType.float32]()
    var x: List[Float32] = [0, 1]
    var y: List[Float64] = [0, 1]
    model.fit(x, y)
