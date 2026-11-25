from numojo import NDArray, Shape, zeros


def training_data[dtype: DType]() raises -> NDArray[dtype]:
    var X = zeros[dtype](Shape(9, 2))
    var values: List[Int] = [-6, -3, -5, -2, -4, -3, -3, -2, 3, 3, 4, 2, 5, 3, 6, 2, 0, 9]
    for i in range(len(values)):
        X.unsafe_set(i, Scalar[dtype](values[i]))
    return X^


def query_data[dtype: DType]() raises -> NDArray[dtype]:
    var X = zeros[dtype](Shape(3, 2))
    var values: List[Int] = [-4, -2, 5, 2, 0, 18]
    for i in range(len(values)):
        X.unsafe_set(i, Scalar[dtype](values[i]))
    return X^
