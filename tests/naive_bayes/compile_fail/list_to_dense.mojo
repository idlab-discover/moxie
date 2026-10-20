from moxie._naive_bayes_dense import validate_matrix

def main() raises:
    var X: List[List[Float64]] = [[1.0]]
    _ = validate_matrix(X)
