"""Array normalization rejects List inputs at its internal boundary."""
from moxie._naive_bayes_dense import log_normalize

def main() raises:
    var scores: List[List[Float64]] = [[0.0, 1.0]]
    _ = log_normalize(scores)
