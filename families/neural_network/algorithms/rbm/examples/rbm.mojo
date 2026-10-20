from numojo import zeros, Shape
from moxie.neural_network import BernoulliRBM


def main() raises:
    # A binary feature matrix: samples are rows, visible units are columns.
    var X = zeros[DType.float64](Shape(4, 3))
    X.unsafe_set(4, 1)
    X.unsafe_set(5, 1)
    X.unsafe_set(6, 1)
    X.unsafe_set(8, 1)
    X.unsafe_set(9, 1)
    X.unsafe_set(10, 1)
    X.unsafe_set(11, 1)
    var model = BernoulliRBM(n_components=2, random_state=42)
    var hidden = model.fit_transform(X)
    print("Hidden probabilities:", hidden)
    var score = model.score_samples(X)
    print("Pseudolikelihood:", score)
    var sample = model.gibbs(X)
    print("One Gibbs step:", sample)
