"""Fit a seeded dense RBF map and an explicit shared-context skewed map."""
from numojo import NDArray, Shape
from moxie.kernel_approximation import RBFSampler, SkewedChi2Sampler
from moxie._random import RandomState

def main() raises:
    var x = NDArray[DType.float64](Shape(2, 2))
    x.unsafe_set(0, 0)
    x.unsafe_set(1, 0)
    x.unsafe_set(2, 1)
    x.unsafe_set(3, 1)
    var rbf = RBFSampler(gamma=String("scale"), n_components=256, random_state=42)
    var features = rbf.fit_transform(x)
    print("RBF output:", features.shape[0], features.shape[1])
    var context = RandomState(42)
    var skewed = SkewedChi2Sampler(skewedness=0.1, n_components=256)
    var other = skewed.fit_transform_with_rng(x, context)
    print("Skewed output:", other.shape[0], other.shape[1])
