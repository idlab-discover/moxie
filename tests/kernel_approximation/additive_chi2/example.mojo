"""Map nonnegative histograms to additive chi-squared features."""
from numojo import NDArray, Shape
from moxie.kernel_approximation import AdditiveChi2Sampler


def main() raises:
    var x = NDArray[DType.float64](Shape(2, 3))
    var values: List[Float64] = [0.2, 0.3, 0.5, 0.0, 0.6, 0.4]
    for k in range(len(values)):
        x.unsafe_set(k, values[k])
    var sampler = AdditiveChi2Sampler(sample_steps=2)
    var features = sampler.fit_transform(x)
    print("Additive chi2 output:", features.shape[0], features.shape[1])
    var names = sampler.get_feature_names_out()
    print("First and last names:", names[0], names[len(names) - 1])
