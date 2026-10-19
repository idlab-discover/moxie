"""Ownership and validation example; this does not fit a Gaussian process."""
import numojo as nm
from moxie.gaussian_process._plumbing import retain_training, validate_uncertainty
from std.testing import assert_equal

def main() raises:
    var X = nm.arange[DType.float64](6).reshape(nm.Shape(2, 3))
    var training = retain_training(X)  # Independent snapshot, default copy=True.
    X[nm.Item(0, 0)] = 100
    assert_equal(training.item(0, 0), 0)
    validate_uncertainty(True, False)
    print("Owned training snapshot:", training.shape[0], "samples")
