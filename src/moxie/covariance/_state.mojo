"""Dense covariance ownership and validation; no covariance computation."""
import numojo as nm
from std.math import isfinite
from moxie.utils.validation import FittedState


def validate_matrix[dtype: DType](x: nm.NDArray[dtype], min_samples: Int = 1) raises:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if min_samples < 1 or x.ndim != 2:
        raise Error("ValueError: expected a dense rank-2 matrix")
    if x.shape[0] < min_samples or x.shape[1] < 1:
        raise Error("ValueError: insufficient samples or features")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not isfinite(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]))):
                raise Error("ValueError: input must be finite")


def validate_square[dtype: DType](x: nm.NDArray[dtype], features: Int) raises:
    validate_matrix(x)
    if x.shape[0] != features or x.shape[1] != features:
        raise Error("ValueError: covariance/precision shape mismatch")


def validate_location[dtype: DType](x: nm.NDArray[dtype], features: Int) raises:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if x.ndim != 1 or x.shape[0] != features or features < 1:
        raise Error("ValueError: location shape mismatch")
    for i in range(features):
        if not isfinite(x.unsafe_get(i * Int(x.strides[0]))):
            raise Error("ValueError: location must be finite")


struct CovarianceState[dtype: DType](Copyable, Movable):
    """Independent learned snapshots. Precision absence never means identity."""
    var fitted: FittedState
    var location_: Optional[nm.NDArray[Self.dtype]]
    var covariance_: Optional[nm.NDArray[Self.dtype]]
    var precision_: Optional[nm.NDArray[Self.dtype]]

    def __init__(out self):
        self.fitted = FittedState()
        self.location_ = None
        self.covariance_ = None
        self.precision_ = None

    def reset(mut self):
        self.fitted.reset()
        self.location_ = None
        self.covariance_ = None
        self.precision_ = None

    def replace(mut self, location: nm.NDArray[Self.dtype], covariance: nm.NDArray[Self.dtype],
                precision: Optional[nm.NDArray[Self.dtype]] = None) raises:
        validate_matrix(covariance)
        var p = covariance.shape[0]
        validate_square(covariance, p)
        validate_location(location, p)
        if precision:
            validate_square(precision.value(), p)
        # Validate and allocate before committing any learned field.
        var loc = location.copy()
        var cov = covariance.copy()
        var prec: Optional[nm.NDArray[Self.dtype]] = None
        if precision:
            prec = precision.value().copy()
        self.location_ = loc^
        self.covariance_ = cov^
        self.precision_ = prec^
        self.fitted.record_fit(1, p)

    def check_input(self, x: nm.NDArray[Self.dtype]) raises:
        if not self.fitted.is_fitted():
            raise Error("NotFittedError: call fit before inference")
        validate_matrix(x)
        self.fitted.check_features(x.shape[1])
