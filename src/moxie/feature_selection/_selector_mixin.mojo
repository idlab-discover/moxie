# Adapted from scikit-learn; SPDX-License-Identifier: BSD-3-Clause
"""Public native selector mixin backed by the family's single mapping engine."""
import numojo as nm
from std.math import isfinite
from ._selection import SelectionState, SupportResult

trait SelectorMixin:
    """Implement dtype and _selection_state; opt into NaNs with _allows_nan.

    Fitting and estimator configuration belong to the implementing estimator.
    The returned state is an owned snapshot, including fitted feature names.
    """
    comptime dtype: DType

    def _selection_state(self) raises -> SelectionState:
        ...

    def _allows_nan(self) -> Bool:
        return False

    def get_support(self, indices: Bool = False) raises -> SupportResult:
        return self._selection_state().get_support(indices)

    def transform(self, X: nm.NDArray[Self.dtype]) raises -> nm.NDArray[Self.dtype]:
        validate_selector_input(X, self._allows_nan())
        return self._selection_state().transform(X)

    def inverse_transform(self, X: nm.NDArray[Self.dtype]) raises -> nm.NDArray[Self.dtype]:
        validate_selector_input(X)
        return self._selection_state().inverse_transform(X)

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        return self._selection_state().get_feature_names_out(input_features)


def validate_selector_input[dtype: DType](X: nm.NDArray[dtype], allow_nan: Bool = False) raises:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise Error("ValueError: expected nonempty rank-2 dense input")
    # sklearn's allow_nan tag disables all finite checks for transform.
    if not allow_nan:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if not isfinite(X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))):
                    raise Error("ValueError: X contains NaN or infinity")
