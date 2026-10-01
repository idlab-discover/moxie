"""Scale some columns, keep others and drop one, then read the output feature names.

Any transformer whose Input/Target/Transformed are `NDArray[dtype]` can be a
column component; implementing `FeatureNamesOut` makes its names available.
"""
from numojo import NDArray, Shape, zeros
from moxie.base import TransformerMixin, UnsupervisedEstimator
from moxie.compose import ColumnTransformer, ColumnEntry, FeatureNamesOut


@fieldwise_init
struct RangeScalerParams(Copyable, Movable):
    pass


struct RangeScaler(Copyable, FeatureNamesOut, TransformerMixin, UnsupervisedEstimator):
    """Maps each column to [0, 1] using the training minimum and maximum."""

    comptime Input = NDArray[DType.float64]
    comptime Target = NDArray[DType.float64]
    comptime Transformed = NDArray[DType.float64]
    comptime Params = RangeScalerParams
    var low_: List[Float64]
    var span_: List[Float64]

    def __init__(out self):
        self.low_ = List[Float64]()
        self.span_ = List[Float64]()

    def get_params(self, deep: Bool = True) raises -> RangeScalerParams:
        return RangeScalerParams()

    def set_params(mut self, params: RangeScalerParams) raises:
        pass

    def __sklearn_clone__(self) raises -> Self:
        return Self()

    def __sklearn_is_fitted__(self) -> Bool:
        return len(self.low_) > 0

    def fit(mut self, X: NDArray[DType.float64], y: Optional[NDArray[DType.float64]] = None) raises:
        self.low_ = List[Float64]()
        self.span_ = List[Float64]()
        for j in range(X.shape[1]):
            var low = X.unsafe_get(j)
            var high = low
            for i in range(X.shape[0]):
                low = min(low, X.unsafe_get(i * X.shape[1] + j))
                high = max(high, X.unsafe_get(i * X.shape[1] + j))
            self.low_.append(low)
            self.span_.append(high - low if high > low else 1.0)

    def transform(self, X: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
        var out = X.copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                var k = i * X.shape[1] + j
                out.unsafe_set(k, (X.unsafe_get(k) - self.low_[j]) / self.span_[j])
        return out^

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        return input_features.copy()


def main() raises:
    # Columns: age, income, height, id.
    var X = zeros[DType.float64](Shape(3, 4))
    var rows: List[Float64] = [30, 52000, 1.80, 7, 45, 61000, 1.65, 8, 25, 38000, 1.72, 9]
    for i in range(12):
        X.unsafe_set(i, rows[i])

    var ct = ColumnTransformer(
        ColumnEntry("scaled", RangeScaler(), [0, 1]),  # age and income -> [0, 1]
        ColumnEntry("raw", "passthrough", [2]),  # height unchanged
        ColumnEntry("id", "drop", [3]),  # never used
    )
    var out = ct.fit_transform(X)
    var names: List[String] = ["age", "income", "height", "id"]
    var features = ct.get_feature_names_out(names)
    for j in range(out.shape[1]):
        print(features[j], ":", out.unsafe_get(j), out.unsafe_get(out.shape[1] + j), out.unsafe_get(2 * out.shape[1] + j))
    for record in ct.transformers_():
        print(record)
