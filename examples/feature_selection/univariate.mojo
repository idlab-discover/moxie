"""Keep the strongest regression feature and reconstruct removed columns."""
import numojo as nm
from moxie.feature_selection import SelectKBest, BuiltinScorer, SelectionParameter

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(6, 2))
    var y = nm.zeros[DType.float64](nm.Shape(6))
    for i in range(6):
        x.unsafe_set(2 * i, Float64(i))
        x.unsafe_set(2 * i + 1, Float64(i % 2))
        y.unsafe_set(i, Float64(i) + 0.1 * Float64(i % 2))
    var selector = SelectKBest(BuiltinScorer("f_regression"), k=SelectionParameter(1))
    _ = selector.fit(x, y^)
    print(selector.get_feature_names_out())
    print(selector.transform(x))
    print(selector.inverse_transform(selector.transform(x)))
