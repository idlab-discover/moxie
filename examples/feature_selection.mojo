"""Apply a known mask; this does not learn which features to select."""
import numojo as nm
from moxie.feature_selection import SelectionState

def main() raises:
    var state = SelectionState()
    state.commit([True, False, True], ["height", "unused", "width"])
    var x = nm.ones[DType.float64](nm.Shape(2, 3))
    var reduced = state.transform(x)
    print("Selected columns:", state.get_feature_names_out())
    print("Reduced shape:", reduced.shape)
    print("Restored shape:", state.inverse_transform(reduced).shape)
