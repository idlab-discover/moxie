"""Inspect a dense composition plan without fitting an estimator."""
from moxie.compose import ColumnSelection, BlockLayout
from moxie.compose._plumbing import feature_name

def main() raises:
    var selected = ColumnSelection([0, -1], 4)
    var remainder = selected.remainder(4)
    var layout = BlockLayout(10)
    layout.append(10, 2)
    layout.append(10, len(remainder.indices))
    print(feature_name("scale", "x0"), layout.width)
