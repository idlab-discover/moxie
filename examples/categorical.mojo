"""A nullable feature with a fixed dictionary and lossless row selection."""
from moxie._categorical import categorical_column
from numojo import Shape, Item
import numojo as nm


def main() raises:
    var cells: List[Optional[String]] = [
        Optional[String]("blue"), None, Optional[String]("amber"),
    ]
    var color = categorical_column(cells)
    print("Categories:", color.domain().values())
    print("First color:", color.get(0).value())
    print("Second row missing:", color.missing_mask()[Item(1)])
    color.set(1, Optional[String]("amber"))
    var rows = nm.zeros[DType.int64](Shape(2))
    rows[Item(0)] = 2
    rows[Item(1)] = 0
    var selected = color.take(rows)
    print("Selected:", selected.get(0).value(), selected.get(1).value())
