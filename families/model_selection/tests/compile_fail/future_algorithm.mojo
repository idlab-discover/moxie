# Historical absence check promoted to a concrete integration contract.
from moxie.model_selection import KFold
from std.testing import assert_equal

def main() raises:
    var cv = KFold(2)
    var cursor = cv.split(4)
    assert_equal(cursor.next().value().test.size, 2)
