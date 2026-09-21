from moxie.utils.validation import FittedState
from std.testing import assert_equal, assert_raises


def main() raises:
    var state = FittedState()
    with assert_raises():
        state.check_features(3)
    with assert_raises():
        state.record_fit(0, 3)
    state.record_fit(2, 3)
    state.check_features(3)
    with assert_raises():
        state.check_features(2)
    state.reset()
    assert_equal(state.is_fitted(), False)
