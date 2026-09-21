from moxie._random import RandomState
from moxie.consumer_a import evaluate, SeededProbe
from std.testing import assert_equal, assert_true

def main() raises:
    var state = RandomState(42)
    print(evaluate(state))
    print(evaluate(state))
    var estimator = SeededProbe(42)
    estimator.fit()
    var previous = estimator.learned_
    estimator.fit()
    assert_equal(estimator.learned_, previous)
    var cloned = estimator.clone()
    assert_equal(cloned.learned_, -1)
    cloned.fit()
    assert_equal(cloned.learned_, estimator.learned_)
    cloned.seed = 43
    cloned.fit()
    assert_true(cloned.learned_ != estimator.learned_)
