from moxie._random import RandomState, our_rand_r
from moxie.consumer_a import evaluate, SeededProbe
from std.testing import assert_equal, assert_true

def main() raises:
    var word = UInt32(0)
    assert_equal(our_rand_r(word), UInt32(270369))
    assert_equal(word, UInt32(270369))
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
