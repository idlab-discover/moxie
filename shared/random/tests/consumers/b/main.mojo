from moxie._random import RandomState, our_rand_r, check_random_state, seed_global_random_state
from moxie.consumer_b import evaluate, evaluate_default, SeededProbe
from std.testing import assert_equal, assert_true

def main() raises:
    var word = UInt32(0)
    assert_equal(our_rand_r(word), UInt32(270369))
    assert_equal(word, UInt32(270369))
    var state = RandomState(42)
    print(evaluate(state))
    print(evaluate(state))
    var estimator = SeededProbe(42)
    _ = estimator.fit()
    var previous = estimator.learned_
    _ = estimator.fit()
    assert_equal(estimator.learned_, previous)
    var cloned = estimator.clone()
    assert_equal(cloned.learned_, -1)
    _ = cloned.fit()
    assert_equal(cloned.learned_, estimator.learned_)
    cloned.seed = 43
    _ = cloned.fit()
    assert_true(cloned.learned_ != estimator.learned_)

    seed_global_random_state(42)
    var reference = RandomState(42)
    assert_equal(evaluate_default(), reference.random_sample())
    with check_random_state() as default_state:
        assert_equal(default_state[].random_sample(), reference.random_sample())
    assert_equal(evaluate_default(), reference.random_sample())
