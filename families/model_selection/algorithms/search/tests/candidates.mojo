from moxie.model_selection import ParameterGrid, ParameterSampler, DiscreteDistribution, ParameterDomain
from moxie._random import RandomState
from std.testing import assert_equal, assert_true, assert_false, assert_raises

def main() raises:
    var mapping = Dict[String, List[Int]]()
    mapping['z'] = [4, 5]
    mapping['a'] = [1, 2, 3]
    var grid = ParameterGrid[Int](mapping)
    assert_equal(len(grid), 6)
    assert_equal(grid[0]['a'], 1)
    assert_equal(grid[1]['z'], 5)
    assert_equal(grid[2]['a'], 2)
    assert_equal(grid[5]['z'], 5)
    mapping['a'] = [99]
    assert_equal(grid[0]['a'], 1)
    var first = grid.next()
    first.value()['a'] = -1
    assert_equal(grid[0]['a'], 1)
    grid.reset()
    for _ in range(6):
        assert_true(Bool(grid.next()))
    assert_false(Bool(grid.next()))
    with assert_raises():
        _ = grid[6]
    with assert_raises():
        var invalid = Dict[String, List[Int]]()
        invalid['bad'] = List[Int]()
        _ = ParameterGrid[Int](invalid)
    var defaults = Dict[String, List[Int]]()
    var empty_grid = ParameterGrid[Int](defaults)
    assert_equal(len(empty_grid), 1)
    assert_equal(len(empty_grid[0]), 0)
    var empty_spaces = List[Dict[String, List[Int]]]()
    assert_equal(len(ParameterGrid[Int](empty_spaces)), 0)
    var sampler = ParameterSampler[DiscreteDistribution[Int]](grid, 20, random_state=42)
    assert_equal(len(sampler), 6)
    var rng = RandomState(42)
    var mutable_sampler = ParameterSampler[DiscreteDistribution[Int]](grid, 20, rng)
    for _ in range(6):
        var a = sampler.next()
        var b = mutable_sampler.next()
        assert_equal(a.value()['a'], b.value()['a'])
        assert_equal(a.value()['z'], b.value()['z'])
    var domains = List[ParameterDomain[DiscreteDistribution[Int]]]()
    domains.append(ParameterDomain[DiscreteDistribution[Int]]('z', [4, 5]))
    domains.append(ParameterDomain[DiscreteDistribution[Int]]('a', DiscreteDistribution[Int]([1, 2, 3])))
    var spaces = List[List[ParameterDomain[DiscreteDistribution[Int]]]]()
    spaces.append(domains^)
    var mixed = ParameterSampler[DiscreteDistribution[Int]](spaces, 10, random_state=1)
    assert_equal(len(mixed), 10)
    for _ in range(10):
        var sample = mixed.next()
        assert_true(sample.value()['a'] >= 1 and sample.value()['a'] <= 3)
        assert_true(sample.value()['z'] == 4 or sample.value()['z'] == 5)
    print('PASS grid and sampler contracts')
