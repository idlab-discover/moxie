from moxie._random import RandomState, NativeRandom
from std.random import seed, random_float64
from std.sys import argv
import numojo as nm
from numojo import Shape, Item

# Decimal CLI parameters are encoded as integer ratios to avoid parser differences.
def main() raises:
    var args = argv()
    var mode = args[1]
    var s = Int(args[2])
    var count = Int(args[3])
    var n = Int(args[4])
    var k = Int(args[5])
    var state = RandomState(s)
    var p = nm.zeros[DType.float64](Shape(len(args) - 7))
    for i in range(p.size):
        p[Item(i)] = Float64(Int(args[7+i])) / 1000000.0
    if mode == "stdlib":
        seed(s)
        for _ in range(count):
            print(random_float64())
        return
    elif mode == "raw":
        for _ in range(count):
            print(state.next_uint32())
    elif mode == "uniform":
        for _ in range(count):
            print(state.uniform(Float64(n), Float64(k)))
    elif mode == "normal":
        for _ in range(count):
            print(state.normal(Float64(n), Float64(k)))
    elif mode == "gamma" or mode == "standard_gamma":
        for _ in range(count):
            if mode == "gamma":
                print(state.gamma(Float64(n) / 1000000.0, Float64(k) / 1000000.0))
            else:
                print(state.standard_gamma(Float64(n) / 1000000.0))
    elif mode == "integer":
        for _ in range(count):
            print(state.randint(n, k))
    elif mode == "binomial":
        for _ in range(count):
            print(state.binomial(n, Float64(k) / 1000000.0))
    elif mode == "choice" or mode == "choice_weighted":
        var result = nm.zeros[DType.int64](Shape(0))
        if mode == "choice":
            result = state.choice(n, count, k != 0)
        else:
            result = state.choice(n, p, count, k != 0)
        for i in range(result.size):
            print(result.item(i))
    elif mode == "shuffle" or mode == "permutation_array":
        var x = nm.zeros[DType.int64](Shape(n) if k == -1 else Shape(n, k))
        for i in range(x.size):
            x.unsafe_set(i, Int64(i * 7 + 3))
        if mode == "shuffle":
            state.shuffle(x)
        else:
            x = state.permutation(x)
        for i in range(x.size):
            print(x.unsafe_get(i))
    elif mode == "categorical":
        for _ in range(count):
            print(state.categorical(p))
    elif mode == "multinomial":
        var result = state.multinomial(n, p, count)
        for i in range(count):
            for j in range(p.size):
                print(result.item(i,j))
    elif mode == "permutation":
        var result = state.permutation(n)
        for i in range(result.size):
            print(result.item(i))
    elif mode == "sample":
        var result = state.sample_without_replacement(n, k, args[6])
        for i in range(result.size):
            print(result.item(i))
    elif mode == "mixed":
        for _ in range(count):
            print(state.standard_normal())
            print(state.randint(-100, 500))
            print(state.random_sample())
            print(state.binomial(1000, 0.2))
            print(state.gamma(100.0, 0.01))
            print(state.gamma(0.3, 2.0))
    elif mode == "dummy_multi_stratified":
        var q = nm.zeros[DType.float64](Shape(2))
        q[Item(0)] = 0.7
        q[Item(1)] = 0.3
        for _ in range(count):
            var first = state.multinomial(1, p, n)
            var second = state.multinomial(1, q, n)
            for i in range(n):
                var selected = 0
                for j in range(p.size):
                    if first.item(i,j) != 0:
                        selected = j
                print(selected)
            for i in range(n):
                print(Int(second.item(i,1)))
    elif mode == "dummy_uniform":
        for _ in range(count):
            for j in range(p.size):
                for _ in range(n):
                    print(state.randint(Int(p.item(j) * 1000000.0)))
    elif mode == "dummy_stratified":
        for _ in range(count):
            var result = state.multinomial(1, p, n)
            for i in range(n):
                for j in range(p.size):
                    print(result.item(i,j))
    else:
        raise Error("Unknown mode")
    print("STATE")
    print(state.state_position())
    print(Int(state.has_gauss()))
    print(state.cached_gauss())
    for i in range(624):
        print(state.state_word(i))
