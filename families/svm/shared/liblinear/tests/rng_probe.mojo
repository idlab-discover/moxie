from moxie._random import RandomState
from moxie.svm._liblinear import _bounded

def main() raises:
    for seed in [0, 42, 4294967295]:
        var rng = RandomState(seed)
        for width in [1, 2, 3, 2147483649, 4294967295]:
            for _ in range(1000):
                print(_bounded(rng, width))
        print(rng.next_uint32())
