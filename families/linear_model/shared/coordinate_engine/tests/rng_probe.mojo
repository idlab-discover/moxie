"""Diagnostic only: existing shared streams are not sklearn our_rand_r."""
from moxie._random import RandomState, NativeRandom

def main() raises:
    var rng = RandomState(0)
    var seed = rng.randint(0, 2147483647)
    print("seed", seed)
    var reseeded = RandomState(seed)
    var native = NativeRandom(UInt64(seed))
    print("philox", native.step()[0])
    for i in range(8):
        print("mt", rng.randint(3), reseeded.randint(3))
