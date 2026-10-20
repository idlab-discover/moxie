from moxie._random import RandomState

def main() raises:
    var rng = RandomState(0)
    print(rng.gamma(100.0, 0.01))
