from moxie._random import RandomState, check_random_state

def escape() raises -> Pointer[RandomState, MutUntrackedOrigin]:
    with check_random_state() as rng:
        return rng

def main() raises:
    var pointer = escape()
    print(pointer[].random_sample())
