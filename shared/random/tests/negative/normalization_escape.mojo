from moxie._random import RandomState, RandomStateHandle, check_random_state

def escape() raises -> RandomStateHandle[MutUntrackedOrigin]:
    var local = RandomState(42)
    return check_random_state(local)

def main() raises:
    var handle = escape()
    print(handle[].random_sample())
