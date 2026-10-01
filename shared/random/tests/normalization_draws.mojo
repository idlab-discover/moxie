"""Validation-only sequences and complete states at normalization boundaries."""
from std.sys import argv
from moxie._random import RandomState, check_random_state, seed_global_random_state


def dump(state: RandomState) raises:
    print("STATE")
    print(state.state_position())
    print(Int(state.has_gauss()))
    print(state.cached_gauss())
    for i in range(624):
        print(state.state_word(i))


def workload(mut state: RandomState) raises:
    print("DRAWS")
    for _ in range(17):
        print(state.random_sample())
    for _ in range(19):
        print(state.standard_normal())
    for _ in range(5):
        print(state.randint(1000))
    dump(state)


def main() raises:
    var args = argv()
    var mode = args[1]
    var seed = Int(args[2])
    var state = RandomState(seed)
    _ = state.standard_normal()
    seed_global_random_state(seed)
    with check_random_state() as rng:
        _ = rng[].standard_normal()
    for _ in range(3):
        if mode == "integer":
            with check_random_state(seed) as rng:
                workload(rng[])
        elif mode == "optional_integer":
            with check_random_state(Optional[Int](seed)) as rng:
                workload(rng[])
        elif mode == "borrowed":
            with check_random_state(state) as rng:
                workload(rng[])
        elif mode == "default":
            with check_random_state() as rng:
                workload(rng[])
        elif mode == "none":
            with check_random_state(None) as rng:
                workload(rng[])
        elif mode == "optional_none":
            with check_random_state(Optional[Int]()) as rng:
                workload(rng[])
        else:
            raise Error("unknown normalization mode")
    print("BORROWED")
    dump(state)
    print("GLOBAL")
    with check_random_state() as rng:
        dump(rng[])
