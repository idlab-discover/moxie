from moxie._random import RandomState, check_random_state

def main() raises:
    var original = RandomState(42)
    var handle = check_random_state(original)
    var moved = original^
    print(handle[].random_sample())
    print(moved.random_sample())
