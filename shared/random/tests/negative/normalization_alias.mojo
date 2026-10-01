from moxie._random import RandomState, check_random_state

def simultaneous(mut a: RandomState, mut b: RandomState):
    print(a.random_sample(), b.random_sample())

def main() raises:
    var original = RandomState(42)
    var handle = check_random_state(original)
    simultaneous(original, handle[])
