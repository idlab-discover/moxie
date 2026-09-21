from moxie._random import RandomState

def main() raises:
    var state = RandomState(42)
    print(state.random_sample())
    var snapshot = state.copy()
    print(state.standard_normal(), snapshot.standard_normal())
    print(state.gamma(100.0, 0.01))
    var selected = state.sample_without_replacement(10, 3)
    for i in range(selected.size):
        print(selected.item(i))
