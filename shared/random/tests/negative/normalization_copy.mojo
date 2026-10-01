from moxie._random import check_random_state

def main() raises:
    var handle = check_random_state()
    var duplicate = handle.copy()
    print(duplicate[].random_sample())
