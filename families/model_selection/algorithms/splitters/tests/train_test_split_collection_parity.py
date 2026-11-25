"""Compare the collection overload to the pinned sklearn train_test_split."""
import subprocess
import sys

import numpy as np

from parity import load_reference


def dataset(dtype):
    rows = np.arange(12, dtype=dtype)
    return [rows[::-1], *(
        100 * index + 10 * rows[:, None] + np.arange(index + 1, dtype=dtype)
        for index in range(1, 4)
    )]


def main():
    namespace, source, _ = load_reference()
    split = namespace["train_test_split"]
    assert split.__code__.co_filename == str(source)
    stream = np.random.RandomState(13)
    groups = ["z", "h", "z", "q", "h", "q", "h", "z", "q", "h", "q", "z"]
    cases = (
        ("ordered", np.float32, dict(shuffle=False, train_size=7, test_size=3)),
        ("seeded", np.float64, dict(random_state=42)),
        ("stratified", np.float64, dict(random_state=42, test_size=0.4, stratify=groups)),
        ("stream_first", np.int64, dict(random_state=stream)),
        ("stream_second", np.int64, dict(random_state=stream)),
    )
    lines = iter(subprocess.check_output([sys.argv[1]], text=True).splitlines())
    for name, dtype, options in cases:
        for index, part in enumerate(split(*dataset(dtype), **options)):
            columns = 1 if part.ndim == 1 else part.shape[1]
            header = next(lines).split()
            assert header == [name, str(index), str(part.ndim), str(len(part)), str(columns)], header
            data = [int(next(lines)) for _ in range(part.size)]
            np.testing.assert_array_equal(data, part.reshape(-1), err_msg=f"{name}, output {index}")
    assert next(lines, None) is None
    print("PASS pinned sklearn train_test_split: 5 cases, 40 exact arrays")


if __name__ == "__main__":
    main()
