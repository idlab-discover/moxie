"""Run with the locked toolchain on PATH: python notebooks/_support/test_moxie_nb_io.py."""

import numpy as np

from moxie_nb import Session


def main():
    session = Session("_support/array_round_trip")
    matrix = np.array([[1.25]])
    vector = np.array([2**60 + 1], dtype=np.int64)
    singleton_axes = np.arange(6, dtype=np.float64).reshape(1, 2, 1, 3)
    mask = np.array([[[True]]])
    empty = np.empty((1, 0, 1))
    session.share(
        matrix=matrix, vector=vector, singleton_axes=singleton_axes,
        mask=mask, empty=empty,
    )
    session.run_mojo(
        "round_trip",
        """
from moxie_nb_io import load, load_ints, save, save_ints


def main() raises:
    save("matrix", load("matrix"))
    save("vector", load[DType.int64]("vector"))
    save_ints("list", load_ints("vector"))
    save("singleton_axes", load[DType.float32]("singleton_axes"))
    save("mask", load[DType.bool]("mask"))
    save("empty", load("empty"))
""",
    )
    expected = {
        "matrix": matrix,
        "vector": vector,
        "list": vector,
        "singleton_axes": singleton_axes.astype(np.float32),
        "mask": mask,
        "empty": empty,
    }
    actual = session.results["round_trip"]
    assert actual.keys() == expected.keys()
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values, strict=True)


if __name__ == "__main__":
    main()
