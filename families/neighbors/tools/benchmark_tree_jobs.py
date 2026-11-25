"""Check pinned sklearn parity and measure Moxie tree queries with pyperf."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pyperf
from sklearn.neighbors import NearestNeighbors, _base

FAMILY = Path(__file__).resolve().parents[1]
BINARY = FAMILY / "builds/benchmark_tree_jobs"
REFERENCE = FAMILY.parents[2] / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
ALGORITHMS = ("kd_tree", "ball_tree")
JOBS = (1, 2)


def native(algorithm, jobs, loops):
    return subprocess.check_output(
        [BINARY, algorithm, str(jobs), str(loops)], text=True
    )


def feature_rows(rows, seed):
    state = seed
    result = np.empty((rows, 5))
    for i in range(result.size):
        state = (1_103_515_245 * state + 12_345) % 2_147_483_648
        result.flat[i] = state / 2_147_483_648
    return result


def check_parity():
    revision = subprocess.check_output(
        ["git", "-C", REFERENCE, "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f"expected sklearn {REVISION}, got {revision}")
    source = subprocess.check_output(
        ["git", "-C", REFERENCE, "show", f"{REVISION}:sklearn/neighbors/_base.py"]
    )
    if Path(_base.__file__).read_bytes() != source:
        raise RuntimeError("loaded sklearn neighbor source differs from the pin")

    training = feature_rows(8192, 42)
    queries = feature_rows(512, 43)
    for algorithm in ALGORITHMS:
        baseline = None
        for jobs in JOBS:
            model = NearestNeighbors(n_neighbors=10, algorithm=algorithm, n_jobs=jobs)
            expected_distances, expected_indices = model.fit(training).kneighbors(queries)
            actual = np.fromstring(native(algorithm, jobs, 0), sep=" ").reshape(512, 10, 2)
            if baseline is None:
                baseline = actual
            else:
                np.testing.assert_array_equal(actual, baseline)
            np.testing.assert_array_equal(actual[:, :, 0], expected_indices)
            np.testing.assert_allclose(
                actual[:, :, 1], expected_distances, rtol=0, atol=1e-14
            )
    print("PASS: exact agreement across job counts and pinned sklearn parity")


def measure(loops, algorithm, jobs):
    elapsed, _checksum = native(algorithm, jobs, loops).split()
    return float(elapsed)


if __name__ == "__main__":
    if "--worker" not in sys.argv:
        check_parity()
    if sys.argv[1:] != ["--check-only"]:
        runner = pyperf.Runner()
        runner.metadata.update(
            compiler="Mojo 1.0.0 / MAX 26.5.0, -O3 --fp-mode=contract=off",
            reference_revision=REVISION,
            input="8192 training rows, 512 query rows, 5 Float64 features, seeds 42 and 43",
            query="10 Euclidean neighbors with distances, leaf_size=30",
            timing="Native query loop after fit and warmup, including result allocation",
        )
        for algorithm in ALGORITHMS:
            for jobs in JOBS:
                runner.bench_time_func(
                    f"{algorithm}/n_jobs={jobs}", measure, algorithm, jobs
                )
