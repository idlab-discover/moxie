"""Check sequential step access against pinned scikit-learn."""

from inspect import getfile
from pathlib import Path
import subprocess
import sys

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import compose_parity_io as io


def main():
    sources = {
        Pipeline: "sklearn/pipeline.py",
        StandardScaler: "sklearn/preprocessing/_data.py",
        DummyRegressor: "sklearn/dummy.py",
    }
    io.verify_reference(sources.values())
    for estimator, relative in sources.items():
        assert Path(getfile(estimator)).read_bytes() == (io.REFERENCE / relative).read_bytes()

    X = [[0.0], [2.0]]
    y = [0.0, 8.0]
    pipe = make_pipeline(StandardScaler(), DummyRegressor()).fit(X, y)
    assert pipe[0] is pipe[-2]
    assert pipe[1] is pipe[-1] is pipe.named_steps["dummyregressor"]
    transformed = pipe[0].transform(X)
    nested = make_pipeline(StandardScaler(), pipe)
    scalers = make_pipeline(StandardScaler(), StandardScaler()).fit(X)
    assert scalers[0] is scalers.named_steps["standardscaler-1"]
    assert scalers[1] is scalers.named_steps["standardscaler-2"]
    both = scalers[-1].transform(scalers[0].transform(X))
    expected = {
        "supervised": [len(pipe), transformed[0, 0], pipe[-1].predict(transformed)[0]],
        "nested": [len(nested), len(nested[-1])],
        "transformers": [len(scalers), both[0, 0]],
    }
    output = subprocess.check_output(
        [ROOT / "builds/tests/pipeline/test_pipeline_steps"], text=True, cwd=ROOT
    )
    actual = {}
    for line in output.splitlines():
        label, *values = line.split()
        actual[label] = [float(value) for value in values]
    assert actual == expected, (actual, expected)
    print("PASS: pipeline lengths and fitted child outputs match exactly.")


if __name__ == "__main__":
    main()
