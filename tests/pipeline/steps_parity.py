"""Check sequential step access against pinned scikit-learn."""

from inspect import getfile
from pathlib import Path
import subprocess
import sys

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

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
    fresh = clone(pipe)
    fresh.set_params(steps=[("standardscaler", fresh[0]), ("regression", fresh[1])])
    renamed_names = list(fresh.named_steps)
    fresh.steps[-1] = ("standardscaler", fresh[-1])
    assert fresh.named_steps["standardscaler"] is fresh[-1]
    transformed = pipe[0].transform(X)
    nested = make_pipeline(StandardScaler(), pipe)
    scalers = make_pipeline(StandardScaler(), StandardScaler()).fit(X)
    assert scalers[0] is scalers.named_steps["standardscaler-1"]
    assert scalers[1] is scalers.named_steps["standardscaler-2"]
    both = scalers[-1].transform(scalers[0].transform(X))
    prefix, tail, full, empty = pipe[:-1], pipe[-1:99], pipe[-99:99], pipe[2:0]
    assert prefix[-1] is pipe[0]
    assert tail.named_steps["dummyregressor"] is pipe[-1]
    assert full[-1] is pipe[-1]
    assert empty.__sklearn_is_fitted__()
    expected = {
        "supervised": [len(pipe), transformed[0, 0], pipe[-1].predict(transformed)[0]],
        "nested": [len(nested), len(nested[-1])],
        "slices": [
            len(prefix), len(tail), len(full), len(empty),
            prefix.transform(X)[0, 0], tail.predict(transformed)[0], full.predict(X)[0],
        ],
        "transformers": [len(scalers), both[0, 0], scalers[:].transform(X)[0, 0]],
    }
    live_prefix = scalers[:1]
    scalers.fit([[4.0], [6.0]])
    expected["refit_slice"] = [live_prefix.transform(X)[0, 0]]
    output = subprocess.check_output(
        [ROOT / "builds/tests/pipeline/test_pipeline_steps"], text=True, cwd=ROOT
    )
    actual = {}
    check_is_fitted(pipe.named_steps["dummyregressor"])
    reference_names = {
        "slice_feature": [prefix.get_feature_names_out()[0]],
        "named_steps": [*pipe.named_steps, str(True)],
        "renamed_steps": renamed_names,
        "duplicate_steps": list(fresh.named_steps),
    }
    names = {}
    for line in output.splitlines():
        label, *values = line.split()
        if label in reference_names:
            names[label] = values
        else:
            actual[label] = [float(value) for value in values]
    assert names == reference_names, (names, reference_names)
    assert actual == expected, (actual, expected)
    print("PASS: pipeline names, lengths, slices and fitted outputs match exactly.")


if __name__ == "__main__":
    main()
