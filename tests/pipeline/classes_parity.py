"""Compare class forwarding to nested pipelines from pinned scikit-learn."""

from inspect import getfile
from pathlib import Path
import sys

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import compose_parity_io as io


def main():
    sources = {
        Pipeline: "sklearn/pipeline.py",
        StandardScaler: "sklearn/preprocessing/_data.py",
        LogisticRegression: "sklearn/linear_model/_logistic.py",
    }
    io.verify_reference(sources.values())
    for estimator, relative in sources.items():
        assert Path(getfile(estimator)).read_bytes() == (io.REFERENCE / relative).read_bytes(), (
            f"Install scikit-learn built from {io.REVISION} to compare {relative}"
        )

    X = np.arange(6, dtype=np.float64).reshape(-1, 1)
    y = np.array([7, 7, 7, -2, -2, -2], dtype=np.int64)
    pipe = make_pipeline(
        StandardScaler(), make_pipeline(StandardScaler(), LogisticRegression())
    ).fit(X, y)
    binary = ROOT / "builds/tests/pipeline/test_pipeline_classes"
    actual = io.parse_results(io.run(ROOT, binary, capture=True).stdout)["logistic"]
    np.testing.assert_array_equal(actual["classes"], pipe[-1].classes_)
    np.testing.assert_array_equal(actual["nested_classes"], pipe.classes_)
    error = io.close(
        actual["probabilities"], pipe.predict_proba(X), 0, 1e-12, "probability columns"
    )
    fresh = clone(pipe).fit(X, y + 10)
    np.testing.assert_array_equal(actual["clone_classes"], fresh.classes_)
    np.testing.assert_array_equal(actual["original_classes"], pipe.classes_)
    pipe.fit(X, y + 20)
    np.testing.assert_array_equal(actual["refit_classes"], pipe.classes_)
    print("PASS: direct, nested, cloned and refitted pipeline classes match exactly. "
          f"Maximum probability error {error:.3g}.")


if __name__ == "__main__":
    main()
