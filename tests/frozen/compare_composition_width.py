"""Compare frozen composition widths using a build of the pinned sklearn tree."""

from pathlib import Path
import subprocess

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"


def main():
    revision = subprocess.check_output(["git", "-C", SOURCE, "rev-parse", "HEAD"], text=True)
    assert revision.strip() == REVISION, revision
    for module in ("pipeline.py", "frozen/_frozen.py", "discriminant_analysis.py", "preprocessing/_data.py"):
        pinned = subprocess.check_output(["git", "-C", SOURCE, "show", f"{REVISION}:sklearn/{module}"])
        assert (Path(sklearn.__file__).parent / module).read_bytes() == pinned, module

    i = np.arange(26)
    y = i % 2
    X = np.column_stack((y * 3 + i % 7 / 4, i % 11 / 3))
    wider = np.zeros((len(i), 4))
    pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    union = FeatureUnion([("scale", StandardScaler())])
    assert not hasattr(FrozenEstimator(pipe), "n_features_in_")
    assert not hasattr(FrozenEstimator(union), "n_features_in_")
    frozen_pipe = FrozenEstimator(pipe.fit(X, y))
    frozen_union = FrozenEstimator(union.fit(wider).fit(X))
    frozen_pipe.fit(wider, y)
    frozen_union.fit(wider, None)
    expected = [clone(frozen_pipe).n_features_in_, clone(frozen_union).n_features_in_]

    native = subprocess.check_output([ROOT / "builds/tests/frozen/test_frozen_composition"], text=True)
    [widths] = [line.split()[1:] for line in native.splitlines() if line.startswith("composition_width ")]
    actual = list(map(int, widths))
    assert actual == expected, (actual, expected)
    print(f"Frozen Pipeline and FeatureUnion input widths match sklearn exactly: {actual}")


if __name__ == "__main__":
    main()
