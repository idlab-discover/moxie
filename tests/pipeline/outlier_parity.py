"""Compare native Pipeline outlier training labels against pinned scikit-learn."""
from inspect import getfile
from pathlib import Path
import sys

import numpy as np
from sklearn.base import OutlierMixin, clone
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import compose_parity_io as io


def main():
    sources = {
        OutlierMixin: "sklearn/base.py",
        Pipeline: "sklearn/pipeline.py",
        StandardScaler: "sklearn/preprocessing/_data.py",
        LocalOutlierFactor: "sklearn/neighbors/_lof.py",
        OneClassSVM: "sklearn/svm/_classes.py",
    }
    io.verify_reference(sources.values())
    for estimator, relative in sources.items():
        assert Path(getfile(estimator)).read_bytes() == (io.REFERENCE / relative).read_bytes(), (
            f"Install scikit-learn built from {io.REVISION} to compare {relative}"
        )

    binary = ROOT / "builds/tests/pipeline/test_pipeline_outliers"
    actual = io.parse_results(io.run(ROOT, binary, capture=True).stdout)
    assert set(actual) == {"lof32", "lof64", "one_class_svm"}
    X = np.array(
        [[-6, -3], [-5, -2], [-4, -3], [-3, -2], [3, 3], [4, 2], [5, 3], [6, 2], [0, 9]],
        dtype=np.float64,
    )
    query = np.array([[-4, -2], [5, 2], [0, 18]], dtype=np.float64)

    for name, dtype, tolerance in (("lof32", np.float32, 2e-6), ("lof64", np.float64, 1e-12)):
        train = X.astype(dtype)
        pipe = make_pipeline(StandardScaler(), LocalOutlierFactor(n_neighbors=2))
        native = actual[name]
        np.testing.assert_array_equal(native["labels"], pipe.fit_predict(train))
        refit = clone(pipe).set_params(
            localoutlierfactor__n_neighbors=4, localoutlierfactor__contamination=0.35,
        )
        np.testing.assert_array_equal(
            native["refitted_labels"], refit.fit_predict(train, np.zeros(len(train))),
        )
        np.testing.assert_array_equal(native["refit_same_instance"], refit.fit_predict(train))
        error = io.close(
            native["original_scores"], pipe[-1].negative_outlier_factor_, 0, tolerance, name,
        )
        novelty = clone(pipe).set_params(localoutlierfactor__novelty=True)
        assert not hasattr(novelty, "fit_predict")
        assert not hasattr(novelty[0], "n_features_in_")
        print(f"PASS: {name} labels and refits match exactly, max training score error {error:.3g}")

    svm = make_pipeline(
        StandardScaler(),
        make_pipeline(StandardScaler(), OneClassSVM(nu=0.6)),
    )
    np.testing.assert_array_equal(actual["one_class_svm"]["labels"], svm.fit_predict(X))
    np.testing.assert_array_equal(actual["one_class_svm"]["prediction"], svm.predict(query))
    print("PASS: nested OneClassSVM training labels and query predictions match exactly")


if __name__ == "__main__":
    main()
