"""Compare dense unsupervised finals against pinned scikit-learn."""
from inspect import getfile
from pathlib import Path
import sys

import numpy as np
from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import compose_parity_io as io


def main():
    sources = {
        Pipeline: "sklearn/pipeline.py",
        StandardScaler: "sklearn/preprocessing/_data.py",
        DBSCAN: "sklearn/cluster/_dbscan.py",
        KMeans: "sklearn/cluster/_kmeans.py",
        KernelDensity: "sklearn/neighbors/_kde.py",
        LocalOutlierFactor: "sklearn/neighbors/_lof.py",
    }
    io.verify_reference(sources.values())
    for estimator, relative in sources.items():
        assert Path(getfile(estimator)).read_bytes() == (io.REFERENCE / relative).read_bytes(), (
            f"Install scikit-learn built from {io.REVISION} to compare {relative}"
        )

    actual = {}
    for binary in ("test_pipeline_cluster", "test_pipeline_density"):
        output = io.run(ROOT, ROOT / "builds/tests/pipeline" / binary, capture=True).stdout
        actual.update(io.parse_results(output))
    assert set(actual) == {"dbscan", "kmeans", "density", "novelty"}

    X = np.array(
        [[-6, -3], [-5, -2], [-4, -3], [-3, -2], [3, 3], [4, 2], [5, 3], [6, 2], [0, 9]],
        dtype=np.float64,
    )
    query = np.array([[-4, -2], [5, 2], [0, 18]], dtype=np.float64)

    dbscan = Pipeline([("transformer", StandardScaler()), ("estimator", DBSCAN(eps=0.55, min_samples=2))])
    train = X.astype(np.float32)
    native = actual["dbscan"]
    labels = dbscan.fit_predict(train)
    np.testing.assert_array_equal(native["labels"], labels)
    np.testing.assert_array_equal(native["original"], labels)
    refit = clone(dbscan).set_params(estimator__eps=0.15).fit(train)
    np.testing.assert_array_equal(native["refit"], refit[-1].labels_)
    print("PASS: DBSCAN Float32 labels, clone independence and refit labels match exactly")

    centers = np.array([[-1, -1], [1, 1]], dtype=np.float64)
    kmeans = make_pipeline(StandardScaler(), make_pipeline(
        StandardScaler(), KMeans(n_clusters=2, init=centers, n_init=1),
    ))
    native = actual["kmeans"]
    np.testing.assert_array_equal(native["labels"], kmeans.fit_predict(X, np.zeros(len(X))))
    np.testing.assert_array_equal(native["predictions"], kmeans.predict(query))
    error = max(
        io.close(native["distances"], kmeans.transform(query), 0, 1e-12, "KMeans query distances"),
        io.close(native["fit_transform"], clone(kmeans).fit_transform(X), 0, 1e-12, "KMeans training distances"),
    )
    print(f"PASS: nested KMeans labels and predictions match exactly, max distance error {error:.3g}")

    density = make_pipeline(StandardScaler(), KernelDensity(bandwidth=0.45)).fit(X)
    native = actual["density"]
    refit = clone(density).set_params(kerneldensity__bandwidth=0.9).fit(X)
    error = max(
        io.close(native["samples"], density.score_samples(query), 0, 1e-12, "KDE samples"),
        io.close(native["score"], density.score(query), 0, 1e-12, "KDE total"),
        io.close(native["refit"], refit.score_samples(query), 0, 1e-12, "KDE refit"),
        io.close(native["original"], density.score_samples(query), 0, 1e-12, "KDE original"),
    )
    print(f"PASS: KernelDensity sample, total and refitted scores, max error {error:.3g}")

    novelty = make_pipeline(StandardScaler(), LocalOutlierFactor(n_neighbors=2, novelty=True)).fit(X)
    native = actual["novelty"]
    np.testing.assert_array_equal(native["prediction"], novelty.predict(query))
    error = max(
        io.close(native["decision"], novelty.decision_function(query), 0, 1e-12, "LOF decisions"),
        io.close(native["samples"], novelty.score_samples(query), 0, 1e-12, "LOF samples"),
    )
    print(f"PASS: LocalOutlierFactor predictions match exactly, max score error {error:.3g}")


if __name__ == "__main__":
    main()
