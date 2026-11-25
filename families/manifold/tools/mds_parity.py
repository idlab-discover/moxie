"""Compare dense MDS/SMACOF to scikit-learn's pinned manifold source."""
import argparse
import json
import subprocess
import sys
import warnings

from joblib import parallel_backend
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.distance import pdist, squareform
from sklearn.exceptions import DataConversionWarning

from check import ROOT, assemble, compile_file, environment
from isomap_parity import pinned, read_matrix

WORK = ROOT / "builds/mds-parity"
BINARY = WORK / "driver"


def oracle():
    mds = pinned("sklearn/manifold/_mds.py", "sklearn.manifold._pinned_mds")
    mds.ClassicalMDS = pinned(
        "sklearn/manifold/_classical_mds.py", "sklearn.manifold._pinned_classical_mds"
    ).ClassicalMDS
    return mds


def matrix(name, values):
    path = WORK / name
    with path.open("w") as stream:
        stream.write(f"{values.shape[0]} {values.shape[1]}\n")
        np.savetxt(stream, values, fmt="%.17g")
    return path


def native(values, initial, estimator, settings):
    args = [
        BINARY, values.dtype, initial.dtype if initial is not None else "float64",
        matrix("input.txt", values), "mds" if estimator else "smacof",
        matrix("init.txt", initial) if initial is not None else None,
        settings.get("metric_mds" if estimator else "metric", True),
        settings.get("n_components", 2), settings.get("n_init", 1),
        settings.get("n_jobs", 1), settings.get("max_iter", 300),
        settings.get("eps", 1e-6), settings.get("random_state", 23),
        settings.get("normalized_stress", "auto"), settings.get("init", "warn"),
        settings.get("metric", "euclidean"), settings.get("dissimilarity", "deprecated"),
        settings.get("metric_params", {}).get("p"),
    ]
    args = [str(x).lower() if isinstance(x, bool) else str(x) for x in args]
    return subprocess.run(args, text=True, capture_output=True, check=True, env=environment(ROOT))


def case(reference, label, values, *, initial=None, estimator=False, **settings):
    settings.setdefault("random_state", 23)
    with warnings.catch_warnings(record=True) as received, parallel_backend("threading", n_jobs=1):
        warnings.simplefilter("always")
        if estimator:
            model = reference.MDS(**settings)
            embedding = model.fit_transform(values, init=initial)
            stress, iterations = model.stress_, model.n_iter_
        else:
            embedding, stress, iterations = reference.smacof(
                values, init=initial, return_n_iter=True, **settings,
            )
    result = native(values, initial, estimator, settings)
    assert not result.stderr, result.stderr
    lines = iter(result.stdout.splitlines())
    actual = next(lines).split()
    actual_stress, actual_iterations = float(actual[0]), int(actual[1])
    if estimator:
        assert int(actual[2]) == model.n_features_in_
    actual_warnings = [(next(lines), next(lines)) for _ in range(int(next(lines)))]
    expected_warnings = [(w.category.__name__, str(w.message)) for w in received
                         if w.category in (FutureWarning, UserWarning, DataConversionWarning)]
    assert actual_warnings == expected_warnings, (label, actual_warnings, expected_warnings)
    actual_embedding = read_matrix(lines, np.dtype("float64"))
    use_float32_tolerance = values.dtype == np.float32 or (
        initial is not None and initial.dtype == np.float32
    )
    tolerance = 1e-4 if use_float32_tolerance else 3e-10
    assert actual_embedding.shape == embedding.shape, label
    assert actual_iterations == iterations, (label, actual_iterations, iterations)
    assert_allclose(actual_embedding, embedding, atol=tolerance, rtol=tolerance, err_msg=label)
    assert_allclose(actual_stress, stress, atol=tolerance, rtol=tolerance, err_msg=label)
    if estimator:
        dissimilarities = read_matrix(lines, values.dtype)
        assert_allclose(dissimilarities, model.dissimilarity_matrix_,
                        atol=tolerance, rtol=tolerance, err_msg=label)
    assert next(lines, None) is None
    print(f"PASS {label}: {iterations} iterations, stress {actual_stress:.10g}")


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    assemble(ROOT)
    compile_file(ROOT / "tests/mds_driver.mojo", BINARY)
    reference = oracle()
    rng = np.random.default_rng(82)
    features = rng.uniform(0.1, 2, size=(15, 4))
    explicit = rng.normal(size=(15, 3))
    for dtype in (np.float32, np.float64):
        name = np.dtype(dtype).name
        x = features.astype(dtype)
        distances = squareform(pdist(x)).astype(dtype)
        z = explicit.astype(dtype)
        nonmetric_d = distances.round(1)
        # Tied dissimilarities and zero-valued missing constraints.
        nonmetric_d[0, 2] = nonmetric_d[2, 0] = 0
        nonmetric_d[1, 3] = nonmetric_d[3, 1] = 0
        for metric in (True, False):
            d = distances if metric else nonmetric_d
            for normalized in ("auto", True, False):
                case(reference, f"{name} explicit metric={metric} stress={normalized}", d,
                     initial=z, metric=metric, n_init=3, n_components=1, max_iter=12,
                     normalized_stress=normalized)
            case(reference, f"{name} seeded serial metric={metric}", d,
                 metric=metric, n_init=4, max_iter=100, eps=3e-5)
            case(reference, f"{name} seeded parallel metric={metric}", d,
                 metric=metric, n_init=4, n_jobs=2, max_iter=29)
        case(reference, f"{name} classical init", x, estimator=True, init="classical_mds",
             n_init=2, max_iter=11)
        case(reference, f"{name} explicit overrides classical init", distances,
             initial=explicit, estimator=True, init="classical_mds", metric="precomputed",
             metric_mds=False, max_iter=20, n_components=1)
        case(reference, f"{name} cosine", x, estimator=True, init="random", metric="cosine",
             metric_mds=False, max_iter=11)
        case(reference, f"{name} Minkowski p=3", x, estimator=True, init="random",
             metric="minkowski", metric_params={"p": 3}, max_iter=11)
        case(reference, f"{name} Bray–Curtis with signed features",
             (x - x.dtype.type(0.5)), estimator=True, init="random", metric="braycurtis", max_iter=9)
        case(reference, f"{name} boolean metric warning",
             (x > 0.75).astype(dtype), estimator=True, init="random", metric="jaccard", max_iter=7)
        for metric_name in ("seuclidean", "mahalanobis"):
            case(reference, f"{name} inferred {metric_name}", x, estimator=True,
                 init="random", metric=metric_name, max_iter=7)
        case(reference, f"{name} legacy metric and dissimilarity", nonmetric_d,
             estimator=True, metric=False, dissimilarity="precomputed", max_iter=11)
        case(reference, f"{name} square data warning", x[:4], estimator=True, max_iter=11)

    tiny = np.array([
        [0, 1.732050807568877, 1.7320508075688772],
        [1.732050807568877, 0, 6.661338147750939e-16],
        [1.7320508075688772, 6.661338147750939e-16, 0],
    ])
    nearly_equal = np.array([[0.08665881585055124, 0.7939114643387546],
                             [0.9959834154297658, 0.7555546025640025],
                             [0.8766008278401566, 0.4227358815811242]])
    case(reference, "nearly equal isotonic thresholds", tiny, initial=nearly_equal, metric=False)
    case(reference, "near-symmetric dissimilarities preserved", tiny + np.triu(np.full((3, 3), 1e-12), 1),
         initial=nearly_equal, max_iter=15)
    case(reference, "all-zero metric raw stress", np.zeros((3, 3)), max_iter=5)
    case(reference, "all-zero metric normalized stress", np.zeros((3, 3)), max_iter=5,
         normalized_stress=True)
    print("PASS: dense MDS and SMACOF against pinned scikit-learn")


def notebook():
    """Run only the two MDS cells from the digits notebook."""
    core = ROOT.parents[1]
    sys.path.insert(0, str(core / "notebooks/_support"))
    from moxie_nb import Session
    from sklearn.datasets import load_digits

    path = core / "notebooks/manifold/plot_lle_digits.ipynb"
    cells = {cell["id"]: "".join(cell["source"]) for cell in json.loads(path.read_text())["cells"]}
    samples = load_digits(n_class=6).data[:200]
    session = Session("manifold/plot_lle_digits_mds_validation")
    session.share(S=samples)
    session.run_mojo("mds_digits", cells["mds-digits-native"].split("\n", 1)[1])
    exec(cells["mds-digits-parity"], {"MDS": oracle().MDS, "S": samples, "nb": session})
    for check in session.checks:
        print(check.name, check.status, check.max_abs)
    session.assert_parity()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", action="store_true")
    if parser.parse_args().notebook:
        notebook()
    else:
        main()
