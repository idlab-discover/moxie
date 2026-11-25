"""Compare scaled mixture probabilities against pinned scikit-learn."""
from pathlib import Path
import sys

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import compose_parity_io as io


def main():
    pipeline_path = "sklearn/pipeline.py"
    scaler_path = "sklearn/preprocessing/_data.py"
    base_path = "sklearn/mixture/_base.py"
    gaussian_path = "sklearn/mixture/_gaussian_mixture.py"
    io.verify_reference([pipeline_path, scaler_path, base_path, gaussian_path])
    pipeline = io.load_pinned(pipeline_path)
    scaler = io.load_pinned(scaler_path)
    io.load_pinned(base_path, register=True)
    mixture = io.load_pinned(gaussian_path)
    io.pinned_code(pipeline.Pipeline.predict_proba, pipeline_path)
    io.pinned_code(scaler.StandardScaler.fit, scaler_path)
    io.pinned_code(mixture.GaussianMixture.predict_proba, base_path)
    io.pinned_code(mixture.GaussianMixture._m_step, gaussian_path)

    binary = ROOT / "builds/tests/pipeline/test_pipeline_mixture_probability"
    actual = io.parse_results(io.run(ROOT, binary, capture=True).stdout)
    assert set(actual) == {"float32", "float64"}
    for dtype, tolerance in ((np.float32, 2e-6), (np.float64, 2e-12)):
        X = np.array(
            [[-6, -3], [-5, -2], [-4, -3], [-3, -2], [3, 3], [4, 2], [5, 3], [6, 2], [0, 9]],
            dtype=dtype,
        )
        query = np.array([[-4, -2], [5, 2], [0, 0]], dtype=dtype)
        pipe = pipeline.make_pipeline(scaler.StandardScaler(), mixture.GaussianMixture(
            n_components=2, covariance_type="diag", reg_covar=0.15, tol=1e-6,
            random_state=0, weights_init=np.full(2, 0.5, dtype=dtype),
            means_init=np.array([[-0.7, 0], [0.7, 0]], dtype=dtype),
            precisions_init=np.ones((2, 2), dtype=dtype),
        ))
        assert not hasattr(pipe, "predict_log_proba")
        assert not hasattr(pipe, "classes_")
        with np.testing.assert_raises(NotFittedError):
            pipe.predict_proba(query)
        fitted = pipe.fit(X).predict_proba(query)
        with np.testing.assert_raises(ValueError):
            pipe.predict_proba(query[:, :1])
        nested = pipeline.make_pipeline(scaler.StandardScaler(), clone(pipe)).fit(X).predict_proba(query)
        fresh = clone(pipe)
        with np.testing.assert_raises(NotFittedError):
            fresh.predict_proba(query)
        clone_refit = fresh.set_params(gaussianmixture__reg_covar=0.35).fit(X).predict_proba(query)
        original = pipe.predict_proba(query)
        refit = pipe.set_params(gaussianmixture__reg_covar=0.35).fit(X).predict_proba(query)
        expected = dict(fitted=fitted, nested=nested, clone_refit=clone_refit, original=original, refit=refit)
        case = str(X.dtype)
        assert set(actual[case]) == set(expected)
        assert np.max(np.abs(fitted - refit)) > 1e-3
        errors = []
        for name, target in expected.items():
            assert target.dtype == dtype
            observed = actual[case][name]
            # Scaler and EM arithmetic can round differently in each dtype.
            errors.append(io.close(observed, target, 0, tolerance, f"{case} {name}"))
            np.testing.assert_allclose(observed.sum(axis=1), 1, rtol=0, atol=tolerance)
        print(f"PASS: {case} scaled mixture probabilities, max absolute error {max(errors):.3g}")


if __name__ == "__main__":
    main()
