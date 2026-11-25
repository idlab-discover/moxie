"""Compare the iterative polynomial Nystroem + Ridge gallery configuration."""

import subprocess
import sys
import warnings

import numpy as np

from parity import FAMILY, REVISION, oracle

# Reuse the Ridge owner's loader for the support wheel's device API alias.
sys.path.insert(0, str(FAMILY.parent / "linear_model" / "tools"))
from parity_io import load_pinned, pinned_code, verify_reference


def fixture(start, rows, dtype, *, repeated_basis=False):
    i = np.arange(start, start + rows)[:, None]
    j = np.arange(3)
    pattern = i % 4 if repeated_basis and start == 0 else i
    values = ((((pattern + 3) * (pattern + 17 * j + 8)) % 67) / 4).astype(dtype)
    if repeated_basis:
        values /= 16
    missing = (i % 5 == j) & ((start != 0) | (j < 2))
    if repeated_basis and start == 0:
        missing = (i % 5 == 0) & (j < 2)
    values[missing] = np.nan
    return values


def matrix(lines, shape):
    assert tuple(map(int, next(lines).split())) == shape
    return np.array([float(next(lines)) for _ in range(np.prod(shape))]).reshape(shape)


def regressors():
    base = "sklearn/linear_model/_base.py"
    ridge_path = "sklearn/linear_model/_ridge.py"
    pairwise = "sklearn/metrics/pairwise.py"
    kernel_path = "sklearn/kernel_approximation.py"
    pipeline_path = "sklearn/pipeline.py"
    verify_reference([base, ridge_path, pairwise, kernel_path, pipeline_path])
    load_pinned(base, register=True)
    ridge = load_pinned(ridge_path)
    load_pinned(pairwise, register=True)
    kernel = load_pinned(kernel_path)
    pipeline = load_pinned(pipeline_path)
    for function, path in (
        (ridge.Ridge.fit, ridge_path),
        (ridge._BaseRidge.fit, ridge_path),
        (ridge.Ridge.predict, ridge_path),
        (kernel.Nystroem.fit, kernel_path),
        (kernel.Nystroem.transform, kernel_path),
        (kernel.pairwise_kernels, pairwise),
        (pipeline.Pipeline.fit, pipeline_path),
        (pipeline.Pipeline.predict, pipeline_path),
    ):
        pinned_code(function, path)
    return kernel.Nystroem, ridge.Ridge, pipeline.make_pipeline


def main(binary):
    iterative, _ = oracle()
    nystroem, ridge, make_pipeline = regressors()
    lines = iter(subprocess.check_output([binary], text=True).splitlines())
    cases = (
        ("full-rank", np.float32, 6, False, 3e-4),
        ("duplicate-basis", np.float32, 17, True, 3e-4),
        ("rank-deficient", np.float64, 17, False, 5e-8),
    )
    for name, dtype, rows, repeated_basis, tolerance in cases:
        train = fixture(0, rows, dtype, repeated_basis=repeated_basis)
        query = fixture(17, 7, dtype, repeated_basis=repeated_basis)
        model = iterative(
            estimator=make_pipeline(
                nystroem(kernel="polynomial", degree=2, random_state=0),
                ridge(alpha=1e4),
            ),
            random_state=0, max_iter=40, tol=1e-1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="n_components > n_samples")
            fitted = model.fit_transform(train)
        replay = model.transform(query)
        assert fitted.dtype == replay.dtype == dtype
        assert tuple(map(int, next(lines).split())) == (
            model.n_iter_, model.n_features_with_missing_, len(model.imputation_sequence_),
        )
        for step in model.imputation_sequence_:
            transformer = step.estimator.steps[0][1]
            indices = transformer.component_indices_
            if repeated_basis and step.feat_idx < 2:
                # All these fitted bases have exact row repetitions. Their
                # Gram matrices are singular even after Float32 rounding.
                assert np.unique(transformer.components_, axis=0).shape[0] == 4
                assert indices.size == 13
            assert tuple(map(int, next(lines).split())) == (
                step.feat_idx, len(step.neighbor_feat_idx), indices.size,
            )
            np.testing.assert_array_equal(
                [int(next(lines)) for _ in step.neighbor_feat_idx], step.neighbor_feat_idx,
            )
            np.testing.assert_array_equal([int(next(lines)) for _ in indices], indices)
        max_error = 0.0
        for source, expected in ((train, fitted), (query, replay)):
            actual = matrix(lines, expected.shape)
            observed = ~np.isnan(source)
            np.testing.assert_array_equal(actual[observed], source[observed])
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            max_error = max(max_error, np.max(np.abs(actual - expected)))
        print(f"PASS {name} {dtype.__name__} against {REVISION}: {model.n_iter_} rounds, "
              f"exact sequence and basis indices, max fill error {max_error:.3g}")
    assert next(lines, None) is None


if __name__ == "__main__":
    main(sys.argv[1])
