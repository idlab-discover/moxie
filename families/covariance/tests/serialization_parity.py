"""Compare restored native estimators to covariance methods from pinned sklearn."""
from pathlib import Path
import sys

import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAMILY / "tools"))
import parity_io
import pinned_sklearn


def main():
    modules = pinned_sklearn.load(["_empirical_covariance", "_shrunk_covariance"])
    empirical = modules["_empirical_covariance"].EmpiricalCovariance
    shrinkage = modules["_shrunk_covariance"]
    for function in (empirical.fit, empirical.score, empirical.mahalanobis,
                     shrinkage.ShrunkCovariance.fit, shrinkage.LedoitWolf.fit, shrinkage.OAS.fit):
        assert pinned_sklearn.executed_from_pinned(function), function

    builds = FAMILY / "builds"
    checks = 0
    for dtype, width, tolerance in ((np.float32, 32, 3e-6), (np.float64, 64, 1e-12)):
        X = np.array([[0, 0], [1, 2], [3, -1], [2, 4], [5, 1], [-1, 3]], dtype=dtype)
        probe = np.array([[1, -1], [2, 1], [4, 5]], dtype=dtype)
        native = parity_io.run(
            builds / "test_serialization", builds / f"serialization-{width}.txt",
            f"{width} {parity_io.matrix(X)} {parity_io.matrix(probe)}",
        )
        estimators = {
            "empirical": empirical(store_precision=False),
            "fixed": shrinkage.ShrunkCovariance(shrinkage=0.3),
            "lw": shrinkage.LedoitWolf(block_size=2, assume_centered=True),
            "oas": shrinkage.OAS(store_precision=False),
        }
        expected = {}
        for name, model in estimators.items():
            model.fit(X)
            expected[f"{name}_score"] = [model.score(probe)]
            expected[f"{name}_mahalanobis"] = model.mahalanobis(probe)
            if hasattr(model, "shrinkage_"):
                expected[f"{name}_shrinkage"] = [model.shrinkage_]
        assert native.keys() == expected.keys()
        for key, ref in expected.items():
            # The fixture is well-conditioned. Allow Float32 accumulation and
            # eigenvalue rounding differences, and Float64 solver roundoff.
            np.testing.assert_allclose(native[key], ref, rtol=tolerance, atol=tolerance, err_msg=key)
            checks += np.size(ref)
        print(f"PASS: Float{width} file round trips, configuration and inference contracts")
    print(f"PASS: {checks} comparisons against pinned sklearn")


if __name__ == "__main__":
    main()
