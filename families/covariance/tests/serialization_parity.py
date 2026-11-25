"""Compare restored native estimators to covariance methods from pinned sklearn."""
from pathlib import Path
import sys

import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAMILY / "tools"))
import parity_io
import pinned_sklearn


def main():
    modules = pinned_sklearn.load([
        "_empirical_covariance", "_shrunk_covariance", "_robust_covariance",
        "_elliptic_envelope", "_graph_lasso",
    ])
    empirical = modules["_empirical_covariance"].EmpiricalCovariance
    shrinkage = modules["_shrunk_covariance"]
    min_cov_det = modules["_robust_covariance"].MinCovDet
    envelope = modules["_elliptic_envelope"].EllipticEnvelope
    graphical = modules["_graph_lasso"]
    for function in (empirical.fit, empirical.score, empirical.mahalanobis,
                     shrinkage.ShrunkCovariance.fit, shrinkage.LedoitWolf.fit, shrinkage.OAS.fit,
                     min_cov_det.fit, envelope.fit, envelope.predict, envelope.decision_function,
                     graphical.GraphicalLasso.fit, graphical.GraphicalLassoCV.fit):
        assert pinned_sklearn.executed_from_pinned(function), function

    builds = FAMILY / "builds"
    checks = 0
    for dtype, width, tolerance in ((np.float32, 32, 3e-6), (np.float64, 64, 1e-12)):
        X = np.array([[0, 0], [1, 2], [3, -1], [2, 4], [5, 1], [-1, 3]], dtype=dtype)
        probe = np.array([[1, -1], [2, 1], [4, 5]], dtype=dtype)
        robust_X = np.vstack((np.random.default_rng(37).normal(size=(24, 2)),
                              [[7, 5], [-4, -7], [8, 1], [2, -9]])).astype(dtype)
        native = parity_io.run(
            builds / "test_serialization", builds / f"serialization-{width}.txt",
            f"{width} {parity_io.matrix(X)} {parity_io.matrix(probe)} {parity_io.matrix(robust_X)}",
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
        mcd = min_cov_det(store_precision=False, support_fraction=0.75, random_state=0).fit(robust_X)
        detector = envelope(contamination=0.2, random_state=0).fit(robust_X)
        expected.update(
            mcd_score=[mcd.score(probe)],
            mcd_mahalanobis=mcd.mahalanobis(probe),
            mcd_raw_location=mcd.raw_location_,
            mcd_raw_covariance=mcd.raw_covariance_.ravel(),
            mcd_raw_support=mcd.raw_support_,
            mcd_support=mcd.support_,
            mcd_dist=mcd.dist_,
            envelope_offset=[detector.offset_],
            envelope_predict=detector.predict(probe),
            envelope_decision=detector.decision_function(probe),
            envelope_score_samples=detector.score_samples(probe),
            envelope_mahalanobis=detector.mahalanobis(probe),
        )
        fixed = graphical.GraphicalLasso(alpha=0.1, mode="lars", max_iter=70).fit(robust_X)
        cv = graphical.GraphicalLassoCV(alphas=[0.2, 0.4, 0.1], cv=3, n_jobs=1, mode="lars").fit(robust_X)
        for name, model in (("graphical", fixed), ("graphical_cv", cv)):
            expected[f"{name}_score"] = [model.score(probe)]
            expected[f"{name}_mahalanobis"] = model.mahalanobis(probe)
            expected[f"{name}_covariance"] = model.covariance_.ravel()
            expected[f"{name}_precision"] = model.precision_.ravel()
            expected[f"{name}_location"] = model.location_
            expected[f"{name}_costs"] = np.asarray(model.costs_).ravel()
            expected[f"{name}_n_iter"] = [model.n_iter_]
        results = cv.cv_results_
        expected.update(
            graphical_cv_alpha=[cv.alpha_],
            graphical_cv_alphas=results["alphas"],
            graphical_cv_splits=np.column_stack([results[f"split{i}_test_score"] for i in range(3)]).ravel(),
            graphical_cv_mean=results["mean_test_score"],
            graphical_cv_std=results["std_test_score"],
        )
        assert native.keys() == expected.keys()
        for key, ref in expected.items():
            # The fixture is well-conditioned. Allow Float32 accumulation and
            # eigenvalue rounding differences, and Float64 solver roundoff.
            if key in {"mcd_raw_support", "mcd_support", "envelope_predict"} or key.endswith("_n_iter"):
                np.testing.assert_array_equal(native[key], ref, err_msg=key)
            else:
                np.testing.assert_allclose(native[key], ref, rtol=tolerance, atol=tolerance, err_msg=key)
            checks += np.size(ref)
        print(f"PASS: Float{width} file round trips, configuration and inference contracts")
    print(f"PASS: {checks} comparisons against pinned sklearn")


if __name__ == "__main__":
    main()
