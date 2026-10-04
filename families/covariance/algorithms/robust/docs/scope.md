# Dense scope and status

Implemented (dense Float32/Float64): fast_mcd (univariate, small-sample,
>500-sample subset/merge and >1500-sample full-refinement branches),
MinCovDet (raw fit, assume_centered refit, consistency correction,
reweighting, inherited score/mahalanobis/error_norm/get_precision) and
EllipticEnvelope (offset_, decision_function, score_samples, predict, fit_predict, score).

`pixi run covariance-test`: chi2 constants, lifecycle, seed reproducibility, shared
stream advancement, fast_mcd/MinCovDet agreement, invalid parameters and
inputs, the zero-support-covariance error, detector label/decision/accuracy
invariants, univariate and full-support branches, both dtypes.
`pixi run covariance-parity`: 8600 checks over 50 seeded cases (all branches above,
support_fraction, assume_centered; Float64/Float32) against the pinned sklearn
sources with identical seeds; supports and labels match exactly (after the
documented tie normalization in tests/parity.py).

Sparse branches: none; sparse input is deferred by SPARSITY-SCOPE.

Run these commands from the Moxie repository root.
