# Dense scope and status

Implemented (dense Float32/Float64): GraphicalLasso and graphical_lasso with
mode='cd' and mode='lars', precomputed covariance, assume_centered, alpha=0 and max_iter
limits; inherited score/mahalanobis/error_norm/get_precision.

GraphicalLassoCV adds automatic alpha refinement, explicit alpha lists, default
or custom folds, CV scores and a final refit. Its private path uses the same
graphical solver; no extra public `graphical_lasso_path` export is
introduced (it is absent from sklearn's covariance `__all__`).

Not implemented: parallel CV execution and Python metadata routing. See [API adaptations](api.md).

`pixi run covariance-test` covers Gram CD closed forms, symmetry and C P = I at
convergence, diagonal precision at large alpha, precomputed/function agreement,
parameter validation, LARS refits and cloning. `pixi run covariance-parity` exercises both
modes against the pinned `_graph_lasso.py` and `_least_angle.py`; the support
wheel supplies compiled CD and array utilities. Iteration counts must match.
Float64 tolerance is 1e-10 and Float32 1e-6 relative, scaled by conditioning.
The private Gram-path suite adds 192 endpoint comparisons including complete
paths, interpolation, ties, duplicate columns, rejected pivots and nondefault
eps; its relative/absolute tolerance is 2e-9. Inputs remain unchanged.
Sparse input: deferred (SPARSITY-SCOPE); dense precision storage only.

The CV suite compares complete alpha grids, every fold score, means/population
standard deviations, selected alpha, refit covariance/precision, costs, iteration
counts, location, score and Mahalanobis distances. It includes unsorted/duplicate
alpha lists, tied scores, custom shuffled folds, refinements, wide data, singleton
test folds, both centering modes and Float32/Float64. See `tests/cv_parity.py`
for deterministic fixtures and tolerances. `tests/cv_contracts.mojo` covers native
API adaptations, ownership, strided inputs, cloning, refits and invalid inputs.

CV parity runs 60 seeded datasets per solver. Single-alpha cases use fixed-alpha
reference fits because pinned sklearn raises IndexError. Float32 CD cases that
hit mixed-buffer errors use Float64 oracle inputs rounded first to Float32.
LARS Float32 centered cases check grids/selection against the original Float32
reference, then numerical outputs against Float64 oracle inputs rounded first
to Float32, matching native computation. Ill-conditioned short paths can amplify
Float32 grid rounding in fold scores beyond the usual storage tolerance.
For rank-deficient folds, the empirical alpha-zero baseline columns (split,
mean, std) are excluded and counted: determinant rounding can change finite
versus -inf scores. Positive-alpha paths and final fits remain compared.

Run these commands from the Moxie repository root.
