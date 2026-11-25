# Dense coordinate estimator scope

Implemented: ElasticNet, Lasso and their single-target paths with the modes
and typed adaptations in [api.md](api.md). Uses family preprocessing and
prediction, and the residual and Gram engines at shared/coordinate_engine.

Validation: `pixi run build`, `pixi run test`, `pixi run parity`. Native tests
cover both dtypes, unfitted calls, clone/params, refits, warm starts, weight
normalization, scalar weights, shape changes, failed fits, positive constraints,
negative strides and explicit unsupported modes. Differential tests compile
and execute the actual local sklearn _cd_fast.pyx as well as _base.py and
_coordinate_descent.py at 4358542a1bd39af2e7e1cbace8b9fed067473c5b.
264 cyclic cases cover tall/wide/tiny/constant/multiple outputs, l1_ratio 0/0.5/1,
weights including zeros, positivity, no intercept, warm starts, one iteration,
zero targets, large alpha, and column targets. The 96 seeded random fits
cover weighted fits, positivity, warm starts and one-sweep truncation.
Float64 tolerance is 1e-8 and
Float32 tolerance is 3e-4 for reduction order. Target-scaled absolute tolerances and exact
Float64 iteration counts are enforced by the script. Raw reports are ignored
under builds/parity. Tests are correctness evidence, not performance claims.

Unfinished dense work includes multi-task paths,
MultiTaskLasso, MultiTaskElasticNet and CV. Sparse matrices/targets and sparse
coefficient conversion remain deferred under session-plan/SPARSITY-SCOPE.md.
Fit parity also checks computed and supplied Gram matrices, weighted and
zero-target fits, cyclic and seeded random selection, warm refits,
iteration limits, and the tall and wide fallbacks for invalidated Gram
matrices. Gram and cross products reuse the family's widened,
stride-aware reductions. NuMojo matrix multiplication would require a
separately allocated transposed design for these products. Numax
`dot` and `nrm2` require compile-time vector extents. The squared target
norm accumulates directly in Float64 before rounding to the solver dtype.
Path tests cover generated and supplied grids, positive constraints,
degenerate inputs, seeded initial coefficients, strided inputs and iteration
limits in both dtypes. The residual path shares a single column layout and
set of norms across alpha values. The Gram path reuses the validated Gram
matrix, feature–target products and squared target norm across alpha values.
Numax `geomspace` and `dot` require compile-time extents. The path uses runtime-size NuMojo `logspace` after handling the
single-point case and uses a Float64 correlation reduction on the engine's
packed columns. NuMojo vector `dot` is excluded by the shared numerical policy.
The correlation reduction avoids a transposed matrix or widened data buffer.

The pinned comparison covers 144 residual paths and 300 Gram or automatic
paths, including computed and supplied products. It checks grids at relative
tolerances of 2e-9 (Float64) and 2e-6 (Float32), allowing for the geometric-grid
implementation and BLAS reduction order. Coefficients and dual gaps use the
fit tolerances above, scaled by target magnitude and its square respectively.
Float64 path iteration counts are exact in these cases. Float32 counts can
differ near the convergence threshold and are recorded with the numerical
comparisons.
