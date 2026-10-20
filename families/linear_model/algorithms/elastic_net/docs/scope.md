# Dense cyclic estimator scope

Implemented: ElasticNet and Lasso with the modes and typed adaptations in
[api.md](api.md). Uses family preprocessing/prediction and the single cyclic
coordinate engine at shared/coordinate_engine; no duplicated solver.

Validation: `pixi run build`, `pixi run test`, `pixi run parity`. Native tests
cover both dtypes, unfitted calls, clone/params, refits, warm starts, weight
normalization, scalar weights, shape changes, failed fits, positive constraints,
negative strides and explicit unsupported modes. Differential tests compile
and execute the actual local sklearn _cd_fast.pyx as well as _base.py and
_coordinate_descent.py at 4358542a1bd39af2e7e1cbace8b9fed067473c5b.
264 cases cover tall/wide/tiny/constant/multiple outputs, l1_ratio 0/0.5/1,
weights including zeros, positivity, no intercept, warm starts, one iteration,
zero targets, large alpha, and column targets. Float64 tolerance 1e-8,
Float32 3e-4 for reduction order; target-scaled absolute tolerances and exact
Float64 iteration counts are enforced by the script. Raw reports are ignored
under builds/parity. Tests are correctness evidence, not performance claims.

Unfinished dense: random selection, Gram precomputation, paths, MultiTaskLasso,
MultiTaskElasticNet and CV. Sparse matrices/targets and sparse coefficient
conversion remain deferred under session-plan/SPARSITY-SCOPE.md. Default cyclic
residual fitting works; precompute=True and selection="random" raise explicitly.
The family queue remains PARTIAL.
