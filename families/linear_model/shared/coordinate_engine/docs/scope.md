# Coordinate engine scope

Implemented: dense Float32/Float64 cyclic elastic-net residual updates, L1/L2
penalties, positivity, Gap Safe screening, initial and per-iteration dual-gap
stopping, iteration counts and convergence status. Owns _cd_fast.mojo;
ElasticNet/Lasso consume it with shared family preprocessing. X is materialized
once as contiguous feature columns; residual and correlation scratch are O(n+p).
Coefficients are independent owned arrays; inputs and initial state are preserved.

`pixi run build`, `pixi run test`, `pixi run parity` now validate an actual
solver. Native tests include exact tiny solution, zero-iteration warm solve,
invalid penalties/shapes and both dtypes. Raw parity compiles the pinned local
sklearn _cd_fast.pyx and compares coefficients, dual gaps and Float64 iteration
counts. Float32 tolerance is 3e-4; Float64 is 1e-8, scaled by target norm.
The elastic_net child separately exercises 264 public estimator cases.

Unfinished dense work: random selection (shared our_rand_r integration pending),
Gram solver, path continuation and multi-task group updates. The old RNG probes
and private handoffs are retained as evidence; they are not solver tests.
No local RNG substitutes for the common primitive. Sparse coordinate branches
remain deferred by session-plan/SPARSITY-SCOPE.md; no implicit densification.
Reference revision: 4358542a1bd39af2e7e1cbace8b9fed067473c5b.
