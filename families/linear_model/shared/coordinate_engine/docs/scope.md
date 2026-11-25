# Coordinate engine scope

The dense Float32/Float64 residual and Gram solvers support cyclic and random
elastic-net updates, L1/L2 penalties, positivity, Gap Safe screening, initial and per-iteration dual-gap
stopping, iteration counts and convergence status. Owns _cd_fast.mojo;
ElasticNet/Lasso consume it with shared family preprocessing. The residual
solver materializes X once as contiguous feature columns and uses O(n+p)
residual and correlation scratch.
Coefficients are independent owned arrays; inputs and initial state are preserved.

`pixi run build`, `pixi run test`, `pixi run parity` now validate an actual
solver. Native tests include exact tiny solution, zero-iteration warm solve,
invalid penalties/shapes and both dtypes. Raw parity compiles the pinned local
sklearn _cd_fast.pyx and compares coefficients, dual gaps and Float64 iteration
counts. Float32 tolerance is 3e-4; Float64 is 1e-8, scaled by target norm.
The elastic_net child checks public estimators and paths, including computed
and supplied Gram matrices in fits and seeded random selection.

Unfinished dense work includes Gram paths and multi-task group updates. The old
RNG probes and private handoffs are retained as evidence.
No local RNG substitutes for the common primitive. Sparse coordinate branches
remain deferred by session-plan/SPARSITY-SCOPE.md; no implicit densification.
Reference revision: 4358542a1bd39af2e7e1cbace8b9fed067473c5b.
