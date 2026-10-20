# Dense smooth-engine scope

Status: **blocked, not implemented**. This project exports no loss or optimizer.
It currently provides a locked dependency assembly and reproducible numerical
prerequisite probes. No estimator default or solver selection is supported yet.
The build and test tasks check this preparation only; parity fails explicitly.

The intended dense engine serves logistic, GLM, ridge and Huber consumers:
weighted loss/gradient/Hessian products, unpenalized intercepts, output-fast
packed coefficients, optimizer state, line search and convergence diagnostics.
Implementation and numerical/behavioral parity remain pending, including the
reference's dense solver branches; they are not excluded from acceptance.

Dense Newton direct solves need suitable singular and ill-conditioned
system diagnostics. Moxie now supplies shared linalg contracts, and the existing
linear-model numerical dependency provides factorizations and iterative solves.
The old missing-owner diagnosis is stale; choosing and integrating the correct
symmetric solve remains future engine work. The pinned reference's
`newton-cholesky` calls SciPy's symmetric solve, so its pivoting and diagnostic
behavior must be preserved. NuMojo's general solve remains excluded.

Sparse exclusions: CSR/CSC/COO input/target/output, sparse sandwich products,
sparse Hessian products and sparse solver dispatch are deferred by the workspace
scope decision. No sparse typed API or implicit densification is provided.
Reference: sklearn `_linear_loss.py`, `_glm/_newton_solver.py`, `_sag.py` at
4358542a1bd39af2e7e1cbace8b9fed067473c5b. Inherited advanced factorization limits
are recorded in the pinned `moxie/shared/numerics/docs/numerics.md`.

Commands:

- `pixi run build`: assemble current dependencies, compile assembled Moxie and the
  numerical prerequisite probe with Mojo 1.0.0.
- `pixi run test`: execute the pinned local sklearn Newton inner solve (AST
  extraction; no wheel substitution), verify its solve/fallback behavior, and
  reproduce NuMojo's excluded solve results in Float32 and Float64.
- `pixi run parity`: compile preparation, then fail explicitly because the
  native engine does not exist. A passing prerequisite probe is not parity.

The negative probe uses duplicate-column Hessians and a zero-pivot symmetric
system. Future engine tests must cover both dtypes, weights, shape/layout,
finite/domain checks, repeated calls, state ownership, initialization, bounds,
iteration limits, nonconvergence and numerical failure. Runtime learning must
remain pure Mojo; Python/SciPy here are validation tooling only.
