# Validated dense-only scope

This private foundation engine assembles dependencies. It implements no
estimators, numerical solvers or new containers. Supported behavior is version-1
lock/export validation, immutable source verification, a single generated Moxie
namespace, genuine baseline deduplication and explicit module relocation.

Native validation uses the real parent `moxie.base` and
`moxie.utils.validation`, and completed numerics `moxie._dense.row_squared_norms`
and `sum_axis2d` over pinned NuMojo Float32/Float64 rank-2 dense arrays.
Two independent consumer projects test shared implementation identity, native
execution, view/copy ownership, repeated fit, cloning, parameter independence,
unfitted/dimension errors and invalid reduction axes. Their estimator-shaped
objects and dummy replacement marker are contract test doubles only.

Deferred assembler-specific sparse branches: **none**. As inherited dependency
restrictions, `moxie._sparse`, sparse containers/conversions/arithmetic/solvers,
sparse input/output and sparse consumer probes are excluded by
`session-plan/SPARSITY-SCOPE.md`. Existing parent sparse implementations are
unchanged. The generated dense fixture namespace does not expose `_sparse`;
this is checked without treating its absence as a readiness failure.
Numerics' excluded dot/solve/variance/RNG paths are not promoted into supported
operations; see `../numerics/docs/numerics.md`. RNG and advanced dense solvers
belong to other owners and are not used by assembly.

Unsupported estimator defaults or required dense selections: **none**, because
this project has no estimator API. The required assembler paths are explicit.
Version 1 initializers support named exports only, with no aliases or arbitrary
initializer bodies; module implementations remain ordinary Mojo source.

Executed successfully: `pixi run build`, `pixi run test` (28 host-side
contracts plus native probes), `pixi run parity`, and both fixtures’ own
`pixi run --locked --manifest-path tests/fixtures/consumer_{a,b}/pixi.toml test`.
The fixture paths were executed separately.

Validation commands:

- `pixi run build`: complete native package/consumer/control compilation.
- `pixi run test`: source/revision drift, missing/unpinned files, traversal and
  symlink escape, module/export/asset collisions, schema validation, override
  failures, baseline provenance, stale output removal and native contracts.
- `pixi run parity`: exact assembled-versus-direct native behavior for both
  dense dtypes, plus original parent validation behavior.

Completion means this dense-only assembly scope. It makes no claim of estimator
parity, sparse support, or full scikit-learn coverage.
