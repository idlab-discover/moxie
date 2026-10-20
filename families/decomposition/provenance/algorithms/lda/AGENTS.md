## Current family development policy

Read [../../docs/development.md](../../docs/development.md) before development. The whole family repository
is the ownership boundary. That policy supersedes conflicting child-only,
read-only family-helper, separate-session, local-repinning and no-commit rules
below. Preserve external dependency ownership and all correctness requirements.

# LDA

Read ../../docs/plumbing.md and ../../../session-plan/SESSION.md before changes.
Reuse pinned `moxie.base`, `moxie.utils.validation.FittedState`,
`moxie.decomposition._base`, `moxie.decomposition._contracts`,
`moxie._dense.sum_axis2d`, `moxie._dense.row_squared_norms`, and
`moxie._random.RandomState`. Arrays use NuMojo `NDArray`, `Shape`, and `Item`.
Read ../../../moxie/shared/numerics/docs/numerics.md before numerical work.
LDA variational updates belong here; RNG distributions belong in
../../../moxie/shared/random. Do not implement a local Gamma sampler.
Use the pinned assembly launcher; never update dependencies during checks.
Keep all writes in this project and leave changes uncommitted.

Run `pixi run reproduce` for the current Gamma capability probe.
Build/test/parity must fail until a real estimator and its checks exist.
