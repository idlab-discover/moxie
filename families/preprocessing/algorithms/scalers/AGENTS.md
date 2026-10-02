## Current family development policy

Read [../../docs/development.md](../../docs/development.md) before development. Preprocessing is integrated
into Moxie. Preserve module ownership, toolchain locks and correctness-reference
revisions; commit changes in the Moxie repository.

# Scalers child

Read `../../docs/plumbing.md`, `docs/scope.md`, `docs/memory.md`, and the session's
PREPROCESSING-MEMORY.md before implementation. Keep generated output local; coordinate changes across the family as needed.

Consume the live-source `DEPENDENCIES.json` through `tools/check.py` and the
current foundation assembler. Statistics and transform sources are live inputs;
maintain module ownership and exports in DEPENDENCIES.json. Preserve Mojo 1.0.0 / MAX 26.5.0 and `--fp-mode=contract=off`.

Reuse `numojo.NDArray`, `Shape`, `Item` (Float32/64), the numerical owner's
approved operations and `moxie._dense.sum_axis2d` / `row_squared_norms`.
Read `../../../../shared/numerics/docs/numerics.md` for exclusions.
Reuse `moxie.base`, `moxie.utils.validation.FittedState`, preprocessing `_base`,
`_validation`, and `_metadata.FeatureNames` for estimator plumbing.

The authorized statistics owner supplies `_moments.MomentsStateV1`,
`update_moments`, `update_moments_weighted`, `update_moments_scalar`,
`ExtremaStateV1`, `update_extrema`, and `robust_quantiles`.
Its versioned raw-batch update supersedes the old scaffold
`_contracts.FeatureMoments` / `FeatureStatistics.compute/merge` requirement.
The transform owner supplies `_dense_transforms.transform`, `transform_into`,
`transform_inplace`, `transform_sink`, and `TileSink`. Scalers consume these
implementations; shared statistics and transform kernels belong in their family engine modules.

Before resuming estimator implementation, run `tests/owner_update_parity.py`
against the assembled dependencies. Both historical owner blockers now have
positive regressions: raw-batch information retention and signed-weight effective
counts. Retain both when implementing estimators; see `docs/scope.md` for scope.

Functional `scale` must consume the distinct `_moments.nanmean_at` and
`nanmean_std_at` / `InputPrecisionStatsV1` APIs. The live statistics module supplies
this extension. `tests/functional_statistics.py` is a positive
consumer regression; retain it alongside the historical moments regressions.

`tests/unweighted_reduction.py --require-parity` is the positive regression for
the resolved pairwise unweighted-sum blocker (historical notes:
`docs/private/scalers/BLOCKED.md`); keep shared reduction work in its owner.

The eight exports live in Moxie's `src/moxie/preprocessing/_scalers.mojo`.
Run `pixi run build`, `test`, `parity`, and `memory` after changing them or the
engines; parity and memory delegate to the family environment. Keep every
operation within the [memory contract](docs/memory.md): no full-table masks,
casts, broadcast weights, residuals or view copies, and no fitted estimator kept
alive inside a function. Run the memory gate under a memory cap.
