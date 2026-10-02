## Current family development policy

Read [../../docs/development.md](../../docs/development.md) before development. Preprocessing is integrated
into Moxie. Preserve module ownership, toolchain locks and correctness-reference
revisions; commit changes in the Moxie repository.

# Statistics owner

This directory owns `moxie.preprocessing._moments`. Reuse `numojo.NDArray`, `Shape`,
stride/offset access from the current numerics owner. Reuse foundation
`moxie.base`, `moxie.utils.validation.FittedState`, family `_base`,
`_contracts`, `_validation.validate_dense` and `_metadata.FeatureNames`.
`moxie._dense.sum_axis2d` and `row_squared_norms` remain the shared ordinary
reductions; this owner supplies corrected moments, extrema and exact quantiles.
Do not use NuMojo variance, axis sums, dot, or random. No replacement container.
Use the current foundation assembler and live dependency paths. Validate family
changes and commit atomically under the family development policy.
Run `pixi run build`, `test`, `parity`, and `memory`.
