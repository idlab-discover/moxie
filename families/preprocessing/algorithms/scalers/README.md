# Preprocessing scalers

Owns the scaler acceptance gates for the family's `StandardScaler`,
`MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `scale`, `minmax_scale`,
`maxabs_scale` and `robust_scale` (`src/moxie/preprocessing/_scalers.mojo`).
The family package assembles the public exports; the
[API page](../../docs/api/scalers.md) documents them. See [scope](docs/scope.md)
for required behavior and sparse exclusions, and [memory](docs/memory.md) for
the allocation contract.

`pixi run build` and `pixi run test` validate the live statistics/transform
assembly and the shared-statistics regressions (raw-batch information loss,
signed-weight counts, input-precision statistics, pairwise unweighted sums).
`pixi run parity` reruns those with `--require-parity` and then runs the family's
1407-check scaler parity against pinned sklearn. `pixi run memory` runs the
family's peak-RSS acceptance of every public operation.
