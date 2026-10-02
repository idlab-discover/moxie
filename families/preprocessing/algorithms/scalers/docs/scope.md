# Dense scaler scope

All eight exports are implemented in the family package
(`src/moxie/preprocessing/_scalers.mojo`, documented in
[docs/api/scalers.md](../../../docs/api/scalers.md)). This child keeps the
shared-engine regressions and runs the family's parity and memory gates.

## Required dense behavior

The assignment includes `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`,
`RobustScaler`, `scale`, `minmax_scale`, `maxabs_scale`, and `robust_scale` on
Float32/Float64 NuMojo storage. Required methods include fit, transform, inverse,
clone/configuration and feature names; partial_fit for standard/minmax/maxabs;
standard-scaler sample weights; constants, NaNs, clipping, robust quantiles and
unit variance. Functional helpers require both axes and vector shape preservation.
Explicit in-place, caller-output and bounded sink APIs and their memory gates
are required by [memory.md](memory.md). In-place methods are implemented;
caller-output and sink scaler methods are not yet exposed.

## Dependencies and current limitation

The manifest composes the live statistics and transform modules with their
foundation dependencies. `_moments.MomentsStateV1`
receives raw batches and resolves the historical summary information-loss issue;
the scaffold's old FeatureMoments interface remains unchanged.

The consumer regression covers a signed-weight reduction mismatch. For one
previous zero observation and an eight-row batch with weights
`[1e16,1,-1e16,1,0,0,0,0]`, the pinned reference produces updated count 1;
the former serial-count engine produced 2 and halved the mean. The corrected
statistics owner now matches the reference count, mean and NaN variance in both
dtypes at tile sizes 1, 3 and 256.
`tests/owner_update_parity.py` executes the original pinned sklearn update body
and the compiled, hash-verified native owner. It is a positive regression required
before scaler implementation resumes; statistical kernels remain in their owner.

Functional `scale` uses NumPy's input-precision `nanmean`/`nanstd`, unlike
StandardScaler's widened, corrected moments. The statistics engine supplies the
distinct `nanmean_at` and `nanmean_std_at` APIs. The positive consumer regression
in `tests/functional_statistics.py` verifies the historical Float32 permutation
collision and Float64 controls, including exact standard deviations, at all
three tile sizes. This closes that capability gap; it implements no public helper.

Unweighted moments were blocked on reduction overflow semantics, now resolved
by the owner's NumPy-order pairwise sums. For the
finite Float64 column `[1e308,0,1e308,-1e308,-1e308,0,0,0]`, the pinned reference
produces sum and mean zero. The owner's serial sum overflows to infinity and
its mean was infinity. Centering with that mean converts finite values to
infinities, whereas reference mean-only scaling preserves this input. With
variance enabled, the reference variance is infinity and the owner variance is
NaN. This is a mismatch of exceptional-value classes, not low-bit rounding.

`tests/unweighted_reduction.py` executes the original pinned sklearn update body
and compiled owner composition. It reproduces the mismatch with and without
variance at tile sizes 1, 3 and 256; corresponding magnitude-1e100 controls pass.
It now passes with `--require-parity`; the `overflow` dataset in the family's
`tests/parity/test_scalers.py` covers the public estimator. The input-precision
API is separate and does not replace the widened, corrected moments contract.

Shared storage is restricted to Float32/Float64. Python containers, Float16,
integer-array conversion, pandas output, metadata routing and callbacks require
separate binding/storage work. No implicit conversions are supplied here.

## Sparse exclusions

CSR/CSC/COO input/output, sparse moment reductions and sparse in-place scaling
are deferred by the dense-only session policy. Source branches are in
`sklearn/preprocessing/_data.py` at revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`: StandardScaler, MaxAbsScaler,
RobustScaler and corresponding helpers. MinMaxScaler rejects sparse input.
No requested estimator has a sparse default requiring an explicit dense choice.
Sparse types are unavailable in the accepted family API; the shared
`require_dense_output(True)` negative contract raises.

## Checks

Run with `PIXI_CACHE_DIR="$PWD/.pixi/pixi-cache"`:

```sh
pixi run build
pixi run test
pixi run parity
pixi run memory
```

`build` reads live sources and compiles the assembled dependencies and local native
probes. `test` checks shared storage independence, sparse-output rejection and
the four historical owner regressions. `parity` requires those to match the
pinned reference and runs the family's scaler parity (1407 checks). `memory`
runs the family's peak-RSS acceptance described in [memory.md](memory.md). Both
delegate through the family's Pixi environment, which carries the sklearn oracle.
Raw check logs stay in ignored `builds/`.

Check results and exact commands are recorded under ignored
`docs/private/scalers/checks.json` with logs under `builds/`. Family engines and consumers can be changed together under the development policy.
