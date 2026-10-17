# Dense isotonic scope

Supported exports are `IsotonicRegression`, `isotonic_regression`, and
`check_increasing` in `moxie.isotonic`, based on scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

Dense Float64 and Float32 lists support weighted PAVA, increasing/decreasing
fits, automatic direction, predictor tie consolidation, threshold compression,
linear interpolation, lower/upper bounds and all three out-of-bounds modes.
The estimator accepts 1-D and one-column X and returns 1-D predictions.
Fit, predict, transform, fit_transform, weighted R², clone, parameter updates,
feature names and repeated fitting are covered by native contracts. Failed
refits preserve the previous fitted state, an explicit transactional adaptation.
The default dtype is Float64; use `IsotonicRegression[DType.float32]` for Float32.

Deferred sparse exports/branches: **none**. The reference estimator explicitly
uses `accept_sparse=False` in `sklearn/isotonic.py::IsotonicRegression.fit`.
No sparse container, conversion, solver or automatic densification is exposed.
There are no unsupported sparse defaults or required dense mode selections.
Shared sparse infrastructure remains deferred by the workspace policy.

The List interface and PAVA workspace are retained. The current numerical
policy and compatible Mojo/MAX toolchain apply; this algorithm adds no generic
container or shared numerical operations. Moxie's live assembly supplies
estimator traits, clone and fitted-state validation. Moxie owns
`src/moxie/isotonic.mojo`; no sibling isotonic checkout or historical framework
snapshot is consumed.

Language adaptations include compile-time dtype selection, explicit conversion
of integer/mixed-dtype inputs, same-dtype weight lists, fluent in-place fit and typed
parameter records. Python warnings, reflection, bindings, pandas output,
metadata routing and serialization are separate binding work. Direct `f_` is a
native method; feature-name inputs are accepted and ignored. See [API](api.md)
for validation differences and copy/ownership semantics.

Validation commands from the Moxie root:

```sh
pixi run build
pixi run test
pixi run parity
```

The build compiles the live assembled package. Ownership checks compare
assembled isotonic bytes with the Moxie-owned source. Native
tests cover invalid shapes, empty/nonfinite inputs, weight errors, unfitted
access and failed refits; compile-failure checks reject missing predictor
capability and implicit mixed-dtype conversion. Parity executes the pinned
local Python module and a locally compiled copy of its Cython tie engine:
623 Float64 cases and 456 Float32 cases, including weighted scores. Threshold
counts, X thresholds, domain and direction must match exactly. Value tolerances
are `2e-12` for Float64 and `2e-6` for Float32, allowing pooling/reduction order
and rounding differences. Python and SciPy are validation tools only.

The reproducible performance driver is `tools/isotonic_benchmark.py`; generated timing,
environment and peak-RSS evidence stays in ignored `benchmarks/`. This scope
does not claim full Python API compatibility or performance superiority.
