# Isotonic regression

Moxie owns `IsotonicRegression`, `isotonic_regression`, and `check_increasing`
in `src/moxie/isotonic.mojo`. Import them from `moxie.isotonic`.
The estimator accepts Float32 and Float64 NDArrays and supports one-column
pipelines. Concrete List overloads and List thresholds are also available.

See [API and adaptations](api.md), [supported scope](scope.md), and
[license notices](THIRD_PARTY_NOTICES). The [example](../../examples/isotonic.mojo)
runs as `builds/examples/isotonic` after a root build.

Root `pixi run build`, `pixi run test`, and `pixi run parity` include isotonic.
Focused checks use `pixi run python tools/isotonic_checks.py test` or `parity`.
Tests in `tests/isotonic/` cover numerical behavior, lifecycle, fluent concrete
and generic calls, dtype preservation, and rejected mixed-dtype inputs.
`test_reference_edges.mojo` pins hand-derived reference values for pooling,
crossed bounds, negative score weights, tied-rank direction, refit after
`set_params` and the documented `check_increasing` validation difference.
The root shared-contract tests cover missing prediction capabilities.
An isolated consumer assembles the current Moxie manifest and exercises all
three exports and both estimator dtypes. Ownership checks verify the assembled
module matches the Moxie source.

The parity runner executes the clean local scikit-learn source at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`, including its compiled Cython tie
engine. The installed scikit-learn wheel supplies validation infrastructure.
It generates 624 Float64 and 457 Float32 cases, including scaler pipeline
predictions. The respective tolerances are `2e-12` and `2e-6`. Threshold
counts, X thresholds, domain, direction and NaN locations must match exactly. Evidence is written to
ignored `builds/isotonic/parity/`. This is bounded core behavior coverage,
not full Python API parity.

The estimator uses the existing List-based solver. Calibration uses its List
overloads for isotonic calibration.
