# Isotonic regression

Moxie owns `IsotonicRegression`, `isotonic_regression`, and `check_increasing`
in `src/moxie/isotonic.mojo`. Import them from `moxie.isotonic`.
The retired sibling checkout is not required. The implementation retains
Float32 and Float64 lists, including one-column estimator inputs.

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
It generates 623 Float64 and 456 Float32 cases. Outputs use tolerances of
`2e-12` and `2e-6` respectively; threshold counts, X thresholds, domain,
direction and NaN locations must match exactly. Evidence is written to
ignored `builds/isotonic/parity/`. This is bounded core behavior coverage,
not full Python API parity.

Moxie's existing Mojo 1.0.0 and MAX 26.5.0 lock remains unchanged.
List storage and the existing solver are retained; this integration adds no
NDArray interface or serialization. Calibration's independent isotonic
implementation remains outside this integration.
