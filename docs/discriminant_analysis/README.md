# Discriminant analysis

Moxie owns `LinearDiscriminantAnalysis` and `QuadraticDiscriminantAnalysis` in
`src/moxie/discriminant_analysis.mojo`, `_discriminant_analysis/`, and
`_discriminant_analysis_contracts.mojo`. Import both from
`moxie.discriminant_analysis`. The original sibling checkout is not required.

See [API and Mojo adaptations](api.md), [tested scope](scope.md), and
[numerical dependencies](numerics.md). The [example](../../examples/discriminant_analysis.mojo)
runs as `builds/examples/discriminant_analysis` after a root build.

Root `pixi run build`, `pixi run test`, and `pixi run parity` include this family.
Focused checks use `pixi run python tools/discriminant_analysis_checks.py test`
or `parity`. Native tests, negative capability probes, and the reference runner
live in `tests/discriminant_analysis/`. Ownership checks compare assembled bytes
against Moxie's sources; an isolated consumer assembles Moxie's live manifest
and fits both public estimators outside the repository.

The oracle executes the clean local scikit-learn checkout at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`; its installed wheel supplies
validation infrastructure only. The existing Float64/Float32 tolerances,
probability comparisons, and eigenvector sign and separation rules remain in
the runner. Reports go to ignored `builds/discriminant_analysis/parity/`.
These checks do not establish full Python API parity.

Mojo 1.0.0 and MAX 26.5.0 remain locked by Moxie's existing Pixi environment.
Scikit-learn notices remain in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) and
`LICENSES/scikit-learn.txt`; assembly includes numerical dependency licenses.
