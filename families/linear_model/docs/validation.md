# Validation

Run these commands from the Moxie root with its locked environment:

```sh
pixi run linear-model-build
pixi run linear-model-test
pixi run linear-model-parity
pixi run linear-model-start
```

The focused commands assemble the complete Moxie namespace, verify that Moxie
owns every linear-model implementation module, compare the private numerical
modules with their external owner, and check the retained license asset.
They precompile the package and execute public estimator examples plus fluent
trait consumers. The family checker then uses isolated dependency trees for
its contracts and algorithm suites. A successful compile alone is not parity.

The test command covers shared capabilities, shape/weight validation, snapshots,
preprocessing, prediction, R2, concrete and generic fluent mutation, clone and
repeat-fit behavior. Algorithm contracts add strided inputs, weights, solver
selection, failure-state preservation, uncertainty and unsupported-mode errors.
Two independent namespace consumers check ownership, separate allocations,
missing-source rejection and module/export collision rejection. Invalid
Predictor conformance must fail compilation for the expected reason.

The parity command executes the original local scikit-learn source at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. It verifies the revision and source
cleanliness before loading estimator modules. The installed sklearn wheel
supplies support utilities, not the estimator implementations under comparison.
Coordinate tests compile the reference Cython engine from that same checkout.

| Suite | Comparisons |
| --- | --- |
| least_squares, 38 cases | Coefficients, intercepts, rank, singular values, predictions and scores |
| ridge, 149 cases | Ridge, RidgeClassifier and ridge_regression; direct, iterative and positive solvers |
| elastic_net, 264 cases | ElasticNet/Lasso cyclic updates, weights, warm starts, dual gaps and stopping |
| bayesian, 86 cases | Coefficients, precisions, covariance, evidence, means and predictive deviations |
| coordinate_engine, 60 cases | Raw cyclic coefficients, dual gaps and iteration counts |

All numerical suites cover Float32 and Float64. The scripts and child API docs
state tolerances for widened arithmetic and factorization differences. Reports,
fixtures, compiler output and Cython oracle binaries stay in ignored `builds/`
directories. Unsupported estimators and solver branches are listed in
[coverage.md](coverage.md); these checks do not establish full sklearn parity.

After a successful build, root `python tools/linear_model_checks.py test-only`
and `parity-only` reuse the compiled artifacts. The root aggregate tasks use
these modes after their build prerequisite. For a fresh full-namespace check
without running another family's suite, use:

```sh
pixi run python tools/linear_model_checks.py namespace
```

All live source paths point to Moxie or declared numerical dependencies.
The standalone `moxie-linear_model` directory is not required.
