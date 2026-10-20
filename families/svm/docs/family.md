# svm family interfaces

Moxie assembles the dense linear and kernel estimators. See [scope.md](scope.md) for implementation status.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](interfaces.md) for typed parameter records,
in-place fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `SupportVectorClassifier` | `ClassifierMixin, DecisionFunction` | Inherited only |
| `ProbabilisticSVC` | `SupportVectorClassifier, LogProbabilityPredictor` | Inherited only |
| `SupportVectorRegressor` | `RegressorMixin` | Inherited only |
| `OneClassSupportVector` | `OutlierMixin, SampleScorer` | Inherited only |

## Behavioral distinctions

SVC/NuSVC probability requires probability=True. LinearSVC has no predict_proba. Kernel and linear solvers remain separate implementations; support_, support_vectors_, dual_coef_, coef_ and intercept_ are estimator-dependent.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../../../../scikit-learn/sklearn/svm`.
- Guide and mathematical references: `../../../../scikit-learn/doc/modules/svm.rst`.
- Shared contracts: `../../../../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`BaseLibSVM`, `BaseSVC`, `LinearSVC`, `LinearSVR`, `SVC`, `NuSVC`, `SVR`, `NuSVR`, `OneClassSVM`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.

## Dense plumbing

Use [plumbing.md](plumbing.md) for shared ownership, layouts, engine extension
points and verified dependency assembly. [estimators.md](estimators.md) records
the exact reference defaults and dense acceptance obligations.
[scope.md](scope.md) explicitly defers sparse branches. All seven estimators and l1_min_c are implemented public exports.
