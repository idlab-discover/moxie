# neural_network family interfaces

This family owns the implemented RBM subset, private MLP parameter helpers and
capability contracts. See [current scope](scope.md).
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](interfaces.md) for typed parameter records,
in-place fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `MLPClassifierAPI` | `IncrementalClassifier, LogProbabilityPredictor` | Inherited only |
| `MLPRegressorAPI` | `IncrementalRegressor` | Inherited only |
| `RBMTransformer` | `UnsupervisedEstimator, TransformerMixin, FitTransformer, SampleScorer` | Inherited only |

## Behavioral distinctions

MLP partial_fit requires a stochastic solver; early stopping and optimizer state need explicit handling. BernoulliRBM is an unsupervised transformer, not a classifier. coefs_, intercepts_, losses and iteration counts are learned state.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../scikit-learn/sklearn/neural_network`.
- Guide and mathematical references: `../scikit-learn/doc/modules/neural_networks_supervised.rst`.
- Shared contracts: `../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`BernoulliRBM`, `BaseMultilayerPerceptron`, `MLPClassifier`, `MLPRegressor`, `BaseOptimizer`, `SGDOptimizer`, `AdamOptimizer`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.

## Dense shared plumbing

Read [plumbing.md](plumbing.md) for exact shared array layouts, ownership,
extension signatures, the three future owners and verified dependency assembly.
[scope.md](scope.md) records the dense-only contract and sparse exclusions.
The RBM child supplies the concrete estimator. MLP training remains unfinished.
