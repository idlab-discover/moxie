# decomposition family interfaces

This repository preserves the original capability scaffold and integrates dense
estimator implementations. See [scope.md](scope.md) for current coverage.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](interfaces.md) for typed parameter records,
in-place fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `DecompositionTransformer` | `UnsupervisedEstimator, TransformerMixin, FitTransformer` | Inherited only |
| `InvertibleDecomposition` | `DecompositionTransformer, InverseTransformer` | Inherited only |
| `IncrementalDecomposition` | `IncrementalTransformer` | Inherited only |

## Behavioral distinctions

PCA/SVD, NMF, ICA and dictionary/topic models share transformations but not constraints or inverse transforms. SparseCoder may be stateless; supervised dictionary variants and sample scoring are algorithm-specific.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../../../../scikit-learn/sklearn/decomposition`.
- Guide and mathematical references: `../../../../scikit-learn/doc/modules/decomposition.rst`.
- Shared contracts: `../../../../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`FastICA`, `SparsePCA`, `MiniBatchSparsePCA`, `PCA`, `KernelPCA`, `LatentDirichletAllocation`, `FactorAnalysis`, `IncrementalPCA`, `SparseCoder`, `DictionaryLearning`, `MiniBatchDictionaryLearning`, `TruncatedSVD`, `NMF`, `MiniBatchNMF`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.

## Dense scaffold integration

Read [plumbing.md](plumbing.md) for exact layouts, helper signatures, engine
boundaries, ownership and assembly requirements. [scope.md](scope.md) defines
the accepted dense-only scope and sparse deferrals. The current implemented modes and remaining raising declarations are listed in
[scope.md](scope.md); reserved names alone do not establish implementation.
