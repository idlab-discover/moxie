# manifold family interfaces

Moxie owns ClassicalMDS, trustworthiness and the manifold capability contracts.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](../../../docs/interfaces.md) for typed parameter records,
in-place fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `EmbeddingEstimator` | `UnsupervisedEstimator, FitTransformer` | Inherited only |
| `OutOfSampleEmbedding` | `EmbeddingEstimator, TransformerMixin` | Inherited only |

## Behavioral distinctions

TSNE/MDS/SpectralEmbedding generally only fit_transform. Isomap and LLE can transform unseen samples. Preserve precomputed-distance input interpretation and embedding_ semantics.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../../../../scikit-learn/sklearn/manifold`.
- Guide and mathematical references: `../../../../scikit-learn/doc/modules/manifold.rst`.
- Shared contracts: `../../../../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`Isomap`, `SpectralEmbedding`, `TSNE`, `MDS`, `ClassicalMDS`, `LocallyLinearEmbedding`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.

## Dense plumbing

The extension and ownership contract is [plumbing.md](plumbing.md); the authorized
exclusions and required explicit dense settings are [scope.md](scope.md).
The verified public reference roster is the ten exports in sklearn.manifold's
`__all__`, including ClassicalMDS, smacof, locally_linear_embedding,
spectral_embedding and trustworthiness. ClassicalMDS and trustworthiness are
implemented; the remaining algorithm exports raise explicit unsupported errors.
