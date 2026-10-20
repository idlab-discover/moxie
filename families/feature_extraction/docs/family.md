# feature_extraction family interfaces

This family owns dense dictionary and image implementations plus explicit deferred declarations.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](interfaces.md) for typed parameter records,
fluent mutable-borrow fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `FeatureExtractor` | `UnsupervisedEstimator, TransformerMixin, FitTransformer` | Inherited only |

## Behavioral distinctions

Input may be documents, mappings or images, not a numeric matrix. text.CountVectorizer/TfidfVectorizer/HashingVectorizer belong in the nested text module. Sparse outputs and vocabulary_ need their own concrete types.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. The dictionary and image API documents specify their constructors, learned state
and overloads. Deferred text and hashing declarations do not imply runtime support.

## Reference sources

- Source: `../../../../scikit-learn/sklearn/feature_extraction`.
- Guide and mathematical references: `../../../../scikit-learn/doc/modules/feature_extraction.rst`.
- Shared contracts: `../../../../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes

`HashingVectorizer`, `CountVectorizer`, `TfidfTransformer`, `TfidfVectorizer`, `DictVectorizer`, `PatchExtractor`, `FeatureHasher`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.

## Dense implementation and extension

The accepted scope and exclusions are in [scope.md](scope.md). Shared ownership,
exact helper signatures, layouts, child responsibilities and verified dependency
assembly are defined in [plumbing.md](plumbing.md). Sparse output descriptions
above are historical reference context, not implemented capabilities.
