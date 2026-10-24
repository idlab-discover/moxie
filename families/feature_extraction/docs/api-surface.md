# Public feature-extraction API

The assembled `moxie.feature_extraction.DictVectorizer` resolves to the native
implementation in `algorithms/dict_hash`, not the former declaration.
The image module resolves to `algorithms/image`. Graph helpers are also exported
from the family root. See the [dictionary API](../algorithms/dict_hash/docs/api.md)
and [image API](../algorithms/image/docs/api.md) for typed parameters, defaults,
learned state, array types and fluent fit/set_params behavior.

`FeatureExtractor` remains the shared optional transformation contract.
`FeatureHasher` and the text vectorizers/TF-IDF classes remain explicit raising
declarations. Deferred helper calls also raise. Their native `APIValue` and
`APIExpressionDefault` markers preserve source signatures without pretending
to implement Python objects or callable defaults. `text.ENGLISH_STOP_WORDS`
is a native immutable tuple, not a Python frozenset.

The focused build precompiles the assembled namespace. Native tests verify
implemented behavior and rejected deferred constructors; numerical parity covers
only the supported dictionary and image modes.

## Composition

`DictVectorizer(sparse=False)` is a `Pipeline` transformer in front of
estimators that take `NDArray` input (for example `RidgeClassifier`); nested
`set_params`, refit and `moxie.base.clone` of such pipelines are covered by
`tests/test_composition.mojo` with sklearn-derived expectations. Its typed
`FeatureBatch` input cannot be a column of `moxie.compose.ColumnTransformer`,
which selects `NDArray` columns. Cross-validation and search over such
pipelines need typed model_selection adapters that do not ship yet
(model_selection tracking PR #3 parts 1 and 3). `PatchExtractor` returns rank-3
or rank-4 patches, so like sklearn it is not followed directly by a 2-D
estimator. The raising declarations implement no estimator trait.
