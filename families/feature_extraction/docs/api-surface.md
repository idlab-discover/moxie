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
