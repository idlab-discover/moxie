# Dense LatentDirichletAllocation

`LatentDirichletAllocation[DType.float32|float64]` learns topic-word Dirichlet
parameters from finite, nonnegative dense document-term counts. Float64 is the
default. `fit` supports batch and online variational Bayes; `partial_fit` makes
online updates and retains its RNG and batch counter. Ordinary `fit` resets state.

```mojo
from moxie.decomposition import LatentDirichletAllocation
var model = LatentDirichletAllocation(3, random_state=0)
var topics = model.fit_transform(counts)
var unnormalized = model.transform(counts, normalize=False)
var bound = model.score(counts)
var perplexity = model.perplexity(counts)
```

Run [the self-contained example](../examples/lda.mojo) with
`MOXIE_ESTIMATORS=lda pixi run test`. Run differential checks with
`MOXIE_ESTIMATORS=lda pixi run parity`.

Constructor names/defaults follow the local sklearn reference
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Learned arrays `components_` and
`exp_dirichlet_component_` are optional owned arrays of shape (topics, vocabulary).
`components_` is not a normalized probability matrix. Transform outputs
(documents, topics), normalized by default. `score` returns the variational
bound; `bound_` stores final training perplexity. `n_iter_`, `n_batch_iter_`,
`doc_topic_prior_`, `topic_word_prior_` and `n_features_in_` follow the reference.
No sample weights or inverse transform are part of this estimator.

Mojo adaptations: fit/partial_fit/set_params mutate and return a mutable reference to self;
get_params returns a typed record and clone is unfitted. The record supports
all settings; named set_params updates support the common iteration, component,
batch, method and seed settings. Random state accepts an integer or None;
None uses the shared entropy context. Arrays preserve the selected compile-time
dtype and accept strided inputs, copied to owned contiguous working arrays.
Inference rejects feature-count mismatches without resetting feature metadata
(including perplexity, unlike the reference's reset during validation).

Only `n_jobs=None` or `1` is supported. Other values raise explicitly: sklearn's
parallel document partitioning changes random initialization. Verbose is
accepted but does not print progress. Python callbacks, metadata routing,
non-native output containers and sparse counts are unsupported.

The E-step uses the reference's AS 103 digamma approximation, not a general
special-function replacement; scalar exponentials/log-gamma use Mojo's standard
library. No Python executes during learning or inference. Parity checks use
pinned `_lda.py` and compile its pinned `_online_lda_fast.pyx` for the oracle.
Float64 tolerance is 1e-7; Float32 is 2e-3, allowing rounding differences through
iterative variational updates and tolerance-dependent stopping. Array comparisons
use that relative tolerance and a maximum-value-scaled absolute tolerance;
iteration counters are exact. Contract tests separately cover unfitted calls,
refits, clone, parameters, negative/nonfinite/empty inputs and reversed strides.
