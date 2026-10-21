# Dense Naive Bayes scope

All five public estimators live in Moxie core. They preserve List and NDArray
inputs, boolean/integer/floating feature dtypes, numeric/string label domains,
weighted fitting, incremental updates and probability prediction. See
[API coverage](api-surface.md) and [behavioral differences](family.md).

Sparse input remains deferred. Gaussian and categorical validation is dense in
the reference; multinomial, complement and Bernoulli sparse counting paths are
not implemented. No sparse input is silently densified. Python reflection,
pandas output, metadata routing, bindings and serialization are outside this
integration. CategoricalNB takes scalar alpha only, like the reference.

Run the root build, test and parity commands or the focused runner described in
[README](README.md). Compilation alone does not establish sklearn parity.
