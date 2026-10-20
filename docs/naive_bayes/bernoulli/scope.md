# BernoulliNB dense scope

The canonical implementation lives in `src/moxie/_nb_bernoulli.mojo` and is
exported by `moxie.naive_bayes`. See [the API](../family.md) for List and NDArray
inputs, numeric/string labels, weights, incremental fitting, binarization,
scalar/per-feature alpha, cloning and independent copies.

Sparse input and Python-specific facilities remain deferred. Per-feature alpha
uses the corrected feature-wise denominator. The pinned sklearn vector path has
a broadcasting bug; `tests/naive_bayes/check_sklearn.py` retains independent
scalar-fit composition comparisons and explicit probes of that bug.

Native coverage includes `test_bernoulli_arrays.mojo`, the shared label/input
contracts and `bernoulli/dependency_contract.mojo`. Run the root checks or the
focused runner described in [validation](../README.md).
