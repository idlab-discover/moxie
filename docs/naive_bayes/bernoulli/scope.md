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

## Native model files

```mojo
from moxie.naive_bayes_serialization import BernoulliNBSchema
from moxie.serialization import load_model, save_model

save_model[BernoulliNBSchema[]](model, "bernoulli.moxie")
var restored = load_model[BernoulliNBSchema[]]("bernoulli.moxie")
```

The experimental schema covers fitted and unfitted models. Choose
the same built-in label domain and declared input dtype when saving and loading.
The file stores constructor configuration, including the optional threshold
and scalar or per-feature smoothing, separately from learned arrays and the
most recent fit warning. It also preserves configured
[sample-weight requests](../family.md#sample-weight-requests). Vector smoothing
is saved in logical order and loaded into owned contiguous storage. Signed counts and nonfinite feature log
probabilities from raw, unbinarized inputs are preserved. A fitted restore can
predict and resume weighted `partial_fit` without passing `classes` again.

After assembling the current dependencies, run
`python tests/naive_bayes/check_sklearn.py --round-trip-bernoulli --batch-size 12`
in the pinned reference environment. The check requires byte-identical
archives after each restore and compares learned arrays and predictions with
scikit-learn. The Float64 tolerance is `1e-7 + 1e-7 * abs(expected)` for
floating-point differences in accumulation and logarithms. Class order and
labels are exact, and NaN and infinity positions must match.
