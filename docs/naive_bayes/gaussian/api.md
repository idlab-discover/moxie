# GaussianNB

The public `moxie.naive_bayes.GaussianNB` is implemented in
`moxie._nb_gaussian`. It accepts boolean, integer and Float32/Float64 NuMojo
matrices, including
strided and offset views, and the existing Float64 nested Lists.

```mojo
from moxie.naive_bayes import GaussianNB, StringDomain
from moxie._labels import StringTarget
import numojo as nm

var X = nm.ones[DType.float32](nm.Shape(4, 2))
var y = StringTarget(["cat", "dog", "cat", "dog"])
var model = GaussianNB[StringDomain, dtype=DType.float32]()
model.fit(X, y)
var predictions = model.predict(X)  # StringTarget
var probabilities = model.predict_proba(X)  # Float32 NDArray, (4, 2)
```

`GaussianNB()` defaults to List[Int] targets. For numeric NDArray targets use
`GaussianNB[NumericDomain[DType.int64]]()`, or Float32/Float64 integral labels.
The domain is a compile-time choice; predictions and `classes_` retain that
target representation. String classes use StringTarget. Classes are sorted;
probability columns follow their order. Annotate the default concrete type as
`GaussianNB[]`; a bare generic annotation does not fix its label domain.

Constructor defaults remain `priors=None`, `var_smoothing=1e-9`. Priors are
optional Float64 Lists. Fit and partial_fit sample weights are Float64 scalars
or Float64 Lists; score weights remain Lists. `get_params(deep=True)` returns
GaussianNBParams; named `set_params` updates and unfitted clone are supported.
Regular fit replaces state. The first partial_fit requires classes; later calls
accept omitted classes or the same sorted class set. Failed fits leave the old
model intact. Fitted copies and parameter records own independent buffers.

Learned `theta_` and `var_` are NDArrays in the selected dtype with shape
`(classes, features)`; `class_count_` and `class_prior_` are vectors in that
dtype. Read scalars with `.item`. `epsilon_` is a scalar in that dtype and
`n_features_in_` is Int. Unfitted learned arrays are empty; inference raises a descriptive NotFittedError.

NDArray probability methods return `(samples, classes)` NDArrays in that dtype;
List input retains nested-List probability output. `predict`, `predict_proba`,
`predict_log_proba`, `predict_joint_log_proba` and weighted/unweighted accuracy
score all accept either feature representation. Native shared estimator traits
retain List feature input as their associated Input; array overloads are concrete
methods, not a new global array trait contract.

The compile-time `dtype` parameter selects `DType.float32` or `DType.float64`
(default). Use `GaussianNB[dtype=DType.float32]()` for List[Int] labels or
`GaussianNB[StringDomain, dtype=DType.float32]()` for string labels. Means,
variances, counts, priors, epsilon, moment arithmetic and array probabilities
use this dtype. Features and sample weights convert to it before arithmetic;
values that become nonfinite on conversion are rejected. Float32 weight totals
must remain positive and finite. Float32 scalar exp/log use Numax's accurate
Float64 libm evaluation rounded back to Float32 before subsequent arithmetic.

Unlike sklearn, dtype is explicit and cannot change on fit or partial_fit.
For same-dtype parity, select the dtype of the reference training and prediction
arrays. A default Float64 model still widens Float32 inputs; a Float32 model
also converts Float64, integer and List features to Float32. List probability
outputs retain the Float64 List interface, representing the computed values.
Clone preserves the selected dtype; parameter records contain only constructor
values, so `set_params` cannot change dtype. These language-driven differences
are covered by contract tests. The implementation borrows arrays and retains no input aliases; List inputs materialize an NDArray.
Weights must be finite, nonnegative and have positive total in each batch.
Priors must be finite, nonnegative and sum to one. These stricter validation
rules are inherited native differences, not sparse deferrals.

Weighted two-pass population moments use the Chan merge. Smoothing follows
the pinned sklearn implementation: unweighted variance of the current batch
sets epsilon, and a later update subtracts that new epsilon before merging.
Absent classes and zero class weights retain reference behavior. Degenerate
likelihoods preserve NaN/infinity; no fallback distribution is substituted.

Run `pixi run build` and `pixi run test` from the Moxie root. With an environment
containing the pinned local sklearn build, run:

```sh
/path/to/reference/python tests/naive_bayes/check_sklearn.py
/path/to/reference/python tests/naive_bayes/gaussian/parity.py
```

Both scripts verify the sklearn revision and exact naive_bayes.py bytes.
The Gaussian differential suite compares learned arrays after every batch,
class order, exact predictions, all probability methods and weighted score.
Float64 tolerance is `1e-7`; Float32 reference comparisons use `3e-5` to allow
different reduction orders and libm rounding. Large-offset fixtures detect
accidental Float64 accumulation; learned and output dtypes are compile-checked.
Nonfinite results are checked explicitly. Contracts cover invalid ranks, feature counts, nonfinite
inputs, continuous labels, failed updates, repeat fits, clone/copy and strides.
No performance or exhaustive sklearn compatibility claim is made.
