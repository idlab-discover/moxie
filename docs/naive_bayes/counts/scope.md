# Dense count models

Moxie owns `MultinomialNB` and `ComplementNB` in `moxie._nb_counts` and
exports both through `moxie.naive_bayes`.

Both accept dense boolean, integer and Float32/Float64 NDArray features and
nested Float64 List inputs. They implement weighted and incremental fitting,
prediction, joint/log probabilities, weighted accuracy, parameter updates,
unfitted cloning and independent fitted copies. Counts use Float64 NDArray;
numeric and string labels follow the shared sorted label-domain contract.

Scalar alpha uses `alpha`; per-feature alpha uses the native keyword
`alpha_values`, an owned rank-one Float64 NDArray. Present vector values take
precedence. Scalar parameter updates clear a previous vector. `force_alpha`
and both complement `norm` modes retain their reference meanings.
Vector alpha accepts valid positive and negative strides and nonzero offsets;
smoothing follows logical feature order, including during weighted `partial_fit`.

Sparse CSR/CSC/COO inputs, conversion, sparse counts and sparse products are
deferred. Reference branches are
`sklearn/naive_bayes.py::_BaseDiscreteNB._check_X`, `_check_X_y`, and
MultinomialNB/ComplementNB `_count`/`_joint_log_likelihood`. No sparse input is
silently densified. No estimator is wholly deferred and no default requires
an explicit dense setting.

Restrictions: single-output targets, finite nonnegative
weights with positive finite total, finite nonnegative priors, and no Python
reflection, pandas or metadata routing. Methods mutate in place and errors
use Mojo Error. Probability results use Float64 NDArrays for NDArray inputs
and nested Float64 Lists for List inputs. Training widens to Float64 and uses
an O(CN) weighted indicator matrix. Transactional updates duplicate O(CF) state.

## MultinomialNB model files

Use `MultinomialNBSchema[D, dtype]` from `moxie.naive_bayes_serialization`
with the [experimental native archive API](../../serialization.md). Choose the
same built-in label domain and input dtype used by the model. The defaults
are `IntListDomain` and Float64.

```mojo
from moxie.naive_bayes_serialization import MultinomialNBSchema
from moxie.serialization import load_model, save_model

save_model[MultinomialNBSchema[]](model, "multinomial.moxie")
var restored = load_model[MultinomialNBSchema[]]("multinomial.moxie")
```

Archives store constructor settings independently of learned counts,
log probabilities, class order and the alpha-floor warning from the last
successful fit or update. They support fitted and unfitted models. A loaded
model can predict and continue weighted `partial_fit` without refitting.
Vector alpha is stored in logical feature order and restored in owned
contiguous storage.

Validation uses Moxie's root build/test/parity commands and focused
`tools/naive_bayes_checks.py`. The count parity runner is
`tests/naive_bayes/counts/parity.py`; it loads the pinned local sklearn module
and checks counts, probabilities, predictions and scores with rtol=atol=1e-7,
exact labels and explicit nonfinite comparisons. Native kernel and tie
regressions run in the same root test command.

The differential suite includes 72 baseline cases and 48 vector-alpha layout
cases: positive/reversed strides, Float32/Float64 inputs, both `force_alpha`
settings, weighted fit and repeated weighted partial fits, and both complement
normalization modes. Native layout tests require bit-identical finalizer results
to an equivalent contiguous alpha vector.

Run `tests/naive_bayes/counts/parity.py --round-trip-multinomial` in the pinned
reference environment to compare MultinomialNB after byte-exact archive round
trips before fit and after each fit or update.

The former ComplementNB `norm=True`, `alpha=0`, `force_alpha=False` tie failure
came from Mojo 1.0's inaccurate `std.math.log` (~1e-10 relative), not from the
complement algorithm; the family's `_log` now uses Numax `core.libm`, and the
two-sample reproducer checks the sklearn feature weights bit for bit. Native tests
also verify invalid shapes/labels/priors/alpha lengths and all-zero weights,
rollback, reversed strides, Float32 fitting, parameter ownership and List input.
