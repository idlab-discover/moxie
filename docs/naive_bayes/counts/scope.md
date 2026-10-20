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
weights with positive finite total, finite nonnegative priors, no Python
reflection/pandas/metadata routing/serialization. Methods mutate in place;
errors use Mojo Error. Probability results remain nested Float64 Lists to reuse
the family normalization implementation. NDArray prediction materializes an
O(NF) List input; training widens/copies to Float64 and uses an O(CN) weighted
indicator matrix. Transactional updates duplicate O(CF) state.

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
The former ComplementNB `norm=True`, `alpha=0`, `force_alpha=False` tie failure
came from Mojo 1.0's inaccurate `std.math.log` (~1e-10 relative), not from the
complement algorithm; the family's `_log` now uses Numax `core.libm`, and the
two-sample reproducer checks the sklearn feature weights bit for bit. Native tests
also verify invalid shapes/labels/priors/alpha lengths and all-zero weights,
rollback, reversed strides, Float32 fitting, parameter ownership and List input.
