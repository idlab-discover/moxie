# Public SelectorMixin

`moxie.feature_selection.SelectorMixin` is a native trait for implementing dense
selectors. It supplies `get_support(indices=False)`, `transform`,
`inverse_transform` and `get_feature_names_out`. Implement `dtype` and `_selection_state() -> SelectionState`:

```mojo
from moxie.feature_selection import SelectorMixin, SelectionState

@fieldwise_init
struct MaskSelector(SelectorMixin, Copyable, Movable):
    comptime dtype = DType.float64
    var state: SelectionState

    def _selection_state(self) raises -> SelectionState:
        self.state.check()
        return self.state.copy()
```

An estimator computes its mask during fitting and commits it to `SelectionState`.
The mixin borrows the input and delegates all gather/scatter and name/mask mapping
to that shared engine. It owns no second mask implementation. State snapshots,
support results, transformed arrays and output names own their memory.
The example above is a mapping adapter, not a feature-selection algorithm.
For a fitted estimator using this trait, see [RFE](rfe.md).

This replaces Python inheritance and `_get_support_mask` with explicit trait
conformance and an owned state snapshot containing mask, names and fitted status.
`_base.FeatureSelector`, `SupervisedSelector` and `UnsupervisedSelector` remain
unchanged; concrete estimators additionally conform to the appropriate existing
capabilities. The mixin does not invent a fit, clone or parameter policy. Unlike Python's
inherited TransformerMixin, `fit_transform` belongs to the concrete estimator's
`FitTransformer` capability. It calls that estimator's fit and then this mixin's
transform. This preserves the existing supervised/unsupervised signatures;
the locked Mojo compiler cannot conditionally provide trait methods with
`where` clauses. RFE's fit-transform behavior is covered by contract tests.

Dense Float32/Float64 rank-2 arrays are supported. Shapes, fitted state and feature
names are checked; masks and ascending indices use the tagged `SupportResult`.
`transform` rejects NaNs and infinities unless `_allows_nan()` is overridden to
return True. Like sklearn's tag, that override disables both finite checks.
`inverse_transform` always requires finite nonempty input and inserts zeros.
An empty support mask transforms to shape `(n_samples, 0)`; inverse transformation
of a zero-column matrix raises. Noncontiguous input is supported and dtype is
preserved. Empty name lists mean omitted names.

Sparse storage, arbitrary Python dtypes/containers, `set_output`, metadata routing
and Python warnings are outside the accepted scope. The mixin does not emit a
warning when no features are selected. See [plumbing](plumbing.md) for the storage
contract and `tests/feature_selection/test_rfe.mojo` for public mixin contract tests.
