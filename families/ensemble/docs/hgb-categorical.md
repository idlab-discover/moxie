# Categorical features in dense histogram boosting

Both histogram boosters accept `categorical_features=List[Int]([1, 3])`
for zero-based column indices or a `List[Bool]` with one flag per input
column. The typed parameter record uses
`moxie.ensemble.HistGradientBoostingCategoricalFeatures` for the selector.
`None`, an empty selection, and the default `"from_dtype"` treat a homogeneous
NuMojo array as numerical. Names and categorical dtype inference require table
input support.

Selected Float32 or Float64 columns can contain any finite category values,
including negative and fractional values. The estimator owns sorted category
domains learned from the full training input before holdout selection, with
at most `max_bins` distinct nonmissing values per column. NaNs and unseen
categories use the tree's missing-value route. Infinities in categorical
columns raise `ValueError`. The fitted `is_categorical_` mask follows input
column order and is `None` if no columns were selected.

Monotonic constraints follow input column order, with a zero required on
categorical columns. The pinned scikit-learn implementation interprets explicit
interaction groups in the preprocessed order, where categorical columns come
first and each group of columns retains its relative input order. Warm starts
require the same fitted categorical columns and domains so the retained trees'
ordinal codes keep their meaning. Ordinary refits learn a new encoding.

The encoder collects distinct categories in a bounded map and reuses
`NumericCategories.find` for lookup in owned sorted domains. Numax
`core.sorting.unique` sorts a full materialized column and its scalar
`searchsorted` copies the values to the host per lookup. The bounded map and
direct stride-aware writes into NuMojo storage avoid these intermediates.
Category encoding materializes one matrix per input being encoded, which all
stages in that fit or prediction reuse. Purely numerical input is borrowed
through a layout-preserving view.

The focused `hgb-classifier` and `hgb-regressor` checks compare fitted masks,
all staged predictions, final predictions, and early-stopping histories with
the pinned scikit-learn build. Fixtures cover unseen and missing values,
categorical columns interleaved with numerical columns, weighted training,
explicit and internal validation, feature subsampling, and warm starts.
The native contracts also exercise Float32 strided input, invalid selectors
and cardinality, and fitted state after failed warm starts.
