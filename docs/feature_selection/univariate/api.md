# Using native univariate selection

Arrays use `numojo.NDArray` with samples in rows. The runnable example is
`examples/feature_selection/univariate.mojo`; compile against the verified `builds/deps` namespace.
These exports are also available from the family root assembly; no separate
univariate installation or second include path is needed.

```mojo
var selector = SelectKBest(BuiltinScorer("f_regression"),
                          k=SelectionParameter(2))
selector.fit(X, y^)
var selected = selector.transform(X)
var restored = selector.inverse_transform(selected)
```

`BuiltinScorer()` defaults to `f_classif`; its names are `f_classif`, `chi2`,
`f_regression` and `r_regression`. Regression configuration supports `center`
and `force_finite`, both true by default. A custom typed callable implements
`ConfiguredScorer`, which extends the family's `UnivariateScorer` and requires
independent `clone_config()` configuration copying and a default constructor.
It declares input `dtype` and result `score_dtype` independently, receives an
optional target and may omit p-values. Existing custom scorers must add an
explicit `score_dtype` declaration (`dtype` for same-dtype results). Missing
p-values raise when requesting a p-value policy's support.

Native adaptations:

- Fit mutates in place. The optional target owns its array: move it with `y^`,
  pass an explicit owned `.copy()`, or construct and reuse an `Optional` target.
- `SelectKBest.k` and generic `param` use `SelectionParameter`, a tagged integer,
  Float64 or String value. `SelectionParameter("all")` is valid for K-best.
- Learned attributes are owned accessors: `scores_()`, `pvalues_()`,
  `n_features_in_()` and `feature_names_in_()`. They reject unfitted access and
  prevent callers from mutating the estimator through returned buffers.
- `fit_named(X, y, names)` records explicit String feature names. Output names and
  support return the family's native types. Empty name lists mean absent names.
- `get_params()` returns typed `SelectorParams`; `set_params(params)` validates
  and updates configuration. Named policy parameter updates are also available.
  `clone(selector)` is unfitted. Policy changes update support from stored scores
  without recomputing the scoring function; changing the scorer needs refitting.
- `f_oneway(groups)` accepts a List of rank-1 or rank-2 arrays and returns
  `ScoreResult` with Float64 rank-1 scores and p-values. Other two-output scorers
  also return `ScoreResult`; `r_regression` returns only an array.
- Classification targets accept Float32, Float64, exact Int64 arrays, and native
  `moxie._labels.StringTarget`, independently of feature dtype.

Library execution is pure Mojo. NuMojo owns storage; approved shared reductions
and rank-2 products compute statistics. Scalar probability tails use standard
math functions and continued fractions. Selection delegates all mapping to the
family `SelectionState`; no replacement mask storage or transform kernel exists.

The standalone scores and all six selectors accept Float32 and Float64. Select
Float32 input with `BuiltinScorer[DType.float32]()` (default ANOVA), or specify
a scorer name such as `BuiltinScorer[DType.float32]("f_regression")`.
Built-in scores and p-values remain Float64; transformed and inverse-transformed
arrays preserve the input dtype. Default constructors continue to use Float64.
See [scope](scope.md) for other unfinished validation and sparse deferrals.

### Strided scoring inputs

The built-in scorers accept dense row/column strides and nonzero offsets.
`chi2`, `r_regression`, and `f_regression` gather non-C-contiguous inputs through
`SelectionState` before Float64 conversion. This allocates a temporary with the
logical input shape; it does not copy unused backing storage. C-contiguous inputs
use direct conversion. Inputs remain unmodified, and selector transforms retain
the input dtype. Tests cover positive/negative steps, column-major layouts,
offset views and strided vector/column targets in Float32 and Float64.

### Independent target precision

Standalone `chi2`, `f_classif`, `r_regression` and `f_regression` infer feature
and target dtypes separately. For selectors, specify the target dtype as the
second scorer parameter, for example:

```mojo
var selector = SelectKBest(BuiltinScorer[DType.float32, DType.float64]("chi2"),
                          k=SelectionParameter(2))
selector.fit(X32, y64.copy())
var selected = selector.transform(X32)  # Float32
```

The second parameter defaults to the feature dtype, preserving existing calls.
Class labels are compared at their original precision; regression arithmetic
and all returned statistics remain Float64. Both dtype combinations support
strided vector/column targets; chi-square also accepts binary indicator matrices.

### Native classification labels

`f_classif` and `chi2` accept `NDArray[DType.int64]` and `StringTarget` directly.
Int64 identities, including adjacent values above `2**53`, stay exact. String
comparison uses the shared label engine's UTF-8 lexicographic ordering. Vectors
and single-column targets are accepted; numeric chi-square indicators retain
their existing support. String multioutput targets are rejected.

All six selectors use the same scorer adapter:

```mojo
from moxie._labels import StringTarget
from moxie.feature_selection import BuiltinScorer, SelectKBest, SelectionParameter

# X is a Float32 matrix with four rows.
var y = StringTarget(["spam", "ham", "spam", "ham"])
var selector = SelectKBest(
    BuiltinScorer[DType.float32, target_type=StringTarget]("chi2"),
    k=SelectionParameter(1),
)
selector.fit(X, y.copy())
```

For Int64 use `BuiltinScorer[DType.float32, DType.int64]("f_classif")`.
The optional `target_type` parameter defaults to `NDArray[target_dt]`; its only
other supported value is `StringTarget`. `List[Label]` targets, which `moxie.svm`
classifiers such as `LinearSVC` use, are not accepted. A `Pipeline` needs one
target type, so `Pipeline(SelectKBest(...), LinearSVC())` does not compile;
chain the fitted steps explicitly (see
`notebooks/feature_selection/plot_feature_selection.ipynb`). String regression scorers raise a
clear error. As in sklearn, ANOVA groups finite fractional numeric labels,
while chi-square requires discrete labels. Column-vector warnings are not emitted.

`pixi run parity` includes `tests/feature_selection/parity_labels.py`: pinned sklearn scorer and
selection-policy bodies check binary/multiclass labels, ordering, column and
strided targets, scores, p-values, masks, clone fitting and Float32/Float64
features. Masks compare exactly; numerical tolerances are 5e-5 relative and
2e-12 absolute to accommodate the documented Float32 accumulation differences.

Exact-mask parity fixtures use separated cutoffs. Equal or nearly equal scores
can select different features after floating-point reduction rounding; sklearn
also leaves selection among tied scores unspecified. This applies to native
labels and numeric labels alike, and is not a promise of bitwise statistics.
