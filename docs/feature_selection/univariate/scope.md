# Dense univariate feature selection

Dense Float32/Float64 selectors and standalone scores are implemented. Broader
sklearn parity remains incomplete. See [the API guide](api.md) for the interface.

## Supported and exercised

- `chi2`, `f_classif`, `f_regression`, `r_regression`, `f_oneway` on native NuMojo
  Float32/Float64 arrays; results use Float64. Target-based scorers accept
  independently typed Float32/Float64 targets. Classification also accepts exact
  Int64 arrays and native `StringTarget` labels. Dense numeric class labels and
  binary multilabel indicators for chi-square are supported.
- Float32/Float64 `SelectKBest`, `SelectPercentile`, `SelectFpr`, `SelectFdr`, `SelectFwe`,
  `GenericUnivariateSelect`, with the default ANOVA scorer or a configured scorer.
- Independent score and threshold engines; K-best ties, fractional percentiles,
  score NaNs/infinities, strict FPR/FWE thresholds and inclusive FDR step-up.
- Strided univariate inputs: stepped/reversed rows and columns, column-major
  layouts and offset views, with strided vector/column targets in both dtypes.
- Regression center/force_finite branches; constants and one-class ANOVA;
  nonnegative chi-square validation; target/feature shapes and finite inputs.
- Shared support masks, transforms, inverse transforms and names, plus unfitted
  errors, clone independence, parameter changes, repeated fitting and owned scores.

## Deferred sparse and binding branches

CSR/CSC/COO inputs, sparse label matrices and sparse gather/scatter are deferred
by the session dense-only decision. No sparse input is densified. `chi2` directly
constructs a dense indicator matrix instead of the reference's sparse internal
LabelBinarizer intermediate. References: sklearn `_univariate_selection.py`
`chi2`, `f_classif`, `r_regression`; `_base.py` `transform`/`inverse_transform`.
The pinned family SelectionState and numerics owner expose only dense arrays.
Sparse capability import has a negative compile test.

Python reflection, metadata routing, pandas output, warning transport and
serialization remain binding work. Warnings about constants, empty selections
and k exceeding feature count are not emitted; native results retain the
corresponding mask/statistic behavior.

## In-scope limitations

The scorer contract separates input `dtype` from result `score_dtype`. Built-in
scorers return Float64 statistics for both input dtypes; selection policies use
those statistics without downcasting. Transform and inverse-transform retain
the input dtype. This preserves the existing native score representation rather
than promising every sklearn scorer's exact dtype promotion rules.

Broader wrapper/custom-scorer parity, additional
layouts, extreme-tail/cancellation campaigns and a performance baseline remain
unfinished. This is not a claim of full dense sklearn parity.
No default requires sparse output. Float64 selector defaults are supported;
Float32 input requires a typed scorer.

## Checks

Commands: `pixi run build`, `pixi run test`, `pixi run parity`. Build compiles owned
modules, example and two isolated consumer assemblies. Test runs score, policy,
selector lifecycle and inherited shared-selection contracts, both consumers,
and negative sparse/mixed-score-dtype compile checks. The mixed-dtype negative
check rejects a scorer whose return type contradicts its declared score dtype;
a positive custom-scorer test exercises Float32 input with Float64 output.
Parity executes function bodies from the pinned local sklearn source and compares
native scores, p-values and policies, plus Float32 selector scores, masks and
transforms for all six selectors and both dtypes on strided fixtures, including
independent target dtypes and Float64 class labels that would merge in Float32.
It covers all four built-in scorers; `r_regression` is exercised only with score-based
selectors. Masks compare exactly; transformed values compare exactly in their
input dtype. It does not substitute wheel algorithms.
The small bounded fixture matrix uses rtol 5e-5 and atol 2e-12 to accommodate
Float32 accumulation differences. That tolerance does not certify ill-conditioned
inputs or tiny relative-tail error. Final executed results are recorded in the
attempt handoff under ignored docs/private/handoffs/.

Root `tests/feature_selection/parity_labels.py` checks string and exact Int64 classification labels
for all six selectors, chi2, f_classif and discrete mutual information, including
reversed/offset targets, Int64 extremes, cloning and refitting. Each fixture is
compiled separately under the shared serial lock and hard 8 GiB cap.
