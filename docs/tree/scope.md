# Dense tree scope

Implemented public estimators: `DecisionTreeClassifier`, `DecisionTreeRegressor`,
`ExtraTreeClassifier`, and `ExtraTreeRegressor`. This is **partial** family
coverage, not complete sklearn parity.

Supported modes:

- Numeric dense Float32 features, Float64 regression targets/weights; string or
  Int64/Float32/Float64 classification labels with preserved identity and dtype;
  single and multiple outputs, strided features and numeric targets.
- Gini, entropy/log-loss, squared error, deprecated friedman_mse alias, absolute error, and Poisson.
- Best/random splitters, reference feature sampling and xorshift draws,
  constant-feature handling, missing-value routing, depth-first and best-first
  leaf limits, integer/fractional sample limits and feature selection.
- Dense numeric and mixed string/numeric categorical encoding, fitted
  dictionaries, unknown-to-missing routing, best bitset and random hash splits
  within the reference category limits.
- Sample and class weights, fit/refit/clone/parameter lifecycle, prediction,
  classifier probabilities/log-probabilities, accuracy/R2, apply, depth/leaf
  inspection, feature importances, cost-complexity pruning and pruning paths.
- Single-output regression/binary-classification monotonic bounds without NaNs.
- Owned feature names and ordered-name validation on TreeTable; categorical
  selectors by indices, boolean masks, names or native dtype metadata.

Native shape/type adaptations and RNG context behavior are explicit in
[estimators.md](estimators.md). In particular probabilities/classes are always
per-output lists, classification target dtype is a compile-time option, and
regression targets and predictions remain Float64. Default numeric dense fitting is usable.

Remaining dense limits:

- Absolute-error regression supports finite numeric features and signed weights;
  missing features and categorical splits are rejected.
- Python dataframe/reflection/visualization bindings, arbitrary object labels,
  and numeric label dtypes beyond Int64/Float32/Float64.
- Absolute-error regression rejects nonpositive total weights: the pinned
  reference median search is undefined for these inputs. Other criteria accept
  signed zero/negative totals, including reference NaN/Inf statistics. All-zero
  input sample weights raise, matching the reference. One-class monotonic
  clipping is supported without writing a nonexistent second class slot.
- Broader adversarial/extreme-value and large-workload validation remains; no
  performance, allocation-efficiency, or complete sklearn parity claim is made.

Sparse exclusions from `../session-plan/SPARSITY-SCOPE.md` remain: CSR/CSC/COO
inputs, sparse partitioners/traversal, and CSR-returning `decision_path`, including
for dense input. No densification or dense replacement path matrix is provided.
`export_text` renders numeric and categorical rules. `export_graphviz` renders
numeric and categorical trees as DOT, including color fills, parallel leaves and
HTML labels. `plot_tree` returns standalone SVG using a native layout. See
[estimators.md](estimators.md) for formatting and rendering adaptations.
`export_treelite` is a Moxie extension (see [estimators.md](estimators.md)).
Forests and boosting belong to ensemble and are outside this family assignment.

Validation commands: `pixi run tree-build`, `pixi run tree-test`,
`pixi run tree-prepare-oracle`, and `pixi run tree-parity`. Root build, test and
parity commands include tree, storage and criteria validation. The root oracle builds sklearn
`4358542a1bd39af2e7e1cbace8b9fed067473c5b` locally and checks tree source provenance.
It compares tree structure, leaf IDs, predictions, importances, scores, and pruning
paths; IDs/class predictions compare exactly, Float64 comparisons use 2e-12
relative/absolute tolerance for accumulation and logarithm roundoff on bounded
rational fixtures. Native tests check lifecycle, strides, class shapes,
capability rejection, and two independently assembled consumers. The private MAE
check uses the reference whole-right-child impurity to avoid its defective
non-root node query.

Current results: the root build, native contracts, example, negative capability
checks, and two isolated consumers pass. The full source-built estimator oracle
passes **21,115 estimator/export comparisons across 208 fitted cases**, including positive, zero,
and signed weights. Storage parity passes. Absolute-error parity adds **4,583
comparisons across 52 fits** for both regressors, including signed weights, multiple outputs, monotonic constraints,
best-first construction and pruning.

Dense numeric categorical validation adds **50,608 comparisons across 160 fits**
against the same source-built reference. It covers best/random search, binary and
multiclass classification, regression, supported multioutput targets, zero and
signed weights, missing/unknown values, leaf limits, pruning, node values and
probabilities. Category words, leaf IDs and class predictions compare exactly;
Float64 statistics use the same 2e-12 tolerance. Unused best-first leaf split kinds
are normalized to the documented native `-1` sentinel. Native contracts cover
category limits, ownership/refit/clone, invalid combinations and strided input.
`tests/categorical_parity.py` runs as part of `pixi run parity` and retains raw
results under ignored `docs/private/categorical/`.

Named mixed tables add **50,928 comparisons across 160 fits**, including 160 exact
category-aware text reports reconstructed from the oracle's fitted dictionaries,
bitsets, hash seeds and missing routes. The same shape, ID and numeric tolerances
apply as above. Numeric text export adds **832 exact comparisons**: 320 reports
across 80 fitted trees and 512 fixed-decimal formatting checks over seeded
binary64 bit patterns. These fits cycle categorical selection by indices, masks, names and dtype metadata.
Native contracts also cover name/order validation, reordered/unused string
dictionaries, all-missing columns, schema errors, snapshots and refitting.
Run `tests/mixed_parity.py` and `tests/export_text_parity.py` with the source-built
oracle Python, or run both through `pixi run parity`. Raw results remain under
ignored `docs/private/mixed_tables/` and `docs/private/export_text/`.

Typed-label parity adds **4,524 comparisons across 48 fits** for DecisionTree and
ExtraTree classifiers with StringTarget, Int64 and Float32 labels. Class identity
and predictions compare exactly, including adjacent Int64 values above `2**53`
and the Int64 minimum; probabilities and impurities use 2e-12 tolerances.
Native contracts check declared output dtypes, string/mixed-table integration,
weights, strides, clone/refit, scoring, probabilities, pruning and text export.
Run `tests/labels_parity.py` with the source-built oracle Python, or `pixi run parity`.

Nonpositive-weight and one-class monotonic edge checks add 818 comparisons
across 120 cases (`tests/weight_edges_parity.py`). They preserve reference
NaN/Inf results where defined and explicitly reject undefined MAE medians.

Seeded randomized differential tests add 38,904 comparisons across 192 fits.
Each typed classifier is reused across 32 changing datasets and class domains,
combining categorical selectors, signed weights, pruning and leaf limits. Seeds
7300–7331 use shared `RandomState` and NumPy's matching stream; labels, leaf IDs
and category words compare exactly, Float64 statistics within 2e-12. Run
`tests/randomized_parity.py` with the oracle Python, or `pixi run parity`.

Forest-member differential tests add **66,531 exact comparisons across 95 member
trees** of 19 reference RandomForest and ExtraTrees classifiers and regressors
(`tests/forest_member_parity.py`). The script recovers each member's seed and
the exact weights sklearn passed to `tree._fit(check_input=False)`: bootstrap
counts drawn with combined weights, balanced_subsample factors, or combined
weights without bootstrap. The native driver builds every member of a forest
from one prepared input and predicts through one shared encoding. Inputs cross
as raw bits: continuous Float32 features with ties and NaNs, every criterion,
multi-output targets, monotonic constraints, leaf and depth limits, pruning,
minimum weight/impurity thresholds and `max_samples`. Structure, thresholds,
impurities, weighted counts, values, importances, leaf IDs and predictions are
bit-identical, and the mean of member predictions equals the forest's
prediction exactly. These realistic floats exposed sample-order and evaluation-
order differences that dyadic fixtures cannot: near-tied splits are decided by
summation roundoff. The numeric partitioner now ports the reference introsort and
missing compaction, and criteria keep the reference evaluation order. Native
`tests/test_prepared.mojo` checks fit equivalence, shared encoding, per-member
weights and seeds, unmodified inputs and rejected mismatched preparations.


Graphical export checks **1,280 exact DOT comparisons and 1,280 SVG checks
across 80 fitted trees** (`tests/export_graphviz_parity.py`). All four estimators
cover weighted numeric inputs, single/multiple outputs, up to seven classes,
entropy/log-loss, absolute error, Poisson, best-first leaf limits, names, depth
truncation, label modes, node IDs, proportions, rotation and rounded boxes.
DOT cases also cover fills, custom palettes, HTML labels and parallel leaves.
SVG checks compare node labels and colors with the reference and verify edges,
font sizes, finite canvas bounds and nonoverlapping boxes. They do not assert
Matplotlib pixel/layout parity. These bounded three-decimal DOT fixtures match
sklearn byte-for-byte; native formatting differences are documented in
[estimators.md](estimators.md). Native contracts cover errors, typed string
labels, categorical bitset/hash rules, escaped names, constant multioutput
colors and deterministic output. Both isolated assembled consumers fit a real
classifier and call the public DOT and SVG exports. Raw outputs and comparison
results remain under ignored `docs/private/export_graphviz/`.
