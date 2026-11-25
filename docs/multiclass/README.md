# Multiclass classifiers

Moxie owns `OneVsRestClassifier`, `OneVsOneClassifier`, and
`OutputCodeClassifier` under `moxie.multiclass`. They accept dense numeric or
string single-label targets. OvR also accepts dense multilabel indicators.
Implementation is in `src/moxie/multiclass.mojo` and `_multiclass_*.mojo`.
The retired sibling is not a source or build dependency.

- [OneVsRestClassifier](one_vs_rest.md)
- [OneVsOneClassifier](one_vs_one.md)
- [OutputCodeClassifier](output_code.md)
- [Component contracts](family.md) and [public API](api-surface.md)
- [Shared estimator interfaces](../interfaces.md)
- Verification notebook: [`plot_multiclass_overview`](../../notebooks/multiclass/plot_multiclass_overview.ipynb) (`pixi run -e notebooks notebooks notebooks/multiclass`)

Run from the Moxie root:

```sh
pixi run build
pixi run test
pixi run parity
# Focused build, native contracts, capability rejections and isolated consumer:
pixi run python tools/multiclass_checks.py test
# Eight parity runners, using the pinned local sklearn installation:
pixi run python tools/multiclass_checks.py parity
```

For parity, create `builds/reference-env` and install the local scikit-learn
checkout at `4358542a1bd39af2e7e1cbace8b9fed067473c5b` as described in
[the runner](../../tests/multiclass/ovr_parity.py). Most comparisons use paired test
components to establish wrapper behavior; the production runner adds real
base estimators.
All original numerical tolerances and exact comparisons are retained.

Moxie's dependency assembly supplies live shared label domains, random state,
preprocessing `label_binarize`, and numerical dependencies. The build retains
Mojo 1.0.0, MAX 26.5.0 and `--fp-mode=contract=off`. Assembly receipts and the
isolated consumer check verify Moxie ownership. No inherited framework copy or
historical source pin is used.

Mutation methods follow current Moxie fluent contracts. `fit`, `set_params`,
and supported `partial_fit` return a mutable borrow of the estimator. Optional
classes still require explicit copies when passed as owned optional values.
Transactional fit and update behavior, class ordering, capability distinctions,
reference quirks and seeded/default random-stream behavior remain unchanged.

The isolated consumer exercises all three wrappers, numeric/string labels,
probabilities, decision values, incremental updates and fluent calls.

Production components: `moxie.linear_model.RidgeClassifier` satisfies the
binary component contract directly (decision values; Float32 or Float64
features). `moxie.tree.DecisionTreeClassifier` does through the dense
`moxie.feature_selection.DecisionTreeClassifierImportance` adapter, which
exposes `NDArray` probabilities. `moxie.svm.LinearSVC` works through
[`BinaryAdapter`](family.md#binary-adapter-for-numeric-list-labels), which
converts its binary labels between `NDArray` and numeric `List` values.
`moxie.discriminant_analysis.LinearDiscriminantAnalysis` works through the
same adapter with its default `Int` labels and Float32 or Float64 features.
`moxie.naive_bayes.GaussianNB` works through the adapter in both feature dtypes,
including incremental OvR and OvO with the default List labels.
`tests/multiclass/production_parity.py` compares all three wrappers around
these production components against pinned sklearn. Pass `--group lda`,
`--group linear-svc` or `--group gaussian-nb` to check one component after assembly.
The GaussianNB group checks the reference's string OvO incremental error, then
compares Moxie's string predictions and decisions against sklearn with labels
encoded as sorted integers. See [OvO string labels](one_vs_one.md#string-labels).

Composition: the wrappers work as `Pipeline` final steps and under `clone`,
`cross_val_score` and `GridSearchCV` (nested `estimator__alpha` through the
typed deep `Params` record). Adapters and scorers may name a concrete wrapper
type (`tests/multiclass/test_concrete_adapters.mojo`) or stay generic over the
estimator (`tests/multiclass/test_composition.mojo`). Using a refitting
`GridSearchCV` as the inner component requires its adapter to expose the
multiclass feature and binary target types and supply default fit metadata.

Sparse inputs, metadata routing, mutable Python RNG objects, bindings and
serialization remain outside the supported scope.
