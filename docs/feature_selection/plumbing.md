# Feature-selection plumbing

The family preserves the existing `_base` capability traits. Shared mechanics
and signatures support the algorithm implementations.
VarianceThreshold is implemented at the family root; see [variance.md](variance.md).
Reference: scikit-learn `4358542a1bd39af2e7e1cbace8b9fed067473c5b`,
`sklearn/feature_selection/_base.py`, `_univariate_selection.py`, `_rfe.py`,
`_from_model.py`, `_sequential.py`, `_mutual_info.py`, `_variance_threshold.py`,
`tests/feature_selection/test_base.py`, and `doc/modules/feature_selection.rst`.
The upstream BSD notice is retained in `LICENSES/feature-selection-scikit-learn.txt`.

## Shared interfaces and layouts

`moxie.feature_selection._selection.SelectionState` owns one `List[Bool]` mask
of length p in original feature order, an optional `List[String]` of p names,
and a fitted flag. These are metadata, not numeric matrix containers.
`commit(mask, names=[])` validates positive p and name length, then copies both
lists; an invalid commit preserves the previous state. Algorithms validate and
compute into temporaries before committing all their learned state. `reset()`
clears all metadata; `is_fitted()` returns the non-raising fitted query. Copying state copies learned metadata; cloning an estimator
constructs a new unfitted state and recursively clones constructor configuration.
`n_features()` returns p and `selected_count()` returns k; both require fit.

`get_support(indices=False)` returns `SupportResult`, a
`Variant[List[Bool], List[Int]]`; False selects the Boolean arm, True the integer
arm. Both own their memory; indices ascend in original column order. This tagged
return preserves the inherited `FeatureSelector.Support` associated-type API.
Algorithms should bind it to `SupportResult` and delegate to this implementation.

`transform[dtype](X)` borrows an accepted rank-2 NuMojo array of shape (n,p)
and returns owned contiguous (n,k). `inverse_transform[dtype](X)` consumes a
borrowed (n,k) array and returns owned contiguous (n,p) with zero-filled removed
columns. Dtype is compile-time Float32 or Float64. Offset and strides are handled
by NuMojo access; positive/negative strides are element counts. No input aliases
are retained. The algorithms own finite/NaN validation according to their tags;
these mapping primitives preserve values, including nonfinite ones. Public
estimators must perform that validation before delegation. n must be positive;
inverse input must have at least one column, matching upstream `check_array`.
Transform with an all-false mask returns (n,0). Python warning transport is not
available: callers can inspect `selected_count()==0`; no warning is emitted by
this primitive. No implicit array coercion or promotion is provided.

`get_feature_names_out(input_features=[])` returns owned selected strings.
An empty list means omitted names (native adaptation); otherwise length must
match p and names must equal fitted names when present. Omitted names use fitted
names or generated `x0`, `x1`, … . Native String lists replace Python object arrays.
Errors use Mojo Error with NotFittedError/ValueError prefixes.

`moxie.feature_selection._contracts` defines:

| Interface | Contract |
| --- | --- |
| `ScoreResult[dtype](scores, pvalues=None)` | Owned rank-1 NuMojo scores (p,), optional same-shaped p-values; constructor moves arrays and validates shapes |
| `UnivariateScorer` | Associated input `dtype`, result `score_dtype` and `Target`; `score(X: NDArray[dtype], y: Optional[Target]) -> ScoreResult[score_dtype]` |
| `SelectionPolicy` | Associated dtype; `select(scores: ScoreResult[dtype]) -> List[Bool]` of exactly p entries |
| `ImportanceProvider` | Associated dtype; `feature_importances() -> NDArray[dtype]`, an owned raw (p,) importance vector or (outputs,p) coefficient matrix from a real fitted estimator |

Scorer implementations must declare `score_dtype` explicitly; use `dtype` for
same-dtype results or Float64 for widened results. Selectors store and threshold
in `score_dtype` while transform/inverse_transform preserve input `dtype`.

Scorers own mathematics, target validation, NaN/constant handling and p-value
availability. Policies own thresholds, tie ordering and statistical corrections;
p-value policies must reject missing p-values. No fabricated p-values are allowed.
Policies preserve score arrays. Results move into learned state; borrowed caller
arrays require a deep copy before retention. The model-based owner adapts real
fitted estimator attributes (coef_ preferred by `auto`, then feature_importances_)
to ImportanceProvider, and owns abs/norm/square aggregation. RFE reuses that owner.
Native typed providers replace Python attribute-path reflection/callable dispatch;
no shared trait forces all estimators to expose importances. CV wrappers consume
model_selection.validation's split, clone, scoring and aggregation contracts;
this family defines no competing fold engine or CV result container. Resolve
those actual interfaces from the completed dependency before implementing wrappers.
Optional prediction/probability/decision capabilities forward only when supported
by the wrapped estimator and reference API. Parallelism must coordinate inner MAX
and outer CV budgets.

## Module organization

All feature-selection sources belong to Moxie under `src/moxie/feature_selection`.
The formerly separate univariate modules now share that directory. The root
manifest assigns every feature-selection module to the `moxie` owner.
`_selection` and `_contracts` remain the shared selection and extension contracts.

| Child | Exports and modules | Additional dependencies |
| --- | --- | --- |
| univariate modules | `_univariate_selection`: SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect, chi2, f_classif, f_regression, r_regression, f_oneway | none |
| family root (variance) | `_variance_threshold`: implemented VarianceThreshold; public `_selector_mixin.SelectorMixin` delegates to SelectionState | none |
| family root (model-based) | `_from_model`: SelectFromModel; `_model_importance` aggregation; `_linear_model_importance`, `_incremental_pca_importance` and `_tree_importance` real adapters | live dense linear-model/coordinate, decomposition and tree modules; shared categorical inputs |
| family root (sequential selection) | `_sequential`: SequentialFeatureSelector; `_sequential_adapter`: DenseSupervisedCV | live model_selection.validation, splitters and label encoding |
| family root (mutual information) | `_mutual_info`: mutual_info_classif, mutual_info_regression; `_mutual_info_scorer`: MutualInfoScorer | live neighbors.nearest/query_engine, shared RNG |
| family root (recursive elimination) | `_rfe`: RFE and RFECV | feature_selection.model_based; RFECV adopts model_selection CV contracts |

SelectorMixin is published from `_selector_mixin` and re-exported through the manifest;
preserve the native `_base` extension traits. Selectors preserve constructor
names/defaults and learned names (scores_, pvalues_, variances_, estimator_,
support_, ranking_, n_features_, cv_results_ as applicable). No scaffold helper
exposes those names as if an algorithm had fitted. Constructor/method details and
statistical references must be audited and contract-tested in the owning session.
Selection count, fraction/auto modes, threshold policies, warm-start/prefit rules,
CV routing and all supported dense branches remain each owner's responsibility.

## Native dependencies and assembly

Read `../moxie/shared/numerics/docs/numerics.md` relative to the repository before
choosing operations. Mojo 1.0.0 and MAX 26.5.0 are fixed by the Pixi lock;
NuMojo sources come from the local path declared in `DEPENDENCIES.json`.
No Python runs during native library calls. Python is build/test tooling only.
See AGENTS.md for the accepted operation subset and exclusions. Dense storage,
RNG and baseline contracts are consumed directly from their shared owners.

Moxie's `DEPENDENCIES.json` declares live project paths, modules, assets and
exports. The shared assembler in `shared/assembly` builds the root namespace
and the two independent consumer namespaces in `tests/feature_selection/consumers`.
No build reads the retired feature-selection checkout or a cached source snapshot.
Keep their export records consistent when changing the public namespace.

Run `pixi run feature-selection-build`, `pixi run feature-selection-test`, and
`pixi run feature-selection-parity` for the complete family checks. Moxie's root
build, test and parity commands include the same checks. Compiler invocations
use the shared lock, hard 8 GiB address-space ceiling and ten-minute timeout in
`tools/compiler_limits.py`. Generated univariate and typed-label parity cases
remain separate compilation units.

The model selector extends ImportanceProvider with the family-owned
SelectionEstimator trait. It consumes five external linear-model modules
under `linear_model.selection`; the root manifest records their source paths.
Fit delegates to that model, and no optimization code is copied into feature_selection. See [from_model.md](from_model.md).

RFE uses `SelectorMixin` and `SelectionState` for every transform. Optional
prediction/scoring/probability/decision capabilities use conditional conformance
to the existing global traits. `SelectionClassLabels` adds owned class-label
access for classifier adapters without requiring it of regression estimators.
RFE reuses `aggregate_importances(square=True)` and clones its adapter at every
elimination step and final fit. Its elimination engine does not execute cross-validation.

Mutual-information scoring uses live neighbor query/contracts modules and
shared RandomState (including global/default-stream handling). Dependency paths
and license assets are recorded in each assembly manifest.
`MutualInfoScorer` implements ConfiguredScorer, returning Float64 ScoreResult
without p-values; selectors continue to delegate every transform to SelectionState.
See [mutual-info.md](mutual-info.md) for native options and validation limits.

Sequential selection adopts the existing model-selection CVAdapter,
cross_val_score, check_cv and split-index contracts. Its SequentialAdapter only
adds dense dtype and default policies. It defines no fold algorithm or CV result
container. Candidate columns and learned masks always use SelectionState.
SequentialCV is copyable constructor configuration for fold counts or snapshots
of shared IndexSplit pairs. See [sequential.md](sequential.md) for the typed API.

RFECV reuses the RFE observer with shared CVAdapter row/metadata slicing,
SequentialCV/check_cv split configuration, TypedScorer and ScoreTable. It owns
only elimination-path orchestration and sklearn-named result columns; no new
fold algorithm or general CV engine is introduced. Training metadata is optional
sample weights; custom scoring metadata uses the same optional dense-array type.
See [rfecv.md](rfecv.md) for typed result access, final refitting and parity scope.

The `decomposition.incremental_selection` contribution consumes six live
modules from the sibling decomposition checkout,
including IncrementalPCA, PCA helpers and the SVD engine. It uses the live runtime SVD dependency. `IncrementalPCAImportance` belongs to this family;
no decomposition algorithms are copied into family-owned source. The adapter's
owned fitted copies and the selector's conditional incremental capability are
covered by root contracts, sklearn comparisons and both isolated consumers.

The `moxie` contribution supplies `src/moxie/tree` modules; `categorical` supplies their existing global input dependency. Source
paths and license assets are recorded in the root and two consumer manifests.
`DecisionTreeClassifierImportance` lives in this family
and owns independent fitted copies. Learning remains in the tree dependency.
`DenseSupervisedCV` uses the estimator target type independently of feature
dtype: numeric rows preserve dtype/strides, strings preserve rank and columns,
and `SplitLabels` sees original labels without floating conversion.
