# Dense imputation plumbing

This family supplies shared contracts, native ownership/validation helpers and
partial SimpleImputer/MissingIndicator implementations and dense KNNImputer. See
[the public API guide](api-surface.md) for the current export surface. Import the existing
`Imputer` capability from `moxie.impute`; private plumbing is imported from
`moxie.impute._plumbing`. The pinned sklearn `impute.__all__` contains
SimpleImputer, MissingIndicator, KNNImputer and IterativeImputer, all directly
public at revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

## Owners and extension boundaries

| Owner | Modules and work | Dependencies |
| --- | --- | --- |
| This family | `_base.Imputer`, `_plumbing` masks/layout/assembly and step ownership | current Moxie, numerics, assembly |
| `algorithms/simple/` | `_simple.SimpleImputer`, `_indicator.MissingIndicator`; statistics, categorical representation, inverse transformation | impute.scaffold |
| `algorithms/knn/` | `_knn.KNNImputer`; donor validity, distances/weight policy, aggregation | impute.scaffold, impute.simple, neighbors.query_engine |
| Deferred iterative implementation | Rounds, bounds, posterior and real model sequence | impute.scaffold, impute.simple, linear_model.bayesian; foundation.random |

The sklearn `_base.py` combines shared plumbing and two concrete estimators.
Native children use `_simple` and `_indicator` to avoid two independent projects
owning one source module; public imports remain `moxie.impute.SimpleImputer` and
`MissingIndicator`. Do not replace the shared `_base` trait with an algorithm.
The existing simple algorithm is assembled into the family build; its sources
remain local under `algorithms/simple`. Engine APIs are consumed from
their completed owners; adapter code belongs to the dependent child, not a fork
of the engine. Missing required native capabilities must be reported to owners.

## Storage and exact interfaces

The locked environment is Mojo 1.0.0 / MAX 26.5.0 / NuMojo 0.10.0 revision
`515fb2856f0ecf3d2740a34d958fe168183e1129`. Read the numerical owner's
[decision](../../../shared/numerics/docs/numerics.md) for allowed operations.
Use `numojo.NDArray[dtype]`, `Shape`, `Item` directly. Missing-mask helpers specialize
Float32/Float64; selection and indicator concatenation also support signed/unsigned
8/16/32/64-bit integers. Masks are `NDArray[DType.bool]`. Rows are samples; columns are
features. A mask is rank 2 `(n,p)`, true precisely where input is missing.
Boolean storage here uses only creation and offset-aware scalar access, not
untested Boolean arithmetic/reductions. It is tested in both consumer trees.

Borrow input; outputs own independent C-contiguous arrays. Indexing uses element
strides, including negative strides and nonzero offsets. `.view()` aliases;
`.copy()` owns an independent snapshot, and a strided copy may copy its entire
backing allocation. Fit retains owned snapshots, not caller aliases. Slicing,
transpose, reshape and astype can allocate; they are not free views. Concurrent
mutation and overlapping in-place operations are outside the shared contract.

| API | Contract |
| --- | --- |
| `MissingValue[dtype]()` / `(value)` | NaN default; explicit scalar equality marker, NaN also accepted, infinite sentinel rejected |
| `missing_mask(x, missing)` | Validates nonempty rank-2 numeric input; NaN permitted only for NaN marker, infinity always rejected; returns independent Boolean `(n,p)` |
| `validate_matrix(x)` | Any dtype; nonempty rank 2, otherwise ValueError-prefixed Error |
| `validate_indices(indices, p)` | List[Int] must be ascending, unique, in `[0,p)`; empty allowed |
| `select_features(x, indices)` | Independent `(n,len(indices))`, including zero output columns; numeric or Boolean dtype preserved |
| `FeatureLayout()` | Unfitted, independent List[Int] metadata and foundation FittedState |
| `layout.fit(mask, keep_empty_features=False, features="missing-only", error_on_new=True)` | Replaces metadata transactionally; invalid fit preserves previous state |
| `layout.set_retained(indices)` | Owned sorted original-index subset, for statistics-dependent invalidity; requires fitted state |
| `layout.check_mask(mask)` | Validates rank, samples, original feature count, fitted state and unexpected missing features |
| `layout.indicator(mask)` | Independent Boolean `(n,k)` in fitted indicator order |
| `append_indicator(filled, original_mask, layout)` | Filled must be `(n,r)`, already selected; returns owned numeric `(n,r+k)`, appended bools cast to input dtype |
| `ImputationStep[Model](feat_idx, neighbor_feat_idx, estimator, n_features)` | Moves an owned fitted model; copies predictor indices, checks bounds/duplicates/target exclusion; predictor order is preserved |

`FeatureLayout.retained_features`, `indicator_features`, `empty_features` use
original feature indices. `state.n_features_in_` is the original feature count.
Fields are inspectable metadata; callers must use fit/set_retained to mutate
mappings and preserve their invariants. A layout copy owns independent Lists.
These Lists are domain metadata, not another generic array implementation.

An imputer captures the original mask before any mutation or column dropping,
fits layout with `error_on_new=False`, and chooses whether to call
append_indicator according to `add_indicator`. Standalone MissingIndicator
uses its default `error_on_new=True`; `features="all"` disables new-feature
rejection. All-empty columns may be removed from filled output but still appear
in indicators. No new indicator column appears merely because transform sees
missingness in a previously complete feature. An all-empty input is valid;
a zero-row or zero-feature input is not. Zero-column *outputs* remain valid.

The family supplies index bookkeeping, not fill values. Simple custom statistics
can return NaN on nonempty columns, so that child must call set_retained with
its valid statistic indices. `keep_empty_features` and constant fill behavior
follow the pinned source; KNN's retained empty columns are zero-filled. Iterative
must align bounds and predictor indices after initial feature selection.

Simple's integer strategies are implemented through IntegerSimpleImputer;
StringSimpleImputer adds owned nullable-text cells and consumes the same Boolean
mask and FeatureLayout. Its values and indicators remain separate typed outputs.
Standalone `StringMissingIndicator` consumes StringMatrix through the same mask
and layout contracts. Broader categorical dense strategies remain child deliverables. This
scaffold does not narrow those future estimator modes to float-only. Children
produce the same Boolean precomputed mask using their domain values (e.g. owned
strings and explicit missing tags), then consume FeatureLayout. The cross-child categorical interface is the rank-2 Boolean mask, so no
consumer needs to understand another child's domain representation. An algorithm
may not introduce an alternative generic dense buffer.
Python object arrays and pandas adapters are separate bindings.

## Numerical engine seams

KNN stores an owned training NDArray and original Boolean mask; query rows keep
original feature width even when the final output drops columns. The neighbor
engine handles dense distances/querying, including required missing-coordinate
semantics, and returns donor row IDs/distances. For each target feature the KNN
child filters donors by that feature's observed mask, applies uniform/distance/
native-callable weights, and handles insufficient/undefined distances according
to sklearn. A generic nearest-k query over unfiltered rows is insufficient.
Indices refer to training rows; undefined distances are not valid zero distances.
KNN donor aggregation is implemented in `algorithms/knn`.

Iterative uses `List[ImputationStep[Model]]` as its ordered per-round sequence.
Each Model must implement the foundation SupervisedEstimator/Predictor contracts
with input NDArray `(observed_rows,predictors)`, target/prediction `(rows,)`, and
clone producing an unfitted configuration. Each step owns a distinct fitted model
snapshot, not the same repeatedly mutated regressor. `ImputationStep` is an
ownership record and deliberately does not impose probability or posterior
methods on every model. The child constrains Model when fitting/predicting.
Posterior mode separately requires mean/std prediction from the real regressor;
nonposterior mode only requires predict. Transform replays the stored ordered
sequence without fitting and preserves sample count. Copying a fitted step is
not sklearn cloning. Random draws use `moxie._random.RandomState`; numerical
sampling and all posterior branches belong to the iterative session.

## Native compatibility obligations

Child constructor names/defaults, learned attributes and numerical branches
must follow the pinned source. The target Simple API supports mean/median/most_frequent/
constant/native-callable strategies, copy and optional indicators/inverse.
MissingIndicator supports missing-only/all, auto dense/explicit dense and
error_on_new. KNN supports uniform/distance/native-callable weights and
nan_euclidean/native-callable metrics. Iterative supports real configurable
regressors, posterior sampling, all five orders, bounds, feature subsampling,
initial strategies, skip_complete, zero/max iterations and early stopping.
The table below records the target contracts. The simple and KNN API guides
document their implemented subsets; IterativeImputer remains deferred. The
reference source remains authoritative for validation and numerical behavior.

| Estimator | Constructor defaults | Learned state / methods |
| --- | --- | --- |
| SimpleImputer | missing_values=NaN, strategy="mean", fill_value=None, copy=True, add_indicator=False, keep_empty_features=False | statistics_ `(p,)`, indicator_, n_features_in_, optional feature_names_in_; fit, transform, fit_transform, inverse_transform when add_indicator, get_feature_names_out |
| MissingIndicator | missing_values=NaN, features="missing-only", sparse="auto", error_on_new=True | features_ `(k,)`, n_features_in_, optional feature_names_in_; fit, transform, fit_transform, get_feature_names_out |
| KNNImputer | missing_values=NaN, n_neighbors=5, weights="uniform", metric="nan_euclidean", copy=True, add_indicator=False, keep_empty_features=False | indicator_, n_features_in_, optional feature_names_in_; owned donor/mask state; fit, transform, fit_transform, get_feature_names_out |
| IterativeImputer | estimator=None (BayesianRidge), missing_values=NaN, sample_posterior=False, max_iter=10, tol=1e-3, n_nearest_features=None, initial_strategy="mean", fill_value=None, imputation_order="ascending", skip_complete=False, min_value=-infinity, max_value=infinity, verbose=0, random_state=None, add_indicator=False, keep_empty_features=False | imputation_sequence_, initial_imputer_, n_iter_, n_features_with_missing_, indicator_, n_features_in_, optional feature_names_in_; fit, transform, fit_transform, get_feature_names_out |

Optional target `y` is ignored by all four. Transform outputs retain input row
order, with imputed retained features followed by fitted indicators; standalone
indicator output is Boolean. Native callbacks are statically typed callable
objects. Per-feature bounds use owned dense vectors with original feature width,
or a scalar broadcast policy. Feature-name metadata uses owned List[String]
with one entry per original input column; output names select retained indices
and append `missingindicator_` plus each original indicator feature name.

Use typed Params, in-place fit/set_params, associated dense output types and
ValueError/NotFittedError-prefixed Mojo Error per [interfaces.md](interfaces.md).
Fit replaces learned state, clone clears it, and parameter changes must not
leave stale fitted caches silently usable. No sample_weight, incremental fitting,
probability or inverse capability is imposed on the family trait. Feature names
follow retained then indicator ordering; string metadata belongs to child public
contracts. Python reflection, metadata routing and serialization remain bindings.

## Dependency assembly and validation

The family, both isolated consumers and the simple algorithm use flattened
`DEPENDENCIES.json` manifests (schema 1, no revision or file-hash pins) with the
live shared assembler at `../../shared/assembly`. Family-owned module entries include concrete simple and KNN algorithm sources
and the deferred API declarations. `EXPORTS.json` selects concrete estimators
where implemented; shared baseline ownership and namespace collision checks
remain enforced. Every build reads the current files at each declared path:
Moxie HEAD (including working edits), its shared numerics/linalg projects,
`../../../mojo-linalg`, `../../../../NuMojo` and the integrated `../neighbors` owner.
KNN imports the neighbors contracts, metric and query engine, with their
flattened numerical dependencies. Missing sources fail assembly; there are no
cached snapshots or Git recovery.

Run `pixi run build`, `pixi run test` and `pixi run parity` from the family root.
Build/test compile and exercise the root namespace, simple and KNN algorithm contracts,
examples and both isolated consumer trees. Consumer probes test concrete public
imports as well as shared layouts, storage and module ownership. A manifest
naming a missing source file and conflicting owners/exports must fail assembly
without replacing the last successful result.

The root parity command compiles differential drivers against the root's
`builds/deps`, using Moxie's locked Python validation environment.
The local scikit-learn estimator bodies remain the reference. Child build/test/
parity commands are also supported and consume the same family modules.

Keep the manifests' module inventory and license assets current when adding,
removing or relocating modules. Generated namespaces, binaries and raw results
stay Git-ignored.
