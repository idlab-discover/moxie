# Preprocessing plumbing

These modules define the native dense contracts and small validation/ownership
helpers that the family's implementations of all 29 sklearn preprocessing
public exports share (see [api.md](api.md)).
The existing `_base` traits and inherited source remain unchanged. Source
semantics follow sklearn `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

## Storage and native compatibility

Consume `moxie/shared/numerics` directly. `numojo.NDArray[DType.float32]` and
`NDArray[DType.float64]` are the accepted numeric inputs/outputs, with samples
in rows and features in columns. `Shape` and `Item` are NuMojo metadata.
Rank-2 input may be contiguous or a valid positive/negative strided view with
nonzero offset. Strides count elements. Helpers borrow arrays and retain no
aliases. `.view()` shares storage and lifetime; `.copy()` makes an independent
snapshot, possibly copying an entire backing allocation for a strided view.
Reshape, transpose and dtype conversion may allocate. Concurrent mutation and
overlapping in-place operations are unsupported. Concrete `copy=False` paths
must separately validate writability/overlap and parity; it is not permission to
retain caller aliases in learned state. Outputs move ownership to callers.

Use the numerical owner's accepted operations and exceptions listed in AGENTS.md.
This scaffold adds validation scans, not replacement numerical kernels. No
backend selection, custom buffer or additional numerical dependency is introduced.
The locked compiler is Mojo 1.0.0 with MAX 26.5.0. NuMojo sources are
read from the live path declared in DEPENDENCIES.json. Library code is pure Mojo.
Python runs assembly/check tooling only. Preserve `--fp-mode=contract=off` when
using foundation RNG; consume `moxie._random.RandomState`, not NuMojo sampling.

## Implemented shared interfaces

All private module names below are under `moxie.preprocessing`.

| Module and interface | Exact contract |
| --- | --- |
| `_validation.validate_dense[dtype](x, *, allow_nan=False, min_samples=1, min_features=1)` | Rank-2 Float32/64; dimensions at least the nonnegative minima; infinity always rejected; NaN optionally allowed. No mutation, allocation or fit-state change. |
| `_validation.missing_at[dtype](x, row, col) -> Bool` | Bounds/rank-checked lazy numeric mask; True iff that element is NaN, not infinity. Uses source strides/offset. No allocation or retained alias. |
| `_validation.require_dense_output(sparse_output)` | False passes; True raises a descriptive error, including defaults requiring sparse output. |
| `_validation.validate_category_code(code, size, *, allow_unknown=False)` | Integer code in `[0,size)`; optional private unknown sentinel -1; negative size always rejected. Missing and unknown are distinct. |
| `_metadata.FeatureNames` | `record_fit(n_samples,n_features,names: Optional[List[String]]=None)` snapshots optional names after validation; `get_feature_names_out(input_features=None)` returns independent names or x0…xN. Supplied names must match known names and count; duplicates allowed. `reset()` restores unfitted state. |
| `_contracts.FeatureMoments` | Constructor `(count,mean,variance)` borrows and copies three Float64 rank-1 NDArrays of equal feature length. Public fields have those names. Copies of the record own independent arrays. Shape validation only; it never computes/repairs statistics. |

Validation errors use Mojo `Error` with ValueError/NotFittedError prefixes.
`FeatureNames` uses foundation `FittedState`, but estimators expose their own
`n_features_in_` and optional `feature_names_in_`. Failed validation leaves old
metadata intact; successful repeat fit replaces names, including removing old
names on unnamed input. One-to-many encoders/polynomials generate their own
output names and reuse input-name validation, not one-to-one output naming.

Numeric missing masks are predicates, an explicit native adaptation of a NumPy
Boolean array: O(1) auxiliary storage and O(1) lookup, no dtype/container fork.
A retained learned mask must be computed from an owned snapshot. String/None
missing categories belong to the typed vocabulary contract, not IEEE NaN tests.
Validation scans cost O(n*p); feature names cost O(p) strings; moments snapshots
cost three independent O(p) Float64 arrays for contiguous vectors. No speed claim.

## Extension interfaces and owners

`CategoryVocabulary` is implemented by `_vocabulary.FeatureVocabulary`, the
family category engine used by the encoders ([encoders](api/encoders.md)).
`LabelTransformer` is implemented by LabelEncoder, LabelBinarizer and
MultiLabelBinarizer ([labels](api/labels.md)) over the shared `moxie._labels`
engine, with label-only canonicalization kept in `_label`. The
other traits in `_contracts` remain signature probes.

- `LabelTransformer(BaseEstimator)` associates `Labels` and `Encoded` types,
  requiring `fit(y)`, `transform(y)`, `fit_transform(y)`, `inverse_transform(y)`.
  Fit mutates in place. This target-only signature is distinct from existing
  `TargetPreprocessor.fit(X,y)` used by TargetEncoder. Label classes must opt
  into only applicable interfaces; do not force feature counts onto target data.
- `CategoryVocabulary` associates a Copyable/Movable `Value`, exposes `size()`,
  `category(index)->Value` and `lookup(value, *, is_missing=False)->Int`.
  Codes are zero-based; lookup returns -1 for unknown, and raises for malformed
  input/unfitted state. `category` rejects out-of-range codes. Canonical categories
  own their values; returned values are copies. Numeric automatic categories
  sort ascending, strings lexically; a numeric NaN occupies the final category.
  Explicit numeric categories must be sorted/unique, with NaN last; explicit
  string categories preserve supplied order. The categorical child owns the
  engine; label, bins and target encoders consume it where rules agree.
  Per-column typed vocabularies permit numeric/string columns without Python
  object arrays; nullable/tagged values must distinguish None, NaN and unknown.
  No mixed-type ordering shortcut or loss of integer label precision is allowed.
  Public integer label outputs require an owner-tested NuMojo dtype extension
  before use; they must not be silently encoded as floating labels.
- `FeatureStatistics.compute[dtype](x, weights: Optional[NDArray[float64]]=None)
  -> FeatureMoments`, `merge(left,right)->FeatureMoments`: scalers own the future
  implementation. Inputs are matrices; weights are sample-length vectors.
  Results have p entries, Float64 accumulation, per-feature nonmissing effective
  counts and population variance (ddof=0). Zero-weight/missing/all-missing and
  near-constant rules follow StandardScaler; merge must match its incremental
  semantics. Robust quantiles and min/max are separate engines, not fake moments.
  Fitted `n_samples_seen_` scalar/vector and integer/weighted forms are estimator
  projections of this internal record, not an ABI change to learned attributes.
- `DenseOutputBuilder.build[dtype](n_samples,n_features_out)->NDArray[dtype]`
  reserves explicit owned, row-major rank-2 construction. Dimensions must be
  nonnegative, arithmetic overflow rejected; every element must be initialized
  before return. Encoder one-hot construction, expansion and dropping categories
  belong to children. Sparse intermediate construction is unnecessary in dense
  modes. Output ordering and dtype are estimator-specific contracts.

Each extension has a module owner. Revise family interfaces and their consumers
together when needed. Genuinely global additions belong in Moxie; keep a
reproducing example and coordinate the shared change there.

## Algorithm ownership and module boundaries

All rows depend on `preprocessing.scaffold`; paths are relative to this family.
They are future sessions, not scaffold deliverables.

| Child | Public exports | Additional dependencies |
| --- | --- | --- |
| `algorithms/scalers/` | StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, scale, minmax_scale, maxabs_scale, robust_scale | none |
| `algorithms/simple/` | Normalizer, Binarizer, KernelCenterer, normalize, binarize, add_dummy_feature | none |
| `algorithms/labels/` | LabelEncoder, LabelBinarizer, MultiLabelBinarizer, label_binarize | none |
| `algorithms/categorical/` | OneHotEncoder, OrdinalEncoder | none |
| `algorithms/quantiles/` | QuantileTransformer, PowerTransformer, quantile_transform, power_transform | none |
| `algorithms/polynomial/` | PolynomialFeatures, SplineTransformer | none |
| `algorithms/function/` | FunctionTransformer | none |
| `algorithms/target_encoder/` | TargetEncoder | preprocessing.categorical, model_selection.splitters |
| `algorithms/bins/` | KBinsDiscretizer | cluster.kmeans, preprocessing.categorical |

Public imports match `sklearn.preprocessing`. `_label`, `_encoders`,
`_function_transformer`, `_polynomial`, `_target_encoder`, `_discretization`
remain the natural private module boundaries. sklearn `_data.py` spans three
independent children: assign unique native modules `_scalers`, `_simple`,
`_quantiles`, with explicit public exports. This deliberate adaptation avoids
three owners claiming `_data`; no source fork or import-order selection.
Categorical vocabulary implementation may live in categorical's `_vocabulary`;
scalers' compatible moments implementation in `_statistics`. Labels precede
categorical in the task graph: consume the vocabulary trait now, and keep
label-only canonicalization separate until a reviewed shared owner dependency
is available. Never copy a future feature vocabulary into the label child.

Preserve constructor defaults and opt-in capabilities. Standard/minmax/maxabs
incremental support does not imply RobustScaler partial_fit. TargetEncoder
fit_transform requires cross-fitting, not fit then transform. FunctionTransformer
uses a typed callable; child tests must cover forward/inverse and feature naming.
FunctionTransformer ([function](api/function.md)) implements it with
`ArrayFunction` struct types and `thin` function adapters.
Quantile/power, polynomial/spline, category frequency/drop/unknown handling and
KBins kmeans branches are substantive dedicated work, not stub results.
TargetEncoder ([encoders](api/encoders.md)) reuses the encoders' vocabulary
core and the label estimators. Until `model_selection.splitters` exists, its
`_target_encoder.kfold_test_folds` / `stratified_kfold_test_folds` are a local
stand-in that reproduces sklearn's KFold/StratifiedKFold folds; replace them
with the splitters dependency when it lands.
KBinsDiscretizer ([discretization](api/discretization.md)) runs its k-means
branch on a family-private 1-D Lloyd stand-in for `cluster.kmeans`, scoped to
sklearn's explicit-init single-run call, and expands its fixed integer
categories directly rather than through `_vocabulary`. Its percentiles share
`_percentiles` with SplineTransformer.
QuantileTransformer and PowerTransformer ([quantiles](api/quantiles.md)) use
two family-private numerical modules: `_special_math` (Cephes normal CDF and
quantile, Box-Cox, NumPy-order sums, percentile and `interp`; scalers' normal
quantile also comes from here) and `_optimize` (SciPy `brent`/`fminbound`).

## Dependency assembly and verification

`DEPENDENCIES.json` declares live dependency paths, module ownership, explicit
exports and license assets. The current Moxie assembler reads these sources on
every build, including working edits. Missing paths or files fail explicitly.
Source hashes and revisions do not gate builds, and no historical cache is used.
Generated `builds/deps/ASSEMBLY.json` receipts record the bytes actually consumed.
Retain Pixi toolchain locks, the scikit-learn oracle revision and baseline proofs.

Each consumer composes one namespace under ignored `builds/deps`. Namespace and
export collisions remain errors, with explicit overrides required where needed.
Two isolated consumers exercise the shared dense operations and family contracts.
Tests verify live source changes, missing-file failures and collision rejection
without replacing the last valid namespace. `pixi run test` builds every declared
module, contract and example before running checks. Numerical engines and public
estimators additionally require their own parity and memory validation.

The [family development policy](development.md) governs the whole repository.
Family helpers and their consumers can be updated together; external framework
components retain their ownership in Moxie.
