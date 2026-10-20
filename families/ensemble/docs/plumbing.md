# Ensemble plumbing

These are the dense contracts shared by every ensemble engine. The forest engine
(`_forest_engine`, `_sampling`, `_aggregation`) implements them; see
[forest_engine.md](forest_engine.md). Existing `moxie.ensemble._base` traits remain unchanged. New contracts live in `_records`,
`_engine_api` and `_plumbing`; those files belong to Moxie core. All child
sessions must consume them through Moxie's shared assembler. Shared base
clone and lifecycle contracts remain owned by Moxie, RNG by `moxie/shared/random`,
and storage by `moxie/shared/numerics`. No child may replace those implementations.

## Implementation owners

The table records the original implementation plan. Only the forest engine and
four public forest estimators are implemented; other entries remain future work.
All implemented modules are in Moxie core, under `src/moxie/ensemble`. All module names are relative to `moxie.ensemble`.

| Task / path | Modules; exports | Dependencies beyond ensemble.scaffold |
| --- | --- | --- |
| hist_bins / shared/hist_bins | `_hist_gradient_boosting.binning`, `.histogram`; private | none |
| voting / algorithms/voting | `_voting`, `_heterogeneous`; VotingClassifier, VotingRegressor | dummy.estimators, naive_bayes.gaussian |
| hist_grower / shared/hist_grower | `_hist_gradient_boosting.splitting`, `.grower`, `.predictor`; private | ensemble.hist_bins |
| hist_gradient_boosting / algorithms/hist_gradient_boosting | `_hist_gradient_boosting.gradient_boosting`; HistGradientBoostingClassifier, HistGradientBoostingRegressor | ensemble.hist_grower, model_selection.splitters, preprocessing.categorical |
| stacking / algorithms/stacking | `_stacking`; StackingClassifier, StackingRegressor | ensemble.voting, model_selection.validation, linear_model.logistic, linear_model.ridge_cv |
| adaboost / algorithms/adaboost | `_weight_boosting`; AdaBoostClassifier, AdaBoostRegressor | tree.decision_tree |
| gradient_boosting / algorithms/gradient_boosting | `_gb`; GradientBoostingClassifier, GradientBoostingRegressor | tree.decision_tree, model_selection.splitters |
| forest_engine (implemented, src/moxie/ensemble) | `_forest_engine`, `_sampling`, `_aggregation`; BaseEnsemble | tree.decision_tree, tree.extra_tree |
| random_forest (implemented) | `_forest`; RandomForestClassifier, RandomForestRegressor | ensemble.forest_engine |
| extra_trees (implemented) | `_forest`; ExtraTreesClassifier, ExtraTreesRegressor | ensemble.forest_engine |
| random_embedding / algorithms/random_embedding | `_random_embedding`; RandomTreesEmbedding | ensemble.forest_engine, preprocessing.categorical |
| bagging / algorithms/bagging | `_bagging`; BaggingClassifier, BaggingRegressor | ensemble.forest_engine, dummy.estimators |
| isolation / algorithms/isolation | `_iforest`; IsolationForest | ensemble.bagging |

The native split of sklearn `_forest.py` avoids overlapping ownership among
independent contributors. BaseEnsemble is a public abstract base in the reference;
its future implementation is exported from `_forest_engine`, leaving the frozen
`_base` capability traits intact. Private BaseForest/BaseBagging/VerboseReporter
are not promised public exports; `ForestEngine` is the native BaseForest. Voting owns the heterogeneous container adapter
implementation shared with stacking; neither imports tree internals.

## Arrays and ownership

Use NuMojo 0.10.0 revision `515fb2856f0ecf3d2740a34d958fe168183e1129`,
Mojo 1.0.0 and MAX 26.5.0. `Values=NDArray[float64]`,
`Features=NDArray[float32]`, `Indices=NDArray[int64]`, `Mask=NDArray[uint8]`
are imported from the tree family's exact records (aliases of the same NuMojo
types). `Bins=NDArray[uint8]`, `Gradients=NDArray[float32]`,
`Counts=NDArray[uint32]` add histogram-specific dtypes, not new containers.
Shape/Item are NuMojo metadata. The integer and byte arrays are structural
indices/counts; no general integer numerical API is implied.

Borrow inputs during computation; retain `.copy()` snapshots when independent
fitted data is required. `_plumbing.snapshot[dtype]` explicitly deep-copies.
`.view()` shares mutation and retains buffer lifetime; strides count elements,
including negative strides and offsets. A strided copy may copy the whole backing
allocation. Slicing, reshape, transpose and casts must not be assumed zero-copy.
Owned outputs transfer with Mojo moves. Concurrent mutation and overlapping
in-place writes are unsupported. Fitted state is distinct from configuration.

Forest and classic boosting tree calls use `moxie.tree._records.TrainingData`,
TreeData and the exact tree estimator/engine traits. X enters trees as Float32;
y/weights/value accumulation are Float64. Forest apply is `(samples,trees)`
Int64; classic boosting apply may add a class axis. Histogram X is Float64,
not silently rounded through ordinary tree Features. Heterogeneous adapters
retain their associated Input/Target types, so classifier labels need not be
numeric. Encoded internal labels are Int64 with stable sorted `classes_` per
output. Public label values stay in the estimator-owned typed label vocabulary.

Numeric prediction matrices use `(samples,outputs)`; probability is a
`List[Values]` of `(samples,classes_for_output)` matrices, avoiding ragged padding.
Single-output estimators remove the output/list dimension at their public API.
`ClassAlignment.columns` is an Int64 vector mapping local class positions into
global columns; absent classes contribute zero probability. Validate uniqueness,
range and length with `validate_alignment`. Hard votes count encoded labels and
break ties at the first global class; soft votes use weighted probabilities.
`EnsembleAggregator` adds into caller-owned sums; normalization divides each row
by its supplied denominator and must apply the consumer's zero-denominator
policy. No universal OOB policy is imposed by this signature. The forest_engine owner
implements this aggregation trait for forests/bagging. Voting consumes the shared
layouts and alignment validation but owns its voting-specific composition of
accepted dense operations; it does not acquire a tree or forest-engine dependency.

`SamplingPlan` contains resolved positive population/draw counts and independent
bootstrap flags. Resolved sample counts may exceed the population with replacement
(e.g. weighted effective counts in the pinned reference), but never without it;
feature counts stay bounded by n_features. `SampleSelection.rows/features` preserve draw order and repeats;
`multiplicities` has one Int64 entry per original sample. Sampling draws features
before rows for bagging. Optional Float64 `(samples,)` weights are sampling mass
when the reference uses weighted choice, not automatically fit weights. Owner
validates mass as NumPy does (on the normalized probabilities), preserves exact
draw consumption and tests weighted sampling without replacement against the
reference (`BootstrapSampler`, `BaggingSampler` in `_sampling`). Shared RandomState supplies primitive draws; a missing primitive
belongs to foundation.random, not a second RNG. OOB indices are sorted original
rows with zero multiplicity, and must reproduce the training selection.

`OOBAccumulator.sums` is Float64 `(samples,outputs_or_classes)` and counts is
Int64 `(samples,)`, one accumulator per output where necessary. Forest zero
counts use divisor one with warning, preserving zero sums. Bagging normalization
can yield NaN where no prediction exists and must reproduce its warning/score
behavior. Forest OOB requires bootstrap, bagging additionally rejects warm_start.
`validate_oob` checks only these configuration rules, when OOB is requested.
OOB scores use each estimator's reference accuracy/R2 or allowed callable;
boosting's loss-based OOB improvements are separate algorithm state.

## Clone, seed, warm start and heterogeneous extension

Reuse `moxie.base.clone`, never copy fitted members to implement clone.
`SeededEstimator.set_random_states` visits every deep parameter ending in
`random_state` in sorted parameter-name order. `_plumbing.next_estimator_seed`
uses shared RandomState.randint with exclusive upper bound 2147483647. A typed
adapter explicitly enumerates fields; there is no Python reflection or dynamic
string parameter dictionary. Apply ensemble overrides to a fresh clone before
seeding. No random draw is consumed merely by constructing the ensemble.

`additional_estimators` validates counts and returns requested for a replacement
fit, or requested-minus-fitted for warm_start; shrinking raises, equality returns
zero. Caller emits the reference no-growth warning, handles validation and keeps
existing members. For integer-seeded forests/bagging replay the reference seed
draws for retained members before adding new members. Mutable RandomState input
retains its reference stream semantics; do not blindly reset it. Gradient and
histogram boosting retain their own stage, validation/early-stop, loss and RNG
state. Normal fit replaces state; cloning always drops it. Failed fit behavior
must be tested by each concrete estimator rather than inferred from this helper.

`NamedEstimator[E]` owns name and Optional[E]; None represents sklearn's 'drop'.
E can be a closed native Variant adapter supplied by voting, using compile-time
capabilities rather than runtime Python objects. `validate_names` accepts all
constructor names as the reserved list, rejects duplicate/nested names and
all-dropped lists. Fit members are fresh recursive clones in original active
order; configuration preserves dropped slots. Named lookup keeps original names.
`EstimatorBank` supplies indexed prediction; ProbabilityBank and DecisionBank
are independent opt-ins, so regression/hard voting do not require probabilities.
Nested updates use typed Params and in-place set_params; adapter implementations
must preserve deep configuration traversal and clone independence. Weighted fit
must reject unsupported active members explicitly, never discard weights.
Stacking keeps cross-validation fitting with model_selection.validation;
`prefit` uses fitted members only when explicitly requested. Binary probability
stacking drops the first class column; passthrough appends dense X after member
outputs. No stacking fitting algorithm is part of this scaffold.

## Histogram boundary

Binner.fit receives categorical feature indices (empty for all numerical).
BinMetadata owns one Float64 threshold vector per feature, Int64 nonmissing bin
counts `(features,)`, byte categorical mask and per-feature known-category
vectors. missing_bin is the dedicated uint8 index `max_bins`, with max_bins
in 2..255 for the public estimators (total storage bins max_bins+1).
Numeric thresholds are sorted; categorical bins refer to encoded categories.
Unknown/negative categories and NaNs follow the reference missing routing.
Training Bins has `(samples,features)` Fortran element strides `(1,samples)`;
inference may be C contiguous. Binner owns materialization and metadata lifetime.
Float32 gradient/Hessian vectors are `(samples,)` for one tree/class; a constant
Hessian is a length-one vector with constant_hessian=True. Gradient boosting
losses supply weighted derivatives; histogram accumulation must not weight twice.

Histogram uses structure-of-arrays: Float64 gradient and Hessian sums and UInt32
counts, all `(features,total_bins)` C contiguous. This deliberately replaces the
Cython structured dtype without changing precision/count semantics. Accumulate
selected sample rows; subtraction creates parent-minus-sibling counts and sums,
with underflow and numerical guards owned by hist_bins. Memory is O(features*bins)
plus O(samples*features) bins, no float64 copy per candidate split.

HistogramSplit carries winning feature/bin, gain, categorical bitset, missing
direction, child counts and derivative sums. No split has feature=-1 and gain
negative infinity. CategoryBits is tree's shared eight UInt32 words (256 bits),
but histogram growth never uses ordinary TreeData/builders. GrowerConfig holds
resolved limits (max_depth/max_leaf_nodes=-1 for unlimited), regularization,
minimum Hessian/gain and shrinkage/feature fraction. Monotonic Int64 signs are
`(features,)`; interaction groups are Lists of feature-index vectors. Allowed
features and current lower/upper prediction bounds enter split search explicitly.
Owners enforce monotonic categorical exclusions, tie order, missing routing and
histogram subtraction reuse against reference tests.

HistogramPredictor owns List[HistogramNode] and raw/binned category bitsets.
Node fields mirror reference PREDICTOR_RECORD_DTYPE by name, not byte ABI;
children refer to zero-based node indices and is_leaf controls validity.
Grower owns best-first expansion, partition scratch, leaf updates and final
prediction nodes. predict/predict_binned return Float64 `(samples,)`. Hist
boosting retains stage/class lists of these predictors and owns loss-specific
leaf updates, early stopping, baseline prediction and preprocessing.

## Dependency assembly and compatibility

Moxie's root `DEPENDENCIES.json` owns the ensemble modules and license assets.
The family checker derives every assembly from that live manifest and verifies
its ensemble modules and exports against the source directory and family
`EXPORTS.json`. It rejects any standalone ensemble source dependency.
Two consumer contribution manifests add only their test modules. Generated
sources remain under ignored `builds/deps` directories.

Children add unique owned modules and export manifests; package initializers are
generated, not concatenated. Duplicate module/export owners fail. A relocation
requires the assembler's explicit override and documented authorization; it is
not permission to fork shared helpers. Two isolated consumers compose both
contributions and call the same family snapshot, dense reduction and RNG code.
The producer in one contribution transfers a Values array to the other, and
shared-view mutation remains visible without an intermediate copy.
Tests reject missing sources and module/export conflicts without replacing good output.
Compile using `-O3 --fp-mode=contract=off` to retain shared RNG arithmetic behavior.
All runtime code is native Mojo/NuMojo/MAX; Python orchestrates checks only.

Extensions may add private engine scratch; changing records/signatures requires
reviewed contract updates and migrated consumers. Public estimator
names/defaults/learned attributes remain sklearn's; language adaptations are
associated types, in-place mutators, typed banks, explicit optional weights and
Mojo Error messages. `DenseForestInspection` omits sparse decision_path while
legacy ForestInspection remains unchanged. See scope.md for exclusions.
