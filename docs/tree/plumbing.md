# Tree plumbing contract

This family defines native dense records and extension points, with storage,
weighted criteria, numeric splitters, builders, pruning and public estimators
implemented in Moxie. See scope.md for remaining gaps. The authoritative signatures are
`moxie.tree._records`, `_engine_api` and `_dense_api`. Existing `_base` contracts
are preserved. Dense consumers opt into `DenseTreeInspection` and the dense
classifier/regressor traits; legacy `TreeInspection` additionally requires sparse
`decision_path` and is not a promise that this method is implemented.

## Ownership and dependency graph

Moxie owns every module below in `src/moxie/tree`.

| Modules | Responsibility |
| --- | --- |
| `_records`, `_engine_api`, `_dense_api`, `_base` | Records and engine/estimator traits |
| `_tree` | Allocation, traversal, inspection and values |
| `_criterion` | Weighted classification and regression criteria |
| `_partitioner`, `_splitter`, `_utils` | Reference partitioning, sorting and split search |
| `_builders` | Depth-first and best-first builds and pruning |
| `_classes`, `_extra_classes` | Four estimators and prepared-input interface |
| `_export` | Text, DOT, SVG and native Balsa Treelite export |

Forests consume these same modules. `_records` owns representations and
`_engine_api` owns signatures. Extra trees specialize the decision-tree engine.
Public names retain sklearn spelling through Moxie's root export manifest.

## Dtypes, shapes and ownership

Supported platform is linux-64: native Int matches signed 64-bit sklearn intp.
`Features = numojo.NDArray[float32]`, `Values = NDArray[float64]`,
`Indices = NDArray[int64]`, `Mask = NDArray[uint8]`,
`Constraints = NDArray[int8]`. Shape/index metadata use NuMojo Shape and Item.
No Python conversion or I/O helper may execute in learning or inference.

TrainingData owns handles to X `(n_samples,n_features)`, y
`(n_samples,n_outputs)`, optional weights `(n_samples,)`, missing mask
`(n_features,)`, category counts `(n_features,)` and monotonic signs
`(n_features,)`. None weights means unit weights. Signs are -1,0,+1. Category
count -1 denotes numerical; nonnegative counts denote encoded categories,
including zero for an all-missing column. Engine boundaries receive rank-2
Float32 X and Float64 y. Classifiers use shared `moxie._labels` to encode
string/Int64/Float32/Float64 labels into class IDs before this boundary, and
decode predictions into their original type afterward. Class domains never
pass through Float64. Native TreeTable feature names and dtype masks belong to
the scaffold records; estimators validate and resolve selectors before encoding. X may be strided with element strides and offset; y must have contiguous
output axis. Workspace feature_values and indices are contiguous vectors.
`validate_training_layout` checks these ranks, lengths, nonempty dimensions
and y output stride before engines inspect data; it does not check finite values
or criterion-specific label/weight restrictions. Do not assume NuMojo reshape, transpose, slices or dtype casts are views.

Borrow TrainingData for a build. It owns independent snapshots when fitting
must retain caller data; `snapshot_values` explicitly deep-copies through
NuMojo. Criterion initialize may keep explicit refcounted read-only views of y,
weights and samples for its current node. The builder ensures those inputs are
not mutated during criterion updates; partitioning must finish before initialize
or reset. Never retain raw pointers beyond a borrow or mutate overlapping views.
Each build owns independent criterion/splitter state. Concurrent mutation is
outside the contract. Fit builds fresh state and publishes after success; clone
copies configuration only. Copying a fitted tree is a separate deep operation.

TreeData owns `List[Node]` and value `(capacity,n_outputs,max_n_classes)` in
C row-major order, plus n_features, n_classes per output, n_categories per
feature, n_outputs, max_n_classes, max_depth, node_count and capacity.
`len(nodes)==node_count<=capacity`; unoccupied value rows are not predictions.
Value stride is n_outputs*max_n_classes; offset is
`node*stride + output*max_n_classes + class`. `value_offset` checks bounds and
signed-size multiplication overflow. Regression has one value per output;
classification stores weighted proportions in sorted classes_ order, with
unused class slots zero. Native Array/List metadata layouts are not a Cython
binary ABI; exchange uses these typed records, never raw struct casts.

Node fields mirror sklearn `_utils.Node`: left/right child and feature Int,
threshold Float64, 8 UInt32 category words (`CategoryBits` SIMD), impurity
Float64, n_node_samples Int, weighted_n_node_samples Float64,
missing_go_to_left UInt8 and split_kind Int8. Nodes use root ID 0, consecutive
creation IDs, child IDs or TREE_LEAF=-1; leaf feature/threshold are
TREE_UNDEFINED=-2. Depth-first builders create preorder IDs; best-first builders
preserve their expansion allocation order. Pruning creates a new compact tree
with remapped children, never changes IDs in an unrelated original tree.
Capacity growth invalidates internal borrows; storage owns resize and publication.

## Split and criterion boundaries

SplitRecord fields: feature, pos, threshold, left_cat_bitset, split_kind,
improvement, impurity_left/right, lower_bound/upper_bound and missing_go_to_left.
The sample interval is half-open `[start,end)`; nonleaf pos separates nonempty
children. A leaf has pos >= end and split_kind=-1. `validate_split` checks record
bounds/kinds, not numerical correctness. ParentInfo carries lower/upper value
bounds, parent impurity and n_constant_features. Bounds may be infinite but
cannot be NaN or reversed. They constrain predictions, not the impurity itself.

SplitWorkspace owns one sample permutation, feature permutation, constant feature
prefix, temporary feature values, start/end and n_missing. A partitioner mutates
that workspace, preserving sample IDs and X/y row correspondence. Its trait
includes sort, next candidate, min/max, final partition, missing shift and
categorical bitset/category order operations. Returned feature ranges and
positions replace C pointer output arguments. Float32 FEATURE_THRESHOLD=1e-7,
midpoint overflow handling and categorical weighted ordering are validated by
the splitter owner. Sample order is observable: it fixes weighted summation
order, and near-tied splits are decided by the last bit. Numeric sorting
therefore ports the reference missing-value compaction and 3-way introsort
(`simultaneous_sort`), and criteria keep the reference evaluation order,
including tree `log` as `ln(x)/ln(2)`. Do not substitute a library sort.

Concrete criteria are configured with CriterionConfig: n_outputs, per-output
n_classes (one for regression), and n_samples for scratch capacity. Concrete
splitter/builder constructors accept their corresponding config records plus
owned engine instances; static type parameters select implementations, not
string-based runtime dispatch. Engine-private scratch fields may extend state
without replacing these records or changing entry-point signatures.

Criterion owns per-node weighted state. initialize binds the samples interval
and global weighted_n_samples; reset starts with empty left, reverse_reset with
empty right, update moves to new_pos. node/children impurity, node_value,
clip_node_value, middle_value, true/proxy improvement and monotonicity checks
are separate entry points. node_value writes Float64 `(n_outputs,max_n_classes)`
into a preallocated contiguous destination. Classification counts/probabilities,
regression sums, MAE weighted median machinery and Poisson statistics belong to
criteria. Gini, entropy/log_loss, squared_error, absolute_error and poisson are
implemented branches (deprecated friedman_mse maps to squared_error); do not substitute one criterion for another.

Splitter initialize receives a borrowed mutable shared RandomState, initializes
its tree-local rand_r state with the exact reference seed draw, and owns that
state thereafter. Shared RandomState does not itself promise sklearn's tree
xorshift stream. The tiny tree-specific stream adapter belongs to `_utils` in
the splitter owner, with exact draw-consumption tests. node_reset returns node
weight; node_split returns a SplitRecord and updates the constant-feature count
in ParentInfo. Best/random implementations reuse one criterion and partitioner.
SplitterConfig carries resolved max_features, min_samples_leaf and
min_weight_leaf. The resolved weight threshold may be negative when signed
training weights have a negative total; accepted children still require positive
weight. Estimators resolve public string/fraction options, not engines.

Split kinds are leaf=-1, numeric=0, categorical bitset=1, categorical hash=2.
Numeric uses `value <= threshold`; NaN uses the stored missing direction before
any other routing. Best search tests missing-left/right, including missing-only
splits; tied direction is right. Unseen missing values use the larger child,
with right for a tie. Random splitting chooses a missing direction as in the
reference. Bitsets encode ordinal 0..255 as word=c//32, bit=c%32. Hash splits
store the UInt32 seed in word 0 and use sklearn lowbias32(seed XOR category)&1;
the remaining words are zero. Random categories allow up to 2**24 distinct
Float32-exact integer codes. Estimator-owned encoding preserves category order
and maps unknown categories to NaN. No string is coerced into a numeric threshold.
Categorical columns cannot carry nonzero monotonic signs. Monotonic constraints
reject multioutput/multiclass and unsupported missing-value configurations;
binary classification flips signs internally for the negative-class probability.
Splitter and builder propagate bounds and clip leaf values consistently.

## Storage and builders

TreeStorage owns resize, add_node, apply, predict, node depths and importances.
apply yields Int64 `(n_samples,)` leaf IDs; predict yields Float64
`(n_samples,n_outputs,max_n_classes)`. Estimators remove singleton dimensions
according to sklearn, choose class argmax (first on ties), and supply log
probability/score semantics. Node depths use root=1 as sklearn inspection does;
builder max_depth uses root=0. Importance uses weighted impurity decreases and
optional normalization; zero-total importance stays zero.

TreeBuilder takes TreeData, TrainingData and mutable RandomState and mutates
fresh tree state. Concrete builders compose the exact storage and splitter
implementations; static trait dispatch replaces Cython object dispatch.
BuilderConfig carries resolved min_samples_split/leaf, min_weight_leaf,
max_depth, min_impurity_decrease and max_leaf_nodes (-1 means unlimited).
Depth-first is used without a leaf limit; best-first respects the leaf budget.
Pruning returns a new TreeData; cost_complexity_pruning_path returns two dense
Float64 vectors in PruningPath. These bodies live in the child owners and are assembled into the family API.

## Ensemble build boundary

Forests consume trees through `moxie.tree._classes`, owned by decision_tree:
`BaseDecisionTree.prepare_fit` returns `PreparedTreeInput`, and
`fit_prepared[_with_random_state]` builds a member from it with per-member
weights, mirroring sklearn's `tree._fit(check_input=False)`. The prepared input
owns the encoded TrainingData parts; each build borrows views of X and y and
owns only its weights. `TreeInputEncoder` holds fitted feature names, categorical
mask and dictionaries; every member of one prepared input shares it, so an
ensemble encodes prediction input once and calls `*_encoded` per member.
Forest bootstrap draws, seeds, aggregation and OOB state belong to ensemble.

## Assembly and native compatibility

Moxie's root `DEPENDENCIES.json` declares the tree modules under the `moxie`
owner. All implementations live in `src/moxie/tree`; tests live in `tests/tree`.
There are no child engine projects or inherited framework copies. Generated
namespace initializers and source receipts live under `builds/deps`.

Use Mojo 1.0.0 and MAX 26.5.0 with `-O3 --fp-mode=contract=off`.
Balsa remains external at `../../../nvidiads/balsa` relative to Moxie, including
its native Treelite v4 codec. NuMojo, numerical kernels, labels and RNG retain
their existing live-source dependency owners. License notices are in
`LICENSES/tree-*.txt`.

`pixi run tree-test` checks storage, criteria, estimators, exports, ownership,
and two independent external assemblies. `pixi run tree-prepare-oracle` builds
the pinned scikit-learn source into `builds/tree/oracle-env`; `pixi run tree-parity`
checks estimator structure, labels, categorical words, prepared forest members,
text/DOT rendering, and Treelite bytes. The storage and criterion suites compile
their original pinned Cython oracle bodies separately. No installed-wheel
fallback is used. Root build, test and parity commands include these suites.
