# Naive Bayes internal contracts

Moxie owns the public `moxie.naive_bayes` facade and all implementations.
`_nb_gaussian`, `_nb_counts`, `_nb_bernoulli` and `_nb_categorical` own
model-specific likelihoods and updates. `_naive_bayes_common` and
`_naive_bayes_dense` supply family validation, sufficient-statistic records and
label adapters. Global labels and array operations remain in their shared owners.
All canonical estimators preserve their List interfaces. Legacy estimators and
unused List counting/prediction implementations are not assembled.

## Existing shared interfaces

The implementations of these functions were moved unchanged from the original
family module. `M` below means `List[List[Float64]]`; `V` is `List[Float64]`.
Borrowed inputs are not retained. Returned lists own their storage.

| Shared signature | Contract |
| --- | --- |
| `_check_X(X: M, expected=0, nonnegative=False) raises -> Int` | Nonempty rectangular finite input; returns feature count; expected=0 skips comparison |
| `_weights(n: Int, sample_weight: Optional[V]) raises -> V` | Owned ones or copied weights; finite nonnegative values, positive finite total |
| `_check_prior(prior: Optional[V], n: Int, normalized=False) raises` | Length and finite nonnegative values; optional Gaussian sum check |
| `_log_priors(counts: V, prior: Optional[V], fit_prior: Bool) raises -> V` | Explicit, empirical, or uniform priors |
| `_alpha(alpha: Float64, force_alpha: Bool) raises -> Float64` | Scalar nonnegative smoothing; optional floor at 1e-10 |

Every update validates before committing state. Ordinary fit clones parameters
into a fresh model; partial_fit copies fitted state, validates labels/features,
updates sufficient statistics, recomputes probabilities, then moves the result
into self. Failed updates preserve the old model. Clone copies configuration
only; copying a fitted object is separate. Parameters are never learned buffers.

## Dense extension ABI

`from numojo import NDArray, Shape, Item` is the storage import. `DenseMatrix`
is exactly `NDArray[DType.float64]`, also used for rank-one numeric vectors;
the name is an alias, not a container or runtime rank guarantee. Dense
validation and snapshots accept Bool, signed/unsigned 8/16/32/64-bit integers,
Float32 or Float64 via their compile-time dtype parameter. Snapshots preserve
storage dtype. Discrete estimator arithmetic widens to Float64; Gaussian
selects Float32 or Float64 explicitly. Integer storage is
validated locally by `tests/test_input_compat.mojo`; floating numerical operation
contracts at the external numerics owner remain unchanged.

Class labels come from Moxie's shared `moxie._labels` (`moxie/shared/labels`,
consumed live from Moxie HEAD): sorted domains, Int64 codes, `type_of_target` and
`check_classification_targets`. The family adds only single-output adapters
implementing `ClassDomain`: `IntListDomain` for the preserved List[Int] target, `NumericDomain[label]` for `NDArray[int64 | float32 |
float64]` targets and `StringDomain` for rank-one `StringTarget`. Each provides
`from_classes` (sklearn `unique_labels`), `check_target` (rank one plus target
kind; returns the length), `encode` (rejects unseen labels), `decode`,
`n_classes`, `same_classes`, `equal_at`, `empty_target` and owned `class_values`. Continuous, NaN or infinite
targets and rank-two targets raise. `Labels` remains `List[Int]` for the
preserved List estimators; `int_classes(domain)` returns an Int64 domain as a
List[Int].

| Type / function | Exact layout and semantics |
| --- | --- |
| `validate_matrix[dtype](X: NDArray[dtype], expected: Int=0, nonnegative: Bool=False) raises -> Int` | Rank 2, nonempty, finite; checks logical elements using NuMojo layout; returns features |
| `owned_snapshot[dtype](X: NDArray[dtype]) raises -> NDArray[dtype]` | Validates matrix then `.copy()`; never retains caller aliases |
| `prepare_batch[dtype, label](X, y: NDArray[label], current: Optional[NumericDomain[label]]=None, classes: Optional[NDArray[label]]=None, sample_weight=None, expected=0, nonnegative=False) raises -> LabeledBatch[NumericDomain[label]]` | First call needs `classes` (fit passes `classes=y`); later calls pass the fitted domain and optionally the same classes |
| `prepare_batch[dtype](X, y: StringTarget, current: Optional[StringDomain]=None, classes: Optional[StringTarget]=None, ...) raises -> LabeledBatch[StringDomain]` | Same contract for string labels |
| `prepare_batch[dtype](X, y: List[Int], current: List[Int], classes: Optional[List[Int]]=None, ...) raises -> LabeledBatch[NumericDomain[int64]]` | List[Int] compatibility; empty `current` means unfitted |
| `LabeledBatch[D](domain: D, batch: BatchContract)` | The domain to store as `classes_` and the batch encoded against it |
| `BatchContract(targets: NDArray[int64], n_classes: Int, weights: List[Float64], n_features: Int)` | Owned validated metadata; `targets[i]` is sample i's sorted class code; changing inputs cannot change the record |
| `CountStatistics(class_count: DenseMatrix, feature_count: DenseMatrix)` | Float64 vectors `(C,)` and matrix `(C,F)` |
| `TypedGaussianStatistics[dtype](class_count, mean, variance: NDArray[dtype], epsilon: Scalar[dtype])` | `(C,)`, `(C,F)`, `(C,F)`; variance is population variance including stored epsilon, removed before next merge. `GaussianStatistics` remains its Float64 alias |
| `CategoryStatistics(class_count: DenseMatrix, category_count: List[DenseMatrix], n_categories: List[Int])` | `(C,)`; F independent `(C,K_f)` matrices; F latest inferred category counts; allocated K_f never shrinks |
| `DenseLikelihood.joint_log_likelihood(self, X: DenseMatrix) raises -> DenseMatrix` | Child computes owned `(N,C)` scores after fitted/input checks; sorted class columns, row order preserved |
| `CountUpdater.accumulate(self, X: DenseMatrix, batch: BatchContract, mut statistics: CountStatistics) raises` | Adds weighted sufficient statistics to caller-owned transaction state; no finalization or hidden parameter mutation |

## Shared dense prediction

`_BaseNB` prediction is family-owned and works on the joint log likelihood
`jll` of shape `(N,C)`, independent of label type and model:

| Function | sklearn behavior reproduced |
| --- | --- |
| `joint_argmax(jll) raises -> NDArray[int64]` | `np.argmax(jll, axis=1)`: first maximum wins; the first NaN wins |
| `predict_labels[D](domain: D, jll) raises -> D.Target` | `classes_[argmax]`; column count must equal `domain.n_classes()` |
| `log_normalize[dtype](jll: NDArray[dtype]) raises -> NDArray[dtype]` | `jll - logsumexp(jll, axis=1)` with scipy 1.18 edge cases: NaN rows stay NaN, all `-inf` rows become NaN, a `+inf` entry gives NaN there and `-inf` elsewhere |
| `exp_normalize[dtype](jll: NDArray[dtype]) raises -> NDArray[dtype]` | `exp(predict_log_proba)`, preserving NaN |
| `weighted_accuracy[D](y_true: D.Target, y_pred: D.Target, sample_weight=None) raises -> Float64` | `accuracy_score` with the family weight validation; labels outside `classes_` count as wrong |

Argmax, label decoding and probability List conversion also accept either floating
dtype. Normalization rounds exponentials and logarithms to the input dtype
before subsequent arithmetic. `DenseNBPredictor` packages the Float64 path as
defaults. An estimator declares
`comptime Domain: ClassDomain`, implements `_class_domain(self) raises ->
Self.Domain` and `predict_joint_log_proba[dtype](self, X: NDArray[dtype]) raises
-> DenseMatrix`, and inherits NDArray `predict` (returning `Self.Domain.Target`),
`predict_log_proba`, `predict_proba` and `score`. Canonical estimators also provide List overloads that delegate to the same
array normalization helpers.

Records are fieldwise ownership transfers, not validators. Owners must validate
shapes before calling a numerical kernel. Copying records deep-copies their
NDArray fields. Public learned names ending in `_` remain estimator fields;
records are internal state, with no independent fitted flag. Gaussian and
categorical engines use their typed records, not CountUpdater's rectangular
count contract. Their update and finalization tests live under `tests/naive_bayes`; raising
trait probes verify method and ownership compatibility only.

Samples are rows; model statistics use class then feature axes. Class vectors
follow sorted `classes_`; probability outputs are `(N,C)`. New allocated matrices
are C-order, contiguous; borrowed valid positive/negative-stride views and offsets
are accepted by validation. `.view()` explicitly shares mutation/lifetime.
`.copy()` is independent but copies the backing buffer even for a small strided
view. `.reshape`, slicing, `.T`, and `.astype` may allocate; never budget them as
free views. Concurrent mutation and overlapping in-place operations are excluded.
Space is O(NF) for copied input, O(CF) rectangular state, O(C sum K_f) categorical
state, and O(NC) scores; transactional partial_fit temporarily duplicates state.

## Assembly and validation

The root `DEPENDENCIES.json` assigns every Naive Bayes module to Moxie and
retains the Numax library used for scalar exp/log. Root validation includes
`tools/naive_bayes_checks.py`. Its receipt checks compare assembled bytes to
core sources; isolated consumers import the public estimators and exchange
NuMojo data. Missing files and module/export collisions must fail without
replacing the previous assembly output.

Use the shared [numerical policy](../../shared/numerics/docs/numerics.md).
The integration preserves the existing numerical kernels and tolerances.
See [validation instructions](README.md) and [Gaussian arithmetic](gaussian/api.md).
