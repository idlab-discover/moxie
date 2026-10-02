# Adapted from scikit-learn sklearn/preprocessing/_target_encoder.py,
# sklearn/preprocessing/_target_encoder_fast.pyx and the KFold/StratifiedKFold
# fold assignment of sklearn/model_selection/_split.py (revision
# 4358542a1bd39af2e7e1cbace8b9fed067473c5b); authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense TargetEncoder over typed categorical columns.

Features are Moxie `CategoricalColumn`s of one `CategoryKind` (as for the
OneHot/Ordinal encoders, through the family `_vocabulary` engine); the target
is a label container of a `LabelKind` (`NumericLabels[dtype]` or
`StringLabels`). Encodings follow `_target_encoder_fast.pyx` (m-estimate and
empirical-Bayes "auto" smoothing). `fit_transform` cross-fits on folds
reproduced from sklearn's KFold / StratifiedKFold (`kfold_test_folds`,
`stratified_kfold_test_folds`): a family-local stand-in for the future
model_selection splitters dependency. Target means and variances use NumPy's
pairwise summation order and dtype, so encodings match sklearn exactly.

Memory: fitted state is O(total categories x classes). A fit or transform
holds the result, one rows-long Int64 code vector, the numeric target (one
Float64 or Int per row), a fold vector and one training-row list; there is no
n x p code or mask table and no n x classes indicator matrix.
See docs/api/encoders.md.
"""
from std.math import isnan
from numojo import NDArray, Shape, Item
from moxie._categorical import CategoricalColumn
from moxie._random import RandomState, check_random_state
from moxie.preprocessing._base import TargetPreprocessor
from moxie.preprocessing._common import (
    matrix, vector, check_fitted, check_set_output, check_names_length, feature_names_in,
)
from moxie.preprocessing._encoders import _EncoderCore
from moxie.preprocessing._label import LabelKind, NumericLabels, LabelEncoder, LabelBinarizer, _check_kind_container
from moxie.preprocessing._moments import _precision_sum
from moxie.preprocessing._vocabulary import CategoryKind, StringKind


# ---------------------------------------------------------------------------
# Cross-fitting folds: sklearn KFold / StratifiedKFold test-fold assignment.
#
# BaseCrossValidator.split yields, for fold f, train = rows whose fold != f and
# test = rows whose fold == f, both ascending. A fold-per-row vector is
# therefore an exact, O(n) description of every split.


def _check_kfold(n_splits: Int, shuffle: Bool, random_state: Optional[Int], n_samples: Int) raises:
    """_BaseKFold.__init__ and _BaseKFold.split checks, in sklearn's order."""
    if n_splits <= 1:
        raise Error("ValueError: k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits=" + String(n_splits) + ".")
    if not shuffle and random_state:
        raise Error("ValueError: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.")
    if n_splits > n_samples:
        raise Error("ValueError: Cannot have number of splits n_splits=" + String(n_splits) + " greater than the number of samples: n_samples=" + String(n_samples) + ".")


def _kfold(n_samples: Int, n_splits: Int, shuffle: Bool, mut rng: RandomState) raises -> List[Int]:
    var indices = vector[DType.int64](n_samples)
    for i in range(n_samples):
        indices.unsafe_set(i, Int64(i))
    if shuffle:
        rng.shuffle(indices)
    var folds = List[Int](length=n_samples, fill=0)
    var position = 0
    for f in range(n_splits):
        var size = n_samples // n_splits + Int(f < n_samples % n_splits)
        for k in range(position, position + size):
            folds[Int(indices.unsafe_get(k))] = f
        position += size
    return folds^


def kfold_test_folds(n_samples: Int, n_splits: Int, shuffle: Bool, random_state: Optional[Int]) raises -> List[Int]:
    """Test fold of each row under `KFold(n_splits, shuffle=, random_state=)`.

    An integer seeds a fresh RandomState; with shuffle, `None` draws from the
    process default stream of `moxie._random`, as sklearn draws from NumPy's.
    """
    _check_kfold(n_splits, shuffle, random_state, n_samples)
    if shuffle and not random_state:
        with check_random_state() as rng:
            return _kfold(n_samples, n_splits, True, rng[])
    var rng = RandomState(random_state.value() if random_state else 0)
    return _kfold(n_samples, n_splits, shuffle, rng)


def kfold_test_folds(n_samples: Int, n_splits: Int, shuffle: Bool, mut random_state: RandomState) raises -> List[Int]:
    """KFold with a caller-owned RandomState (sklearn's RandomState instance), advanced in place."""
    _check_kfold(n_splits, shuffle, Optional[Int](0), n_samples)
    return _kfold(n_samples, n_splits, shuffle, random_state)


def _below(x: Int, i: Int, s: Int) -> Int:
    """Number of positions p in [0, x) with p % s == i."""
    return (x - i + s - 1) // s if x > i else 0


def _stratified(labels: List[Int], n_splits: Int, shuffle: Bool, mut rng: RandomState) raises -> List[Int]:
    var n = len(labels)
    var rank = Dict[Int, Int]()
    var y = List[Int](capacity=n)
    for label in labels:
        var found = rank.get(label)
        if found:
            y.append(found.value())
        else:
            var code = len(rank)
            rank[label] = code
            y.append(code)
    var n_classes = len(rank)
    var counts = List[Int](length=n_classes, fill=0)
    for code in y:
        counts[code] += 1
    var too_small = True
    for count in counts:
        too_small = too_small and n_splits > count
    if too_small:
        raise Error("ValueError: n_splits=" + String(n_splits) + " cannot be greater than the number of members in each class.")
    # Class k's fold sequence: arange(n_splits).repeat(allocation[:, k]), where
    # allocation[i][k] counts class k among sorted positions i, i+s, i+2s, ...
    var per_class = List[NDArray[DType.int64]]()
    var start = 0
    for k in range(n_classes):
        var stop = start + counts[k]
        var block = vector[DType.int64](counts[k])
        var at = 0
        for i in range(n_splits):
            var share = _below(stop, i, n_splits) - _below(start, i, n_splits)
            for _ in range(share):
                block.unsafe_set(at, Int64(i))
                at += 1
        if shuffle:
            rng.shuffle(block)
        per_class.append(block^)
        start = stop
    var seen = List[Int](length=n_classes, fill=0)
    var folds = List[Int](length=n, fill=0)
    for r in range(n):
        var k = y[r]
        folds[r] = Int(per_class[k].unsafe_get(seen[k]))
        seen[k] += 1
    return folds^


def stratified_kfold_test_folds(labels: List[Int], n_splits: Int, shuffle: Bool, random_state: Optional[Int]) raises -> List[Int]:
    """Test fold of each row under `StratifiedKFold(n_splits, shuffle=, random_state=)`.

    `labels` are any integer class codes (equal codes = equal classes). As in
    sklearn, classes are ranked by first appearance, each fold receives a
    round-robin share of the class-sorted rows, and with shuffle each class's
    fold sequence is shuffled, class by class, with one RandomState. sklearn's
    UserWarning for a class smaller than n_splits is not emitted.
    """
    _check_kfold(n_splits, shuffle, random_state, len(labels))
    if shuffle and not random_state:
        with check_random_state() as rng:
            return _stratified(labels, n_splits, True, rng[])
    var rng = RandomState(random_state.value() if random_state else 0)
    return _stratified(labels, n_splits, shuffle, rng)


def stratified_kfold_test_folds(labels: List[Int], n_splits: Int, shuffle: Bool, mut random_state: RandomState) raises -> List[Int]:
    """StratifiedKFold with a caller-owned RandomState, advanced in place."""
    _check_kfold(n_splits, shuffle, Optional[Int](0), len(labels))
    return _stratified(labels, n_splits, shuffle, random_state)


# ---------------------------------------------------------------------------
# Target statistics and encodings (_target_encoder_fast.pyx)


struct _Target(Movable):
    """Numeric target used for encoding.

    Continuous: `values` holds y. Binary: `values` holds LabelEncoder codes.
    Multiclass: `codes` holds LabelEncoder codes and class c's target is the
    one-vs-all indicator (LabelBinarizer column c; all zeros with one class).
    """
    var values: List[Float64]
    var codes: List[Int]
    var n_columns: Int  # 1, or n_classes when multiclass
    var multiclass: Bool
    var single_class: Bool

    def __init__(out self, var values: List[Float64], var codes: List[Int], n_columns: Int, multiclass: Bool, single_class: Bool):
        self.values = values^
        self.codes = codes^
        self.n_columns = n_columns
        self.multiclass = multiclass
        self.single_class = single_class

    @always_inline
    def get(self, row: Int, column: Int) -> Float64:
        if not self.multiclass:
            return self.values[row]
        if self.single_class:
            return 0.0
        return 1.0 if self.codes[row] == column else 0.0


def _mean_var[w: DType](target: _Target, column: Int, rows: List[Int], all_rows: Bool, n: Int, need_var: Bool) raises -> Tuple[Float64, Float64]:
    """NumPy `mean(y)` and `var(y)` of y[rows] in dtype `w`, widened to Float64.

    Sums and squared residuals use NumPy's pairwise tree over the gathered
    (contiguous) training vector; each quotient is formed in Float64 and
    rounded to `w`, as NumPy divides a `w` sum by its intp count.
    """
    var size = n if all_rows else len(rows)

    def values[d: DType, width: Int](i: Int) capturing -> SIMD[d, width]:
        var result = SIMD[d, width](0)
        for lane in range(width):
            var row = i + lane if all_rows else rows[i + lane]
            result[lane] = target.get(row, column).cast[d]()
        return result

    var total = _precision_sum[w, values](size, True)
    var mean = (Float64(total) / Float64(size)).cast[w]()
    var variance = Scalar[w](0)
    if need_var:
        def squares[d: DType, width: Int](i: Int) capturing -> SIMD[d, width]:
            var result = SIMD[d, width](0)
            for lane in range(width):
                var row = i + lane if all_rows else rows[i + lane]
                var delta = target.get(row, column).cast[d]() - mean.cast[d]()
                result[lane] = delta * delta
            return result

        var residual = _precision_sum[w, squares](size, True)
        variance = (Float64(residual) / Float64(size)).cast[w]()
    return (Float64(mean), Float64(variance))


def _fit_encoding(
    codes: NDArray[DType.int64], n_categories: Int, target: _Target, column: Int,
    rows: List[Int], all_rows: Bool, smooth: Optional[Float64], y_mean: Float64, y_variance: Float64,
) raises -> NDArray[DType.float64]:
    """One feature's encoding over training rows (pyx `_fit_encoding_fast[_auto_smooth]`).

    Rows coded -1 (unknown) are skipped. Accumulation is Float64 in row order.
    """
    var size = codes.size if all_rows else len(rows)
    var encoding = vector[DType.float64](n_categories)
    if smooth:
        var m = smooth.value()
        var sums = List[Float64](length=n_categories, fill=m * y_mean)
        var counts = List[Float64](length=n_categories, fill=m)
        for s in range(size):
            var row = s if all_rows else rows[s]
            var code = Int(codes.unsafe_get(row))
            if code == -1:
                continue
            sums[code] += target.get(row, column)
            counts[code] += 1.0
        for c in range(n_categories):
            encoding.unsafe_set(c, y_mean if counts[c] == 0 else sums[c] / counts[c])
        return encoding^
    var means = List[Float64](length=n_categories, fill=0.0)
    var counts = List[Int](length=n_categories, fill=0)
    var squares = List[Float64](length=n_categories, fill=0.0)
    for s in range(size):
        var row = s if all_rows else rows[s]
        var code = Int(codes.unsafe_get(row))
        if code == -1:
            continue
        counts[code] += 1
        means[code] += target.get(row, column)
    for c in range(n_categories):
        means[c] /= Float64(counts[c])
    for s in range(size):
        var row = s if all_rows else rows[s]
        var code = Int(codes.unsafe_get(row))
        if code == -1:
            continue
        var diff = target.get(row, column) - means[code]
        squares[code] += diff * diff
    for c in range(n_categories):
        var count = Float64(counts[c])
        var lam = y_variance * count / (y_variance * count + squares[c] / count)
        if isnan(lam):
            # No training rows, or zero variance with zero residuals.
            encoding.unsafe_set(c, y_mean)
        else:
            encoding.unsafe_set(c, lam * means[c] + (1 - lam) * y_mean)
    return encoding^


def _training_rows(folds: List[Int], fold: Int, mut train: List[Int]):
    train.clear()
    for r in range(len(folds)):
        if folds[r] != fold:
            train.append(r)


# ---------------------------------------------------------------------------
# TargetEncoder


@fieldwise_init
struct TargetEncoderParams[K: CategoryKind](Copyable):
    var categories: Optional[List[List[Optional[Self.K.Domain.Value]]]]
    var target_type: String
    var smooth: Optional[Float64]
    var cv: Int
    var shuffle: Optional[Bool]
    var random_state: Optional[Int]


struct TargetEncoder[
    K: CategoryKind = StringKind, Y: LabelKind = NumericLabels[DType.float64],
    L: Copyable & Deinitable = Y.Labels,
](TargetPreprocessor):
    """Encode categories by shrunk target means; Float64 output.

    `K` is the feature kind (as for OneHotEncoder) and `Y` the target label
    kind: `NumericLabels[Float64/Float32/Int64]` (any target type) or
    `StringLabels` (binary/multiclass). `L` is Y's container and must stay at
    its default. `smooth=None` is sklearn's "auto"; `shuffle=None` and
    `random_state=None` are its "deprecated" defaults (shuffle, unseeded).
    """
    comptime Params = TargetEncoderParams[Self.K]
    comptime Input = List[CategoricalColumn[Self.K.Domain]]
    comptime Target = Self.L
    comptime Transformed = NDArray[DType.float64]
    var categories: Optional[List[List[Optional[Self.K.Domain.Value]]]]
    var target_type: String
    var smooth: Optional[Float64]
    var cv: Int
    var shuffle: Optional[Bool]
    var random_state: Optional[Int]
    var encodings_: List[NDArray[DType.float64]]
    var categories_: List[List[Optional[Self.K.Domain.Value]]]
    var target_type_: String
    var target_mean_: NDArray[DType.float64]
    var classes_: Optional[Self.L]
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]
    var _core: _EncoderCore[Self.K]

    def __init__(out self, categories: Optional[List[List[Optional[Self.K.Domain.Value]]]] = None,
        target_type: String = "auto", smooth: Optional[Float64] = None, cv: Int = 5,
        shuffle: Optional[Bool] = None, random_state: Optional[Int] = None,
    ) raises:
        _check_kind_container[Self.Y, Self.L]()
        self.categories = categories.copy()
        self.target_type = target_type
        self.smooth = smooth
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.encodings_ = List[NDArray[DType.float64]]()
        self.categories_ = List[List[Optional[Self.K.Domain.Value]]]()
        self.target_type_ = ""
        self.target_mean_ = vector[DType.float64](0)
        self.classes_ = None
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self._core = _EncoderCore[Self.K]()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.categories.copy(), self.target_type, self.smooth, self.cv, self.shuffle, self.random_state)

    def set_params(mut self, params: Self.Params) raises:
        self.categories = params.categories.copy()
        self.target_type = params.target_type
        self.smooth = params.smooth
        self.cv = params.cv
        self.shuffle = params.shuffle
        self.random_state = params.random_state

    def set_params(
        mut self, *, target_type: Optional[String] = None, smooth: Optional[Optional[Float64]] = None,
        cv: Optional[Int] = None, shuffle: Optional[Optional[Bool]] = None,
        random_state: Optional[Optional[Int]] = None,
    ) raises:
        """Outer None leaves a parameter unchanged; an inner None sets it to None."""
        if target_type:
            self.target_type = target_type.value()
        if smooth:
            self.smooth = smooth.value()
        if cv:
            self.cv = cv.value()
        if shuffle:
            self.shuffle = shuffle.value()
        if random_state:
            self.random_state = random_state.value()

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._core.fitted

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    # -- fitting ------------------------------------------------------------

    def _validate_params(self) raises:
        var t = self.target_type
        if t != "auto" and t != "continuous" and t != "binary" and t != "multiclass":
            raise Error("ValueError: The 'target_type' parameter of TargetEncoder must be a str among {'auto', 'binary', 'continuous', 'multiclass'}. Got '" + t + "' instead.")
        if self.smooth and (isnan(self.smooth.value()) or self.smooth.value() < 0):
            raise Error("ValueError: The 'smooth' parameter of TargetEncoder must be a str among {'auto'} or a float in the range [0.0, inf). Got " + String(self.smooth.value()) + " instead.")
        if self.cv < 2:
            raise Error("ValueError: The 'cv' parameter of TargetEncoder must be an int in the range [2, inf), an object implementing 'split' and 'get_n_splits', an iterable or None. Got " + String(self.cv) + " instead.")
        if self.random_state and (self.random_state.value() < 0 or self.random_state.value() > 4294967295):
            raise Error("ValueError: The 'random_state' parameter of TargetEncoder must be an int in the range [0, 4294967295], an instance of 'numpy.random.mtrand.RandomState' or None or a str among {'deprecated'}. Got " + String(self.random_state.value()) + " instead.")

    def _target(mut self, y: Self.Y.Labels, n: Int) raises -> _Target:
        """Infer target_type_, set classes_, build the numeric target (_fit_encodings_all)."""
        var kind = self.target_type
        if kind == "auto":
            kind = Self.Y.target_type(y)
            if kind != "binary" and kind != "multiclass" and kind != "continuous":
                raise Error("ValueError: Unknown label type: Target type was inferred to be '" + kind + "'. Only ('binary', 'multiclass', 'continuous') are supported.")
        if kind == "continuous":
            _ = Self.Y.column_length(y)
            var values = List[Float64](capacity=n)
            for i in range(n):
                var value = Self.Y.get(y, i)
                if Self.Y.is_nan(value):
                    raise Error("ValueError: Input y contains NaN.")
                var number = Self.Y.number(value)
                if number - number != 0:
                    raise Error("ValueError: Input y contains infinity or a value too large for dtype('" + String(Self.Y.number_dtype) + "').")
                values.append(number)
            self.target_type_ = kind
            self.classes_ = None
            return _Target(values^, List[Int](), 1, False, False)
        if kind == "multiclass":
            # LabelBinarizer's target checks; its columns are one-vs-all codes.
            var binarizer = LabelBinarizer[Self.Y]()
            binarizer.fit(y)
            if binarizer.y_type_ == "multilabel-indicator":
                _ = Self.Y.column_length(y)  # raises: TargetEncoder targets are 1d
            if Self.Y.rows(binarizer.classes_) == 2:
                raise Error("ValueError: target_type='multiclass' cannot encode a target with exactly 2 classes (sklearn raises IndexError); use target_type='binary'")
        var encoder = LabelEncoder[Self.Y]()
        var encoded = encoder.fit_transform(y)
        var codes = List[Int](capacity=n)
        for i in range(n):
            codes.append(Int(encoded.unsafe_get(i)))
        var n_classes = Self.Y.rows(encoder.classes_)
        self.target_type_ = kind
        self.classes_ = rebind_var[Self.L](encoder.classes_.copy())
        if kind == "binary":
            var values = List[Float64](capacity=n)
            for code in codes:
                values.append(Float64(code))
            return _Target(values^, List[Int](), 1, False, False)
        return _Target(List[Float64](), codes^, n_classes, True, n_classes == 1)

    def _moments(self, target: _Target, column: Int, rows: List[Int], all_rows: Bool, n: Int) raises -> Tuple[Float64, Float64]:
        var need_var = not self.smooth
        if self.target_type_ == "continuous":
            return _mean_var[Self.Y.number_dtype](target, column, rows, all_rows, n, need_var)
        return _mean_var[DType.float64](target, column, rows, all_rows, n, need_var)

    def _fit(mut self, X: Self.Input, y: Self.Y.Labels) raises -> _Target:
        """sklearn `_fit_encodings_all`: categories, target, encodings on all rows."""
        self._validate_params()
        var n_y = Self.Y.rows(y)
        if len(X) and X[0].rows() != n_y:
            raise Error("ValueError: Found input variables with inconsistent numbers of samples: [" + String(X[0].rows()) + ", " + String(n_y) + "]")
        var core = _EncoderCore[Self.K]()
        core.fit(X, self.categories, "ignore", None, None, False)
        var n = X[0].rows()
        var target = self._target(y, n)
        var width = target.n_columns
        var none = List[Int]()
        var means = vector[DType.float64](width)
        var variances = List[Float64]()
        for c in range(width):
            var stats = self._moments(target, c, none, True, n)
            means.unsafe_set(c, stats[0])
            variances.append(stats[1])
        var encodings = List[NDArray[DType.float64]]()
        for i in range(core.n_features()):
            var codes = core.encode(i, X[i], "ignore")
            for c in range(width):
                encodings.append(_fit_encoding(codes, core.vocabularies[i].size(), target, c, none, True, self.smooth, means.unsafe_get(c), variances[c]))
        self.categories_ = core.categories()
        self.n_features_in_ = core.n_features()
        self.feature_names_in_ = None
        self.encodings_ = encodings^
        self.target_mean_ = means^
        self._core = core^
        return target^

    def _stratify_labels(self, y: Self.Y.Labels, n: Int) raises -> Optional[List[Int]]:
        """sklearn check_cv: StratifiedKFold labels when the target is classified and
        type_of_target(y) is binary/multiclass, else None (KFold)."""
        if self.target_type_ == "continuous":
            return None
        var kind = Self.Y.target_type(y)
        if kind != "binary" and kind != "multiclass":
            return None
        # Any equality coding of y works: classes are ranked by first appearance.
        var encoder = LabelEncoder[Self.Y]()
        var encoded = encoder.fit_transform(y)
        var labels = List[Int](capacity=n)
        for i in range(n):
            labels.append(Int(encoded.unsafe_get(i)))
        return labels^

    def _shuffle(self) -> Bool:
        return self.shuffle.value() if self.shuffle else True

    def _cross_fit(self, X: Self.Input, target: _Target, folds: List[Int]) raises -> NDArray[DType.float64]:
        """sklearn fit_transform after check_cv: encodings fitted on each fold's training rows."""
        var n = X[0].rows()
        var width = target.n_columns
        var p = self._core.n_features()
        # Training mean and variance per fold and target column.
        var fold_means = List[List[Float64]]()
        var fold_variances = List[List[Float64]]()
        var train = List[Int](capacity=n)
        for f in range(self.cv):
            _training_rows(folds, f, train)
            var mean_row = List[Float64]()
            var variance_row = List[Float64]()
            for c in range(width):
                var stats = self._moments(target, c, train, False, n)
                mean_row.append(stats[0])
                variance_row.append(stats[1])
            fold_means.append(mean_row^)
            fold_variances.append(variance_row^)
        var out_width = p * width
        var out = matrix[DType.float64](n, out_width)
        for i in range(p):
            var codes = self._core.encode(i, X[i], "ignore")
            var n_categories = self._core.vocabularies[i].size()
            for f in range(self.cv):
                _training_rows(folds, f, train)
                for c in range(width):
                    var encoding = _fit_encoding(codes, n_categories, target, c, train, False, self.smooth, fold_means[f][c], fold_variances[f][c])
                    var column = i * width + c
                    for r in range(n):
                        if folds[r] == f:
                            var code = Int(codes.unsafe_get(r))
                            if code >= 0:
                                out.unsafe_set(r * out_width + column, encoding.unsafe_get(code))
            # sklearn overwrites every unknown row once per fold, so the last
            # fold's training mean remains.
            for r in range(n):
                if Int(codes.unsafe_get(r)) < 0:
                    for c in range(width):
                        out.unsafe_set(r * out_width + i * width + c, fold_means[self.cv - 1][c])
        return out^

    def _folds(self, y: Self.Y.Labels, n: Int) raises -> List[Int]:
        var labels = self._stratify_labels(y, n)
        if labels:
            return stratified_kfold_test_folds(labels.value(), self.cv, self._shuffle(), self.random_state)
        return kfold_test_folds(n, self.cv, self._shuffle(), self.random_state)

    def _folds(self, y: Self.Y.Labels, n: Int, mut random_state: RandomState) raises -> List[Int]:
        var labels = self._stratify_labels(y, n)
        if labels:
            return stratified_kfold_test_folds(labels.value(), self.cv, self._shuffle(), random_state)
        return kfold_test_folds(n, self.cv, self._shuffle(), random_state)

    def fit(mut self, X: Self.Input, y: Self.L) raises:
        """Learn encodings on all of X (fit then transform is not cross-fitted)."""
        _ = self._fit(X, rebind[Self.Y.Labels](y))

    def fit_named(mut self, X: Self.Input, feature_names: List[String], y: Self.L) raises:
        check_names_length(feature_names, len(X))
        _ = self._fit(X, rebind[Self.Y.Labels](y))
        self.feature_names_in_ = feature_names.copy()

    def fit_transform(mut self, X: Self.Input, y: Self.L) raises -> NDArray[DType.float64]:
        """Fit on all rows, then return the cross-fitted encoding of X.

        Folds shuffle with `random_state` (an integer reseeds a fresh
        RandomState; None draws from the process default stream).
        """
        ref labels = rebind[Self.Y.Labels](y)
        var target = self._fit(X, labels)
        var folds = self._folds(labels, X[0].rows())
        return self._cross_fit(X, target, folds)

    def fit_transform(mut self, X: Self.Input, y: Self.L, mut random_state: RandomState) raises -> NDArray[DType.float64]:
        """fit_transform shuffling folds with a caller-owned RandomState, advanced in place.

        The configured `random_state` is ignored, like passing a RandomState
        instance as sklearn's `random_state`.
        """
        ref labels = rebind[Self.Y.Labels](y)
        var target = self._fit(X, labels)
        var folds = self._folds(labels, X[0].rows(), random_state)
        return self._cross_fit(X, target, folds)

    def fit_transform_named(mut self, X: Self.Input, feature_names: List[String], y: Self.L) raises -> NDArray[DType.float64]:
        check_names_length(feature_names, len(X))
        ref labels = rebind[Self.Y.Labels](y)
        var target = self._fit(X, labels)
        self.feature_names_in_ = feature_names.copy()
        var folds = self._folds(labels, X[0].rows())
        return self._cross_fit(X, target, folds)

    # -- transformation -------------------------------------------------------

    def transform(self, X: Self.Input) raises -> NDArray[DType.float64]:
        """Encode with `encodings_`; unknown categories get `target_mean_`."""
        check_fitted(self._core.fitted, "TargetEncoder")
        self._core.check_transform(X, "TargetEncoder")
        var n = X[0].rows()
        var width = self.target_mean_.size
        var out_width = len(X) * width
        var out = matrix[DType.float64](n, out_width)
        for i in range(len(X)):
            var codes = self._core.encode(i, X[i], "ignore")
            for c in range(width):
                ref encoding = self.encodings_[i * width + c]
                var mean = self.target_mean_.unsafe_get(c)
                for r in range(n):
                    var code = Int(codes.unsafe_get(r))
                    out.unsafe_set(r * out_width + i * width + c, encoding.unsafe_get(code) if code >= 0 else mean)
        return out^

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        """Input names; a multiclass target gives `<feature>_<class>` per class."""
        check_fitted(self._core.fitted, "TargetEncoder")
        var names = feature_names_in(self.n_features_in_, self.feature_names_in_, input_features)
        if self.target_type_ != "multiclass":
            return names^
        ref classes = rebind[Self.Y.Labels](self.classes_.value())
        var result = List[String]()
        for name in names:
            for c in range(Self.Y.rows(classes)):
                result.append(name + "_" + Self.Y.python_str(Self.Y.get(classes, c)))
        return result^
