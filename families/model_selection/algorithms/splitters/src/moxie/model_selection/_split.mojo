"""Native splitters. Adapted from scikit-learn; see LICENSES/scikit-learn.txt."""
from numojo import NDArray, Shape, Item, zeros
from moxie.model_selection._indices import IndexArray, IndexSplit, SplitCursor
from moxie._random import RandomState
from moxie.model_selection._base import CrossValidator
from moxie._labels import StringTarget, encode_column, type_of_target


def _sorted_codes[dtype: DType](y: NDArray[dtype], column: Int = 0) raises -> IndexArray:
    return encode_column(y, column, classification=False).index.inverse.copy()


def _key_codes(var keys: List[String]) raises -> IndexArray:
    return encode_column(StringTarget(keys), classification=False).index.inverse.copy()


def _numeric_row_keys[dtype: DType](y: NDArray[dtype]) raises -> IndexArray:
    # sklearn StratifiedShuffleSplit: " ".join(row.astype(str)), sorted as strings.
    var keys = List[String]()
    for i in range(y.shape[0]):
        var key = String()
        for j in range(y.shape[1]):
            if j: key += " "
            key += String(y[Item(i, j)])
        keys.append(key^)
    return _key_codes(keys^)


struct SplitLabels(Copyable, Movable, Sized):
    """Float32/Float64, String or explicit Int64 labels and groups for splitters.

    sklearn splitters observe labels only through equality, sorted order and
    first appearance. Sorted-rank Int64 codes preserve all three, so the Int64
    splitter implementations produce the reference folds unchanged.
    """
    var kind: String
    var _rows: Int
    var _rank: Int
    var _columns: Int
    var _column: Optional[IndexArray]
    var _row_keys: Optional[IndexArray]
    var _row_error: String

    @implicit
    def __init__(out self, y: NDArray[DType.float64]) raises:
        self = Self._numeric(y)

    @implicit
    def __init__(out self, y: NDArray[DType.float32]) raises:
        self = Self._numeric(y)

    @implicit
    def __init__(out self, values: List[String]) raises:
        self = Self._strings(StringTarget(values))

    @implicit
    def __init__(out self, y: StringTarget) raises:
        self = Self._strings(y)

    def __init__(out self, y: IndexArray) raises:
        """Explicit Int64 wrapper, for mixing with non-Int64 labels or groups."""
        self = Self._numeric(y)

    def __init__(out self, values: List[String], rows: Int, columns: Int) raises:
        """Row-major rank-two String labels."""
        self = Self._strings(StringTarget(values, rows, columns))

    def __init__(out self, *, var kind: String, rows: Int, rank: Int, columns: Int,
                 var column: Optional[IndexArray], var row_keys: Optional[IndexArray], var row_error: String):
        self.kind = kind^
        self._rows = rows
        self._rank = rank
        self._columns = columns
        self._column = column^
        self._row_keys = row_keys^
        self._row_error = row_error^

    @staticmethod
    def _numeric[dtype: DType](y: NDArray[dtype]) raises -> Self:
        var kind = type_of_target(y)  # rejects nonfinite values and unsupported ranks
        if y.ndim != 1 and y.ndim != 2:
            raise Error("ValueError: labels require rank one or two")
        var columns = 1 if y.ndim == 1 else y.shape[1]
        var column = Optional[IndexArray](None)
        if columns == 1:
            column = _sorted_codes(y)
        var row_keys = Optional[IndexArray](None)
        var row_error = String()
        if y.ndim == 2:
            comptime if dtype == DType.float32:
                # NumPy's float32 str() switches notation at different exponents
                # than Mojo's shortest formatting; keys would not be reproduced.
                row_error = "ValueError: rank-two Float32 stratification labels are unsupported; convert to Float64"
            else:
                row_keys = _numeric_row_keys(y)
        return Self(kind=kind^, rows=y.shape[0], rank=y.ndim, columns=columns,
                    column=column^, row_keys=row_keys^, row_error=row_error^)

    @staticmethod
    def _strings(y: StringTarget) raises -> Self:
        var kind = type_of_target(y)
        var column = Optional[IndexArray](None)
        if y.columns() == 1:
            column = encode_column(y, classification=False).index.inverse.copy()
        var row_keys = Optional[IndexArray](None)
        if y.rank() == 2:
            var keys = List[String]()
            for i in range(y.rows()):
                var key = String()
                for j in range(y.columns()):
                    if j: key += " "
                    key += y.get(i, j)
                keys.append(key^)
            row_keys = _key_codes(keys^)
        return Self(kind=kind^, rows=y.rows(), rank=y.rank(), columns=y.columns(),
                    column=column^, row_keys=row_keys^, row_error=String())

    def __len__(self) -> Int:
        return self._rows

    def class_codes(self) raises -> IndexArray:
        """Classification labels, as StratifiedKFold/StratifiedGroupKFold validate them."""
        if self.kind != "binary" and self.kind != "multiclass":
            raise Error("ValueError: Supported target types are: ('binary', 'multiclass'). Got '" + self.kind + "' instead.")
        return self._column.value().copy()

    def group_codes(self) raises -> IndexArray:
        if self._rank != 1:
            raise Error("ValueError: groups must be rank one")
        return self._column.value().copy()

    def stratify_codes(self) raises -> IndexArray:
        """StratifiedShuffleSplit keys: any finite values; rank-two rows join as strings."""
        if self._rank == 1:
            return self._column.value().copy()
        if not self._row_keys:
            raise Error(self._row_error)
        return self._row_keys.value().copy()


def _range(start: Int, stop: Int) raises -> IndexArray:
    var result = IndexArray(Shape(stop - start))
    for i in range(stop - start):
        result[Item(i)] = Int64(i + start)
    return result^


struct _FoldCursor(SplitCursor):
    var assignments: IndexArray
    var folds: IndexArray
    var position: Int

    def __init__(out self, var assignments: IndexArray, var folds: IndexArray):
        self.assignments = assignments^
        self.folds = folds^
        self.position = 0

    def next(mut self) raises -> Optional[IndexSplit]:
        if self.position == self.folds.size:
            return None
        var fold = Int(self.folds[Item(self.position)])
        self.position += 1
        var n_test = 0
        for i in range(self.assignments.size):
            if Int(self.assignments[Item(i)]) == fold:
                n_test += 1
        var train = IndexArray(Shape(self.assignments.size - n_test))
        var test = IndexArray(Shape(n_test))
        var a = 0
        var b = 0
        for i in range(self.assignments.size):
            if Int(self.assignments[Item(i)]) == fold:
                test[Item(b)] = Int64(i)
                b += 1
            else:
                train[Item(a)] = Int64(i)
                a += 1
        return IndexSplit(train^, test^)


trait BaseCrossValidator(CrossValidator):
    """Native base consuming the inherited family contract."""
    pass


struct KFold(BaseCrossValidator, Copyable):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _FoldCursor
    var n_splits: Int
    var shuffle: Bool
    var random_state: Optional[Int]

    def __init__(out self, n_splits: Int = 5, *, shuffle: Bool = False, random_state: Optional[Int] = None) raises:
        if n_splits < 2:
            raise Error("ValueError: n_splits must be at least 2")
        if not shuffle and random_state:
            raise Error("ValueError: random_state has no effect without shuffle")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        return self.n_splits

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        var n_samples = X
        _validate_lengths(n_samples, y, groups)
        if self.shuffle and not self.random_state:
            raise Error("ValueError: pass an application RNG context for random_state=None")
        var seed = 0
        if self.random_state:
            seed = self.random_state.value()
        var rng = RandomState(seed)
        return self.split(n_samples, rng)

    def split(self, n_samples: Int, mut random_state: RandomState) raises -> Self.Splits:
        if n_samples < self.n_splits:
            raise Error("ValueError: n_splits exceeds sample count")
        var indices = _range(0, n_samples)
        if self.shuffle:
            indices = random_state.permutation(n_samples)
        var assignments = IndexArray(Shape(n_samples))
        var position = 0
        for fold in range(self.n_splits):
            var size = n_samples // self.n_splits
            if fold < n_samples % self.n_splits:
                size += 1
            for _ in range(size):
                assignments[Item(Int(indices[Item(position)]))] = Int64(fold)
                position += 1
        return _FoldCursor(assignments^, _range(0, self.n_splits))


struct _TimeCursor(SplitCursor):
    var n_samples: Int
    var test_size: Int
    var gap: Int
    var max_train_size: Optional[Int]
    var position: Int

    def __init__(out self, n: Int, k: Int, size: Int, gap: Int, max_train_size: Optional[Int]):
        self.n_samples = n
        self.test_size = size
        self.gap = gap
        self.max_train_size = max_train_size
        self.position = n - k * size

    def next(mut self) raises -> Optional[IndexSplit]:
        if self.position >= self.n_samples:
            return None
        var stop = self.position - self.gap
        var start = 0
        if self.max_train_size:
            if self.max_train_size.value() > 0 and self.max_train_size.value() < stop:
                start = stop - self.max_train_size.value()
        var result = IndexSplit(_range(start, stop), _range(self.position, self.position + self.test_size))
        self.position += self.test_size
        return result^


struct TimeSeriesSplit(BaseCrossValidator, Copyable):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _TimeCursor
    var n_splits: Int
    var max_train_size: Optional[Int]
    var test_size: Optional[Int]
    var gap: Int

    def __init__(out self, n_splits: Int = 5, *, max_train_size: Optional[Int] = None, test_size: Optional[Int] = None, gap: Int = 0) raises:
        if n_splits < 2:
            raise Error("ValueError: n_splits must be at least 2")
        if gap < 0:
            raise Error("ValueError: gap must be nonnegative")
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        return self.n_splits

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        var n_samples = X
        _validate_lengths(n_samples, y, groups)
        if n_samples < self.n_splits + 1:
            raise Error("ValueError: too many folds")
        var size = n_samples // (self.n_splits + 1)
        if self.test_size:
            size = self.test_size.value()
        if size <= 0 or n_samples - self.gap - size * self.n_splits <= 0:
            raise Error("ValueError: insufficient samples for test size and gap")
        return _TimeCursor(n_samples, self.n_splits, size, self.gap, self.max_train_size)


def _validate_lengths(n_samples: Int, y: Optional[IndexArray], groups: Optional[IndexArray]) raises:
    if n_samples < 0:
        raise Error("ValueError: negative sample count")
    if y:
        if y.value().ndim != 1 or y.value().size != n_samples:
            raise Error("ValueError: y must be rank one with matching length")
    if groups:
        if groups.value().ndim != 1 or groups.value().size != n_samples:
            raise Error("ValueError: groups must be rank one with matching length")


struct _CombinationCursor(SplitCursor):
    var labels: IndexArray
    var chosen: IndexArray
    var n_labels: Int
    var done: Bool

    def __init__(out self, var labels: IndexArray, n_labels: Int, p: Int) raises:
        if p < 0 or p >= n_labels:
            raise Error("ValueError: leave-out count must be nonnegative and less than population")
        self.labels = labels^
        self.chosen = _range(0, p)
        self.n_labels = n_labels
        self.done = False

    def next(mut self) raises -> Optional[IndexSplit]:
        if self.done:
            return None
        var assignments = IndexArray(Shape(self.labels.size))
        for i in range(self.labels.size):
            assignments[Item(i)] = -1
            for j in range(self.chosen.size):
                if self.labels[Item(i)] == self.chosen[Item(j)]:
                    assignments[Item(i)] = 0
        var cursor = _FoldCursor(assignments^, _range(0, 1))
        var result = cursor.next()
        var j = self.chosen.size - 1
        while j >= 0:
            if Int(self.chosen[Item(j)]) != j + self.n_labels - self.chosen.size:
                break
            j -= 1
        if j < 0:
            self.done = True
        else:
            self.chosen[Item(j)] = self.chosen[Item(j)] + 1
            for k in range(j + 1, self.chosen.size):
                self.chosen[Item(k)] = self.chosen[Item(k - 1)] + 1
        return result^


def _comb(n: Int, p: Int) raises -> Int:
    if p < 0 or p > n:
        return 0
    var result = 1
    for k in range(1, p + 1):
        if result > 9223372036854775807 // (n - k + 1):
            raise Error("OverflowError: split count exceeds native Int")
        result = result * (n - k + 1) // k
    return result


struct LeavePOut(BaseCrossValidator, Copyable):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _CombinationCursor
    var p: Int

    def __init__(out self, p: Int):
        self.p = p

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        _validate_lengths(X, y, groups)
        return _CombinationCursor(_range(0, X), X, self.p)

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        if not X:
            raise Error("ValueError: X is required")
        return _comb(X.value(), self.p)


struct LeaveOneOut(BaseCrossValidator, Copyable):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _CombinationCursor

    def __init__(out self):
        pass

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        _validate_lengths(X, y, groups)
        return _CombinationCursor(_range(0, X), X, 1)

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        if not X:
            raise Error("ValueError: X is required")
        return X.value()


def _unique(values: IndexArray) raises -> IndexArray:
    # Stable insertion of distinct labels; integer comparisons remain exact.
    var scratch = IndexArray(Shape(values.size))
    var count = 0
    for i in range(values.size):
        var value = values[Item(i)]
        var position = 0
        while position < count:
            if scratch[Item(position)] >= value:
                break
            position += 1
        if position < count:
            if scratch[Item(position)] == value:
                continue
        var j = count
        while j > position:
            scratch[Item(j)] = scratch[Item(j - 1)]
            j -= 1
        scratch[Item(position)] = value
        count += 1
    var result = IndexArray(Shape(count))
    for i in range(count):
        result[Item(i)] = scratch[Item(i)]
    return result^


def _encode(values: IndexArray, unique: IndexArray) raises -> IndexArray:
    var result = IndexArray(Shape(values.size))
    for i in range(values.size):
        for j in range(unique.size):
            if values[Item(i)] == unique[Item(j)]:
                result[Item(i)] = Int64(j)
                break
    return result^


struct PredefinedSplit(BaseCrossValidator):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _FoldCursor
    var test_fold: IndexArray
    var unique_folds: IndexArray

    def __init__(out self, test_fold: IndexArray) raises:
        if test_fold.ndim != 1:
            raise Error("ValueError: test_fold must be rank one")
        self.test_fold = test_fold.copy()
        var unique = _unique(test_fold)
        var count = 0
        for i in range(unique.size):
            if unique[Item(i)] != -1:
                count += 1
        self.unique_folds = IndexArray(Shape(count))
        count = 0
        for i in range(unique.size):
            if unique[Item(i)] != -1:
                self.unique_folds[Item(count)] = unique[Item(i)]
                count += 1

    def split(self, X: Int = 0, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        return _FoldCursor(self.test_fold.copy(), self.unique_folds.copy())

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        return self.unique_folds.size


struct LeavePGroupsOut(BaseCrossValidator, Copyable):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _CombinationCursor
    var n_groups: Int

    def __init__(out self, n_groups: Int):
        self.n_groups = n_groups

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        _validate_lengths(X, y, groups)
        if not groups:
            raise Error("ValueError: groups are required")
        var unique = _unique(groups.value())
        return _CombinationCursor(_encode(groups.value(), unique), unique.size, self.n_groups)

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        if not groups:
            raise Error("ValueError: groups are required")
        var unique = _unique(groups.value())
        return _comb(unique.size, self.n_groups)

    def split(self, X: Int, y: Optional[IndexArray] = None, *, groups: SplitLabels) raises -> Self.Splits:
        return self.split(X, y, groups.group_codes())

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, *, groups: SplitLabels) raises -> Int:
        return self.get_n_splits(X, y, groups.group_codes())


struct LeaveOneGroupOut(BaseCrossValidator, Copyable):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _CombinationCursor

    def __init__(out self):
        pass

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        return LeavePGroupsOut(1).split(X, y, groups)

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        return LeavePGroupsOut(1).get_n_splits(X, y, groups)

    def split(self, X: Int, y: Optional[IndexArray] = None, *, groups: SplitLabels) raises -> Self.Splits:
        return LeavePGroupsOut(1).split(X, y, groups=groups)

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, *, groups: SplitLabels) raises -> Int:
        return LeavePGroupsOut(1).get_n_splits(X, y, groups=groups)


struct StratifiedKFold(BaseCrossValidator, Copyable):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _FoldCursor
    var n_splits: Int
    var shuffle: Bool
    var random_state: Optional[Int]

    def __init__(out self, n_splits: Int = 5, *, shuffle: Bool = False, random_state: Optional[Int] = None) raises:
        var checked = KFold(n_splits, shuffle=shuffle, random_state=random_state)
        self.n_splits = checked.n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        return self.n_splits

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        _validate_lengths(X, y, groups)
        var rng = _seed(self.random_state, self.shuffle)
        if not y:
            raise Error("ValueError: stratification requires y")
        return self.split(X, y.value(), rng)

    def split(self, X: Int, y: SplitLabels, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        _validate_lengths(X, None, groups)
        var rng = _seed(self.random_state, self.shuffle)
        return self.split(X, y, rng)

    def split(self, X: Int, y: SplitLabels, mut rng: RandomState) raises -> Self.Splits:
        return self.split(X, y.class_codes(), rng)

    def split(self, X: Int, y: IndexArray, mut rng: RandomState) raises -> Self.Splits:
        _validate_vector(X, y)
        if X < self.n_splits:
            raise Error("ValueError: n_splits exceeds sample count")
        var unique = _unique(y)
        var encoded = _encode(y, unique)
        # Classes are processed in first-occurrence order, not label order.
        var order = IndexArray(Shape(unique.size))
        var seen = zeros[DType.int64](Shape(unique.size))
        var classes = 0
        for i in range(X):
            var c = Int(encoded[Item(i)])
            if seen[Item(c)] == 0:
                seen[Item(c)] = 1
                order[Item(classes)] = Int64(c)
                classes += 1
        var counts = _counts(encoded, unique.size)
        var enough = False
        for c in range(classes):
            enough = enough or Int(counts[Item(c)]) >= self.n_splits
        if not enough:
            raise Error("ValueError: n_splits exceeds every class count")
        var assignments = IndexArray(Shape(X))
        var offset = 0
        for c in range(classes):
            var label = Int(order[Item(c)])
            var count = Int(counts[Item(label)])
            var folds = IndexArray(Shape(count))
            var pos = 0
            for f in range(self.n_splits):
                for j in range(count):
                    if (offset + j) % self.n_splits == f:
                        folds[Item(pos)] = Int64(f)
                        pos += 1
            if self.shuffle:
                folds = _permute(folds, rng)
            pos = 0
            for i in range(X):
                if Int(encoded[Item(i)]) == label:
                    assignments[Item(i)] = folds[Item(pos)]
                    pos += 1
            offset += count
        return _FoldCursor(assignments^, _range(0, self.n_splits))


def _seed(seed: Optional[Int], needed: Bool = True) raises -> RandomState:
    if seed:
        return RandomState(seed.value())
    if needed:
        raise Error("ValueError: random_state=None requires an application-owned mutable RandomState")
    return RandomState(0)


def _counts(encoded: IndexArray, n: Int) raises -> IndexArray:
    var counts = zeros[DType.int64](Shape(n))
    for i in range(encoded.size):
        var c = Int(encoded[Item(i)])
        counts[Item(c)] = counts[Item(c)] + 1
    return counts^


struct GroupKFold(BaseCrossValidator, Copyable):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _FoldCursor
    var n_splits: Int
    var shuffle: Bool
    var random_state: Optional[Int]

    def __init__(out self, n_splits: Int = 5, *, shuffle: Bool = False, random_state: Optional[Int] = None) raises:
        var checked = KFold(n_splits, shuffle=shuffle, random_state=random_state)
        self.n_splits = checked.n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        return self.n_splits

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        _validate_lengths(X, y, groups)
        if not groups:
            raise Error("ValueError: groups are required")
        var rng = _seed(self.random_state, self.shuffle)
        return self.split(X, groups.value(), rng)

    def split(self, X: Int, y: Optional[IndexArray] = None, *, groups: SplitLabels) raises -> Self.Splits:
        return self.split(X, y, groups.group_codes())

    def split(self, X: Int, groups: SplitLabels, mut rng: RandomState) raises -> Self.Splits:
        return self.split(X, groups.group_codes(), rng)

    def split(self, X: Int, groups: IndexArray, mut rng: RandomState) raises -> Self.Splits:
        _validate_vector(X, groups)
        var unique = _unique(groups)
        var n = unique.size
        if n < self.n_splits:
            raise Error("ValueError: n_splits exceeds group count")
        var encoded = _encode(groups, unique)
        var counts = _counts(encoded, n)
        var group_folds = IndexArray(Shape(n))
        var order = _range(0, n)
        if self.shuffle:
            order = rng.permutation(n)
            var pos = 0
            for f in range(self.n_splits):
                var count = n // self.n_splits + Int(f < n % self.n_splits)
                for _ in range(count):
                    group_folds[Item(Int(order[Item(pos)]))] = Int64(f)
                    pos += 1
        else:
            # Stable ascending argsort followed by reversal: ties use larger id.
            for i in range(n):
                for j in range(i + 1, n):
                    var a = Int(order[Item(i)])
                    var b = Int(order[Item(j)])
                    if counts[Item(b)] > counts[Item(a)] or (counts[Item(b)] == counts[Item(a)] and b > a):
                        order[Item(i)] = Int64(b)
                        order[Item(j)] = Int64(a)
            var loads = zeros[DType.int64](Shape(self.n_splits))
            for i in range(n):
                var best = 0
                for f in range(1, self.n_splits):
                    if loads[Item(f)] < loads[Item(best)]:
                        best = f
                var group = Int(order[Item(i)])
                group_folds[Item(group)] = Int64(best)
                loads[Item(best)] = loads[Item(best)] + counts[Item(group)]
        var assignments = IndexArray(Shape(X))
        for i in range(X):
            assignments[Item(i)] = group_folds[Item(Int(encoded[Item(i)]))]
        return _FoldCursor(assignments^, _range(0, self.n_splits))


def _validate_vector(n: Int, values: IndexArray) raises:
    if n < 0 or values.ndim != 1 or values.size != n:
        raise Error("ValueError: expected a rank-one array of matching length")


def _permute(values: IndexArray, mut rng: RandomState) raises -> IndexArray:
    var order = rng.permutation(values.size)
    var result = IndexArray(Shape(values.size))
    for i in range(values.size):
        result[Item(i)] = values[Item(Int(order[Item(i)]))]
    return result^


struct _SplitList(SplitCursor):
    var pairs: List[IndexSplit]
    var position: Int

    def __init__(out self, var pairs: List[IndexSplit]):
        self.pairs = pairs^
        self.position = 0

    def next(mut self) raises -> Optional[IndexSplit]:
        if self.position == len(self.pairs):
            return None
        var result = self.pairs[self.position].copy()
        self.position += 1
        return result^


struct SplitSize(ImplicitlyCopyable, Movable):
    """Explicit integer-count versus floating-fraction constructor argument."""
    var count: Int
    var fraction: Float64
    var is_fraction: Bool

    @implicit
    def __init__(out self, count: Int):
        self.count = count
        self.fraction = 0
        self.is_fraction = False

    @implicit
    def __init__(out self, fraction: Float64):
        self.count = 0
        self.fraction = fraction
        self.is_fraction = True

    def resolve(self, n: Int, test: Bool) raises -> Int:
        from std.math import ceil, floor, isfinite
        if self.is_fraction:
            if not isfinite(self.fraction) or self.fraction <= 0 or self.fraction >= 1:
                raise Error("ValueError: size fraction must be in (0, 1)")
            var value = self.fraction * Float64(n)
            return Int(ceil(value)) if test else Int(floor(value))
        if self.count <= 0 or self.count >= n:
            raise Error("ValueError: size count must be positive and smaller than population")
        return self.count


def _sizes(n: Int, test_size: Optional[SplitSize], train_size: Optional[SplitSize], default_test: Float64) raises -> Tuple[Int, Int]:
    var n_test = 0
    var n_train = 0
    if test_size:
        n_test = test_size.value().resolve(n, True)
    if train_size:
        n_train = train_size.value().resolve(n, False)
    if not test_size and not train_size:
        n_test = SplitSize(default_test).resolve(n, True)
    if not train_size:
        n_train = n - n_test
    elif not test_size:
        n_test = n - n_train
    if train_size and test_size:
        if train_size.value().is_fraction and test_size.value().is_fraction:
            if train_size.value().fraction + test_size.value().fraction > 1:
                raise Error("ValueError: size fractions sum to more than one")
    if n_train <= 0 or n_test <= 0 or n_train + n_test > n:
        raise Error("ValueError: invalid train/test sizes")
    return (n_train, n_test)


trait BaseShuffleSplit(BaseCrossValidator):
    pass


def _take_indices(values: IndexArray, start: Int, count: Int) raises -> IndexArray:
    var result = IndexArray(Shape(count))
    for i in range(count):
        result[Item(i)] = values[Item(start + i)]
    return result^


struct RepeatedKFold(BaseCrossValidator, Copyable):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _SplitList
    var n_splits: Int
    var n_repeats: Int
    var random_state: Optional[Int]

    def __init__(out self, *, n_splits: Int = 5, n_repeats: Int = 10, random_state: Optional[Int] = None) raises:
        if n_splits < 2 or n_repeats <= 0:
            raise Error("ValueError: invalid fold or repeat count")
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        return self.n_splits * self.n_repeats

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        var rng = _seed(self.random_state)
        return self.split(X, rng, y, groups)

    def split(self, X: Int, mut rng: RandomState, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        _validate_lengths(X, y, groups)
        var pairs = List[IndexSplit]()
        var cv = KFold(self.n_splits, shuffle=True)
        for _ in range(self.n_repeats):
            var cursor = cv.split(X, rng)
            while True:
                var pair = cursor.next()
                if not pair:
                    break
                pairs.append(pair.value().copy())
        return _SplitList(pairs^)


struct RepeatedStratifiedKFold(BaseCrossValidator, Copyable):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _SplitList
    var n_splits: Int
    var n_repeats: Int
    var random_state: Optional[Int]

    def __init__(out self, *, n_splits: Int = 5, n_repeats: Int = 10, random_state: Optional[Int] = None) raises:
        if n_splits < 2 or n_repeats <= 0:
            raise Error("ValueError: invalid fold or repeat count")
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        return self.n_splits * self.n_repeats

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        var rng = _seed(self.random_state)
        return self.split(X, rng, y, groups)

    def split(self, X: Int, mut rng: RandomState, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        _validate_lengths(X, y, groups)
        if not y:
            raise Error("ValueError: stratification requires y")
        var pairs = List[IndexSplit]()
        var cv = StratifiedKFold(self.n_splits, shuffle=True)
        for _ in range(self.n_repeats):
            var cursor = cv.split(X, y.value(), rng)
            while True:
                var pair = cursor.next()
                if not pair:
                    break
                pairs.append(pair.value().copy())
        return _SplitList(pairs^)

    def split(self, X: Int, y: SplitLabels, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        var rng = _seed(self.random_state)
        return self.split(X, rng, y, groups)

    def split(self, X: Int, mut rng: RandomState, y: SplitLabels, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        return self.split(X, rng, y.class_codes(), groups)


struct ShuffleSplit(BaseShuffleSplit, Copyable):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _SplitList
    var n_splits: Int
    var test_size: Optional[SplitSize]
    var train_size: Optional[SplitSize]
    var random_state: Optional[Int]

    def __init__(out self, n_splits: Int = 10, *, test_size: Optional[SplitSize] = None, train_size: Optional[SplitSize] = None, random_state: Optional[Int] = None) raises:
        if n_splits <= 0:
            raise Error("ValueError: n_splits must be positive")
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        return self.n_splits

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        var rng = _seed(self.random_state)
        return self.split(X, rng, y, groups)

    def split(self, X: Int, mut rng: RandomState, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        _validate_lengths(X, y, groups)
        var population = X
        var sizes = _sizes(population, self.test_size, self.train_size, 0.1)
        var pairs = List[IndexSplit]()
        for _ in range(self.n_splits):
            var order = rng.permutation(population)
            pairs.append(IndexSplit(_take_indices(order, sizes[1], sizes[0]), _take_indices(order, 0, sizes[1])))
        return _SplitList(pairs^)


struct GroupShuffleSplit(BaseShuffleSplit, Copyable):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _SplitList
    var n_splits: Int
    var test_size: Optional[SplitSize]
    var train_size: Optional[SplitSize]
    var random_state: Optional[Int]

    def __init__(out self, n_splits: Int = 5, *, test_size: Optional[SplitSize] = None, train_size: Optional[SplitSize] = None, random_state: Optional[Int] = None) raises:
        if n_splits <= 0:
            raise Error("ValueError: n_splits must be positive")
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        return self.n_splits

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        var rng = _seed(self.random_state)
        return self.split(X, rng, y, groups)

    def split(self, X: Int, y: Optional[IndexArray] = None, *, groups: SplitLabels) raises -> Self.Splits:
        var rng = _seed(self.random_state)
        return self.split(X, rng, y, groups=groups)

    def split(self, X: Int, mut rng: RandomState, y: Optional[IndexArray] = None, *, groups: SplitLabels) raises -> Self.Splits:
        return self.split(X, rng, y, groups.group_codes())

    def split(self, X: Int, mut rng: RandomState, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        _validate_lengths(X, y, groups)
        var population: Int
        if not groups:
            raise Error("ValueError: groups are required")
        var unique = _unique(groups.value())
        population = unique.size
        var encoded = _encode(groups.value(), unique)
        var sizes = _sizes(population, self.test_size, self.train_size, 0.2)
        var pairs = List[IndexSplit]()
        for _ in range(self.n_splits):
            var order = rng.permutation(population)
            var assignments = IndexArray(Shape(population))
            for i in range(population):
                assignments[Item(i)] = -1
            for i in range(sizes[1]):
                assignments[Item(Int(order[Item(i)]))] = 1
            for i in range(sizes[1], sizes[1] + sizes[0]):
                assignments[Item(Int(order[Item(i)]))] = 0
            var train_count = 0
            var test_count = 0
            for i in range(X):
                var f = Int(assignments[Item(Int(encoded[Item(i)]))])
                train_count += Int(f == 0)
                test_count += Int(f == 1)
            var train = IndexArray(Shape(train_count))
            var test = IndexArray(Shape(test_count))
            train_count = 0
            test_count = 0
            for i in range(X):
                var f = Int(assignments[Item(Int(encoded[Item(i)]))])
                if f == 0:
                    train[Item(train_count)] = Int64(i)
                    train_count += 1
                elif f == 1:
                    test[Item(test_count)] = Int64(i)
                    test_count += 1
            pairs.append(IndexSplit(train^, test^))
        return _SplitList(pairs^)


# Class allocation from sklearn.utils.extmath._approximate_mode.
# Tie sampling is a full shared-RNG permutation, as NumPy choice(replace=False).
def _allocate_classes(counts: IndexArray, draws: Int, mut rng: RandomState) raises -> IndexArray:
    from std.math import floor
    var total = 0
    for i in range(counts.size):
        total += Int(counts[Item(i)])
    var result = IndexArray(Shape(counts.size))
    var remainder = NDArray[DType.float64](Shape(counts.size))
    var left = draws
    for i in range(counts.size):
        var exact = Float64(counts[Item(i)]) / Float64(total) * Float64(draws)
        result[Item(i)] = Int64(floor(exact))
        remainder[Item(i)] = exact - floor(exact)
        left -= Int(result[Item(i)])
    while left > 0:
        var best: Float64 = -1
        for i in range(counts.size):
            if remainder[Item(i)] > best:
                best = remainder[Item(i)]
        var n_ties = 0
        for i in range(counts.size):
            n_ties += Int(remainder[Item(i)] == best)
        var ties = IndexArray(Shape(n_ties))
        var pos = 0
        for i in range(counts.size):
            if remainder[Item(i)] == best:
                ties[Item(pos)] = Int64(i)
                pos += 1
                remainder[Item(i)] = -1
        ties = _permute(ties, rng)
        var take = min(left, n_ties)
        for j in range(take):
            var i = Int(ties[Item(j)])
            result[Item(i)] = result[Item(i)] + 1
        left -= take
    return result^


struct StratifiedShuffleSplit(BaseShuffleSplit, Copyable):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _SplitList
    var n_splits: Int
    var test_size: Optional[SplitSize]
    var train_size: Optional[SplitSize]
    var random_state: Optional[Int]

    def __init__(out self, n_splits: Int = 10, *, test_size: Optional[SplitSize] = None, train_size: Optional[SplitSize] = None, random_state: Optional[Int] = None) raises:
        if n_splits <= 0:
            raise Error("ValueError: n_splits must be positive")
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        return self.n_splits

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        var rng = _seed(self.random_state)
        return self.split(X, rng, y, groups)

    def split(self, X: Int, y: SplitLabels, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        var rng = _seed(self.random_state)
        return self.split(X, rng, y, groups)

    def split(self, X: Int, mut rng: RandomState, y: SplitLabels, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        return self.split(X, rng, y.stratify_codes(), groups)

    def split(self, X: Int, mut rng: RandomState, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        _validate_lengths(X, y, groups)
        if not y:
            raise Error("ValueError: stratification requires y")
        var sizes = _sizes(X, self.test_size, self.train_size, 0.1)
        var unique = _unique(y.value())
        var encoded = _encode(y.value(), unique)
        var counts = _counts(encoded, unique.size)
        if sizes[0] < unique.size or sizes[1] < unique.size:
            raise Error("ValueError: train and test size must cover every class")
        for c in range(unique.size):
            if counts[Item(c)] < 2:
                raise Error("ValueError: every class needs at least two samples")
        var pairs = List[IndexSplit]()
        for _ in range(self.n_splits):
            var n_train = _allocate_classes(counts, sizes[0], rng)
            var remaining = counts.copy()
            for c in range(unique.size):
                remaining[Item(c)] = remaining[Item(c)] - n_train[Item(c)]
            var n_test = _allocate_classes(remaining, sizes[1], rng)
            var train = IndexArray(Shape(sizes[0]))
            var test = IndexArray(Shape(sizes[1]))
            var a = 0
            var b = 0
            for c in range(unique.size):
                var rows = IndexArray(Shape(Int(counts[Item(c)])))
                var pos = 0
                for i in range(X):
                    if Int(encoded[Item(i)]) == c:
                        rows[Item(pos)] = Int64(i)
                        pos += 1
                rows = _permute(rows, rng)
                for i in range(Int(n_train[Item(c)])):
                    train[Item(a)] = rows[Item(i)]
                    a += 1
                for i in range(Int(n_test[Item(c)])):
                    test[Item(b)] = rows[Item(Int(n_train[Item(c)]) + i)]
                    b += 1
            pairs.append(IndexSplit(_permute(train, rng), _permute(test, rng)))
        return _SplitList(pairs^)


struct StratifiedGroupKFold(BaseCrossValidator, Copyable):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _FoldCursor
    var n_splits: Int
    var shuffle: Bool
    var random_state: Optional[Int]

    def __init__(out self, n_splits: Int = 5, *, shuffle: Bool = False, random_state: Optional[Int] = None) raises:
        var checked = KFold(n_splits, shuffle=shuffle, random_state=random_state)
        self.n_splits = checked.n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        return self.n_splits

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        var rng = _seed(self.random_state, self.shuffle)
        return self.split(X, rng, y, groups)

    def split(self, X: Int, y: SplitLabels, groups: SplitLabels) raises -> Self.Splits:
        var rng = _seed(self.random_state, self.shuffle)
        return self.split(X, rng, y, groups)

    def split(self, X: Int, y: SplitLabels, groups: IndexArray) raises -> Self.Splits:
        return self.split(X, y, SplitLabels(groups))

    def split(self, X: Int, y: IndexArray, groups: SplitLabels) raises -> Self.Splits:
        return self.split(X, SplitLabels(y), groups)

    def split(self, X: Int, mut rng: RandomState, y: SplitLabels, groups: SplitLabels) raises -> Self.Splits:
        return self.split(X, rng, y.class_codes(), groups.group_codes())

    def split(self, X: Int, mut rng: RandomState, y: SplitLabels, groups: IndexArray) raises -> Self.Splits:
        return self.split(X, rng, y, SplitLabels(groups))

    def split(self, X: Int, mut rng: RandomState, y: IndexArray, groups: SplitLabels) raises -> Self.Splits:
        return self.split(X, rng, SplitLabels(y), groups)

    def split(self, X: Int, mut rng: RandomState, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        from std.math import sqrt, abs
        from moxie._dense import sum_axis2d
        _validate_lengths(X, y, groups)
        if not y or not groups:
            raise Error("ValueError: y and groups are required")
        var labels = _unique(y.value())
        var group_labels = _unique(groups.value())
        var classes = labels.size
        var n_groups = group_labels.size
        if n_groups < self.n_splits:
            raise Error("ValueError: n_splits exceeds group count")
        var ye = _encode(y.value(), labels)
        var ge = _encode(groups.value(), group_labels)
        var counts = _counts(ye, classes)
        var enough = False
        for c in range(classes):
            enough = enough or Int(counts[Item(c)]) >= self.n_splits
        if not enough:
            raise Error("ValueError: n_splits exceeds every class count")
        var group_counts = zeros[DType.float64](Shape(n_groups, classes))
        for i in range(X):
            var g = Int(ge[Item(i)])
            var c = Int(ye[Item(i)])
            group_counts[Item(g, c)] = group_counts[Item(g, c)] + 1
        var order = _range(0, n_groups)
        if self.shuffle:
            order = rng.permutation(n_groups)
        # Shared reductions compose the within-group class variance. Only the
        # standard deviation is required: sqrt rounding can create stable ties.
        var group_means = sum_axis2d(group_counts, 1) / Float64(classes)
        var residual = group_counts.copy()
        for g in range(n_groups):
            for c in range(classes):
                residual[Item(g, c)] = residual[Item(g, c)] - group_means[Item(g)]
        var dispersion = sum_axis2d(residual * residual, 1)
        for g in range(n_groups):
            dispersion[Item(g)] = sqrt(dispersion[Item(g)] / Float64(classes))
        # Stable insertion sort preserves shuffled order on equal dispersion.
        for i in range(1, n_groups):
            var current = order[Item(i)]
            var j = i
            while j > 0:
                if dispersion[Item(Int(order[Item(j - 1)]))] >= dispersion[Item(Int(current))]:
                    break
                order[Item(j)] = order[Item(j - 1)]
                j -= 1
            order[Item(j)] = current
        var fold_counts = zeros[DType.float64](Shape(self.n_splits, classes))
        var assignments = IndexArray(Shape(n_groups))
        var loads = zeros[DType.int64](Shape(self.n_splits))
        for pos in range(n_groups):
            var g = Int(order[Item(pos)])
            var best = 0
            var best_eval: Float64 = 1.0e300
            for f in range(self.n_splits):
                var ratios = fold_counts.copy()
                for k in range(self.n_splits):
                    for c in range(classes):
                        if k == f:
                            ratios[Item(k, c)] = ratios[Item(k, c)] + group_counts[Item(g, c)]
                        ratios[Item(k, c)] = ratios[Item(k, c)] / Float64(counts[Item(c)])
                var means = sum_axis2d(ratios, 0) / Float64(self.n_splits)
                var centered = ratios - means
                var variances = sum_axis2d(centered * centered, 0) / Float64(self.n_splits)
                var score: Float64 = 0
                for c in range(classes):
                    score += sqrt(variances[Item(c)])
                score /= Float64(classes)
                if score < best_eval or (abs(score - best_eval) <= 1.0e-8 + 1.0e-5 * abs(best_eval) and loads[Item(f)] < loads[Item(best)]):
                    best = f
                    best_eval = score
            assignments[Item(g)] = Int64(best)
            for c in range(classes):
                fold_counts[Item(best, c)] = fold_counts[Item(best, c)] + group_counts[Item(g, c)]
                loads[Item(best)] = loads[Item(best)] + Int64(group_counts[Item(g, c)])
        var rows = IndexArray(Shape(X))
        for i in range(X):
            rows[Item(i)] = assignments[Item(Int(ge[Item(i)]))]
        return _FoldCursor(rows^, _range(0, self.n_splits))


struct CheckedCV(BaseCrossValidator, Copyable):
    """Native tagged KFold/StratifiedKFold selection returned by check_cv."""
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _FoldCursor
    var folds: KFold
    var stratified: Bool

    def __init__(out self, var folds: KFold, stratified: Bool):
        self.folds = folds^
        self.stratified = stratified

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        return self.folds.n_splits

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        if self.stratified:
            return StratifiedKFold(self.folds.n_splits, shuffle=self.folds.shuffle, random_state=self.folds.random_state).split(X, y, groups)
        return self.folds.split(X, y, groups)

    def split(self, X: Int, mut rng: RandomState, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        _validate_lengths(X, y, groups)
        if self.stratified:
            if not y:
                raise Error("ValueError: stratification requires y")
            return StratifiedKFold(self.folds.n_splits, shuffle=self.folds.shuffle).split(X, y.value(), rng)
        return self.folds.split(X, rng)

    def split(self, X: Int, y: SplitLabels, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        if self.stratified:
            return StratifiedKFold(self.folds.n_splits, shuffle=self.folds.shuffle, random_state=self.folds.random_state).split(X, y, groups)
        _validate_lengths(X, None, groups)
        if len(y) != X:
            raise Error("ValueError: y must have one entry per sample")
        return self.folds.split(X, None, groups)

    def split(self, X: Int, mut rng: RandomState, y: SplitLabels, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        _validate_lengths(X, None, groups)
        if self.stratified:
            return StratifiedKFold(self.folds.n_splits, shuffle=self.folds.shuffle).split(X, y, rng)
        if len(y) != X:
            raise Error("ValueError: y must have one entry per sample")
        return self.folds.split(X, rng)


def check_cv(cv: Optional[Int] = None, y: Optional[IndexArray] = None, *, classifier: Bool = False, shuffle: Bool = False, random_state: Optional[Int] = None) raises -> CheckedCV:
    var n = 5
    if cv:
        n = cv.value()
    return CheckedCV(KFold(n, shuffle=shuffle, random_state=random_state), classifier and Bool(y))


def check_cv(cv: Optional[Int], y: SplitLabels, *, classifier: Bool = False, shuffle: Bool = False, random_state: Optional[Int] = None) raises -> CheckedCV:
    """Stratify only binary/multiclass targets, as sklearn's type_of_target dispatch does."""
    var n = 5
    if cv:
        n = cv.value()
    var stratified = classifier and (y.kind == "binary" or y.kind == "multiclass")
    return CheckedCV(KFold(n, shuffle=shuffle, random_state=random_state), stratified)


def check_cv[C: CrossValidator](var cv: C, y: Optional[IndexArray] = None, *, classifier: Bool = False, shuffle: Bool = False, random_state: Optional[Int] = None) -> C:
    """Pass an explicit splitter through by ownership transfer."""
    return cv^


def check_cv[C: CrossValidator](var cv: C, y: SplitLabels, *, classifier: Bool = False, shuffle: Bool = False, random_state: Optional[Int] = None) -> C:
    return cv^


struct IterableCV(BaseCrossValidator):
    comptime Input = Int
    comptime Target = IndexArray
    comptime Groups = IndexArray
    comptime Splits = _SplitList
    var pairs: List[IndexSplit]

    def __init__(out self, pairs: List[IndexSplit]) raises:
        self.pairs = _copy_splits(pairs)

    def get_n_splits(self, X: Optional[Int] = None, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Int:
        return len(self.pairs)

    def split(self, X: Int, y: Optional[IndexArray] = None, groups: Optional[IndexArray] = None) raises -> Self.Splits:
        _validate_lengths(X, y, groups)
        for i in range(len(self.pairs)):
            self.pairs[i].validate(X)
        return _SplitList(_copy_splits(self.pairs))


def check_cv(cv: List[IndexSplit], y: Optional[IndexArray] = None, *, classifier: Bool = False, shuffle: Bool = False, random_state: Optional[Int] = None) raises -> IterableCV:
    return IterableCV(cv)


def check_cv(cv: List[IndexSplit], y: SplitLabels, *, classifier: Bool = False, shuffle: Bool = False, random_state: Optional[Int] = None) raises -> IterableCV:
    return IterableCV(cv)


def _rows[dtype: DType](X: NDArray[dtype], indices: IndexArray) raises -> NDArray[dtype]:
    if X.ndim == 1:
        var result = NDArray[dtype](Shape(indices.size))
        for i in range(indices.size):
            result[Item(i)] = X[Item(Int(indices[Item(i)]))]
        return result^
    if X.ndim != 2:
        raise Error("ValueError: dense row splitting supports rank one or two")
    var result = NDArray[dtype](Shape(indices.size, X.shape[1]))
    for i in range(indices.size):
        for j in range(X.shape[1]):
            result[Item(i, j)] = X[Item(Int(indices[Item(i)]), j)]
    return result^


def _train_test_indices(n: Int, mut rng: RandomState, test_size: Optional[SplitSize], train_size: Optional[SplitSize], shuffle: Bool, stratify: Optional[IndexArray]) raises -> IndexSplit:
    var sizes = _sizes(n, test_size, train_size, 0.25)
    if not shuffle:
        if stratify:
            raise Error("ValueError: stratification requires shuffle")
        return IndexSplit(_range(0, sizes[0]), _range(sizes[0], sizes[0] + sizes[1]))
    if stratify:
        var cv = StratifiedShuffleSplit(1, test_size=SplitSize(sizes[1]), train_size=SplitSize(sizes[0]))
        var cursor = cv.split(n, rng, y=stratify)
        return cursor.next().value().copy()
    var order = rng.permutation(n)
    return IndexSplit(_take_indices(order, sizes[1], sizes[0]), _take_indices(order, 0, sizes[1]))


def train_test_split[dtype: DType](X: NDArray[dtype], mut rng: RandomState, *, test_size: Optional[SplitSize] = None, train_size: Optional[SplitSize] = None, shuffle: Bool = True, stratify: Optional[IndexArray] = None) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    if X.ndim != 1 and X.ndim != 2:
        raise Error("ValueError: expected rank-one or rank-two dense input")
    var pair = _train_test_indices(X.shape[0], rng, test_size, train_size, shuffle, stratify)
    return (_rows(X, pair.train), _rows(X, pair.test))


def train_test_split[dtype: DType](X: NDArray[dtype], *, test_size: Optional[SplitSize] = None, train_size: Optional[SplitSize] = None, random_state: Optional[Int] = None, shuffle: Bool = True, stratify: Optional[IndexArray] = None) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    var rng = _seed(random_state, shuffle)
    return train_test_split(X, rng, test_size=test_size, train_size=train_size, shuffle=shuffle, stratify=stratify)


def train_test_split[dx: DType, dy: DType](X: NDArray[dx], y: NDArray[dy], mut rng: RandomState, *, test_size: Optional[SplitSize] = None, train_size: Optional[SplitSize] = None, shuffle: Bool = True, stratify: Optional[IndexArray] = None) raises -> Tuple[NDArray[dx], NDArray[dx], NDArray[dy], NDArray[dy]]:
    if (X.ndim != 1 and X.ndim != 2) or (y.ndim != 1 and y.ndim != 2):
        raise Error("ValueError: expected rank-one or rank-two dense inputs")
    if X.shape[0] != y.shape[0]:
        raise Error("ValueError: inconsistent sample counts")
    var pair = _train_test_indices(X.shape[0], rng, test_size, train_size, shuffle, stratify)
    return (_rows(X, pair.train), _rows(X, pair.test), _rows(y, pair.train), _rows(y, pair.test))


def train_test_split[dx: DType, dy: DType](X: NDArray[dx], y: NDArray[dy], *, test_size: Optional[SplitSize] = None, train_size: Optional[SplitSize] = None, random_state: Optional[Int] = None, shuffle: Bool = True, stratify: Optional[IndexArray] = None) raises -> Tuple[NDArray[dx], NDArray[dx], NDArray[dy], NDArray[dy]]:
    var rng = _seed(random_state, shuffle)
    return train_test_split(X, y, rng, test_size=test_size, train_size=train_size, shuffle=shuffle, stratify=stratify)


def _stratify(stratify: SplitLabels, n: Int) raises -> IndexArray:
    if len(stratify) != n:
        raise Error("ValueError: stratify must have one entry per sample")
    return stratify.stratify_codes()


def _row_strings(values: List[String], indices: IndexArray) raises -> List[String]:
    var result = List[String]()
    for i in range(indices.size):
        result.append(values[Int(indices[Item(i)])].copy())
    return result^


def train_test_split[dtype: DType](X: NDArray[dtype], mut rng: RandomState, *, stratify: SplitLabels, test_size: Optional[SplitSize] = None, train_size: Optional[SplitSize] = None, shuffle: Bool = True) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    return train_test_split(X, rng, test_size=test_size, train_size=train_size, shuffle=shuffle, stratify=_stratify(stratify, X.shape[0]))


def train_test_split[dtype: DType](X: NDArray[dtype], *, stratify: SplitLabels, test_size: Optional[SplitSize] = None, train_size: Optional[SplitSize] = None, random_state: Optional[Int] = None, shuffle: Bool = True) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    return train_test_split(X, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=_stratify(stratify, X.shape[0]))


def train_test_split[dx: DType, dy: DType](X: NDArray[dx], y: NDArray[dy], mut rng: RandomState, *, stratify: SplitLabels, test_size: Optional[SplitSize] = None, train_size: Optional[SplitSize] = None, shuffle: Bool = True) raises -> Tuple[NDArray[dx], NDArray[dx], NDArray[dy], NDArray[dy]]:
    return train_test_split(X, y, rng, test_size=test_size, train_size=train_size, shuffle=shuffle, stratify=_stratify(stratify, X.shape[0]))


def train_test_split[dx: DType, dy: DType](X: NDArray[dx], y: NDArray[dy], *, stratify: SplitLabels, test_size: Optional[SplitSize] = None, train_size: Optional[SplitSize] = None, random_state: Optional[Int] = None, shuffle: Bool = True) raises -> Tuple[NDArray[dx], NDArray[dx], NDArray[dy], NDArray[dy]]:
    return train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=_stratify(stratify, X.shape[0]))


def train_test_split[dx: DType](X: NDArray[dx], y: List[String], mut rng: RandomState, *, test_size: Optional[SplitSize] = None, train_size: Optional[SplitSize] = None, shuffle: Bool = True, stratify: Optional[IndexArray] = None) raises -> Tuple[NDArray[dx], NDArray[dx], List[String], List[String]]:
    """String targets split like sklearn lists: owned train/test String lists."""
    if X.ndim != 1 and X.ndim != 2:
        raise Error("ValueError: expected rank-one or rank-two dense input")
    if X.shape[0] != len(y):
        raise Error("ValueError: inconsistent sample counts")
    var pair = _train_test_indices(X.shape[0], rng, test_size, train_size, shuffle, stratify)
    return (_rows(X, pair.train), _rows(X, pair.test), _row_strings(y, pair.train), _row_strings(y, pair.test))


def train_test_split[dx: DType](X: NDArray[dx], y: List[String], *, test_size: Optional[SplitSize] = None, train_size: Optional[SplitSize] = None, random_state: Optional[Int] = None, shuffle: Bool = True, stratify: Optional[IndexArray] = None) raises -> Tuple[NDArray[dx], NDArray[dx], List[String], List[String]]:
    var rng = _seed(random_state, shuffle)
    return train_test_split(X, y, rng, test_size=test_size, train_size=train_size, shuffle=shuffle, stratify=stratify)


def train_test_split[dx: DType](X: NDArray[dx], y: List[String], mut rng: RandomState, *, stratify: SplitLabels, test_size: Optional[SplitSize] = None, train_size: Optional[SplitSize] = None, shuffle: Bool = True) raises -> Tuple[NDArray[dx], NDArray[dx], List[String], List[String]]:
    return train_test_split(X, y, rng, test_size=test_size, train_size=train_size, shuffle=shuffle, stratify=_stratify(stratify, X.shape[0]))


def train_test_split[dx: DType](X: NDArray[dx], y: List[String], *, stratify: SplitLabels, test_size: Optional[SplitSize] = None, train_size: Optional[SplitSize] = None, random_state: Optional[Int] = None, shuffle: Bool = True) raises -> Tuple[NDArray[dx], NDArray[dx], List[String], List[String]]:
    return train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=_stratify(stratify, X.shape[0]))


def _copy_splits(pairs: List[IndexSplit]) raises -> List[IndexSplit]:
    var result = List[IndexSplit]()
    for i in range(len(pairs)):
        result.append(pairs[i].copy())
    return result^
