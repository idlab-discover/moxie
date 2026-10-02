"""Parity: LabelEncoder, LabelBinarizer, MultiLabelBinarizer and label_binarize.

Native labels are `List[String]` or NuMojo Int64/Float32/Float64 vectors (and
rank-2 indicators); sklearn receives the equivalent NumPy arrays (Python lists
of lists for MultiLabelBinarizer). Every output is compared exactly: codes and
indicators are integers, classes and inverse transforms are copies of input
labels, and thresholds are compared in sklearn's floating dtype. Error cases
check that both sides raise, plus a message fragment.
"""
import json
import math

import numpy as np

from harness import Driver, oracle

oracle()
from sklearn import preprocessing as P

d = Driver('labels')
expected = {}  # (case, name) -> value
errors = {}  # (case, name) -> needle
cases = {}  # case -> list of Mojo lines

MOJO = {'str': 'StringLabels', 'int64': 'NumericLabels[DType.int64]',
        'float32': 'NumericLabels[DType.float32]', 'float64': 'NumericLabels[DType.float64]'}
NP = {'int64': np.int64, 'float32': np.float32, 'float64': np.float64}

d.prelude('''
from moxie.preprocessing._label import StringLabels, NumericLabels

def emit_labels(name: String, x: List[String]) raises:
    emit_names(name, x)

def emit_labels[d: DType](name: String, x: NDArray[d]) raises:
    emit(name, x)

def emit_rows(name: String, rows: List[List[String]]) raises:
    emit_int(name + "_n", len(rows))
    for i in range(len(rows)):
        emit_names(name + "_" + String(i), rows[i])

def emit_rows(name: String, rows: List[List[Int64]]) raises:
    emit_int(name + "_n", len(rows))
    for i in range(len(rows)):
        var row = zeros[DType.int64](Shape(len(rows[i])))
        for j in range(len(rows[i])):
            row.unsafe_set(j, rows[i][j])
        emit(name + "_" + String(i), row)
''')


def sk_call(fn):
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 - any sklearn error is the expectation
        return None, f'{type(exc).__name__}: {exc}'


def line(case, text):
    cases.setdefault(case, []).append(text)


def expect(case, name, value):
    expected[(case, name)] = value


def expect_error(case, name, stmt, fn, needle):
    """Native statement must raise; sklearn `fn` must raise too (unless fn is None)."""
    if fn is not None:
        _, err = sk_call(fn)
        assert err is not None, f'sklearn did not raise for {case}.{name}'
        assert needle in err, f'{case}.{name}: sklearn says {err!r}, needle {needle!r}'
    line(case, f'try:\n    {stmt}\n    print("I {name} 0")\nexcept e:\n    emit_error("{name}", e)')
    errors[(case, name)] = needle


def labels(values, kind):
    if kind == 'str':
        return d.strings(list(values))
    return d.input(np.asarray(values, dtype=NP[kind]))


def as_expected(value, kind):
    if kind == 'str':
        return [str(v) for v in value]
    return np.asarray(value)


# ---------------------------------------------------------------------------
# LabelEncoder

ENC = {
    'int_a': ([2, 1, 3, 1, 3], 'int64', [3, 1]),
    'int_neg': ([1, 1, 4, 5, -1, 0], 'int64', [0, -1, 5]),
    'int_extreme': ([2 ** 62, -2 ** 63, 0, 2 ** 63 - 1, 0], 'int64', [0, -2 ** 63]),
    'float_nonfinite': ([1.5, np.nan, -np.inf, 2.0, np.inf, np.nan, 1.5, -0.5], 'float64', [np.nan, -0.5, np.inf]),
    'float_plain': ([0.25, -3.0, 0.25, 1e20], 'float64', [1e20]),
    'f32_nonfinite': ([1.5, np.nan, -np.inf, 2.0, 1.5, 0.1], 'float32', [0.1, np.nan]),
    'str_a': (['b', 'a', 'c', 'a', 'c'], 'str', ['c', 'b']),
    'str_unicode': (['é', 'e', 'z', 'ä', 'Z', 'e'], 'str', ['ä', 'Z']),
}
for key, (values, kind, known) in ENC.items():
    case = f'enc_{key}'
    y = np.asarray(values, dtype=object if kind == 'str' else NP[kind])
    if kind == 'str':
        y = np.asarray(values)
    ref = P.LabelEncoder()
    codes = ref.fit_transform(y)
    line(case, f'var model = LabelEncoder[{MOJO[kind]}]()')
    line(case, f'var y = {labels(values, kind)}')
    line(case, 'var codes = model.fit_transform(y)')
    line(case, 'emit("codes", codes)')
    line(case, 'emit_labels("classes", model.classes_)')
    line(case, 'emit_labels("inverse", model.inverse_transform(codes))')
    expect(case, 'codes', codes)
    expect(case, 'classes', as_expected(ref.classes_, kind))
    expect(case, 'inverse', as_expected(ref.inverse_transform(codes), kind))
    known_y = np.asarray(known, dtype=y.dtype)
    line(case, f'emit("known", model.transform({labels(known, kind)}))')
    expect(case, 'known', ref.transform(known_y))
    # Refit on a subset replaces the classes; transform uses the new ones.
    line(case, f'model.fit({labels(values[:3], kind)})')
    line(case, 'emit_labels("refit_classes", model.classes_)')
    expect(case, 'refit_classes', as_expected(P.LabelEncoder().fit(y[:3]).classes_, kind))
    # Unseen labels and codes.
    unseen = ['zz', 'q'] if kind == 'str' else [77, 66]
    unseen_y = np.asarray(unseen, dtype=y.dtype)
    expect_error(case, 'unseen', f'_ = model.transform({labels(unseen, kind)})',
                 lambda: P.LabelEncoder().fit(y[:3]).transform(unseen_y), 'previously unseen labels')
    expect_error(case, 'bad_code', f'_ = model.inverse_transform({d.input(np.array([0, 9, -1], dtype=np.int64))})',
                 lambda: P.LabelEncoder().fit(y[:3]).inverse_transform(np.array([0, 9, -1])), 'previously unseen labels')

# Strided, column-vector, empty, shape and unfitted behavior.
y_int = np.array([5, 1, 9, 1, 5, 7], dtype=np.int64)
case = 'enc_layout'
line(case, f'var base = {d.input(y_int)}')
line(case, 'var reversed = base.view_with_layout(Shape(6), nm.NDArrayStrides(-1), 5)')
line(case, 'var model = LabelEncoder[NumericLabels[DType.int64]]()')
line(case, 'emit("reversed", model.fit_transform(reversed))')
line(case, 'emit("every_other", model.transform(base.view_with_layout(Shape(3), nm.NDArrayStrides(2), 0)))')
line(case, f'emit("column", model.fit_transform({d.input(y_int.reshape(-1, 1))}))')
line(case, f'var empty = model.transform({d.input(np.zeros(0, dtype=np.int64))})')
line(case, 'emit_int("empty_size", empty.size)')
line(case, f'var empty_inverse = model.inverse_transform({d.input(np.zeros(0, dtype=np.int64))})')
line(case, 'emit_int("empty_inverse_size", empty_inverse.size)')
ref = P.LabelEncoder()
expect(case, 'reversed', ref.fit_transform(y_int[::-1]))
expect(case, 'every_other', ref.transform(y_int[::2]))
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    expect(case, 'column', P.LabelEncoder().fit_transform(y_int.reshape(-1, 1)))
expect(case, 'empty_size', ref.transform(np.zeros(0, dtype=np.int64)).size)
expect(case, 'empty_inverse_size', ref.inverse_transform(np.zeros(0, dtype=np.int64)).size)
expect_error(case, 'matrix', f'model.fit({d.input(np.ones((2, 2), dtype=np.int64))})',
             lambda: P.LabelEncoder().fit(np.ones((2, 2), dtype=np.int64)), 'y should be a 1d array')
line(case, 'var unfitted = LabelEncoder()')
expect_error(case, 'unfitted', f'_ = unfitted.transform({d.strings(["a"])})',
             lambda: P.LabelEncoder().transform(np.array(['a'])), 'not fitted')

# ---------------------------------------------------------------------------
# LabelBinarizer

rng = np.random.RandomState(0)
LB_TARGETS = {
    'int_binary': ([3, 5, 3, 5, 5], 'int64', [5, 4, 3]),
    'int_multi': ([1, 4, 6, 4, 1, 2], 'int64', [6, 0, 2, 9]),
    'int_single': ([8, 8, 8], 'int64', [8, 1]),
    'float_multi': ([0.0, 2.0, 1.0, 2.0], 'float64', [1.0, 3.0]),
    'f32_binary': ([-1.0, 1.0, 1.0], 'float32', [1.0, 0.0]),
    'str_binary': (['no', 'yes', 'no'], 'str', ['yes', 'maybe']),
    'str_multi': (['cat', 'dog', 'emu', 'dog'], 'str', ['emu', 'yak', 'cat']),
}
LB_PARAMS = [(0, 1), (-1, 1), (-2, 0), (0, 3), (-3, 5), (2, 7)]
for key, (values, kind, test) in LB_TARGETS.items():
    y = np.asarray(values) if kind == 'str' else np.asarray(values, dtype=NP[kind])
    y_test = np.asarray(test, dtype=y.dtype)
    for neg, pos in LB_PARAMS:
        case = f'lb_{key}_{neg}_{pos}'.replace('-', 'm')
        ref = P.LabelBinarizer(neg_label=neg, pos_label=pos)
        Y = ref.fit_transform(y)
        line(case, f'var model = LabelBinarizer[{MOJO[kind]}](neg_label={neg}, pos_label={pos})')
        line(case, f'var Y = model.fit_transform({labels(values, kind)})')
        line(case, 'emit("Y", Y)')
        line(case, 'emit_labels("classes", model.classes_)')
        line(case, 'emit_names("type", [model.y_type_])')
        line(case, 'emit_labels("inverse", model.inverse_transform(Y))')
        line(case, f'emit("test", model.transform({labels(test, kind)}))')
        expect(case, 'Y', Y)
        expect(case, 'classes', as_expected(ref.classes_, kind))
        expect(case, 'type', [ref.y_type_])
        expect(case, 'inverse', as_expected(ref.inverse_transform(Y), kind))
        expect(case, 'test', ref.transform(y_test))
        # Decision scores: default threshold and an explicit one, Float64 and Float32.
        width = Y.shape[1]
        scores = (rng.rand(len(values), width) * (pos - neg) + neg).astype(np.float64)
        scores[0, :] = (pos + neg) / 2.0  # exactly on the default threshold
        for dt in (np.float64, np.float32):
            S = scores.astype(dt)
            S_in = d.input(S)
            tag = dt.__name__
            line(case, f'emit_labels("scores_{tag}", model.inverse_transform({S_in}, None))')
            line(case, f'emit_labels("scores_{tag}_t", model.inverse_transform({S_in}, Optional[Float64](0.3)))')
            expect(case, f'scores_{tag}', as_expected(ref.inverse_transform(S), kind))
            expect(case, f'scores_{tag}_t', as_expected(ref.inverse_transform(S, threshold=0.3), kind))

# Multilabel indicators (Int64 and Float64), a column vector and binary 2-column scores.
INDICATOR = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.int64)
for dt in ('int64', 'float64'):
    for neg, pos in [(0, 1), (-1, 1), (-2, 0), (0, 4)]:
        case = f'lb_multilabel_{dt}_{neg}_{pos}'.replace('-', 'm')
        Yin = INDICATOR.astype(NP[dt])
        ref = P.LabelBinarizer(neg_label=neg, pos_label=pos)
        Y = ref.fit_transform(Yin)
        line(case, f'var model = LabelBinarizer[{MOJO[dt]}](neg_label={neg}, pos_label={pos})')
        line(case, f'var Y = model.fit_transform({d.input(Yin)})')
        line(case, 'emit("Y", Y)')
        line(case, 'emit_labels("classes", model.classes_)')
        line(case, 'emit_names("type", [model.y_type_])')
        line(case, 'emit_labels("inverse", model.inverse_transform(Y))')
        line(case, f'emit("one_d", model.transform({d.input(np.array([2, 0, 1], dtype=NP[dt]))}))')
        expect(case, 'Y', Y)
        expect(case, 'classes', ref.classes_)
        expect(case, 'type', [ref.y_type_])
        expect(case, 'inverse', ref.inverse_transform(Y))
        expect(case, 'one_d', ref.transform(np.array([2, 0, 1], dtype=NP[dt])))
        scores = rng.rand(4, 3)
        line(case, f'emit_labels("scores", model.inverse_transform({d.input(scores)}, Optional[Float64](0.4)))')
        expect(case, 'scores', ref.inverse_transform(scores, threshold=0.4))

case = 'lb_shapes'
ref = P.LabelBinarizer().fit(np.array([4, 9, 4], dtype=np.int64))
line(case, 'var model = LabelBinarizer[NumericLabels[DType.int64]]()')
line(case, f'model.fit({d.input(np.array([[4], [9], [4]], dtype=np.int64))})')
line(case, 'emit_labels("classes", model.classes_)')
two = np.array([[0.2, 0.9], [0.8, 0.1]])
one = np.array([0.2, 0.9, 0.5])
line(case, f'emit_labels("two_columns", model.inverse_transform({d.input(two)}, None))')
line(case, f'emit_labels("vector", model.inverse_transform({d.input(one)}, None))')
expect(case, 'classes', np.array([4, 9]))
expect(case, 'two_columns', ref.inverse_transform(two))
expect(case, 'vector', ref.inverse_transform(one))
multi = P.LabelBinarizer().fit(np.array([0, 1, 2], dtype=np.int64))
nan_scores = np.array([[0.1, np.nan, 0.9], [0.5, 0.5, 0.2], [np.nan, 1.0, 2.0]])
line(case, 'var multi = LabelBinarizer[NumericLabels[DType.int64]]()')
line(case, f'multi.fit({d.input(np.array([0, 1, 2], dtype=np.int64))})')
line(case, f'emit_labels("argmax_nan", multi.inverse_transform({d.input(nan_scores)}, None))')
line(case, f'emit_labels("argmax_wide", multi.inverse_transform({d.input(np.array([[0, 0, 0, 5], [1, 0, 0, 0]], dtype=np.int64))}))')
expect(case, 'argmax_nan', multi.inverse_transform(nan_scores))
expect(case, 'argmax_wide', multi.inverse_transform(np.array([[0, 0, 0, 5], [1, 0, 0, 0]])))

# LabelBinarizer errors.
case = 'lb_errors'
line(case, 'var bad = LabelBinarizer[NumericLabels[DType.int64]](neg_label=2, pos_label=2)')
expect_error(case, 'neg_ge_pos', f'bad.fit({d.input(np.array([0, 1], dtype=np.int64))})',
             lambda: P.LabelBinarizer(neg_label=2, pos_label=2).fit([0, 1]), 'neg_label=2 must be strictly less than pos_label=2.')
line(case, 'var sparse_bad = LabelBinarizer[NumericLabels[DType.int64]](neg_label=1, pos_label=2, sparse_output=True)')
expect_error(case, 'sparse_labels', f'sparse_bad.fit({d.input(np.array([0, 1], dtype=np.int64))})',
             lambda: P.LabelBinarizer(neg_label=1, pos_label=2, sparse_output=True).fit([0, 1]),
             'Sparse binarization is only supported with non zero pos_label and zero neg_label, got pos_label=2 and neg_label=1')
line(case, 'var sparse = LabelBinarizer[NumericLabels[DType.int64]](sparse_output=True)')
expect_error(case, 'sparse_deferred', f'sparse.fit({d.input(np.array([0, 1], dtype=np.int64))})', None, 'sparse output is deferred')
line(case, 'var model = LabelBinarizer[NumericLabels[DType.int64]]()')
expect_error(case, 'multioutput', f'model.fit({d.input(np.array([[1, 3], [2, 1]], dtype=np.int64))})',
             lambda: P.LabelBinarizer().fit(np.array([[1, 3], [2, 1]])), 'Multioutput target data is not supported')
expect_error(case, 'unfitted', f'_ = model.transform({d.input(np.array([0, 1], dtype=np.int64))})',
             lambda: P.LabelBinarizer().transform(np.array([0, 1])), 'not fitted')
expect_error(case, 'empty', f'model.fit({d.input(np.zeros(0, dtype=np.int64))})',
             lambda: P.LabelBinarizer().fit(np.zeros(0, dtype=np.int64)), 'y has 0 samples')
line(case, 'var floats = LabelBinarizer[NumericLabels[DType.float64]]()')
expect_error(case, 'continuous', f'floats.fit({d.input(np.array([1.5, 2.0]))})',
             lambda: P.LabelBinarizer().fit(np.array([1.5, 2.0])), 'Unknown label type')
expect_error(case, 'nan', f'floats.fit({d.input(np.array([1.0, np.nan]))})',
             lambda: P.LabelBinarizer().fit(np.array([1.0, np.nan])), 'Input y contains NaN')
expect_error(case, 'inf', f'floats.fit({d.input(np.array([1.0, np.inf]))})',
             lambda: P.LabelBinarizer().fit(np.array([1.0, np.inf])), 'Input y contains infinity')
line(case, f'model.fit({d.input(np.array([1, 2, 3], dtype=np.int64))})')
expect_error(case, 'not_multilabel', f'_ = model.transform({d.input(np.array([[1, 0], [0, 1]], dtype=np.int64))})',
             lambda: P.LabelBinarizer().fit(np.array([1, 2, 3])).transform(np.array([[1, 0], [0, 1]])), 'not fitted with multilabel input')
line(case, 'var binary = LabelBinarizer[NumericLabels[DType.int64]]()')
line(case, f'binary.fit({d.input(np.array([0, 1], dtype=np.int64))})')
expect_error(case, 'binary_wide', f'_ = binary.inverse_transform({d.input(np.zeros((2, 3), dtype=np.int64))})',
             lambda: P.LabelBinarizer().fit(np.array([0, 1])).inverse_transform(np.zeros((2, 3))), "output_type='binary', but y.shape = (2, 3)")
line(case, 'var indicator = LabelBinarizer[NumericLabels[DType.int64]]()')
line(case, f'indicator.fit({d.input(INDICATOR)})')
expect_error(case, 'multilabel_width', f'_ = indicator.inverse_transform({d.input(np.zeros((2, 2), dtype=np.int64))})',
             lambda: P.LabelBinarizer().fit(INDICATOR).inverse_transform(np.zeros((2, 2))), 'The number of class is not equal')

# ---------------------------------------------------------------------------
# label_binarize

LBZ = [
    ('order', [1, 6], [1, 2, 4, 6], 'int64', 0, 1),
    ('unsorted', [1, 2, 4, 6, 2], [6, 1, 4, 2], 'int64', 0, 1),
    ('binary_unsorted', [1, 2, 1], [2, 1], 'int64', 0, 1),
    ('binary_superset', [1, 2, 1], [0, 1, 2], 'int64', -1, 1),
    ('multiclass_two', [1, 2, 7, 1], [2, 1], 'int64', 0, 1),
    ('unseen', [1, 0, 2, 9], [3, 4, 5], 'int64', 0, 1),
    ('duplicates', [1, 2, 5], [1, 1, 2], 'int64', 0, 1),
    ('dup_unsorted', [1, 2, 5, 2], [2, 1, 1, 5], 'int64', 0, 1),
    ('pos_switch', [1, 2, 1], [1, 2], 'int64', -2, 0),
    ('pos_switch_multi', [1, 2, 3, 1], [3, 2, 1], 'int64', -1, 0),
    ('shifted', [5, 7, 9], [9, 5, 7], 'int64', 2, 7),
    ('float', [1.0, 2.0, 1.0], [2.0, 1.0], 'float64', 0, 1),
    ('f32', [3.0, 1.0, 2.0], [1.0, 2.0, 3.0], 'float32', -4, -1),
    ('strings', ['cat', 'dog', 'emu'], ['dog', 'cat', 'bird'], 'str', 0, 1),
    ('strings_binary', ['yes', 'no', 'no'], ['yes', 'no'], 'str', -1, 2),
]
for key, y, classes, kind, neg, pos in LBZ:
    case = f'lbz_{key}'
    y_np = np.asarray(y) if kind == 'str' else np.asarray(y, dtype=NP[kind])
    c_np = np.asarray(classes) if kind == 'str' else np.asarray(classes, dtype=NP[kind])
    line(case, f'emit("Y", label_binarize({labels(y, kind)}, classes={labels(classes, kind)}, neg_label={neg}, pos_label={pos}))')
    expect(case, 'Y', P.label_binarize(y_np, classes=c_np, neg_label=neg, pos_label=pos))

for dt in ('int64', 'float64'):
    for neg, pos in [(0, 1), (-1, 0), (-1, 1), (0, 3)]:
        for classes in ([0, 1, 2], [2, 0, 1], [5, 3, 9]):
            case = f'lbz_multilabel_{dt}_{neg}_{pos}_{"_".join(map(str, classes))}'.replace('-', 'm')
            Yin = np.array([[2, 0, 2], [0, 2, 0], [2, 2, 0]], dtype=NP[dt]) if pos == 3 else INDICATOR.astype(NP[dt])
            line(case, f'emit("Y", label_binarize({d.input(Yin)}, classes={d.input(np.array(classes, dtype=NP[dt]))}, neg_label={neg}, pos_label={pos}))')
            expect(case, 'Y', P.label_binarize(Yin, classes=np.array(classes, dtype=NP[dt]), neg_label=neg, pos_label=pos))

case = 'lbz_errors'
ii = lambda v: d.input(np.asarray(v, dtype=np.int64))  # noqa: E731
expect_error(case, 'continuous', f'_ = label_binarize({d.input(np.array([1.2, 2.7]))}, classes={d.input(np.array([0.0, 1.0]))})',
             lambda: P.label_binarize(np.array([1.2, 2.7]), classes=[0.0, 1.0]), 'continuous target data is not supported')
expect_error(case, 'mismatch', f'_ = label_binarize({ii([[1, 0]])}, classes={ii([0, 1, 2])})',
             lambda: P.label_binarize(np.array([[1, 0]]), classes=[0, 1, 2]), 'mismatch with the labels [0 1] found in the data')
expect_error(case, 'empty', f'_ = label_binarize({ii([])}, classes={ii([1, 2])})',
             lambda: P.label_binarize(np.array([], dtype=np.int64), classes=[1, 2]), 'Found array with 0 sample(s) (shape=(0,))')
expect_error(case, 'multioutput', f'_ = label_binarize({ii([[1, 3], [2, 1]])}, classes={ii([1, 2, 3])})',
             lambda: P.label_binarize(np.array([[1, 3], [2, 1]]), classes=[1, 2, 3]), 'Multioutput target data is not supported')
expect_error(case, 'neg_ge_pos', f'_ = label_binarize({ii([0, 2])}, classes={ii([0, 2])}, neg_label=1, pos_label=0)',
             lambda: P.label_binarize(np.array([0, 2]), classes=[0, 2], neg_label=1, pos_label=0), 'neg_label=1 must be strictly less than pos_label=0.')
expect_error(case, 'sparse_labels', f'_ = label_binarize({ii([1, 2])}, classes={ii([1, 2])}, neg_label=-1, sparse_output=True)',
             lambda: P.label_binarize(np.array([1, 2]), classes=[1, 2], neg_label=-1, sparse_output=True), 'Sparse binarization is only supported')
expect_error(case, 'sparse_deferred', f'_ = label_binarize({ii([1, 2])}, classes={ii([1, 2])}, sparse_output=True)', None, 'sparse output is deferred')
expect_error(case, 'nan', f'_ = label_binarize({d.input(np.array([1.0, np.nan]))}, classes={d.input(np.array([1.0]))})',
             lambda: P.label_binarize(np.array([1.0, np.nan]), classes=[1.0]), 'Input y contains NaN')

# ---------------------------------------------------------------------------
# MultiLabelBinarizer


def mojo_rows(rows, kind):
    if kind == 'str':
        inner = ', '.join('[' + ', '.join(json.dumps(v) for v in row) + ']' for row in rows)
        return f'List[List[String]] = [{inner}]'
    inner = ', '.join('[' + ', '.join(f'Int64({v})' for v in row) + ']' for row in rows)
    return f'List[List[Int64]] = [{inner}]'


def mojo_classes(classes, kind):
    if kind == 'str':
        return 'List[String] = [' + ', '.join(json.dumps(v) for v in classes) + ']'
    return 'List[Int64] = [' + ', '.join(f'Int64({v})' for v in classes) + ']'


MLB = [
    ('str_auto', [['sci-fi', 'thriller'], ['comedy'], [], ['comedy', 'comedy', 'drama']], 'str', None,
     [['drama', 'noir'], ['thriller']]),
    ('str_given', [['sci-fi', 'thriller'], ['comedy'], [], ['noir']], 'str', ['thriller', 'comedy', 'western'],
     [['comedy', 'sci-fi'], ['western']]),
    ('int_auto', [[3, 1], [2], [1, 1, 0], []], 'int64', None, [[0, 7], [3]]),
    ('int_given', [[3, 1], [2], [1, 1, 0]], 'int64', [3, 2, 9, 1], [[9, 1, 4]]),
]
for key, y, kind, classes, test in MLB:
    case = f'mlb_{key}'
    ref = P.MultiLabelBinarizer(classes=classes)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Y = ref.fit_transform(y)
        Y_fit = P.MultiLabelBinarizer(classes=classes).fit(y).transform(y)
        Y_test = ref.transform(test)
    line(case, f'var y: {mojo_rows(y, kind)}')
    line(case, f'var test: {mojo_rows(test, kind)}')
    if classes is None:
        line(case, f'var model = MultiLabelBinarizer[{MOJO[kind]}]()')
    else:
        line(case, f'var classes: {mojo_classes(classes, kind)}')
        line(case, f'var model = MultiLabelBinarizer[{MOJO[kind]}](classes=classes^)')
    line(case, 'var Y = model.fit_transform(y)')
    line(case, 'emit("Y", Y)')
    line(case, 'emit_labels("classes", model.classes_)')
    line(case, 'emit_rows("inverse", model.inverse_transform(Y))')
    line(case, 'emit("test", model.transform(test))')
    line(case, 'var fresh = clone(model)')
    line(case, 'fresh.fit(y)')
    line(case, 'emit("fit_then_transform", fresh.transform(y))')
    expect(case, 'Y', Y)
    expect(case, 'classes', as_expected(ref.classes_, kind))
    expect(case, 'test', Y_test)
    expect(case, 'fit_then_transform', Y_fit)
    inverse = ref.inverse_transform(Y)
    expect(case, 'inverse_n', len(inverse))
    for i, row in enumerate(inverse):
        expect(case, f'inverse_{i}', [str(v) for v in row] if kind == 'str' else np.asarray(row, dtype=np.int64))

case = 'mlb_errors'
line(case, 'var model = MultiLabelBinarizer[NumericLabels[DType.int64]]()')
line(case, 'var y: List[List[Int64]] = [[Int64(1), Int64(1), Int64(1), Int64(0)]]')
line(case, 'emit("Y", model.fit_transform(y))')
expect(case, 'Y', P.MultiLabelBinarizer().fit_transform([(1, 1, 1, 0)]))
fitted = P.MultiLabelBinarizer().fit([(1, 1, 1, 0)])
expect_error(case, 'not_binary', f'_ = model.inverse_transform({ii([[1, 3]])})',
             lambda: fitted.inverse_transform(np.array([[1, 3]])), 'Expected only 0s and 1s in label indicator. Also got [3]')
expect_error(case, 'narrow', f'_ = model.inverse_transform({ii([[1]])})',
             lambda: fitted.inverse_transform(np.array([[1]])), 'Expected indicator for 2 classes, but got 1')
expect_error(case, 'wide', f'_ = model.inverse_transform({ii([[1, 1, 1]])})',
             lambda: fitted.inverse_transform(np.array([[1, 1, 1]])), 'Expected indicator for 2 classes, but got 3')
line(case, 'var duplicate_classes: List[String] = ["a", "b", "a"]')
line(case, 'var duplicate = MultiLabelBinarizer(classes=duplicate_classes^)')
line(case, 'var words: List[List[String]] = [["a"]]')
expect_error(case, 'duplicates', 'duplicate.fit(words)',
             lambda: P.MultiLabelBinarizer(classes=['a', 'b', 'a']).fit([['a']]), 'contains duplicate classes')
line(case, 'var unfitted = MultiLabelBinarizer()')
expect_error(case, 'unfitted', '_ = unfitted.transform(words)',
             lambda: P.MultiLabelBinarizer().transform([['a']]), 'not fitted')
line(case, 'var sparse = MultiLabelBinarizer(sparse_output=True)')
expect_error(case, 'sparse_deferred', 'sparse.fit(words)', None, 'sparse output is deferred')

# ---------------------------------------------------------------------------

for name, body in cases.items():
    d.case(name, '\n'.join(body))
out = d.run(imports='from moxie.base import clone\n'
                    'from moxie.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer, label_binarize')
for (case, name), value in expected.items():
    got = out.get(case, {})
    label = f'{case}.{name}'
    if 'error' in got:
        d.close(label, got['error'], np.zeros(0))
    elif isinstance(value, list):
        d.equal(label, got.get(name), value)
    elif isinstance(value, (int, np.integer)):
        d.equal(label, got.get(name), int(value))
    else:
        value = np.asarray(value)
        # Float multilabel classes_ and inverse transforms keep the label dtype natively (sklearn: int64).
        dtype = value.dtype if not (case.startswith('lb_multilabel_float64') and name in ('classes', 'inverse', 'scores')) else None
        d.close(label, got.get(name, 'missing'), value, rtol=0.0, dtype=dtype)
for (case, name), needle in errors.items():
    d.raises(f'{case}.{name}', out.get(case, {}).get(name), needle)
d.finish()
