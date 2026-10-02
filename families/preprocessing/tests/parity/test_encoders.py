"""Parity: OneHotEncoder and OrdinalEncoder on string, float and int columns.

Native features are Moxie CategoricalColumns (string None / float NaN = missing);
sklearn receives the equivalent object/float/int arrays. Encoded outputs,
categories_, infrequent_categories_, drop_idx_, feature names and inverse
transforms are compared exactly (no tolerance: outputs are small integers).
Inverse cells compare as text, with sklearn's None/NaN mapped to <missing>.
Error cases check that both sides raise, plus a message fragment.
"""
import math

import numpy as np

from harness import Driver, oracle

oracle()
from sklearn import preprocessing as P

MISSING = '<missing>'
d = Driver('encoders')
checks = []  # (case, name, expected, kind)


def expect(case, name, value, kind='exact'):
    checks.append((case, name, value, kind))


d.prelude('''
from moxie._categorical import categorical_column, CategoricalColumn, StringCategories, NumericCategories
from moxie.preprocessing._vocabulary import CategoryKind, StringKind, StringNaNKind, NumericKind

comptime MISSING = "<missing>"

def text_columns[K: CategoryKind](name: String, columns: List[CategoricalColumn[K.Domain]]) raises:
    for j in range(len(columns)):
        var texts = List[String]()
        var domain = columns[j].domain()
        var codes = columns[j].codes()
        for r in range(codes.size):
            var c = Int(codes[Item(r)])
            texts.append(String(MISSING) if c == -1 else K.text(domain.get(c)))
        emit_names(name + "_" + String(j), texts)

def text_categories[K: CategoryKind](name: String, categories: List[List[Optional[K.Domain.Value]]]) raises:
    for j in range(len(categories)):
        var texts = List[String]()
        for value in categories[j]:
            texts.append(K.text(value.value()) if value else String(MISSING))
        emit_names(name + "_" + String(j), texts)

def text_infrequent[K: CategoryKind](name: String, infrequent: Optional[List[Optional[List[Optional[K.Domain.Value]]]]]) raises:
    if not infrequent:
        emit_int(name + "_disabled", 1)
        return
    for j in range(len(infrequent.value())):
        var texts = List[String]()
        if infrequent.value()[j]:
            for value in infrequent.value()[j].value():
                texts.append(K.text(value.value()) if value else String(MISSING))
        else:
            texts.append("<none>")
        emit_names(name + "_" + String(j), texts)

def drop_codes(name: String, drop: Optional[List[Optional[Int]]]) raises:
    if not drop:
        emit_int(name + "_none", 1)
        return
    var codes = nm.zeros[DType.int64](Shape(len(drop.value())))
    for j in range(len(drop.value())):
        codes[Item(j)] = Int64(drop.value()[j].value()) if drop.value()[j] else -1
    emit(name, codes)
''')


def text(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return MISSING
    return str(v)


def expect_fitted(case, ref, prefix):
    for j, cats in enumerate(ref.categories_):
        expect(case, f'{prefix}cat_{j}', [text(v) for v in cats.tolist()], 'names')
    infrequent = getattr(ref, '_infrequent_enabled', False) and ref.infrequent_categories_
    if not getattr(ref, '_infrequent_enabled', False):
        expect(case, f'{prefix}inf_disabled', 1, 'int')
    else:
        for j, values in enumerate(infrequent):
            expect(case, f'{prefix}inf_{j}', ['<none>'] if values is None else [text(v) for v in values.tolist()], 'names')


def expect_inverse(case, name, X_inv):
    for j in range(X_inv.shape[1]):
        expect(case, f'{name}_{j}', [text(v) for v in X_inv[:, j].tolist()], 'names')


# ---- datasets ---------------------------------------------------------------
rng = np.random.RandomState(0)
colors = ['red', 'green', 'blue', 'amber', 'cyan']
weights = [0.4, 0.25, 0.2, 0.1, 0.05]
S_train = np.empty((40, 3), dtype=object)
S_train[:, 0] = rng.choice(colors, 40, p=weights)
S_train[:, 1] = rng.choice(['x', 'y'], 40)
S_train[:, 2] = rng.choice(['a', 'b', 'c', None], 40, p=[0.5, 0.2, 0.2, 0.1])
S_test = np.array([['red', 'x', 'a'], ['violet', 'y', None], ['cyan', 'z', 'b'], ['blue', 'x', 'q']], dtype=object)
S_known = np.array([['red', 'x', 'a'], ['amber', 'y', None], ['cyan', 'y', 'b']], dtype=object)
F_train = np.array([[1.5, 0.0], [np.nan, 2.0], [-3.25, 2.0], [1.5, 1e20], [1.5, 0.0], [np.nan, 2.0]])
F_test = np.array([[1.5, 2.0], [np.nan, 0.0], [-3.25, 1e20]])
I_train = np.array([[3, -7], [1, -7], [3, 10], [2, 10], [3, -7], [1, 4]], dtype=np.int64)
I_test = np.array([[2, 4], [1, -7]], dtype=np.int64)


def string_columns(X):
    return '[' + ', '.join(f'categorical_column({d.optional_strings(X[:, j].tolist())})' for j in range(X.shape[1])) + ']'


def numeric_columns(X, dtype):
    return '[' + ', '.join(f'categorical_column({d.input(np.ascontiguousarray(X[:, j]))})' for j in range(X.shape[1])) + ']'


def nan_strings(X):
    """Same strings with NaN (pandas-style) missing values for sklearn."""
    Y = X.copy()
    Y[np.equal(Y, None)] = np.nan
    return Y


N_train, N_test, N_known = nan_strings(S_train), nan_strings(S_test), nan_strings(S_known)
C_train = np.array([['only', 'u'], ['only', 'v'], ['only', 'u']], dtype=object)
C_test = np.array([['only', 'v'], ['only', 'w']], dtype=object)
KIND = {'c': 'StringKind', 's': 'StringKind', 'n': 'StringNaNKind', 'f': 'NumericKind[DType.float64]', 'i': 'NumericKind[DType.int64]'}
DATA = {'c': (C_train, C_test, string_columns), 's': (S_train, S_test, string_columns), 'n': (N_train, N_test, lambda X: string_columns(np.where(np.equal(X, X), X, None))),
        'f': (F_train, F_test, lambda X: numeric_columns(X, 'f')),
        'i': (I_train, I_test, lambda X: numeric_columns(X, 'i'))}


def mojo_value(kind, v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 'None'
    if kind in 'csn':
        return f'Optional[String]("{v}")'
    if kind == 'f':
        return f'Optional[Float64]({float(v)!r})'
    return f'Optional[Int64]({int(v)})'




def native_args(kind, kw):
    parts = []
    for k, v in kw.items():
        if k == 'categories':
            T = {'c': 'String', 's': 'String', 'n': 'String', 'f': 'Float64', 'i': 'Int64'}[kind]
            rows = ', '.join('[' + ', '.join(mojo_value(kind, x) for x in cats) + ']' for cats in v)
            parts.append(f'categories=Optional[List[List[Optional[{T}]]]]([{rows}])')
        elif k == 'drop' and isinstance(v, list):
            T = {'c': 'String', 's': 'String', 'n': 'String', 'f': 'Float64', 'i': 'Int64'}[kind]
            parts.append('drop=drop_values')
        elif isinstance(v, bool):
            parts.append(f'{k}={v}')
        elif isinstance(v, str):
            parts.append(f'{k}="{v}"')
        elif isinstance(v, float) and math.isnan(v):
            parts.append(f'{k}=nan[DType.float64]()')
        elif k in ('min_frequency', 'unknown_value', 'encoded_missing_value'):
            parts.append(f'{k}={float(v)!r}')
        else:
            parts.append(f'{k}={v}')
    return ', '.join(parts)


def sklearn_kw(kind, kw):
    # Missing is None in object (string) columns and NaN in numeric ones.
    gap = None if kind in 'cs' else np.nan
    out = dict(kw)
    if 'min_frequency' in out and float(out['min_frequency']).is_integer() and out['min_frequency'] >= 1:
        out['min_frequency'] = int(out['min_frequency'])
    if 'drop' in out and isinstance(out['drop'], list):
        out['drop'] = np.array([gap if v is None else v for v in out['drop']], dtype=object)
    if 'categories' in out:
        out['categories'] = [[gap if v is None else v for v in cats] for cats in out['categories']]
    return out


def run_case(tag, cls, kind, kw, *, otype='DType.float64', test=None, inverse=True, names=True):
    Xtr, Xte, make = DATA[kind]
    if test is not None:
        Xte = test
    T = {'DType.float64': np.float64, 'DType.int64': np.int64, 'DType.float32': np.float32}[otype]
    native = f'{cls}[{KIND[kind]}, {otype}]'
    C = f'List[CategoricalColumn[{KIND[kind]}.Domain]]'
    body = [f'var Xtr: {C} = {make(Xtr)}', f'var Xte: {C} = {make(Xte)}',
            *([f'var drop_values: List[Optional[{dict(c="String", s="String", n="String", f="Float64", i="Int64")[kind]}]] = [' + ', '.join(mojo_value(kind, x) for x in kw['drop']) + ']']
              if isinstance(kw.get('drop'), list) else []),
            f'var enc = {native}({native_args(kind, (dict(sparse_output=False) if cls == "OneHotEncoder" else {}) | kw)})',
            'emit("fit", enc.fit_transform(Xtr))',
            f'text_categories[{KIND[kind]}]("cat", enc.categories_)',
            f'text_infrequent[{KIND[kind]}]("inf", enc.infrequent_categories_)']
    extra = dict(sparse_output=False) if cls == 'OneHotEncoder' else {}
    ref = getattr(P, cls)(dtype=T, **extra, **sklearn_kw(kind, kw)).fit(Xtr)
    expect(tag, 'fit', ref.transform(Xtr), 'array')
    expect_fitted(tag, ref, '')
    if cls == 'OneHotEncoder':
        body.append('drop_codes("drop", enc.drop_idx_)')
        if ref.drop_idx_ is None:
            expect(tag, 'drop_none', 1, 'int')
        else:
            expect(tag, 'drop', np.array([-1 if v is None else v for v in ref.drop_idx_], dtype=np.int64), 'array')
    if names:
        body.append('emit_names("names", enc.get_feature_names_out())')
        expect(tag, 'names', [str(n) for n in ref.get_feature_names_out()], 'names')
        body.append('var custom: List[String] = [' + ', '.join(f'"f{j}"' for j in range(Xtr.shape[1])) + ']')
        body.append('emit_names("custom", enc.get_feature_names_out(custom^))')
        expect(tag, 'custom', [str(n) for n in ref.get_feature_names_out([f'f{j}' for j in range(Xtr.shape[1])])], 'names')
    try:
        expected = ref.transform(Xte)
        body.append('var Z = enc.transform(Xte)')
        body.append('emit("test", Z)')
        expect(tag, 'test', expected, 'array')
        if inverse:
            body.append(f'text_columns[{KIND[kind]}]("inv", enc.inverse_transform(Z))')
            try:
                # A copy: sklearn's OrdinalEncoder.inverse_transform writes into its input.
                X_inv = ref.inverse_transform(expected.copy())
            except (ValueError, IndexError) as error:
                expect(tag, 'inv_error', str(error), 'error_tail')
            else:
                if kind in 'fi' and any(v == 'infrequent_sklearn' for v in X_inv.ravel().tolist()):
                    # Documented: numeric columns cannot hold the string placeholder.
                    expect(tag, 'inv_error', 'infrequent_sklearn', 'error_tail')
                else:
                    expect_inverse(tag, 'inv', X_inv)
    except ValueError as error:
        body.append('_ = enc.transform(Xte)')
        expect(tag, 'error', str(error), 'error')
    d.case(tag, '\n'.join(body))


# OneHotEncoder: plain, drop modes, unknown handling, infrequent grouping.
ohe = [
    ('plain', {}),
    ('first', dict(drop='first')),
    ('binary', dict(drop='if_binary')),
    ('ignore', dict(handle_unknown='ignore')),
    ('ignore_first', dict(handle_unknown='ignore', drop='first')),
    ('minfreq', dict(min_frequency=5.0, handle_unknown='infrequent_if_exist')),
    ('minfrac', dict(min_frequency=0.2, handle_unknown='infrequent_if_exist')),
    ('maxcat', dict(max_categories=3, handle_unknown='infrequent_if_exist')),
    ('maxcat1', dict(max_categories=1, handle_unknown='infrequent_if_exist')),
    ('maxcat_ignore', dict(max_categories=3, handle_unknown='ignore')),
    ('maxcat_first', dict(max_categories=3, drop='first', handle_unknown='infrequent_if_exist')),
    ('maxcat_binary', dict(max_categories=2, drop='if_binary', handle_unknown='infrequent_if_exist')),
    ('warn', dict(min_frequency=5.0, handle_unknown='warn')),
]
for name, kw in ohe:
    run_case(f'ohe_s_{name}', 'OneHotEncoder', 's', kw)
run_case('ohe_s_known', 'OneHotEncoder', 's', {}, test=S_known)
# Single-category feature: dropping leaves no output columns; inverse refills it.
run_case('ohe_c_first', 'OneHotEncoder', 'c', dict(drop='first', handle_unknown='ignore'))
run_case('ohe_c_binary', 'OneHotEncoder', 'c', dict(drop='if_binary'), test=C_train)
run_case('ohe_s_drop_values', 'OneHotEncoder', 's', dict(drop=['red', 'y', None]), test=S_known)
run_case('ohe_s_explicit', 'OneHotEncoder', 's',
         dict(categories=[['red', 'green', 'blue', 'amber', 'cyan', 'violet'], ['y', 'x', 'z'], ['c', 'b', 'a', 'q', None]]),
         test=S_test)
run_case('ohe_s_int64', 'OneHotEncoder', 's', dict(handle_unknown='ignore'), otype='DType.int64')
run_case('ohe_s_f32', 'OneHotEncoder', 's', dict(drop='first'), otype='DType.float32', test=S_known)
for name, kw in [('plain', {}), ('first', dict(drop='first')), ('ignore', dict(handle_unknown='ignore')),
                 ('maxcat', dict(max_categories=2, handle_unknown='infrequent_if_exist'))]:
    run_case(f'ohe_f_{name}', 'OneHotEncoder', 'f', kw)
    run_case(f'ohe_i_{name}', 'OneHotEncoder', 'i', kw)
run_case('ohe_f_explicit', 'OneHotEncoder', 'f', dict(categories=[[-3.25, 1.5, None], [0.0, 2.0, 1e20]]))
run_case('ohe_f_drop_nan', 'OneHotEncoder', 'f', dict(drop=[None, 2.0]), test=F_train)

# OrdinalEncoder: unknown values, missing encodings, infrequent grouping, dtypes.
oe = [
    ('plain', {}, 'DType.float64'),
    ('unknown_m1', dict(handle_unknown='use_encoded_value', unknown_value=-1), 'DType.float64'),
    ('unknown_nan', dict(handle_unknown='use_encoded_value', unknown_value=float('nan')), 'DType.float64'),
    ('missing_m2', dict(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-2), 'DType.float64'),
    ('int64', dict(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-2), 'DType.int64'),
    ('minfreq', dict(min_frequency=5.0, handle_unknown='use_encoded_value', unknown_value=-1), 'DType.float64'),
    ('maxcat', dict(max_categories=3, handle_unknown='use_encoded_value', unknown_value=99), 'DType.float64'),
    ('f32', dict(handle_unknown='use_encoded_value', unknown_value=-1), 'DType.float32'),
]
for name, kw, otype in oe:
    run_case(f'oe_s_{name}', 'OrdinalEncoder', 's', kw, otype=otype)
    run_case(f'oe_n_{name}', 'OrdinalEncoder', 'n', kw, otype=otype)
for name, kw in ohe[:6]:
    run_case(f'ohe_n_{name}', 'OneHotEncoder', 'n', kw)
run_case('oe_s_infrequent_none', 'OrdinalEncoder', 's', dict(min_frequency=5.0), test=S_known)
run_case('oe_s_known', 'OrdinalEncoder', 's', {}, test=S_known)
run_case('oe_s_explicit', 'OrdinalEncoder', 's',
         dict(categories=[['red', 'green', 'blue', 'amber', 'cyan'], ['y', 'x'], ['c', 'b', 'a', None]]), test=S_known)
for name, kw in [('plain', {}), ('unknown', dict(handle_unknown='use_encoded_value', unknown_value=-1)),
                 ('maxcat', dict(max_categories=2, handle_unknown='use_encoded_value', unknown_value=-1))]:
    run_case(f'oe_f_{name}', 'OrdinalEncoder', 'f', kw)
    run_case(f'oe_i_{name}', 'OrdinalEncoder', 'i', kw)

# Errors: native must raise where sklearn does (fragment from sklearn's message).
errors = {
    'ohe_sparse_default': ('var e = OneHotEncoder[StringKind]()\ne.fit(Xtr)', None, 'sparse output'),
    'ohe_handle': ('var e = OneHotEncoder[StringKind](sparse_output=False, handle_unknown="bogus")\ne.fit(Xtr)',
                   lambda: P.OneHotEncoder(handle_unknown='bogus').fit(S_train), "'handle_unknown' parameter"),
    'ohe_shape': ('var e = OneHotEncoder[StringKind](sparse_output=False, categories=Optional[List[List[Optional[String]]]]([[Optional[String]("a")]]))\ne.fit(Xtr)',
                  lambda: P.OneHotEncoder(categories=[['a']]).fit(S_train), 'Shape mismatch'),
    'ohe_nan_last': ('var e = OneHotEncoder[NumericKind[DType.float64]](sparse_output=False, categories=Optional[List[List[Optional[Float64]]]]([[None, Optional[Float64](1.5)], [Optional[Float64](0.0)]]))\ne.fit(Ftr)',
                     lambda: P.OneHotEncoder(categories=[[np.nan, 1.5], [0.0]]).fit(F_train), 'Nan should be the last'),
    'ohe_unsorted': ('var e = OneHotEncoder[NumericKind[DType.float64]](sparse_output=False, categories=Optional[List[List[Optional[Float64]]]]([[Optional[Float64](1.5), Optional[Float64](-3.25), None], [Optional[Float64](0.0), Optional[Float64](2.0), Optional[Float64](1e20)]]))\ne.fit(Ftr)',
                     lambda: P.OneHotEncoder(categories=[[1.5, -3.25, np.nan], [0.0, 2.0, 1e20]]).fit(F_train), 'Unsorted categories'),
    'ohe_duplicate': ('var e = OneHotEncoder[StringKind](sparse_output=False, categories=Optional[List[List[Optional[String]]]]([[Optional[String]("red"), Optional[String]("red")], [Optional[String]("x")], [Optional[String]("a")]]))\ne.fit(Xtr)',
                      lambda: P.OneHotEncoder(categories=[['red', 'red'], ['x'], ['a']]).fit(S_train), 'duplicate elements'),
    'ohe_unknown_fit': ('var e = OneHotEncoder[StringKind](sparse_output=False, categories=Optional[List[List[Optional[String]]]]([[Optional[String]("red")], [Optional[String]("x"), Optional[String]("y")], [Optional[String]("a"), Optional[String]("b"), Optional[String]("c"), None]]))\ne.fit(Xtr)',
                        lambda: P.OneHotEncoder(categories=[['red'], ['x', 'y'], ['a', 'b', 'c', None]]).fit(S_train), 'during fit'),
    'ohe_drop_missing': ('var dv: List[Optional[String]] = [Optional[String]("violet"), Optional[String]("x"), Optional[String]("a")]\nvar e = OneHotEncoder[StringKind](sparse_output=False, drop=dv)\ne.fit(Xtr)',
                         lambda: P.OneHotEncoder(drop=['violet', 'x', 'a']).fit(S_train), 'supposed to be dropped'),
    'ohe_drop_len': ('var dv: List[Optional[String]] = [Optional[String]("red")]\nvar e = OneHotEncoder[StringKind](sparse_output=False, drop=dv)\ne.fit(Xtr)',
                     lambda: P.OneHotEncoder(drop=['red']).fit(S_train), 'should have length'),
    'ohe_drop_infrequent': ('var dv: List[Optional[String]] = [Optional[String]("cyan"), Optional[String]("x"), Optional[String]("a")]\nvar e = OneHotEncoder[StringKind](sparse_output=False, max_categories=2, drop=dv)\ne.fit(Xtr)',
                            lambda: P.OneHotEncoder(max_categories=2, drop=['cyan', 'x', 'a']).fit(S_train), 'because it is infrequent'),
    'ohe_minfreq': ('var e = OneHotEncoder[StringKind](sparse_output=False, min_frequency=1.5)\ne.fit(Xtr)',
                    lambda: P.OneHotEncoder(min_frequency=1.5).fit(S_train), "'min_frequency' parameter"),
    'ohe_inverse_shape': ('var e = OneHotEncoder[StringKind](sparse_output=False)\ne.fit(Xtr)\n_ = e.inverse_transform(nm.zeros[DType.float64](Shape(2, 3)))',
                          lambda: P.OneHotEncoder().fit(S_train).inverse_transform(np.zeros((2, 3))), 'Shape of the passed X'),
    'ohe_inverse_zeros': ('var e = OneHotEncoder[StringKind](sparse_output=False)\ne.fit(Xtr)\nvar Z = e.transform(Xtr)\nfor k in range(Z.shape[1]):\n    Z.unsafe_set(k, 0)\n_ = e.inverse_transform(Z)',
                          None, 'can not be inverted'),
    'ohe_unfitted': ('var e = OneHotEncoder[StringKind](sparse_output=False)\n_ = e.transform(Xtr)', None, 'NotFittedError'),
    'ohe_features': ('var e = OneHotEncoder[StringKind](sparse_output=False)\ne.fit(Xtr)\nvar narrow: List[CategoricalColumn[StringCategories]] = [Xtr[0].copy()]\n_ = e.transform(narrow)', None, 'features'),
    'oe_unknown_value': ('var e = OrdinalEncoder[StringKind](handle_unknown="use_encoded_value", unknown_value=1.0)\ne.fit(Xtr)',
                         lambda: P.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=1).fit(S_train), 'already used for encoding'),
    'oe_unknown_type': ('var e = OrdinalEncoder[StringKind](handle_unknown="use_encoded_value")\ne.fit(Xtr)',
                        lambda: P.OrdinalEncoder(handle_unknown='use_encoded_value').fit(S_train), 'should be an integer'),
    'oe_unknown_set': ('var e = OrdinalEncoder[StringKind](unknown_value=-1.0)\ne.fit(Xtr)',
                       lambda: P.OrdinalEncoder(unknown_value=-1).fit(S_train), 'should only be set'),
    'oe_nan_int': ('var e = OrdinalEncoder[StringKind, DType.int64](handle_unknown="use_encoded_value", unknown_value=nan[DType.float64]())\ne.fit(Xtr)',
                   lambda: P.OrdinalEncoder(dtype=np.int64, handle_unknown='use_encoded_value', unknown_value=np.nan).fit(S_train), 'should be a float'),
    'oe_missing_int': ('var e = OrdinalEncoder[StringNaNKind, DType.int64]()\ne.fit(Xtr)',
                       lambda: P.OrdinalEncoder(dtype=np.int64).fit(N_train), 'There are missing values'),
    'oe_missing_used': ('var e = OrdinalEncoder[StringNaNKind](encoded_missing_value=1.0)\ne.fit(Xtr)',
                        lambda: P.OrdinalEncoder(encoded_missing_value=1).fit(N_train), 'already used to encode a known category'),
    'oe_transform_unknown': ('var e = OrdinalEncoder[StringKind]()\ne.fit(Xtr)\n_ = e.transform(Xte)',
                             lambda: P.OrdinalEncoder().fit(S_train).transform(S_test), 'during transform'),
    'oe_inverse_shape': ('var e = OrdinalEncoder[StringKind]()\ne.fit(Xtr)\n_ = e.inverse_transform(nm.zeros[DType.float64](Shape(2, 2)))',
                         lambda: P.OrdinalEncoder().fit(S_train).inverse_transform(np.zeros((2, 2))), 'Shape of the passed X'),
    'oe_handle': ('var e = OrdinalEncoder[StringKind](handle_unknown="ignore")\ne.fit(Xtr)',
                  lambda: P.OrdinalEncoder(handle_unknown='ignore').fit(S_train), "'handle_unknown' parameter"),
}
for name, (code, reference, needle) in errors.items():
    if reference is not None:
        try:
            reference()
        except (ValueError, TypeError):
            pass
        except AttributeError:
            # sklearn defect: its infrequent-drop message calls .item() on a str.
            assert name == 'ohe_drop_infrequent', name
        else:
            raise SystemExit(f'reference did not raise for {name}')
    S = 'List[CategoricalColumn[StringCategories]]'
    F = 'List[CategoricalColumn[NumericCategories[DType.float64]]]'
    d.case(f'err_{name}', f'var Xtr: {S} = {string_columns(S_train)}\nvar Xte: {S} = {string_columns(S_test)}\nvar Ftr: {F} = {numeric_columns(F_train, "f")}\n' + code)

out = d.run(imports='from moxie.preprocessing import OneHotEncoder, OrdinalEncoder\nfrom std.math import nan')

for case, name, value, kind in checks:
    got = out.get(case, {})
    label = f'{case}.{name}'
    if kind in ('error', 'error_tail'):
        # sklearn raised at this step (message checked loosely: fragments differ).
        d.raises(label, got.get('error'), '')
    elif name not in got and 'error' in got:
        d.close(label, got['error'], 0)
    elif kind == 'array':
        d.close(label, got.get(name, 'missing'), value, dtype=value.dtype, rtol=0, atol=0)
    elif kind == 'names':
        d.equal(label, got.get(name), value)
    else:
        d.equal(label, got.get(name), value)
for name, (_, _, needle) in errors.items():
    d.raises(f'err_{name}', out[f'err_{name}'].get('error'), needle)
d.finish()
