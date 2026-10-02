"""Parity: TargetEncoder and its KFold / StratifiedKFold fold stand-in.

Native features are Moxie CategoricalColumns (string None / float NaN =
missing); sklearn receives the equivalent object/float/int arrays. Targets are
NumericLabels (Float64, Float32, Int64, including a one-column matrix) or
StringLabels. Cross-fitted fit_transform outputs, transform outputs,
encodings_, target_mean_, categories_, classes_, target_type_ and feature names
are compared exactly (rtol=0): the native code reproduces sklearn's fold
indices, NumPy's pairwise mean/variance and the pyx accumulation order.
Fold vectors from the native splitter stand-ins are compared directly with
sklearn's KFold / StratifiedKFold test indices, including random_state=None
with both global streams seeded alike. Error cases check that both sides raise.
"""
import math
import warnings

import numpy as np

from harness import Driver, oracle

oracle()
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import TargetEncoder

warnings.simplefilter('ignore')
MISSING = '<missing>'
d = Driver('target_encoder')
checks = []  # (case, name, expected, kind)


def expect(case, name, value, kind='array'):
    checks.append((case, name, value, kind))


d.prelude('''
from moxie._categorical import categorical_column, CategoricalColumn, StringCategories, NumericCategories
from moxie._random import seed_global_random_state, RandomState
from moxie.preprocessing._vocabulary import CategoryKind, StringKind, StringNaNKind, NumericKind
from moxie.preprocessing._label import LabelKind, StringLabels, NumericLabels
from moxie.preprocessing._target_encoder import kfold_test_folds, stratified_kfold_test_folds

comptime MISSING = "<missing>"

def emit_list(name: String, values: List[Int]) raises:
    var array = nm.zeros[DType.int64](Shape(len(values)))
    for i in range(len(values)):
        array[Item(i)] = Int64(values[i])
    emit(name, array)

def text_categories[K: CategoryKind](name: String, categories: List[List[Optional[K.Domain.Value]]]) raises:
    for j in range(len(categories)):
        var texts = List[String]()
        for value in categories[j]:
            texts.append(K.text(value.value()) if value else String(MISSING))
        emit_names(name + "_" + String(j), texts)

def text_labels[Y: LabelKind](name: String, classes: Y.Labels) raises:
    var texts = List[String]()
    for i in range(Y.rows(classes)):
        texts.append(Y.python_str(Y.get(classes, i)))
    emit_names(name, texts)
''')


# ---- fold stand-ins ------------------------------------------------------------
def sk_folds(splitter, n, y=None):
    folds = np.full(n, -1, dtype=np.int64)
    for f, (train, test) in enumerate(splitter.split(np.zeros((n, 1)), y)):
        assert np.all(folds[test] == -1)
        folds[test] = f
        # train is the ascending complement of test, so a fold vector is exact.
        assert np.array_equal(train, np.setdiff1d(np.arange(n), test)) and np.all(np.diff(test) > 0)
    return folds


fold_cases = 0
for n, k in [(10, 3), (17, 5), (5, 5), (101, 7), (2, 2)]:
    for shuffle, seed in [(False, None), (True, 0), (True, 42), (True, 4294967295)]:
        fold_cases += 1
        tag = f'kfold_{fold_cases}'
        d.case(tag, f'emit_list("f", kfold_test_folds({n}, {k}, {shuffle}, {"None" if seed is None else seed}))')
        expect(tag, 'f', sk_folds(KFold(k, shuffle=shuffle, random_state=seed), n))
rng = np.random.RandomState(7)
label_sets = {
    'balanced': np.repeat([3, 1, 2], 10),
    'imbalanced': rng.choice([5, -1, 9, 0], 53, p=[0.5, 0.3, 0.15, 0.05]),
    'binary': rng.choice([1, 0], 31),
    'interleaved': np.tile([2, 0, 1, 1], 9),
    'singleton': np.array([0, 0, 0, 0, 1, 1, 1, 2]),
}
for name, labels in label_sets.items():
    for k in (2, 3, 5):
        for shuffle, seed in [(False, None), (True, 0), (True, 123)]:
            if np.all(k > np.bincount(np.unique(labels, return_inverse=True)[1])):
                continue
            tag = f'strat_{name}_{k}_{int(shuffle)}_{seed}'
            vec = '[' + ', '.join(str(int(v)) for v in labels) + ']'
            d.case(tag, f'var labels: List[Int] = {vec}\nemit_list("f", stratified_kfold_test_folds(labels, {k}, {shuffle}, {"None" if seed is None else seed}))')
            expect(tag, 'f', sk_folds(StratifiedKFold(k, shuffle=shuffle, random_state=seed), len(labels), labels))
# random_state=None draws from the process stream: seed both globals alike.
np.random.seed(11)
expected_none_k = sk_folds(KFold(4, shuffle=True), 23)
expected_none_s = sk_folds(StratifiedKFold(3, shuffle=True), 30, label_sets['balanced'])
d.case('kfold_global', 'seed_global_random_state(11)\nemit_list("k", kfold_test_folds(23, 4, True, None))\n'
       'var labels: List[Int] = [' + ', '.join(str(int(v)) for v in label_sets['balanced']) + ']\n'
       'emit_list("s", stratified_kfold_test_folds(labels, 3, True, None))')
expect('kfold_global', 'k', expected_none_k)
expect('kfold_global', 's', expected_none_s)
for tag, code, needle in [
    ('fold_err_samples', 'emit_list("f", kfold_test_folds(4, 5, False, None))', 'n_splits=5 greater'),
    ('fold_err_seed', 'emit_list("f", kfold_test_folds(9, 3, False, 0))', 'no effect since shuffle is False'),
    ('fold_err_members', 'var labels: List[Int] = [0, 0, 1, 1, 2]\nemit_list("f", stratified_kfold_test_folds(labels, 3, False, None))', 'number of members in each class'),
]:
    d.case(tag, code)
    expect(tag, 'error', needle, 'error')


# ---- datasets -------------------------------------------------------------------
n = 60
S = np.empty((n, 3), dtype=object)
S[:, 0] = rng.choice(['red', 'green', 'blue', 'amber', 'cyan'], n, p=[0.4, 0.25, 0.2, 0.1, 0.05])
S[:, 1] = rng.choice(['x', 'y'], n)
S[:, 2] = rng.choice(['a', 'b', 'c', None], n, p=[0.5, 0.2, 0.2, 0.1])
S_test = np.array([['red', 'x', 'a'], ['violet', 'y', None], ['cyan', 'z', 'b'], ['blue', 'x', 'q']], dtype=object)
F = np.round(rng.randn(n, 2), 0) * 1.5 + 0.0  # no negative zeros (category sign is not compared)
F[rng.rand(n, 2) < 0.1] = np.nan
F_test = np.array([[1.5, np.nan], [np.nan, 0.0], [99.0, -1.5]])
I = rng.randint(-2, 3, size=(n, 2)).astype(np.int64)
I_test = np.array([[0, 7], [2, -2]], dtype=np.int64)

y_cont = rng.randn(n) * 10 + 3
y_f32 = (rng.randn(n) * 7 + 1).astype(np.float32)
y_int = rng.randint(0, 100, n).astype(np.int64)
y_bin = rng.randint(0, 2, n).astype(np.int64)
y_bin_s = np.where(rng.rand(n) < 0.3, 'yes', 'no')
y_multi = rng.choice([4, 0, 2, 9], n).astype(np.int64)
y_multi_f = rng.choice([1.0, 2.0, 4.0], n)
y_multi_s = rng.choice(['cat', 'dog', 'eel'], n, p=[0.5, 0.3, 0.2])
y_one = np.full(n, 3, dtype=np.int64)

KIND = {'s': 'StringKind', 'n': 'StringNaNKind', 'f': 'NumericKind[DType.float64]', 'i': 'NumericKind[DType.int64]'}
DOMAIN_T = {'s': 'String', 'n': 'String', 'f': 'Float64', 'i': 'Int64'}


def nan_strings(X):
    Y = X.copy()
    Y[np.equal(Y, None)] = np.nan
    return Y


DATA = {'s': (S, S_test), 'n': (nan_strings(S), nan_strings(S_test)), 'f': (F, F_test), 'i': (I, I_test)}


def columns(kind, X):
    if kind in 'sn':
        Xs = np.where(np.equal(X, X) & np.not_equal(X, None), X, None) if kind == 'n' else X
        return '[' + ', '.join(f'categorical_column({d.optional_strings(Xs[:, j].tolist())})' for j in range(X.shape[1])) + ']'
    return '[' + ', '.join(f'categorical_column({d.input(np.ascontiguousarray(X[:, j]))})' for j in range(X.shape[1])) + ']'


def mojo_value(kind, v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 'None'
    if kind in 'sn':
        return f'Optional[String]("{v}")'
    if kind == 'f':
        return f'Optional[Float64]({float(v)!r})'
    return f'Optional[Int64]({int(v)})'


def target(y):
    if y.dtype.kind == 'U':
        return 'StringLabels', d.strings(y.tolist())
    dtype = {np.dtype(np.float64): 'DType.float64', np.dtype(np.float32): 'DType.float32', np.dtype(np.int64): 'DType.int64'}[y.dtype]
    return f'NumericLabels[{dtype}]', d.input(y)


def native_args(kind, kw):
    parts = []
    for k, v in kw.items():
        if k == 'categories':
            rows = ', '.join('[' + ', '.join(mojo_value(kind, x) for x in cats) + ']' for cats in v)
            parts.append(f'categories=Optional[List[List[Optional[{DOMAIN_T[kind]}]]]]([{rows}])')
        elif k == 'smooth':
            parts.append(f'smooth=Optional[Float64]({float(v)!r})')
        elif isinstance(v, str):
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f'{k}={v}')
    return ', '.join(parts)


def sklearn_kw(kind, kw):
    gap = None if kind == 's' else np.nan
    out = dict(kw)
    if 'categories' in out:
        out['categories'] = [[gap if v is None else v for v in cats] for cats in out['categories']]
    return out


def text(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return MISSING
    return str(v)


def run_case(tag, kind, y, kw, *, column_y=False, global_seed=None, test=True):
    Xtr, Xte = DATA[kind]
    label_kind, y_expr = target(y.reshape(-1, 1) if column_y else y)
    native = f'TargetEncoder[{KIND[kind]}, {label_kind}]'
    C = f'List[CategoricalColumn[{KIND[kind]}.Domain]]'
    body = [f'var Xtr: {C} = {columns(kind, Xtr)}', f'var Xte: {C} = {columns(kind, Xte)}',
            f'var y = {y_expr}', f'var enc = {native}({native_args(kind, kw)})']
    if global_seed is not None:
        body.append(f'seed_global_random_state({global_seed})')
        np.random.seed(global_seed)
    ref = TargetEncoder(**sklearn_kw(kind, kw))
    expected = ref.fit_transform(Xtr, y.reshape(-1, 1) if column_y else y)
    body += ['emit("fit", enc.fit_transform(Xtr, y))',
             'for k in range(len(enc.encodings_)):\n    emit("enc_" + String(k), enc.encodings_[k])',
             'emit("mean", enc.target_mean_)',
             'emit_names("type", [enc.target_type_])',
             f'text_categories[{KIND[kind]}]("cat", enc.categories_)',
             'emit_names("names", enc.get_feature_names_out())',
             'emit("again", enc.transform(Xtr))']
    expect(tag, 'fit', expected)
    for k, e in enumerate(ref.encodings_):
        expect(tag, f'enc_{k}', np.asarray(e, dtype=np.float64))
    expect(tag, 'mean', np.atleast_1d(np.asarray(ref.target_mean_, dtype=np.float64)))
    expect(tag, 'type', [ref.target_type_], 'names')
    for j, cats in enumerate(ref.categories_):
        expect(tag, f'cat_{j}', [text(v) for v in cats.tolist()], 'names')
    expect(tag, 'names', [str(v) for v in ref.get_feature_names_out()], 'names')
    expect(tag, 'again', ref.transform(Xtr))
    if ref.classes_ is None:
        body.append('emit_int("no_classes", 0 if enc.classes_ else 1)')
        expect(tag, 'no_classes', 1, 'int')
    else:
        body.append(f'text_labels[{label_kind}]("classes", enc.classes_.value())')
        expect(tag, 'classes', [str(v) for v in ref.classes_.tolist()], 'names')
    if test:
        body.append('emit("test", enc.transform(Xte))')
        expect(tag, 'test', ref.transform(Xte))
    body.append('var custom: List[String] = [' + ', '.join(f'"f{j}"' for j in range(Xtr.shape[1])) + ']')
    body.append('emit_names("custom", enc.get_feature_names_out(custom^))')
    expect(tag, 'custom', [str(v) for v in ref.get_feature_names_out([f'f{j}' for j in range(Xtr.shape[1])])], 'names')
    # Plain fit learns the same state as fit_transform.
    body.append('var plain = ' + f'{native}({native_args(kind, kw)})' + '\nplain.fit(Xtr, y)\nemit("plain", plain.transform(Xtr))')
    expect(tag, 'plain', TargetEncoder(**sklearn_kw(kind, kw)).fit(Xtr, y.reshape(-1, 1) if column_y else y).transform(Xtr))
    d.case(tag, '\n'.join(body))


seeded = dict(random_state=0)
for kind in 'snfi':
    run_case(f'cont_{kind}', kind, y_cont, seeded)
    run_case(f'bin_{kind}', kind, y_bin, dict(random_state=3))
    run_case(f'multi_{kind}', kind, y_multi, dict(random_state=5, cv=3))
for smooth in (0.0, 5.0, 1e6):
    run_case(f'smooth_{smooth}_cont', 's', y_cont, dict(random_state=1, smooth=smooth))
    run_case(f'smooth_{smooth}_multi', 'f', y_multi_s, dict(random_state=1, smooth=smooth))
run_case('f32', 's', y_f32, dict(random_state=2))
run_case('f32_smooth', 'f', y_f32, dict(random_state=2, smooth=2.5))
run_case('int_cont', 's', y_int, dict(random_state=4, target_type='continuous'))
run_case('int_auto_multi', 'i', y_int, dict(random_state=4, cv=2))
run_case('bin_strings', 's', y_bin_s, dict(random_state=9))
run_case('multi_strings', 'n', y_multi_s, dict(random_state=9, cv=4))
run_case('multi_floats', 's', y_multi_f, dict(random_state=8))
run_case('noshuffle_cont', 's', y_cont, dict(shuffle=False, cv=4))
run_case('noshuffle_bin', 'f', y_bin, dict(shuffle=False, cv=2))
run_case('noshuffle_multi', 'i', y_multi, dict(shuffle=False))
run_case('shuffle_true', 's', y_bin, dict(shuffle=True, random_state=77, cv=2))
run_case('column_y', 's', y_cont, dict(random_state=6), column_y=True)
run_case('forced_binary', 's', y_multi, dict(random_state=6, target_type='binary'))
run_case('forced_binary_cont', 'f', y_cont, dict(random_state=6, target_type='binary', smooth=1.0))
run_case('forced_cont', 's', y_bin, dict(random_state=6, target_type='continuous'))
run_case('forced_multi', 's', y_multi_f, dict(random_state=6, target_type='multiclass'))
run_case('one_class', 'i', y_one, dict(random_state=6, target_type='multiclass'))
run_case('global_default', 's', y_cont, {}, global_seed=1234)
run_case('global_multi', 'n', y_multi, dict(cv=3), global_seed=99)
# A caller-owned RandomState (sklearn: a RandomState instance) advances across calls.
for tag, kind, y in [('rng_cont', 's', y_cont), ('rng_multi', 'f', y_multi_s)]:
    Xtr, _ = DATA[kind]
    label_kind, y_expr = target(y)
    ref = TargetEncoder(random_state=np.random.RandomState(5), cv=3)
    first, second = ref.fit_transform(Xtr, y), ref.fit_transform(Xtr, y)
    d.case(tag, f'var Xtr: List[CategoricalColumn[{KIND[kind]}.Domain]] = {columns(kind, Xtr)}\nvar y = {y_expr}\n'
           f'var enc = TargetEncoder[{KIND[kind]}, {label_kind}](cv=3)\nvar rng = RandomState(5)\n'
           'emit("first", enc.fit_transform(Xtr, y, rng))\nemit("second", enc.fit_transform(Xtr, y, rng))')
    expect(tag, 'first', first)
    expect(tag, 'second', second)
rs = np.random.RandomState(21)
expected_rng = [sk_folds(KFold(3, shuffle=True, random_state=rs), 14) for _ in range(2)]
d.case('kfold_rng', 'var rng = RandomState(21)\nemit_list("a", kfold_test_folds(14, 3, True, rng))\nemit_list("b", kfold_test_folds(14, 3, True, rng))')
expect('kfold_rng', 'a', expected_rng[0])
expect('kfold_rng', 'b', expected_rng[1])
rs = np.random.RandomState(22)
labels = label_sets['imbalanced']
expected_rng = [sk_folds(StratifiedKFold(3, shuffle=True, random_state=rs), len(labels), labels) for _ in range(2)]
d.case('strat_rng', 'var rng = RandomState(22)\nvar labels: List[Int] = [' + ', '.join(str(int(v)) for v in labels) + ']\n'
       'emit_list("a", stratified_kfold_test_folds(labels, 3, True, rng))\nemit_list("b", stratified_kfold_test_folds(labels, 3, True, rng))')
expect('strat_rng', 'a', expected_rng[0])
expect('strat_rng', 'b', expected_rng[1])
# Explicit categories: an unseen training value is unknown (-1) and every
# unknown row ends with the last fold's training mean.
run_case('explicit_s', 's', y_cont, dict(random_state=0, categories=[['red', 'green', 'blue', 'amber', 'violet'], ['y', 'x'], ['c', 'b', 'a', None]]))
run_case('explicit_multi', 's', y_multi, dict(random_state=0, smooth=3.0, categories=[['red', 'green', 'blue'], ['x', 'y', 'z'], ['a', 'b', 'c', None]]))
run_case('explicit_f', 'f', y_bin, dict(random_state=0, categories=[[-1.5, 0.0, 1.5, None], [-3.0, 0.0, 3.0]]))

# Feature names recorded by fit_named.
Xtr, _ = DATA['s']
d.case('named', f'var Xtr: List[CategoricalColumn[StringCategories]] = {columns("s", Xtr)}\nvar y = {target(y_multi)[1]}\n'
       'var enc = TargetEncoder[StringKind, NumericLabels[DType.int64]](random_state=0)\n'
       'var names: List[String] = ["colour", "axis", "letter"]\nenc.fit_named(Xtr, names, y)\nemit_names("names", enc.get_feature_names_out())')
named = TargetEncoder(random_state=0).fit(Xtr, y_multi)
named.feature_names_in_ = np.array(['colour', 'axis', 'letter'], dtype=object)  # as a DataFrame fit records
expect('named', 'names', [str(v) for v in named.get_feature_names_out()], 'names')


# ---- errors ---------------------------------------------------------------------
def sk_raises(fn):
    try:
        fn()
    except (ValueError, TypeError, IndexError):
        return
    raise SystemExit(f'reference did not raise: {fn}')


S_small, y_small = S[:6], y_cont[:6]
errors = {
    'target_type': ('TargetEncoder[StringKind](target_type="foo")', 'fit(Xs, yc)',
                    lambda: TargetEncoder(target_type='foo').fit(S_small, y_small), "'target_type' parameter"),
    'smooth': ('TargetEncoder[StringKind](smooth=Optional[Float64](-1.0))', 'fit(Xs, yc)',
               lambda: TargetEncoder(smooth=-1.0).fit(S_small, y_small), "'smooth' parameter"),
    'cv': ('TargetEncoder[StringKind](cv=1)', 'fit(Xs, yc)', lambda: TargetEncoder(cv=1).fit(S_small, y_small), "'cv' parameter"),
    'random_state': ('TargetEncoder[StringKind](random_state=-1)', 'fit(Xs, yc)',
                     lambda: TargetEncoder(random_state=-1).fit(S_small, y_small), "'random_state' parameter"),
    'length': ('TargetEncoder[StringKind]()', 'fit(Xs, y5)', lambda: TargetEncoder().fit(S_small, y_small[:5]), 'inconsistent numbers of samples: [6, 5]'),
    'multioutput': ('TargetEncoder[StringKind]()', 'fit(Xs, y2)', lambda: TargetEncoder().fit(S_small, np.c_[y_small, y_small]), "inferred to be 'continuous-multioutput'"),
    'nan_y': ('TargetEncoder[StringKind]()', 'fit(Xs, ynan)', lambda: TargetEncoder().fit(S_small, np.r_[y_small[:5], np.nan]), 'contains NaN'),
    'nan_y_cont': ('TargetEncoder[StringKind](target_type="continuous")', 'fit(Xs, ynan)',
                   lambda: TargetEncoder(target_type='continuous').fit(S_small, np.r_[y_small[:5], np.nan]), 'contains NaN'),
    'nan_y_binary_cv': ('TargetEncoder[StringKind](target_type="binary", random_state=0, cv=2)', 'fit_transform(Xs, ynan)',
                        lambda: TargetEncoder(target_type='binary', random_state=0, cv=2).fit_transform(S_small, np.r_[y_small[:5], np.nan]), 'contains NaN'),
    'noshuffle_seed': ('TargetEncoder[StringKind](shuffle=False, random_state=0)', 'fit_transform(Xs, yc)',
                       lambda: TargetEncoder(shuffle=False, random_state=0).fit_transform(S_small, y_small), 'no effect since shuffle is False'),
    'cv_samples': ('TargetEncoder[StringKind](cv=7, random_state=0)', 'fit_transform(Xs, yc)',
                   lambda: TargetEncoder(cv=7, random_state=0).fit_transform(S_small, y_small), 'n_splits=7 greater'),
    'cv_members': ('TargetEncoder[StringKind](cv=4, random_state=0)', 'fit_transform(Xs, yb)',
                   lambda: TargetEncoder(cv=4, random_state=0).fit_transform(S_small, np.array([0, 0, 0, 1, 1, 1])), 'number of members in each class'),
    'multi_two': ('TargetEncoder[StringKind](target_type="multiclass")', 'fit(Xs, yb)',
                  lambda: TargetEncoder(target_type='multiclass').fit(S_small, np.array([0, 0, 0, 1, 1, 1])), 'exactly 2 classes'),
    'multi_continuous': ('TargetEncoder[StringKind](target_type="multiclass")', 'fit(Xs, yc)',
                         lambda: TargetEncoder(target_type='multiclass').fit(S_small, y_small), 'Unknown label type'),
    'unfitted': ('TargetEncoder[StringKind]()', 'transform(Xs)', None, 'NotFittedError'),
    'features': ('TargetEncoder[StringKind]()', 'fit(Xs, yc)\nvar narrow: List[CategoricalColumn[StringCategories]] = [Xs[0].copy()]\n_ = enc.transform(narrow)',
                 lambda: TargetEncoder().fit(S_small, y_small).transform(S_small[:, :1]), 'is expecting 3 features'),
    'string_cont': ('TargetEncoder[StringKind, StringLabels](target_type="continuous")', 'fit(Xs, ys)',
                    lambda: TargetEncoder(target_type='continuous').fit(S_small, np.array(list('abcabc'))), 'TypeError'),
}
for name, (ctor, call, reference, needle) in errors.items():
    if reference is not None:
        sk_raises(reference)
    yb = d.input(np.array([0, 0, 0, 1, 1, 1], dtype=np.int64))
    body = (f'var Xs: List[CategoricalColumn[StringCategories]] = {columns("s", S_small)}\n'
            f'var yc = {d.input(y_small)}\nvar y5 = {d.input(y_small[:5])}\nvar y2 = {d.input(np.c_[y_small, y_small])}\n'
            f'var ynan = {d.input(np.r_[y_small[:5], np.nan])}\nvar yb = {yb}\nvar ys = {d.strings(list("abcabc"))}\n')
    if 'yb' in call or name == 'multi_two':
        ctor = ctor.replace('TargetEncoder[StringKind]', 'TargetEncoder[StringKind, NumericLabels[DType.int64]]')
    first, _, rest = call.partition('\n')
    body += f'var enc = {ctor}\n_ = enc.{first}' if first.startswith(('transform', 'fit_transform')) else f'var enc = {ctor}\nenc.{first}'
    if rest:
        body += '\n' + rest
    d.case(f'err_{name}', body)
    expect(f'err_{name}', 'error', needle, 'error')

out = d.run(imports='from moxie.preprocessing import TargetEncoder')

for case, name, value, kind in checks:
    got = out.get(case, {})
    label = f'{case}.{name}'
    if kind == 'error':
        d.raises(label, got.get('error'), value)
    elif name not in got and 'error' in got:
        d.close(label, got['error'], 0)
    elif kind == 'array':
        d.close(label, got.get(name, 'missing'), value, rtol=0, atol=0)
    else:
        d.equal(label, got.get(name), value)
d.finish()
