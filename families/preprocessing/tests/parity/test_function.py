"""Parity: FunctionTransformer with typed native callables against Python callables.

Each native callable is paired with the NumPy callable sklearn receives:
identity (func=None), affine with kw_args, column add/drop, Float64->Float32
cast, sqrt/round, x*(1+delta) perturbations, an inverse that corrupts negative
values (exercising the X[::max(1, n // 100)] subsample), integer doubling and
boolean negation. Outputs, inverse outputs, n_features_in_, the check_inverse
warning (sklearn UserWarning vs native inverse_check_warning_), feature names
and error cases are compared on Float64 and Float32, including strided views.

Tolerances: exact (rtol 0) wherever both sides perform the same IEEE
operations (identity copies, affine, casts, sqrt, round, perturbations, integer
and boolean maps). Only the log1p/expm1 pair uses rtol 1e-9 (Float64) and
2e-6 (Float32): Mojo 1.0's std.math.log1p differs from NumPy's libm by up to
~3e-10 relative in Float64 — a property of the test callable, not of the
transformer, which forwards the callable's output unchanged.
"""
import warnings

import numpy as np

from harness import Driver, oracle

oracle()
from sklearn import preprocessing as P

d = Driver('function')
checks = []  # (case, name, expected, kind)
MOJO = {np.float64: 'DType.float64', np.float32: 'DType.float32'}
LO, HI, EDGE = 1.00000005, 1.0000002, 1.0000001


def expect(case, name, value, kind='exact'):
    checks.append((case, name, value, kind))


def sk(fn):
    """Run an sklearn callable; return (result, warning texts) or an error string."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = fn()
        except Exception as e:  # noqa: BLE001 - mirrored as a native error
            return f'Error: {type(e).__name__}: {e}', []
    return result, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


d.prelude(f'''
from std.math import sqrt, round, log1p, expm1
from moxie.preprocessing._function_transformer import (
    Identity, Elementwise, ArrayFn, KwArrayFn, NamesFn, NoKwArgs,
)

def sqrt64(x: Float64) -> Float64:
    return sqrt(x)

def round64(x: Float64) -> Float64:
    return round(x)

def log1p64(x: Float64) -> Float64:
    return log1p(x)

def expm164(x: Float64) -> Float64:
    return expm1(x)

def log1p32(x: Float32) -> Float32:
    return log1p(x)

def expm132(x: Float32) -> Float32:
    return expm1(x)

def to_f32(x: Float64) -> Float32:
    return Float32(x)

def to_f64(x: Float32) -> Float64:
    return Float64(x)

def lo64(x: Float64) -> Float64:
    return x * {LO!r}

def hi64(x: Float64) -> Float64:
    return x * {HI!r}

def edge64(x: Float64) -> Float64:
    return x * {EDGE!r}

def lo32(x: Float32) -> Float32:
    return x * Float32({LO!r})

def hi32(x: Float32) -> Float32:
    return x * Float32({HI!r})

def edge32(x: Float32) -> Float32:
    return x * Float32({EDGE!r})

def corrupt_negative(x: Float64) -> Float64:
    return x + 1.0 if x < 0 else x

def double_int(x: Int64) -> Int64:
    return 2 * x

def half_int(x: Int64) -> Int64:
    return x // 2

def negate_bool(x: Scalar[DType.bool]) -> Scalar[DType.bool]:
    return not x

@fieldwise_init
struct AffineKw(Copyable, Movable):
    var factor: Float64
    var shift: Float64

def affine(X: NDArray[DType.float64], kw: Optional[AffineKw]) raises -> NDArray[DType.float64]:
    var factor = kw.value().factor if kw else 1.0
    var shift = kw.value().shift if kw else 0.0
    var out = zeros[DType.float64](Shape(X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            out.unsafe_set(i * X.shape[1] + j, X.item(i, j) * factor + shift)
    return out^

def unaffine(X: NDArray[DType.float64], kw: Optional[AffineKw]) raises -> NDArray[DType.float64]:
    var factor = kw.value().factor if kw else 1.0
    var shift = kw.value().shift if kw else 0.0
    var out = zeros[DType.float64](Shape(X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            out.unsafe_set(i * X.shape[1] + j, (X.item(i, j) - shift) / factor)
    return out^

def drop_last(X: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
    var p = X.shape[1] - 1
    var out = zeros[DType.float64](Shape(X.shape[0], p))
    for i in range(X.shape[0]):
        for j in range(p):
            out.unsafe_set(i * p + j, X.item(i, j))
    return out^

def add_ones(X: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
    var p = X.shape[1] + 1
    var out = zeros[DType.float64](Shape(X.shape[0], p))
    for i in range(X.shape[0]):
        for j in range(p):
            out.unsafe_set(i * p + j, X.item(i, j) if j < p - 1 else 1.0)
    return out^

def suffix_out(input_features: Optional[List[String]]) raises -> List[String]:
    var names = List[String]()
    for name in input_features.value():
        names.append(name + "_out")
    names.append("extra")
    return names^

def reversed_rows[d: DType](x: NDArray[d]) raises -> NDArray[d]:
    var n = x.shape[0]
    var p = x.shape[1]
    return x.view_with_layout(Shape(n, p), nm.NDArrayStrides(-p, 1), x.offset + (n - 1) * p)

def fortran[d: DType](x: NDArray[d]) raises -> NDArray[d]:
    var n = x.shape[0]
    var p = x.shape[1]
    var owner = zeros[d](Shape(n * p))
    for i in range(n):
        for j in range(p):
            owner.unsafe_set(j * n + i, x.item(i, j))
    return owner.view_with_layout(Shape(n, p), nm.NDArrayStrides(1, n), 0)

def emit_nfi(name: String, value: Optional[Int]):
    emit_int(name, value.value() if value else -1)

def emit_warning(name: String, value: Optional[String]):
    print("S", name, 1 if value else 0)
    if value:
        print(value.value())
''')


def sk_nfi(est):
    return getattr(est, 'n_features_in_', -1)


rng = np.random.RandomState(0)

# ---- identity, validate on/off, strided views --------------------------------
for dt in (np.float64, np.float32):
    X = (rng.rand(6, 3) * 10).astype(dt)
    T = d.input(X)
    m = MOJO[dt]
    for validate in (False, True):
        tag = f'identity_{dt.__name__}_{validate}'
        body = f'''var X = {T}
var ft = FunctionTransformer[Identity[{m}]](validate={validate})
emit("unfitted", ft.transform(X))
emit_nfi("nfi0", ft.n_features_in_)
emit("fit_transform", ft.fit_transform(X))
emit_nfi("nfi", ft.n_features_in_)
emit("inverse", ft.inverse_transform(X))
emit("reversed", ft.transform(reversed_rows(X)))
emit("fortran", ft.transform(fortran(X)))
emit_warning("warning", ft.inverse_check_warning_)'''
        d.case(tag, body)
        est = P.FunctionTransformer(validate=validate)
        expect(tag, 'unfitted', est.transform(X))
        expect(tag, 'nfi0', sk_nfi(est), 'int')
        (out, warns) = sk(lambda: est.fit_transform(X))
        expect(tag, 'fit_transform', out)
        expect(tag, 'nfi', sk_nfi(est), 'int')
        expect(tag, 'inverse', est.inverse_transform(X))
        expect(tag, 'reversed', est.transform(X[::-1]))
        expect(tag, 'fortran', est.transform(np.asfortranarray(X)))
        expect(tag, 'warning', warns, 'names')

# validate=False passes NaN/inf through; rank-1 input leaves n_features_in_ alone.
Xn = np.array([[1.0, np.nan], [np.inf, -2.0]])
v = np.array([1.0, 2.0, 3.0])
d.case('passthrough', f'''var ft = FunctionTransformer()
emit("nan", ft.fit_transform({d.input(Xn)}))
emit_nfi("nfi", ft.n_features_in_)
emit("vector", ft.fit_transform({d.input(v)}))
emit_nfi("nfi_after_vector", ft.n_features_in_)''')
est = P.FunctionTransformer()
expect('passthrough', 'nan', est.fit_transform(Xn))
expect('passthrough', 'nfi', sk_nfi(est), 'int')
expect('passthrough', 'vector', est.fit_transform(v))
expect('passthrough', 'nfi_after_vector', sk_nfi(est), 'int')

# ---- forward/inverse pairs and check_inverse ---------------------------------
X64 = np.array([1, 4, 9, 16, 2, 3], dtype=np.float64).reshape(3, 2)
pairs = [
    # (tag, native func, native inverse, numpy func, numpy inverse, dtype, kind)
    ('sqrt_round', 'Elementwise[sqrt64]()', 'Elementwise[round64]()', np.sqrt, np.around, np.float64, 'exact'),
    ('lo64', 'Elementwise[lo64]()', 'Identity[DType.float64]()', lambda X: X * LO, lambda X: X, np.float64, 'exact'),
    ('hi64', 'Elementwise[hi64]()', 'Identity[DType.float64]()', lambda X: X * HI, lambda X: X, np.float64, 'exact'),
    ('edge64', 'Elementwise[edge64]()', 'Identity[DType.float64]()', lambda X: X * EDGE, lambda X: X, np.float64, 'exact'),
    ('lo32', 'Elementwise[lo32]()', 'Identity[DType.float32]()', lambda X: X * LO, lambda X: X, np.float32, 'exact'),
    ('hi32', 'Elementwise[hi32]()', 'Identity[DType.float32]()', lambda X: X * HI, lambda X: X, np.float32, 'exact'),
    ('edge32', 'Elementwise[edge32]()', 'Identity[DType.float32]()', lambda X: X * EDGE, lambda X: X, np.float32, 'exact'),
    ('log1p64', 'Elementwise[log1p64]()', 'Elementwise[expm164]()', np.log1p, np.expm1, np.float64, 'log64'),
    ('expm164', 'Elementwise[expm164]()', 'Elementwise[log1p64]()', np.expm1, np.log1p, np.float64, 'log64'),
    ('log1p32', 'Elementwise[log1p32]()', 'Elementwise[expm132]()', np.log1p, np.expm1, np.float32, 'log32'),
]
for tag, nf, ni, pf, pi, dt, kind in pairs:
    for validate in (False, True):
        case = f'{tag}_{validate}'
        X = (rng.rand(7, 3) * 50 + 1e-3).astype(dt) if tag != 'sqrt_round' else X64
        T = d.input(X)
        d.case(case, f'''var X = {T}
var ft = FunctionTransformer({nf}, {ni}, validate={validate})
var Xt = ft.fit_transform(X)
emit("t", Xt)
emit("inv", ft.inverse_transform(Xt))
emit_warning("warning", ft.inverse_check_warning_)
ft.set_params(check_inverse=False)
ft.fit(X)
emit_warning("warning_off", ft.inverse_check_warning_)''')
        est = P.FunctionTransformer(pf, pi, validate=validate)
        (Xt, warns) = sk(lambda: est.fit_transform(X))
        expect(case, 't', Xt, kind)
        expect(case, 'inv', est.inverse_transform(Xt), kind)
        expect(case, 'warning', warns, 'names')
        est.set_params(check_inverse=False)
        expect(case, 'warning_off', sk(lambda: est.fit(X))[1], 'names')

# Subsample X[::max(1, n // 100)]: odd rows negative, corrupted by the inverse.
for n in (99, 150, 199, 200, 250, 1000, 1001):
    Y = np.where(np.arange(n) % 2 == 0, np.arange(n) + 1.0, -np.arange(n, dtype=np.float64))[:, None]
    case = f'subsample_{n}'
    d.case(case, f'''var ft = FunctionTransformer(Identity[DType.float64](), Elementwise[corrupt_negative]())
ft.fit({d.input(Y)})
emit_warning("warning", ft.inverse_check_warning_)''')
    est = P.FunctionTransformer(lambda X: X, lambda X: np.where(X < 0, X + 1.0, X))
    expect(case, 'warning', sk(lambda: est.fit(Y))[1], 'names')

# ---- kw_args, shape changes, dtype changes, integers --------------------------
X = rng.randn(5, 3)
T = d.input(X)
d.case('kw_args', f'''var X = {T}
var ft = FunctionTransformer(KwArrayFn[affine](), KwArrayFn[unaffine](), kw_args=AffineKw(2.0, -0.5), inv_kw_args=AffineKw(2.0, -0.5))
var Xt = ft.fit_transform(X)
emit("t", Xt)
emit("inv", ft.inverse_transform(Xt))
emit_warning("warning", ft.inverse_check_warning_)
ft.set_params(kw_args=AffineKw(3.0, 1.0))
emit("updated", ft.transform(X))
emit_warning("warning_updated", ft.inverse_check_warning_)
ft.fit(X)
emit_warning("warning_refit", ft.inverse_check_warning_)
var params = ft.get_params()
params.kw_args = None
ft.set_params(params)
emit("reset", ft.transform(X))''')
affine = lambda X, factor=1.0, shift=0.0: X * factor + shift  # noqa: E731
unaffine = lambda X, factor=1.0, shift=0.0: (X - shift) / factor  # noqa: E731
est = P.FunctionTransformer(affine, unaffine, kw_args=dict(factor=2.0, shift=-0.5), inv_kw_args=dict(factor=2.0, shift=-0.5))
(Xt, warns) = sk(lambda: est.fit_transform(X))
expect('kw_args', 't', Xt)
expect('kw_args', 'inv', est.inverse_transform(Xt))
expect('kw_args', 'warning', warns, 'names')
est.set_params(kw_args=dict(factor=3.0, shift=1.0))
expect('kw_args', 'updated', est.transform(X))
expect('kw_args', 'warning_updated', warns, 'names')   # no refit: still the old result
expect('kw_args', 'warning_refit', sk(lambda: est.fit(X))[1], 'names')
est.set_params(kw_args=None)
expect('kw_args', 'reset', est.transform(X))

d.case('columns', f'''var X = {T}
var ft = FunctionTransformer(ArrayFn[add_ones](), ArrayFn[drop_last](), validate=True, feature_names_out=NamesFn[suffix_out]())
var Xt = ft.fit_transform(X)
emit("t", Xt)
emit("inv", ft.inverse_transform(Xt))
emit_nfi("nfi", ft.n_features_in_)
emit_warning("warning", ft.inverse_check_warning_)
emit_names("names", ft.get_feature_names_out())
var given: List[String] = ["a", "b", "c"]
emit_names("names_given", ft.get_feature_names_out(given^))''')
est = P.FunctionTransformer(lambda X: np.concatenate([X, np.ones((X.shape[0], 1))], axis=1), lambda X: X[:, :-1],
                            validate=True, feature_names_out=lambda self, names: [f'{n}_out' for n in names] + ['extra'])
(Xt, warns) = sk(lambda: est.fit_transform(X))
expect('columns', 't', Xt)
expect('columns', 'inv', est.inverse_transform(Xt))
expect('columns', 'nfi', sk_nfi(est), 'int')
expect('columns', 'warning', warns, 'names')
expect('columns', 'names', list(est.get_feature_names_out()), 'names')
expect('columns', 'names_given', list(est.get_feature_names_out(['a', 'b', 'c'])), 'names')

d.case('cast', f'''var X = {T}
var ft = FunctionTransformer(Elementwise[to_f32](), Elementwise[to_f64]())
var Xt = ft.fit_transform(X)
emit("t", Xt)
emit("inv", ft.inverse_transform(Xt))
emit_warning("warning", ft.inverse_check_warning_)''')
est = P.FunctionTransformer(lambda X: X.astype(np.float32), lambda X: X.astype(np.float64))
(Xt, warns) = sk(lambda: est.fit_transform(X))
expect('cast', 't', Xt)
expect('cast', 'inv', est.inverse_transform(Xt))
expect('cast', 'warning', warns, 'names')

Xi = rng.randint(-50, 50, size=(4, 3)).astype(np.int64)
d.case('ints', f'''var X = {d.input(Xi)}
var ft = FunctionTransformer(Elementwise[double_int](), Elementwise[half_int](), validate=True)
var Xt = ft.fit_transform(X)
emit("t", Xt)
emit("inv", ft.inverse_transform(Xt))
emit_warning("warning", ft.inverse_check_warning_)''')
est = P.FunctionTransformer(lambda X: 2 * X, lambda X: X // 2, validate=True)
(Xt, warns) = sk(lambda: est.fit_transform(Xi))
expect('ints', 't', Xt)
expect('ints', 'inv', est.inverse_transform(Xt))
expect('ints', 'warning', warns, 'names')

Xb = rng.rand(3, 2) > 0.5
d.case('bools', f'''var X = {d.input(Xb)}
var ft = FunctionTransformer(Elementwise[negate_bool](), Elementwise[negate_bool](), check_inverse=False)
emit("t", ft.fit_transform(X))
ft.set_params(check_inverse=True)
ft.fit(X)''')
est = P.FunctionTransformer(np.logical_not, np.logical_not, check_inverse=False)
expect('bools', 't', est.fit_transform(Xb))
est.set_params(check_inverse=True)
expect('bools', 'error', sk(lambda: est.fit(Xb))[0], 'error')

# ---- feature names -----------------------------------------------------------
X = rng.rand(4, 3)
T = d.input(X)
d.case('names', f'''var X = {T}
var one = FunctionTransformer(feature_names_out="one-to-one")
var given: List[String] = ["a", "b"]
emit_names("unfitted_given", one.get_feature_names_out(given^))
one.fit(X)
emit_names("generated", one.get_feature_names_out())
var three: List[String] = ["p", "q", "r"]
emit_names("given", one.get_feature_names_out(three^))
var plain = FunctionTransformer()
plain.fit(X)
try:
    _ = plain.get_feature_names_out()
except e:
    emit_error("none_error", e)
try:
    var short: List[String] = ["a"]
    _ = one.get_feature_names_out(short^)
except e:
    emit_error("length_error", e)
var unvalidated = FunctionTransformer(validate=False, feature_names_out="one-to-one")
_ = unvalidated.fit_transform(X)
var ab: List[String] = ["x", "y", "z"]
emit_names("unvalidated", unvalidated.get_feature_names_out(ab^))''')
one = P.FunctionTransformer(feature_names_out='one-to-one')
expect('names', 'unfitted_given', list(one.get_feature_names_out(['a', 'b'])), 'names')
one.fit(X)
expect('names', 'generated', list(one.get_feature_names_out()), 'names')
expect('names', 'given', list(one.get_feature_names_out(['p', 'q', 'r'])), 'names')
plain = P.FunctionTransformer().fit(X)
expect('names', 'none_error', sk(lambda: plain.get_feature_names_out())[0], 'error')
expect('names', 'length_error', sk(lambda: one.get_feature_names_out(['a']))[0], 'error')
unvalidated = P.FunctionTransformer(validate=False, feature_names_out='one-to-one')
unvalidated.fit_transform(X)
expect('names', 'unvalidated', list(unvalidated.get_feature_names_out(['x', 'y', 'z'])), 'names')

# ---- validation errors -------------------------------------------------------
Xn = np.array([[1.0, np.nan], [2.0, 3.0]])
Xinf = np.array([[1.0, np.inf], [2.0, 3.0]])
error_cases = {
    'nan_fit': ('var ft = FunctionTransformer(validate=True)\nft.fit({Xn})', lambda: P.FunctionTransformer(validate=True).fit(Xn)),
    'inf_transform': ('var ft = FunctionTransformer(validate=True)\n_ = ft.transform({Xinf})', lambda: P.FunctionTransformer(validate=True).transform(Xinf)),
    'nan_inverse': ('var ft = FunctionTransformer(validate=True)\n_ = ft.inverse_transform({Xn})', lambda: P.FunctionTransformer(validate=True).inverse_transform(Xn)),
    'vector_fit': ('var ft = FunctionTransformer(validate=True)\nft.fit({v})', lambda: P.FunctionTransformer(validate=True).fit(v)),
    'empty_fit': ('var ft = FunctionTransformer(validate=True)\nft.fit({empty})', lambda: P.FunctionTransformer(validate=True).fit(np.zeros((0, 2)))),
    'no_features_fit': ('var ft = FunctionTransformer(validate=True)\nft.fit({nofeat})', lambda: P.FunctionTransformer(validate=True).fit(np.zeros((2, 0)))),
    'bad_names_out': ('var ft = FunctionTransformer(feature_names_out="one-to-many")\nft.fit({X})', lambda: P.FunctionTransformer(feature_names_out='one-to-many').fit(X)),
    'identity_kw': ('var ft = FunctionTransformer[KwArrayFn[affine]](kw_args=AffineKw(2.0, 0.0))\n_ = ft.transform({X})', lambda: P.FunctionTransformer(kw_args=dict(factor=2.0)).transform(X)),
    'feature_mismatch': ('var ft = FunctionTransformer(validate=True)\nft.fit({X})\n_ = ft.transform({Xn})', None),
}
inputs = dict(Xn=d.input(Xn), Xinf=d.input(Xinf), v=d.input(v), empty=d.input(np.zeros((0, 2))),
              nofeat=d.input(np.zeros((2, 0))), X=d.input(X))
mism = P.FunctionTransformer(validate=True).fit(X)
error_cases['feature_mismatch'] = (error_cases['feature_mismatch'][0], lambda: mism.transform(Xn))
body = []
for name, (code, fn) in error_cases.items():
    code = code.format(**inputs)
    body.append('try:')
    body += ['    ' + line for line in code.split('\n')]
    body.append(f'    print("I {name}_ok 1")')
    body.append('except e:')
    body.append(f'    emit_error("{name}", e)')
    expect('errors', name, sk(fn)[0], 'error')
d.case('errors', '\n'.join(body))

out = d.run(imports='from moxie.preprocessing import FunctionTransformer')
TOLS = {'exact': dict(rtol=0.0), 'log64': dict(rtol=1e-9), 'log32': dict(rtol=2e-6)}
for case, name, expected, kind in checks:
    got_case = out.get(case, {})
    label = f'{case}.{name}'
    if 'error' in got_case and name not in got_case:
        d._fail(label, f'native case raised {got_case["error"]}')
        d.checks += 1
        continue
    got = got_case.get(name, 'missing')
    if kind == 'int':
        d.equal(label, got, expected)
    elif kind == 'names':
        d.equal(label, got, list(expected))
    elif kind == 'error':
        # Both sides raise the same exception class; message prefixes agree.
        if not (isinstance(expected, str) and expected.startswith('Error: ')):
            d._fail(label, f'sklearn did not raise: {expected!r}')
            d.checks += 1
            continue
        cls_and_msg = expected[len('Error: '):]
        cls, _, msg = cls_and_msg.partition(': ')
        if cls == 'InvalidParameterError':
            cls = 'ValueError'   # a ValueError subclass in sklearn
        d.raises(label, got, f'{cls}: {msg.split(chr(10))[0][:40]}')
    else:
        d.close(label, got, expected, dtype=np.asarray(expected).dtype, **TOLS[kind])
warned = sorted(f'{case}.{name}' for case, name, value, kind in checks if name.startswith('warning') and value)
print('sklearn warned (native must match):', ', '.join(warned))
assert len(warned) >= 8, 'the check_inverse cases must exercise both outcomes'
d.finish(extra=dict(warned=warned))
