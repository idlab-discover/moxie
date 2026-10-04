"""Record and compare unmodified pinned sklearn dense promotion branches."""
from reference import *
import itertools

out = ROOT / 'builds/dtype-parity'
out.mkdir(parents=True, exist_ok=True)
records = []
fixtures = []


def enc(a):
    a = np.asarray(a)
    return f'{a.shape[0]}:{a.shape[1] if a.ndim == 2 else 0}:' + ','.join(repr(float(x)) for x in a.ravel())


def case(kernel, X, y, Z, weight, penalties, branch):
    metric = (lambda x, y: x @ y) if kernel == 'callable' else kernel
    weights = {'none': None, 'scalar1': 1., 'scalar2': 2., 'zero': 0., 'list': [.5, 1., 2.]}[weight]
    record = dict(kernel=kernel, X=str(X.dtype), y=str(y.dtype), Z=str(Z.dtype),
                  weights=weight, alpha=branch, target_shape=y.shape)
    fields = [kernel, str(X.dtype), str(y.dtype), str(Z.dtype),
              {'scalar1': '1.0', 'scalar2': '2.0', 'zero': '0.0'}.get(weight, weight),
              ','.join(map(str, np.atleast_1d(penalties))), enc(X), enc(y), enc(Z)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            model = KR(kernel=metric, alpha=penalties, gamma=.2).fit(X, y, sample_weight=weights)
            K = model._get_kernel(X)
            p = model.predict(Z)
            record.update(K=str(K.dtype), X_fit=str(model.X_fit_.dtype),
                          coef=str(model.dual_coef_.dtype), prediction=str(p.dtype))
            tolerance = 2e-4 if np.float32 in (X.dtype, y.dtype, Z.dtype) else 3e-9
            fields += [str(model.dual_coef_.dtype), str(p.dtype), enc(model.dual_coef_), enc(p), str(tolerance), repr(float(model.score(Z, y[:2])))]
        except (ValueError, np.linalg.LinAlgError) as error:
            record['error'] = type(error).__name__ + ': ' + str(error)
            fields += ['error']
    records.append(record)
    fixtures.append('|'.join(fields))


for kernel, xd, yd, zd, weight, alpha, rank in itertools.product(
    [*ns['PAIRWISE_KERNEL_FUNCTIONS'], 'precomputed', 'callable'],
    [np.float32, np.float64, np.int32, np.int64],
    [np.float32, np.float64, np.int32, np.int64],
    [np.float32, np.float64, np.int32, np.int64],
    ['none', 'scalar1', 'scalar2', 'list'], ['scalar', 'equal', 'unequal'], [1, 2, 3],
):
    if alpha == 'unequal' and rank != 3:
        continue
    X = np.array([[2.13, 1.24, 1.35], [1.24, 3.47, 1.58], [1.35, 1.58, 4.69]], dtype=xd)
    y = np.array([[1.17, 2.21], [3.39, 1.41], [2.57, 4.63]], dtype=yd)
    if rank == 1:
        y = y[:, 0].copy()
    elif rank == 2:
        y = y[:, :1].copy()
    Z = X[:2].astype(zd)
    penalties = {'scalar': 20., 'equal': [20.] * (2 if rank == 3 else 1), 'unequal': [20., 30.]}[alpha]
    case(kernel, X, y, Z, weight, penalties, alpha)

# Rank-cutoff-sensitive fallback and unequal-alpha failure after promotion.
for xd, yd, zd, weight in itertools.product(
    [np.float32, np.float64], [np.float32, np.float64, np.int32],
    [np.float32, np.float64], ['none', 'scalar2', 'list', 'zero'],
):
    for diagonal in [[-1., 1e-8, 0.], [0., 0., 0.], [1., 1., 1.]]:
        X = np.diag(diagonal).astype(xd)
        y = np.array([[1., 2.], [1e-8, 2e-8], [3., 4.]], dtype=yd)
        for penalties in [0., [0., 0.], [0., .1], [1., 1. + 1e-9]]:
            case('precomputed', X, y, X[:2].astype(zd), weight, penalties, 'edge')

(out / 'reference.json').write_text(json.dumps(records, indent=2))
(out / 'fixtures.txt').write_text('\n'.join(fixtures) + '\n')
(out / 'metadata.json').write_text(json.dumps(dict(
    revision=REV, numpy=np.__version__, scipy=scipy.__version__,
    support_sklearn=__import__("sklearn").__version__, cases=len(records),
    source_hashes={p: hashlib.sha256((REFERENCE / p).read_bytes()).hexdigest() for p in paths},
    tolerances={'float32_inputs': 2e-4, 'otherwise': 3e-9},
    note='Float32 rounding is retained at kernel, regularization, and output-storage boundaries.',
), indent=2))
print('Recorded dtype reference cases:', len(records))
if '--record-only' not in sys.argv:
    compile(ROOT / 'tests/dtype_parity_runner.mojo', ROOT / 'builds/dtype_parity_runner')
    result = subprocess.run([str(ROOT / 'builds/dtype_parity_runner'), str(out / 'fixtures.txt')],
                            cwd=ROOT, env=environment(), capture_output=True, text=True)
    (out / 'result.txt').write_text(result.stdout + result.stderr)
    if result.returncode:
        print(result.stdout + result.stderr, file=sys.stderr)
    result.check_returncode()
    print(result.stdout, end='')
