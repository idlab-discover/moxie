"""Differential dense KNNImputer checks against the pinned sklearn source body."""
from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT.parents[2]/'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip() == REVISION
files = ['sklearn/impute/_knn.py', 'sklearn/impute/_base.py', 'sklearn/impute/tests/test_knn.py']
assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',*files],text=True)
sys.path.insert(0, str(ROOT/'algorithms/simple/tests'))
from oracle import load
base = load()
import sklearn.impute._base
previous = sys.modules['sklearn.impute._base']
sys.modules['sklearn.impute._base'] = base
try:
    path = SOURCE/'sklearn/impute/_knn.py'
    spec = importlib.util.spec_from_file_location('sklearn.impute._knn_pinned', path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
finally:
    sys.modules['sklearn.impute._base'] = previous
assert inspect.unwrap(oracle.KNNImputer.transform).__code__.co_filename == str(path)
assert oracle.KNNImputer.__mro__[1] is base._BaseImputer
sys.path.insert(0, str(ROOT/'tools'))
from check import compile_source, run

lines = ['''import numojo as nm
from moxie.impute import KNNImputer
from moxie.impute._plumbing import MissingValue
from moxie.neighbors._contracts import DenseMetric
from moxie.neighbors._distance import Metric
from moxie.neighbors._query_engine import NeighborWeights
from std.math import isnan, abs

@no_inline
def emit[dtype: DType](name: String, X: nm.NDArray[dtype], names: List[String]) raises:
    print("R", name, X.shape[0], X.shape[1], len(names))
    for i in range(X.size): print(Float64(X.unsafe_get(i)))
    for feature in names: print(feature)

@no_inline
def exercise[dtype: DType](name: String, X: nm.NDArray[dtype], T: nm.NDArray[dtype], k: Int, weights: String, keep: Bool, indicator: Bool, marker: SIMD[dtype, 1]) raises:
    var model = KNNImputer[dtype](missing_values=MissingValue[dtype](marker), n_neighbors=k, weights=weights, keep_empty_features=keep, add_indicator=indicator)
    model.fit(X)
    emit(name, model.transform(T), model.get_feature_names_out())

struct Ones(NeighborWeights):
    def __init__(out self):
        pass
    def apply(self, distances: nm.NDArray[DType.float64], offsets: List[Int]) raises -> nm.NDArray[DType.float64]:
        return nm.ones[DType.float64](nm.Shape(distances.size))

struct Zeros(NeighborWeights):
    def __init__(out self):
        pass
    def apply(self, distances: nm.NDArray[DType.float64], offsets: List[Int]) raises -> nm.NDArray[DType.float64]:
        return nm.zeros[DType.float64](nm.Shape(distances.size))

struct WideWeights(NeighborWeights):
    def __init__(out self):
        pass
    def apply(self, distances: nm.NDArray[DType.float64], offsets: List[Int]) raises -> nm.NDArray[DType.float64]:
        return nm.full[DType.float64](nm.Shape(distances.size), 1e50)

struct SignedWeights(NeighborWeights):
    def __init__(out self):
        pass
    def apply(self, distances: nm.NDArray[DType.float64], offsets: List[Int]) raises -> nm.NDArray[DType.float64]:
        var result = nm.zeros[DType.float64](nm.Shape(distances.size))
        for i in range(distances.size): result.unsafe_set(i, Float64(1) if distances.unsafe_get(i) == 0 else Float64(-1))
        return result^

struct L1(DenseMetric):
    comptime dtype = DType.float64
    def __init__(out self):
        pass
    def validate_features(self, n_features: Int) raises:
        pass
    def distance(self, x: nm.NDArray[DType.float64], i: Int, y: nm.NDArray[DType.float64], j: Int) raises -> Float64:
        var total = Float64(0)
        for col in range(x.shape[1]):
            var a = x.unsafe_get(i * Int(x.strides[0]) + col * Int(x.strides[1]))
            var b = y.unsafe_get(j * Int(y.strides[0]) + col * Int(y.strides[1]))
            if not isnan(a) and not isnan(b): total += abs(a-b)
        return total
''']
expected = {}
promoted = {}
cases = 0

def scalar(value):
    return 'Float64("nan")' if np.isnan(value) else f'Float64({float(value)!r})'

def array(name, values, dtype):
    lines.append(f'    var {name} = nm.zeros[DType.{dtype}](nm.Shape({values.shape[0]}, {values.shape[1]}))')
    for i, value in enumerate(values.flat):
        lines.append(f'    {name}.unsafe_set({i}, {scalar(value)}.cast[DType.{dtype}]())')

def add(X, T, dtype, *, k=5, weights='uniform', keep=False, indicator=False, marker=np.nan):
    global cases
    X, T = np.array(X, dtype=dtype), np.array(T, dtype=dtype)
    model = oracle.KNNImputer(n_neighbors=k, weights=weights, missing_values=marker,
                              keep_empty_features=keep, add_indicator=indicator).fit(X)
    name = f'c{cases}'
    expected[name] = (model.transform(T), list(model.get_feature_names_out()), dtype)
    if dtype == 'float32':
        promoted[name] = oracle.KNNImputer(n_neighbors=k, weights=weights, missing_values=marker,
            keep_empty_features=keep, add_indicator=indicator).fit(X.astype('float64')).transform(T.astype('float64'))
    lines.extend(['@no_inline', f'def case{cases}() raises:'])
    array('X', X, dtype); array('T', T, dtype)
    lines.append(f'    exercise("{name}", X, T, {k}, "{weights}", {keep}, {indicator}, {scalar(marker)}.cast[DType.{dtype}]())')
    cases += 1

rng = np.random.default_rng(417)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for dtype in ('float32', 'float64'):
        for weights in ('uniform', 'distance'):
            for k in (1, 2, 5, 20):
                for marker in (np.nan, -1.0):
                    for keep in (False, True):
                        X = rng.normal(size=(9, 5)).astype(dtype)
                        T = rng.normal(size=(4, 5)).astype(dtype)
                        X[rng.random(X.shape)<.25] = marker
                        T[rng.random(T.shape)<.5] = marker
                        X[:, 3] = marker
                        T[0, :] = marker
                        add(X, T, dtype, k=k, weights=weights, keep=keep, indicator=True, marker=marker)
            # No overlap, fewer defined distances than k, zero-distance donors,
            # fit-empty columns including observed transform values, no indicators.
            fixtures = [
                ([[1,np.nan],[np.nan,4]], [[np.nan,np.nan],[np.nan,3]]),
                ([[2,0],[4,0],[8,np.nan]], [[np.nan,0]]),
                ([[1,2,np.nan],[3,4,np.nan]], [[np.nan,2,99],[5,6,9]]),
                ([[np.nan,np.nan],[np.nan,np.nan]], [[1,2],[np.nan,np.nan]]),
                ([[1,2,3]], [[np.nan,2,np.nan]]),
                ([[np.nan,0],[100,.125],[10,10],[20,20]], [[np.nan,0]]),
            ]
            for X, T in fixtures:
                for keep in (False, True):
                    add(X,T,dtype,weights=weights,keep=keep,k=3)
    # Reciprocal overflow is intentional dtype-specific behavior: compare to
    # the original Float32 oracle, rather than removing overflow by promotion.
    overflow_case=f'c{cases}'
    add([[2,1e-40],[6,2e-40]],[[np.nan,0]],'float32',k=2,weights='distance')
    del promoted[overflow_case]
    for callback, native in [(lambda d: np.ones_like(d), 'Ones'), (lambda d: np.zeros_like(d), 'Zeros')]:
        X = np.array([[2.,0.],[4.,0.],[8.,np.nan]])
        T = np.array([[np.nan,0.],[np.nan,np.nan]])
        model = oracle.KNNImputer(weights=callback, n_neighbors=3).fit(X)
        name=f'c{cases}';expected[name]=(model.transform(T),list(model.get_feature_names_out()),'float64')
        lines.extend(['@no_inline',f'def case{cases}() raises:']);array('X',X,'float64');array('T',T,'float64')
        lines.append(f'    var model = KNNImputer[DType.float64, Metric[DType.float64], {native}](n_neighbors=3, weights="callable", weights_callback={native}())')
        lines.extend(['    model.fit(X)',f'    emit("{name}", model.transform(T), model.get_feature_names_out())']);cases+=1
    X=np.array([[2.,0.],[4.,1.]], dtype='float32')
    T=np.array([[np.nan,0.]], dtype='float32')
    model=oracle.KNNImputer(weights=lambda d: np.full(d.shape,1e50,dtype='float64'),n_neighbors=2).fit(X)
    name=f'c{cases}';expected[name]=(model.transform(T),list(model.get_feature_names_out()),'float32')
    lines.extend(['@no_inline',f'def case{cases}() raises:']);array('X',X,'float32');array('T',T,'float32')
    lines.append('    var model = KNNImputer[DType.float32, Metric[DType.float32], WideWeights](n_neighbors=2, weights="callable", weights_callback=WideWeights())')
    lines.extend(['    model.fit(X)',f'    emit("{name}", model.transform(T), model.get_feature_names_out())']);cases+=1
    X=np.array([[1.,0.],[2.,1.]])
    T=np.array([[np.nan,0.]])
    model=oracle.KNNImputer(weights=lambda d: np.where(d == 0, 1., -1.),n_neighbors=2).fit(X)
    name=f'c{cases}';expected[name]=(model.transform(T),list(model.get_feature_names_out()),'float64')
    lines.extend(['@no_inline',f'def case{cases}() raises:']);array('X',X,'float64');array('T',T,'float64')
    lines.append('    var model = KNNImputer[DType.float64, Metric[DType.float64], SignedWeights](n_neighbors=2, weights="callable", weights_callback=SignedWeights())')
    lines.extend(['    model.fit(X)',f'    emit("{name}", model.transform(T), model.get_feature_names_out())']);cases+=1
    def metric(x,y,*,missing_values=np.nan):
        return np.nansum(np.abs(x-y))
    X=np.array([[4,3,3,np.nan],[6,9,6,9],[4,8,6,9],[np.nan,9,11,10.]])
    model=oracle.KNNImputer(metric=metric,n_neighbors=2).fit(X)
    name=f'c{cases}';expected[name]=(model.transform(X),list(model.get_feature_names_out()),'float64')
    lines.extend(['@no_inline',f'def case{cases}() raises:']);array('X',X,'float64')
    lines.append('    var model = KNNImputer[DType.float64, L1](n_neighbors=2, metric="callable", metric_callback=L1())')
    lines.extend(['    model.fit(X)',f'    emit("{name}", model.transform(X), model.get_feature_names_out())']);cases+=1
lines.append('def main() raises:')
lines.extend(f'    case{i}()' for i in range(cases))
source=ROOT/'builds/knn/parity.mojo';source.parent.mkdir(exist_ok=True)
source.write_text('\n'.join(lines)+'\n')
compile_source(source,source.with_suffix(''))
result=run(source.with_suffix(''),capture=True)
(source.parent/'parity-output.txt').write_text(result.stdout)
output=iter(result.stdout.splitlines());seen=set();rounding_cases={}
for line in output:
    tag,name,rows,cols,names=line.split();assert tag=='R' and name not in seen
    seen.add(name);wanted,feature_names,dtype=expected[name]
    rows,cols,names=map(int,(rows,cols,names))
    actual=np.array([float(next(output)) for _ in range(rows*cols)]).reshape(rows,cols)
    assert actual.shape==wanted.shape,(name,actual.shape,wanted.shape)
    # The shared metric computes differences in Float64. sklearn's Float32
    # norm expansion can suffer catastrophic cancellation at small distances.
    # Validate every Float32 result against the pinned source on promoted input,
    # and record its discrepancy from the source's native Float32 calculation.
    tolerance=3e-6 if dtype=='float32' else 3e-12
    numerical_reference=promoted.get(name,wanted)
    np.testing.assert_allclose(actual,numerical_reference,rtol=tolerance,atol=tolerance,
                               equal_nan=True,err_msg=name)
    if name in promoted and not np.allclose(wanted,numerical_reference,rtol=tolerance,atol=tolerance,equal_nan=True):
        rounding_cases[name]=float(np.max(np.abs(wanted-numerical_reference)))
    assert [next(output) for _ in range(names)]==feature_names,name
assert seen==set(expected)
evidence=ROOT/'algorithms/knn/docs/private/knn';evidence.mkdir(parents=True,exist_ok=True)
(evidence/'parity.json').write_text(json.dumps(dict(cases=cases,seed=417,sklearn_revision=REVISION,files={f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in files},float32_promoted_tolerance=3e-6,float32_reference_rounding_cases=rounding_cases,float64_tolerance=3e-12),indent=2)+'\n')
print(f'PASS: {cases} KNNImputer differential cases, including feature names and callbacks')
