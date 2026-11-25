"""Compare native results with pinned estimator bodies, using bounded driver code."""
from pathlib import Path
import argparse
import json
import sys
import warnings
import numpy as np
from oracle import load
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import check
from check import compile, run
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build-dir', type=Path, default=ROOT/'builds')
BUILDS = parser.parse_args().build_dir.resolve()
check.BUILDS = BUILDS
oracle=load()
lines=['''import numojo as nm
from moxie.impute import SimpleImputer, MissingIndicator
from moxie.impute._plumbing import MissingValue
from moxie.impute._simple import ColumnStrategy
from std.testing import assert_equal, assert_raises
from std.utils import Variant

@no_inline
def emit[dtype: DType](name: String, X: nm.NDArray[dtype]) raises:
    var rows = X.shape[0] if X.ndim == 2 else 1
    var cols = X.shape[1] if X.ndim == 2 else X.size
    print("R", name, rows, cols)
    for i in range(X.size):
        print(Float64(X.unsafe_get(i)))

@no_inline
def exercise[dtype: DType](name: String, X: nm.NDArray[dtype], T: nm.NDArray[dtype], strategy: String, marker: SIMD[dtype, 1], keep: Bool, indicator: Bool) raises:
    var model = SimpleImputer[dtype](strategy=strategy, missing_values=MissingValue[dtype](marker), keep_empty_features=keep, add_indicator=indicator, fill_value=SIMD[dtype, 1](7))
    model.fit(X)
    emit(name + "s", model.statistics_)
    var output = model.transform(T)
    emit(name + "o", output)
    if indicator:
        try:
            var inverse = model.inverse_transform(output)
            emit(name + "i", inverse)
        except:
            print("E", name + "i")

@no_inline
def indicate[dtype: DType](name: String, X: nm.NDArray[dtype], features: String, error_on_new: Bool, sparse: Variant[Bool, String]) raises:
    var model = MissingIndicator[dtype](features=features, error_on_new=error_on_new, sparse=sparse)
    emit(name, model.fit_transform(X))

@no_inline
def update_case[dtype: DType](name: String, X: nm.NDArray[dtype]) raises:
    var model = SimpleImputer[dtype](strategy="constant", fill_value=SIMD[dtype, 1](9))
    model.fit(X)
    model.set_params(missing_values=MissingValue[dtype](-1), add_indicator=True)
    emit(name + "marker", model.fit_transform(X))
    model.set_params(fill_value=SIMD[dtype, 1](7), add_indicator=False)
    emit(name + "fill", model.fit_transform(X))
    model.set_params(fill_value=None)
    emit(name + "default", model.fit_transform(X))
    var indicator = MissingIndicator[dtype]()
    indicator.fit(X)
    indicator.set_params(missing_values=MissingValue[dtype](5), sparse="auto")
    emit(name + "indicator", indicator.fit_transform(X))

struct Minimum(ColumnStrategy):
    def __init__(out self):
        pass
    def __call__(self, values: nm.NDArray[DType.float64]) raises -> Float64:
        if values.size == 0:
            return 17
        var value = values.unsafe_get(0)
        for i in range(values.size):
            value = min(value, values.unsafe_get(i))
        return value

@no_inline
def callback_case(name: String, X: nm.NDArray[DType.float64], keep: Bool) raises:
    var model=SimpleImputer[DType.float64, Minimum](strategy="callable", keep_empty_features=keep)
    emit(name + "o", model.fit_transform(X))
    emit(name + "s", model.statistics_)

@no_inline
def ownership_case[dtype: DType](name: String, mut X: nm.NDArray[dtype]) raises:
    var model = SimpleImputer[dtype](
        strategy="constant", fill_value=SIMD[dtype, 1](7), copy=False,
        missing_values=MissingValue[dtype](-1),
    )
    comptime assert type_of(model.statistics_) == nm.NDArray[DType.float64]
    var fitted = model.fit_transform(X)
    var replay = model.transform(X)
    emit(name + "s", model.statistics_)
    emit(name + "f", fitted)
    emit(name + "r", replay)
    assert_equal(X.unsafe_get(0), -1)
    fitted.unsafe_set(0, 19)
    replay.unsafe_set(1, 21)
    assert_equal(X.unsafe_get(0), -1)
    assert_equal(X.unsafe_get(1), 2)
    assert_equal(replay.unsafe_get(0), 7)
    assert_equal(fitted.unsafe_get(1), 2)

@no_inline
def inverse_edge_case[dtype: DType](name: String, X: nm.NDArray[dtype], query: nm.NDArray[dtype]) raises:
    var model = SimpleImputer[dtype](add_indicator=True)
    _ = model.fit(X)
    var output = model.transform(query)
    emit(name, output)
    with assert_raises(contains="ValueError: inverse indicator layout exhausts original features"):
        _ = model.inverse_transform(output)
''']
expected={}
cases=0

def scalar(x):
    if np.isnan(x): return 'Float64("nan")'
    if np.isposinf(x): return 'Float64("inf")'
    if np.isneginf(x): return 'Float64("-inf")'
    return f'Float64({float(x)!r})'

def array(name,x,dtype):
    lines.append(f'    var {name} = nm.zeros[DType.{dtype}](nm.Shape({x.shape[0]}, {x.shape[1]}))')
    for i,v in enumerate(x.flat): lines.append(f'    {name}.unsafe_set({i}, {scalar(v)}.cast[DType.{dtype}]())')

def record(name,value,dtype):
    expected[name]=(None if value is None else np.asarray(value,dtype=float).reshape(1,-1) if np.asarray(value).ndim==1 else np.asarray(value,dtype=float),3e-6 if dtype=='float32' else 3e-13)

rng=np.random.default_rng(278)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for dtype in ('float32','float64'):
        for strategy in ('mean','median','most_frequent','constant'):
            for keep in (False,True):
                for indicator in (False,True):
                    for marker in (np.nan,-1.0):
                        X=rng.integers(1,9,size=(7,4)).astype(dtype)
                        X[rng.random(X.shape)<.3]=marker
                        X[:,2]=marker
                        X[:,3]=2
                        T=X[:3].copy();T[0,3]=marker
                        model=oracle.SimpleImputer(strategy=strategy,missing_values=marker,keep_empty_features=keep,add_indicator=indicator,fill_value=7).fit(X)
                        output=model.transform(T)
                        name=f'c{cases}'
                        record(name+'s',model.statistics_,dtype);record(name+'o',output,dtype)
                        if indicator:
                            try: inverse=model.inverse_transform(output)
                            except IndexError: inverse=None
                            record(name+'i',inverse,dtype)
                        lines.extend(['@no_inline',f'def case{cases}() raises:'])
                        array('X',X,dtype);array('T',T,dtype)
                        lines.append(f'    exercise("{name}", X, T, "{strategy}", {scalar(marker)}.cast[DType.{dtype}](), {keep}, {indicator})')
                        cases+=1
        for features in ('all','missing-only'):
            for error_on_new in (False,True):
                for sparse in (False, 'auto'):
                    X=np.array([[1,np.nan,np.nan],[np.nan,2,np.nan]],dtype=dtype)
                    output=oracle.MissingIndicator(features=features,error_on_new=error_on_new,sparse=sparse).fit_transform(X)
                    assert isinstance(output, np.ndarray) and output.dtype == np.bool_
                    name=f'c{cases}';record(name,output,dtype)
                    lines.extend(['@no_inline',f'def case{cases}() raises:'])
                    array('X',X,dtype)
                    sparse_literal = '"auto"' if sparse == 'auto' else 'False'
                    lines.append(f'    indicate("{name}", X, "{features}", {error_on_new}, {sparse_literal})')
                    cases+=1
    for dtype in ('float32', 'float64'):
        X = np.array([[-1], [5]], dtype=dtype)
        model = oracle.SimpleImputer(strategy='constant', fill_value=9).fit(X)
        model.set_params(missing_values=-1, add_indicator=True)
        name = f'c{cases}'
        record(name+'marker', model.fit_transform(X), dtype)
        model.set_params(fill_value=7, add_indicator=False)
        record(name+'fill', model.fit_transform(X), dtype)
        model.set_params(fill_value=None)
        record(name+'default', model.fit_transform(X), dtype)
        indicator = oracle.MissingIndicator().fit(X)
        indicator.set_params(missing_values=5, sparse='auto')
        record(name+'indicator', indicator.fit_transform(X), dtype)
        lines.extend(['@no_inline', f'def case{cases}() raises:'])
        array('X', X, dtype)
        lines.append(f'    update_case("{name}", X)')
        cases += 1
    for strategy in ('mean','median','callable'):
        for keep in (False,True):
            X=np.array([[1e308,np.nan],[1e308,np.nan],[np.nan,np.nan]])
            callback=lambda a: np.min(a) if len(a) else 17
            model=oracle.SimpleImputer(strategy=callback if strategy=='callable' else strategy,keep_empty_features=keep)
            output=model.fit_transform(X)
            name=f'c{cases}';record(name+'s',model.statistics_,'float64');record(name+'o',output,'float64')
            lines.extend(['@no_inline',f'def case{cases}() raises:'])
            array('X',X,'float64')
            if strategy=='callable': lines.append(f'    callback_case("{name}", X, {keep})')
            else: lines.append(f'    exercise("{name}", X, X, "{strategy}", Float64("nan"), {keep}, False)')
            cases+=1

for dtype in ('float32', 'float64'):
    X = np.array([[-1, 2], [3, 4]], dtype=dtype)
    reference = oracle.SimpleImputer(strategy='constant', fill_value=7, missing_values=-1, copy=False)
    fit_input, replay_input = X.copy(), X.copy()
    fitted = reference.fit_transform(fit_input)
    replay = reference.transform(replay_input)
    assert reference.statistics_.dtype == np.dtype(object)
    for data, output in ((fit_input, fitted), (replay_input, replay)):
        assert np.shares_memory(data, output) and data[0, 0] == 7
        assert output.dtype == X.dtype
    name = f'c{cases}'
    record(name + 's', reference.statistics_, dtype)
    record(name + 'f', fitted, dtype)
    record(name + 'r', replay, dtype)
    lines.extend(['@no_inline', f'def case{cases}() raises:'])
    array('X', X, dtype)
    lines.append(f'    ownership_case("{name}", X)')
    cases += 1

    # Column 0 is dropped. Column 1 was partially observed in fit, but all
    # query rows flag it as missing, exhausting the inverse traversal.
    X = np.array([[np.nan, np.nan], [np.nan, 4]], dtype=dtype)
    query = np.full((1, 2), np.nan, dtype=dtype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        reference = oracle.SimpleImputer(add_indicator=True).fit(X)
        assert not caught
        output = reference.transform(query)
    assert len(caught) == 1 and caught[0].category is UserWarning
    assert 'Skipping features without any observed values: [0]' in str(caught[0].message)
    try:
        reference.inverse_transform(output)
    except IndexError:
        pass
    else:
        raise AssertionError('Pinned inverse unexpectedly accepted an entirely flagged retained column')
    name = f'c{cases}'
    record(name, output, dtype)
    lines.extend(['@no_inline', f'def case{cases}() raises:'])
    array('X', X, dtype)
    array('query', query, dtype)
    lines.append(f'    inverse_edge_case("{name}", X, query)')
    cases += 1

lines.append('def main() raises:')
for i in range(cases): lines.append(f'    case{i}()')
source=BUILDS/'parity_cases.mojo';source.write_text('\n'.join(lines)+'\n')
compile(source,BUILDS/'parity_cases')
result=run(BUILDS/'parity_cases',capture_output=True,text=True)
(BUILDS/'parity-output.txt').write_text(result.stdout)
output=iter(result.stdout.splitlines())
seen=set()
for line in output:
    header=line.split()
    assert header[0] in ('R','E'),line
    name=header[1]
    assert name in expected and name not in seen,name
    seen.add(name)
    wanted,tolerance=expected[name]
    if header[0]=='E':
        assert wanted is None,name
    else:
        assert wanted is not None,name
        rows,cols=map(int,header[2:])
        values=np.array([float(next(output)) for _ in range(rows*cols)]).reshape(rows,cols)
        assert values.shape==wanted.shape,(name,values.shape,wanted.shape)
        np.testing.assert_allclose(values,wanted,rtol=tolerance,atol=tolerance,equal_nan=True,err_msg=name)
assert seen==set(expected),set(expected)-seen
report=dict(cases=cases,records=len(seen),seed=278,float32_rtol=3e-6,float64_rtol=3e-13,scope='floating strategies, callbacks, indicators, inverse and overflow; not full dense acceptance')
(ROOT/'docs/private/impute.simple/parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'PASS: {cases} pinned sklearn differential cases, {len(seen)} result records')
