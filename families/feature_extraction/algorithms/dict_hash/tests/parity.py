"""Generate native assertions from the pinned sklearn encoder, then execute them."""
from pathlib import Path
import hashlib,json,re,sys,warnings
import numpy as np
from oracle import load,ROOT,REV
sys.path.insert(0,str(ROOT/'tools'))
from check import compile,run
DV=load()
def string(s):return json.dumps(s,ensure_ascii=False)
def key(k):return f'FeatureKey({string(k) if isinstance(k,str) else k})'
def val(v):
    if v is None:return 'FeatureValue()'
    if isinstance(v,str):return f'FeatureValue({string(v)})'
    if isinstance(v,list) and not v:return 'FeatureValue(List[String]())'
    if isinstance(v,list) and len(v)>5 and len(set(v))==1:return f'FeatureValue(List[String](length={len(v)}, fill={string(v[0])}))'
    if isinstance(v,list):return 'FeatureValue(List[String](['+','.join(map(string,v))+']))'
    if isinstance(v,(bool,int,np.integer)):
        return f'FeatureValue({"UInt64" if v>2**63-1 else "Int64"}({int(v)}))'
    if not np.isfinite(v):return f'FeatureValue(Float64({string(str(v))}))'
    return f'FeatureValue(Float64({v!r}))'
def batch(name,rows):
    s=[f'    var {name} = FeatureBatch()']
    for i,row in enumerate(rows):
        s.append(f'    var {name}{i} = FeatureRow()')
        for k,v in row.items():s.append(f'    {name}{i}.append(FeatureEntry({key(k)}, {val(v)}))')
        s.append(f'    {name}.append({name}{i}^)')
    return s

def matrix(name,x,dt):
    s=[f'    assert_equal({name}.shape[0], {x.shape[0]})',f'    assert_equal({name}.shape[1], {x.shape[1]})']
    for i,v in enumerate(x.flat):
        if np.isnan(v):s.append(f'    assert_true(isnan({name}.unsafe_get({i})))')
        elif np.isinf(v):s.append(f'    assert_equal({name}.unsafe_get({i}), Scalar[DType.{dt}](Float64({string(str(v))})))')
        else:s.append(f'    assert_equal({name}.unsafe_get({i}), Scalar[DType.{dt}]({str(bool(v)) if dt=="bool" else repr(v.item())}))')
    return s

def metadata(model,oracle):
    s=[f'    assert_equal(len({model}.feature_names_), {len(oracle.feature_names_)})']
    for i,k in enumerate(oracle.feature_names_):
        s += [f'    assert_true({model}.feature_names_[{i}] == {key(k)})',f'    assert_equal({model}.vocabulary_.column({key(k)}), {oracle.vocabulary_[k]})',f'    assert_equal({model}.get_feature_names_out()[{i}], {string(str(k))})']
    return s
cases=[
 ([{'foo':1,'bar':3},{'bar':4,'baz':2},{'bar':1,'quux':1,'quuux':2}], [{'foo':2,'unknown':8},{}], '='),
 ([{'version':['1','2','1'],'ham':2},{'version':'2','spam':0.3},{'version=3':True,'spam':-1}], [{'version':['3','?'],'ham':0}], '='),
 ([{'city':'東京','year':2003},{'city':'é','year':2011},{'city':'😀'},{'category':[]}],[{'city':'東京','other':'new'}],'='),
 ([{'a':['b','b'],'a=b':5},{'a=b':-2,'a':'b'}],[{'a':['b','b','b']}],'='),
 ([{'z':0,'a':0},{}],[{}, {'unknown':None}],''),
 ([{},{}],[{}],'='),
 ([{3:4,1:2},{-2:3,12:9}],[{1:5,2:1}],'='),
 ([{'z':'','a':['é','é','😀']}],[{'a':['é']}],'::'),
 ([{'a':None,'b':float('inf'),'c':float('-inf')}],[{'a':None}],'='),
 ([{'a':['b']*130,'a=b':120}],[{'a':['b']*260}],'='),
 ([{'v':2**63-1}],[{'v':2**63-1}],'='),
 ([{'v':2**64-1}],[{'v':2**64-1}],'='),
 ([{'v':-(2**63)}],[{'v':-(2**63)}],'='),
 ([{'a=b':16777216.0,'a':['b','b'],'y':-0.0}],[{'a':['b','b'],'a=b':16777216.0}],'='),
]
dtypes=['float32','float64','int8','int16','int32','int64','uint8','uint16','uint32','uint64','bool']
lines=['from std.testing import assert_equal, assert_true, assert_raises','from std.math import isnan','from numojo import zeros, Shape','from moxie.feature_extraction import DictVectorizer','from moxie.feature_extraction._dict_vectorizer import FeatureBatch, FeatureRow, FeatureEntry, FeatureKey, FeatureValue','from moxie.base import clone','']
records=[]
with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 for dt in dtypes:
  for sorting in [False,True]:
   for ci,(train,query,sep) in enumerate(cases):
    n=len(records);name=f'case_{n}'
    model=DV(dtype=getattr(np,dt),sparse=False,sort=sorting,separator=sep)
    lines += ['@no_inline', f'def {name}() raises:',f'    var v = DictVectorizer[DType.{dt}](sparse=False, sort={sorting}, separator={string(sep)})']+batch('train',train)+batch('query',query)
    try:x=model.fit_transform(train)
    except (ValueError,TypeError,OverflowError):
        lines += ['    with assert_raises():','        _ = v.fit_transform(train)','']
        records.append(dict(dtype=dt,sort=sorting,case=ci,result='conversion error'));continue
    lines += ['    var x = v.fit_transform(train)']+matrix('x',x,dt)+metadata('v',model)
    try:y=model.transform(query)
    except (ValueError,TypeError,OverflowError):
        lines += ['    with assert_raises():','        _ = v.transform(query)']
    else:
        lines += ['    var y = v.transform(query)']+matrix('y',y,dt)
    lines += ['    var cloned = clone(v)','    assert_true(not cloned.__sklearn_is_fitted__())','    cloned.fit(train)','    var xx = cloned.transform(train)']+matrix('xx',x,dt)
    if x.size and np.isfinite(x).all():
        inv=model.inverse_transform(x)
        lines+=['    var inverse = v.inverse_transform(x)',f'    assert_equal(len(inverse), {len(inv)})']
        for i,row in enumerate(inv):
            lines += [f'    assert_equal(len(inverse[{i}]), {len(row)})']
            for k,value in row.items():
                lines += [f'    assert_equal(inverse[{i}][{key(k)}], Scalar[DType.{dt}]({str(bool(value)) if dt=="bool" else repr(value.item())}))']
    if len(model.feature_names_)>1:
        indices=list(range(len(model.feature_names_)-1,-1,-2));model.restrict(indices,indices=True)
        lines += ['    v.restrict(List[Int](['+','.join(map(str,indices))+']), indices=True)']+metadata('v',model)
        lines += ['    var restricted = v.transform(train)']+matrix('restricted',model.transform(train),dt)
    lines+=['    v.fit(FeatureBatch())','    assert_equal(v.vocabulary_.size(),0)','']
    records.append(dict(dtype=dt,sort=sorting,case=ci,result='matrix/lifecycle'))
# Heterogeneous raw keys are legal with sort=False, mixed sorting raises.
lines += ['def mixed() raises:']+batch('rows',[{1:2,'1':3,2:'cat'}])+['    var v = DictVectorizer(sparse=False, sort=False)','    var x = v.fit_transform(rows)','    assert_equal(x.shape[1],3)','    assert_equal(x.unsafe_get(0),2)','    assert_equal(x.unsafe_get(1),3)','    v.set_params(sort=True)','    with assert_raises():','        v.fit(rows)','']
lines += ['def restrictions() raises:']+batch('data',[{'a':1,'b':2,'c':3}])+['    var v = DictVectorizer(sparse=False)']
for ri,(support,is_indices) in enumerate([([0,0],True),([0,1,0],True),([0,1,0,2],True),([-1,0],True),([0,1,0],False),([],False)]):
    oracle=DV(sparse=False);oracle.fit([{'a':1,'b':2,'c':3}]);oracle.restrict(support,indices=is_indices)
    literal='List[Int](['+','.join(map(str,support))+'])' if support else 'List[Int]()'
    lines += ['    v.fit(data)',f'    v.restrict({literal}, indices={is_indices})']+metadata('v',oracle)
    if any(index >= len(oracle.vocabulary_) for index in oracle.vocabulary_.values()):
        # Duplicate restrict can produce invalid CSR column metadata in sklearn.
        # Validate our safe failure; do not execute its unchecked toarray kernel.
        lines+=['    with assert_raises():','        _ = v.transform(data)']
        continue
    try: x=oracle.transform([{'a':1,'b':2,'c':3}])
    except ValueError: lines+=['    with assert_raises():','        _ = v.transform(data)']
    else: lines += [f'    var selected{ri} = v.transform(data)']+matrix(f'selected{ri}',x,'float64')
lines += ['']
array_cases = []
def scalar(value, dt):
    if dt == 'bool':
        return f'Scalar[DType.bool]({bool(value)})'
    if np.isnan(value):
        return f'Scalar[DType.{dt}](Float64("nan"))'
    return f'Scalar[DType.{dt}]({value.item()!r})'
for model_dt in ['float64', 'float32', 'int8']:
    for input_dt in dtypes:
        name = f'array_case_{len(array_cases)}'
        array_cases.append(name)
        oracle = DV(dtype=getattr(np,model_dt),sparse=False)
        oracle.fit([{'a':1,'b':2,'c':3}])
        values = np.array([[1,0,3],[0,2,0]],dtype=input_dt)
        if input_dt == 'uint64': values[0,0] = 2**64-1
        if input_dt == 'int64': values[0,0] = -(2**63)
        if input_dt.startswith('float'): values[0,0] = 1.25
        lines += ['@no_inline',f'def {name}() raises:']+batch('data',[{'a':1,'b':2,'c':3}])
        lines += [f'    var v = DictVectorizer[DType.{model_dt}](sparse=False)',
                  '    _ = v.fit(data)',f'    var values = zeros[DType.{input_dt}](Shape(2,3))']
        for j,value in enumerate(values.flat):
            lines += [f'    values.unsafe_set({j},{scalar(value,input_dt)})']
        lines += [f'    var inverse: List[Dict[FeatureKey,Scalar[DType.{input_dt}]]] = v.inverse_transform(values)']
        for j,row in enumerate(oracle.inverse_transform(values)):
            lines += [f'    assert_equal(len(inverse[{j}]),{len(row)})']
            for k,value in row.items():
                lines += [f'    assert_equal(inverse[{j}][{key(k)}],{scalar(value,input_dt)})']
        lines += ['']
selections = [
    (np.array([True,False,True]),False),
    (np.array([True,False]),True),
    (np.array([-1,0],dtype='int16'),True),
    (np.array([2,0],dtype='uint64'),True),
    (np.array([0,2,0],dtype='int8'),False),
    (np.array([np.nan,0,-2.5]),False),
    (np.array([],dtype='int64'),True),
    (np.array([],dtype='float64'),True),
    (np.array([1.0]),True),
    (np.array([2**64-1],dtype='uint64'),True),
]
for support,indices in selections:
    name = f'array_case_{len(array_cases)}'
    array_cases.append(name)
    oracle = DV(sparse=False)
    oracle.fit([{'a':1,'b':2,'c':3}])
    dt = str(support.dtype)
    lines += ['@no_inline',f'def {name}() raises:']+batch('data',[{'a':1,'b':2,'c':3}])
    lines += ['    var v = DictVectorizer(sparse=False)','    _ = v.fit(data)',
              f'    var support = zeros[DType.{dt}](Shape({len(support)}))']
    for j,value in enumerate(support):
        lines += [f'    support.unsafe_set({j},{scalar(value,dt)})']
    try:
        assert oracle.restrict(support,indices=indices) is oracle
    except (TypeError,IndexError,OverflowError):
        lines += ['    with assert_raises():',f'        _ = v.restrict(support,indices={indices})']
    else:
        lines += [f'    var selected = v.restrict(support,indices={indices}).transform(data)']
        lines += metadata('v',oracle)+matrix('selected',oracle.transform([{'a':1,'b':2,'c':3}]),'float64')
    lines += ['']
lines += ['def main() raises:']+[f'    case_{i}()' for i in range(len(records))]+[f'    {name}()' for name in array_cases]+['    mixed()','    restrictions()',f'    print("PASS: {len(records)} encoder and {len(array_cases)} array API sklearn parity cases")']
source = ROOT/'builds/parity_generated.mojo'
source.write_text('\n'.join(lines)+'\n')
# Bound compiler memory without removing any generated case or assertion.
text = source.read_text()
functions = list(re.finditer(r"(?m)^(?:@no_inline\n)?def (\w+)\(", text))
assert functions[-1].group(1) == 'main'
entries = [f'case_{i}' for i in range(len(records))] + array_cases + ['mixed', 'restrictions']
blocks = {match.group(1): text[match.start():functions[i+1].start()]
          for i, match in enumerate(functions[:-1])}
assert len(blocks) == len(entries) and set(blocks) == set(entries)
preamble = text[:functions[0].start()]
batch_size = 28
for start in range(0, len(entries), batch_size):
    names = entries[start:start+batch_size]
    batch = ROOT/'builds'/f'parity_batch_{start//batch_size:02d}.mojo'
    batch.write_text(preamble + ''.join(blocks[name] for name in names)
                     + 'def main() raises:\n'
                     + ''.join(f'    {name}()\n' for name in names))
    compile(batch, batch.with_suffix(''))
    run(batch.with_suffix(''))
    print(f'PASS: dictionary parity cases {start+1}-{start+len(names)}', flush=True)
print(f'PASS: {len(records)} encoder and {len(array_cases)} array API sklearn parity cases')
report = ROOT/'docs/private/dict_vectorizer/parity.json'
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps(dict(reference=REV,cases=records,array_api_cases=len(array_cases),compiler_batch_size=batch_size,tolerance='exact equality after dtype conversion; NaN masks compared',numpy=np.__version__,generated_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),oracle='local unchanged class AST; decorator/base/check_array tooling adapters; no sklearn wheel'),indent=2)+'\n')
