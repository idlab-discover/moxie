"""Input-precision reductions and test-only scale composition against pinned AST."""
import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import warnings
import numpy as np
from oracle import SKLEARN, PIN, update
ROOT = Path(__file__).resolve().parents[1]

def check_array(x, *, copy, **kwargs):
    assert x.ndim == 2 and x.dtype in (np.float32, np.float64)
    assert not np.isinf(x).any()
    return x.copy(order='K') if copy else x

source = SKLEARN/'sklearn/preprocessing/_data.py'
names = {'scale', '_handle_zeros_in_scale'}
nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
assert {n.name for n in nodes} == names
for node in nodes: node.decorator_list = []
ns = dict(np=np, warnings=warnings, check_array=check_array, FLOAT_DTYPES=(np.float64, np.float32),
          sparse=SimpleNamespace(issparse=lambda x: False), get_namespace=lambda x: (np, False))
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
for name in names: assert ns[name].__code__.co_filename == str(source)

records = []
def probe(raw, axis, layout=0, center=True, scaling=True):
    n, p = raw.shape
    # Reproduce the actual signed/gapped strides. np.nanmean copies in K order.
    storage = np.zeros(3*n*p+20, dtype=raw.dtype)
    rs, cs, offset = p, 1, 0
    if layout == 1: rs, cs, offset = 3*p, 2, 3
    elif layout == 2: rs, cs, offset = -3*p, -2, 3+(n-1)*3*p+(p-1)*2
    elif layout == 3: rs, cs = 1, n
    elif layout == 4: rs, cs, offset = -1, -n, n*p-1
    x = np.ndarray((n,p),dtype=raw.dtype,buffer=storage,offset=offset*raw.itemsize,
                   strides=(rs*raw.itemsize,cs*raw.itemsize))
    x[:] = raw
    args = [ROOT/'builds/evaluate_input_precision', 'f32' if x.dtype==np.float32 else 'f64',
            n,p,axis,layout,int(center),int(scaling),*x.ravel()]
    out = subprocess.check_output(list(map(str,args)),text=True)
    got = {}
    for line in out.splitlines():
        key, value = line.split(); got.setdefault(key,[]).append(float(value))
    context = f'{x.dtype}, {n}x{p}, axis={axis}, layout={layout}, center={center}, scaling={scaling}'
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        expected = {'mean': np.nanmean(x,axis), 'stddev':np.nanstd(x,axis),
                    'count':np.sum(~np.isnan(x),axis)}
        # Capture reduction calls inside the original scale body for corrections.
        calls=[]
        def traced_mean(a, axis):
            value=np.nanmean(a,axis); calls.append(value.copy()); return value
        proxy=SimpleNamespace(**{name:getattr(np,name) for name in ('asarray','nanstd','rollaxis','allclose','isscalar')},nanmean=traced_mean)
        ns['np']=proxy
        try:
            y=ns['scale'](x,axis=axis,with_mean=center,with_std=scaling,copy=False)
        finally: ns['np']=np
        expected['output']=y.ravel()
        if center:
            expected['mean1']=calls[1]
            expected['correct1']=[not np.allclose(calls[1],0)]
            if scaling:
                expected['mean2']=calls[2]
                expected['correct2']=[not np.allclose(calls[2],0)]
        for key, value in expected.items():
            # Input-precision reductions require exact rounded results: loose
            # tolerances could conceal the cancellation/correction blocker.
            if key != 'output':
                np.testing.assert_array_equal(np.asarray(got[key],dtype=value.dtype if hasattr(value,'dtype') else None),value,
                                              err_msg=f'{key}: {context}')
            else:
                np.testing.assert_allclose(got[key],value,rtol=2e-7 if raw.dtype==np.float32 else 3e-15,atol=0,equal_nan=True,
                                           err_msg=f'{key}: {context}')
    records.append(dict(dtype=str(raw.dtype),shape=raw.shape,axis=axis,layout=layout,center=center,scaling=scaling,
                        correction1=got.get('correct1'),correction2=got.get('correct2')))

rng=np.random.default_rng(6291)
with warnings.catch_warnings(), np.errstate(all='ignore'):
    warnings.simplefilter('ignore')
    # Old engine remains correct but insufficient; prove identical summaries.
    for dtype in (np.float32,np.float64):
        a=np.array([100000000,100000008,100000016],dtype=dtype)[:,None]
        b=a[[0,2,1]]
        for u,v in zip(update(a,0.,0.,0.),update(b,0.,0.,0.)): np.testing.assert_array_equal(u,v)
        for raw in (a,b):
            for axis in (0,1):
                x=raw if axis==0 else raw.T
                for layout in range(5):
                    for center,scaling in ((False,False),(True,False),(False,True),(True,True)):
                        probe(x,axis,layout,center,scaling)
        for length in (1,2,3,4,7,8,9,15,16,31,32,33,127,128,129,255,256,257,8191,8192,8193,16385):
            for axis in (0,1):
                # Different layouts choose serial vs pairwise with identical data.
                raw=rng.normal(size=(length,3)).astype(dtype)
                raw[:,0]=dtype(1e8 if dtype==np.float32 else 1e16)
                raw[1::2,0]*=-1
                raw[::7,1]=np.nan
                raw[:,2]=np.nan
                x=raw if axis==0 else raw.T
                for layout in range(5): probe(x,axis,layout)
        extras=[[[1,np.nan,3]],[[np.nan,np.nan,np.nan]],[[1,1,1]],[[0.,-0.,0.]],
                [[3e38,3e38,3e38]],[[-3e38,3e38,0]],[[1e-30,2e-30,3e-30]],[[1e-20,2e-20,3e-20]],
                [[1e-160,2e-160,3e-160]] if dtype==np.float64 else [[1e-38,2e-38,3e-38]],
                [[1e-320,2e-320,3e-320]] if dtype==np.float64 else [[1e-44,-1e-44,0]],
                np.full((1,10),np.log(1e-5)),np.full((1,10),1e100 if dtype==np.float64 else 1e30),
                np.full((1,10),1e-100),[[1e308,-1e308,1e308]] if dtype==np.float64 else [[1e-44,2e-44,3e-44]]]
        for raw in extras:
            for axis in (0,1):
                x=np.asarray(raw,dtype=dtype)
                if axis==0: x=x.T
                for layout in range(5):
                    for center,scaling in ((False,True),(True,False),(True,True)):
                        probe(x,axis,layout,center,scaling)

assert any(r['correction1']==[1.] for r in records)
assert any(r['correction2']==[1.] for r in records)
(ROOT/'benchmarks').mkdir(exist_ok=True)
(ROOT/'benchmarks/functional-parity.json').write_text(json.dumps(dict(reference=PIN,numpy=np.__version__,cases=records),indent=2)+'\n')
print(f'PASS input-precision parity: {len(records)} cases, exact reductions and both scale correction passes')
