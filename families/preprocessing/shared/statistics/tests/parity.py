"""Differential engine validation: pinned source AST, not an installed estimator."""
from pathlib import Path
import json, subprocess, warnings, itertools
import numpy as np
from oracle import update, constant, namespace, PIN
ROOT=Path(__file__).resolve().parents[1]
records=[]
blocker=[]
def probe(x, weights=None, old=None, tile=256, layout=0, track=True, q=(25.,75.), scalar=None):
    if layout == 3:
        x = np.asfortranarray(x)
    n,p=x.shape
    if old is None: old=(np.zeros(p),np.zeros(p),np.zeros(p)) # count, mean, variance
    mode=0 if weights is None and scalar is None else 1 if scalar is None else 2
    w=np.ones(n,dtype=x.dtype) if weights is None else np.asarray(weights,dtype=x.dtype)
    args=[ROOT/'builds/evaluate','f32' if x.dtype==np.float32 else 'f64',n,p,mode,tile,int(track),layout,*q,1 if scalar is None else scalar]
    args+=list(x.ravel())+list(w)+[v for a in old for v in a]
    output=subprocess.check_output([str(v) for v in args],text=True)
    got={}
    for line in output.splitlines():
        k,v=line.split();got.setdefault(k,[]).append(float(v))
    weights_ref=None if mode==0 else w if mode==1 else np.full(n,scalar,dtype=x.dtype)
    if mode:
        # Execute reference validation too, proving accepted signed / nonzero weights.
        namespace['_check_sample_weight'](scalar if mode==2 else weights_ref,x,dtype=weights_ref.dtype)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m,v,c=update(x,old[1],old[2] if track else None,old[0],sample_weight=weights_ref)
        expected={'count':c,'mean':m,'min':np.nanmin(x,axis=0),'max':np.nanmax(x,axis=0),
                  'abs':np.nanmax(np.abs(x),axis=0),'median':np.nanmedian(x,axis=0),
                  'lower':np.nanpercentile(x,np.array(q),axis=0)[0],'upper':np.nanpercentile(x,np.array(q),axis=0)[1]}
        expected['sum']=np.nansum(x.astype('f8'),axis=0) if mode==0 else weights_ref.astype('f8') @ np.where(np.isnan(x),0,x).astype('f8')
        expected['variance']=v if track else old[2]
        if track: expected['constant']=constant(v,m,c)
        expected['scalar']=[np.max(c)==np.min(c)]
        for k,a in expected.items():
            if k == 'count':
                np.testing.assert_array_equal(got[k], a, err_msg=f'count: dtype={x.dtype}, n={n}, p={p}, layout={layout}')
            # F64 serial reductions versus NumPy pairwise/BLAS: tight relative
            # tolerance, absolute 2e-10 for cancellation around zero.
            np.testing.assert_allclose(got[k],a,rtol=3e-11,atol=2e-10,equal_nan=True,err_msg=f'{k}, dtype={x.dtype}, tile={tile}, layout={layout}, old={old}, X={x}, weights={weights_ref}')
    records.append(dict(n=n,p=p,dtype=str(x.dtype),mode=mode,tile=tile,layout=layout,track=track))
    return np.array(got['count']),np.array(got['mean']),np.array(got['variance'])

# Regression: an eight-row signed-weight batch has pairwise effective count 0,
# although serial accumulation produces 1. This changes IEEE variance classes.
for dtype in (np.float32, np.float64):
    x = np.arange(1, 9, dtype=dtype)[:, None]
    weights = np.array([1e16, 1, -1e16, 1, 0, 0, 0, 0], dtype=dtype)
    for tile in (1, 3, 256):
        got = probe(x, weights, old=(np.ones(1), np.zeros(1), np.zeros(1)), tile=tile)
        assert got[0][0] == 1 and np.isnan(got[2][0])

# Count reduction boundaries and layouts. Include variance around the pairwise
# leaf boundaries, and isolate count/sum for the large widening-buffer probes.
for dtype in (np.float32, np.float64):
    for n in (7, 8, 9, 15, 16, 127, 128, 129, 255, 256, 257, 8191, 8192, 8193, 16385):
        x = np.arange(1, n + 1, dtype=dtype)[:, None]
        weights = np.zeros(n, dtype=dtype)
        weights[:4] = [1e16, 1, -1e16, 1]
        for layout in (0, 1, 2, 3):
            for tile in (1, 3, 256):
                probe(x, weights, old=(np.ones(1), np.zeros(1), np.zeros(1)), tile=tile, layout=layout, track=n <= 257)
        if n <= 257:
            # Broadcast mask has a different reduction axis order in C/F order.
            multi = np.repeat(x, 2, axis=1)
            multi[4::5, 1] = np.nan
            for layout in (0, 1, 2, 3):
                probe(multi, weights, old=(np.ones(2), np.zeros(2), np.zeros(2)), layout=layout, track=False)

    # Widened reduction buffer boundaries: separated large weights among ones
    # distinguish whole-array pairwise reduction from 8192-element chunks.
    for n in (8200, 16385, 20000):
        x = np.arange(1, n + 1, dtype=dtype)[:, None]
        weights = np.ones(n, dtype=dtype)
        weights[17], weights[n - 19] = 1e16, -1e16
        probe(x, weights, track=False)

    # Zero effective count must preserve exceptional classifications and old
    # NaNs; memory tiles and semantic partial-fit partitions are separate.
    x = np.arange(1, 17, dtype=dtype)[:, None]
    weights = np.zeros(16, dtype=dtype)
    weights[:4] = [1e16, 1, -1e16, 1]
    for missing in (False, True):
        if missing:
            x[6] = np.nan
        for old in ((1, 0, 0), (0, np.nan, np.nan), (1, np.nan, 0)):
            for tile in (1, 3, 256):
                probe(x, weights, tuple(np.array([v], dtype='f8') for v in old), tile=tile)
    for split in (4, 7, 8, 9):
        # Give each partition a nonzero weight, as required by the validator.
        weights[-1] = 1
        state = probe(x[:split], weights[:split], track=False)
        probe(x[split:], weights[split:], state, track=False)

# Original impossibility counterexample, both isolated summaries and old state.
for vals in ([1,2],[2,4]):
    x=np.array(vals,dtype='f8')[:,None]
    isolated=probe(x,[1,-1])
    assert isolated[0][0]==0 and isolated[1][0]==-np.inf and np.isnan(isolated[2][0])
    for tile in (1,2,7):
        got=probe(x,[1,-1],old=(np.ones(1),np.zeros(1),np.zeros(1)),tile=tile)
        assert got[1][0]==-vals[0]
        blocker.append(dict(input=vals,weights=[1,-1],tile=tile,updated_mean=got[1].tolist(),count=got[0].tolist(),variance_is_nan=bool(np.isnan(got[2][0]))))

rng=np.random.default_rng(291)
for dtype in (np.float32,np.float64):
    cases=[np.array([[1.]]),np.array([[1,2],[3,4]]),np.array([[np.nan,1,3],[np.nan,np.nan,4],[np.nan,2,5]]),
           rng.normal(size=(37,5)),np.full((19,3),1e7),1e7+rng.normal(scale=1e-5,size=(61,3)),
           np.array([[1e-30,0,1e30],[2e-30,0,-1e30],[3e-30,0,0]]),
           np.array([[3e38,3e38],[-3e38,3e38]]),np.array([[0.,-0.],[-0.,0.]])]
    for raw in cases:
        x=raw.astype(dtype);n,p=x.shape
        weights=rng.uniform(.1,2,n).astype(dtype)
        signed=weights.copy();signed[::2]*=-1
        for layout,tile in ((0,256),(1,1),(2,7)):
            for w in (None,weights,signed):
                probe(x,w,layout=layout,tile=tile)
            probe(x,weights,layout=layout,tile=tile,track=False)
        probe(x,scalar=-2.,q=(0,100))
    # Semantic partitions (not memory tiles), including NaN-poisoned old states.
    x=rng.normal(size=(31,4)).astype(dtype);x[::3,1]=np.nan;x[:4,2]=np.nan
    w=rng.uniform(.2,2,len(x)).astype(dtype)
    for weights in (None,w):
        for partition in (1,4,13):
            state=None
            for i in range(0,len(x),partition):
                state=probe(x[i:i+partition],None if weights is None else weights[i:i+partition],state,tile=2)
    for counts,means,variances in [(0,np.nan,np.nan),(0,2,np.nan),(1,np.nan,0),(-1,0,-1),(1,0,0)]:
        old=tuple(np.full(3,z,dtype='f8') for z in (counts,means,variances))
        probe(np.array([[1,2,np.nan],[2,4,np.nan]],dtype=dtype),[1,-1],old)
    # Signed cancellation in a tile must not poison the complete batch.
    x=np.array([[1,2],[2,4],[5,8],[7,11]],dtype=dtype)
    for tile in (1,2,3,4,1024): probe(x,[1,-1,2,3],tile=tile)
    for scalar in (np.nan,np.inf,-np.inf,0.123456789):
        probe(np.array([[1.]],dtype=dtype),scalar=scalar)
        probe(np.array([[np.nan]],dtype=dtype),scalar=scalar)
        probe(x,scalar=scalar)
        y=x.copy();y[0,0]=np.nan;y[:,1]=np.nan
        probe(y,scalar=scalar)
    # Exact quantile order statistics, duplicates, tails, singleton and q==q.
    for n in (2,3,4,5,7,16,33,100):
        x=rng.integers(-10,11,(n,4)).astype(dtype);x[0,0]=np.nan
        for q in ((0,100),(10,90),(50,50),(23.7,84.2)):
            probe(x,q=q)

(ROOT/'benchmarks').mkdir(exist_ok=True)
(ROOT/'benchmarks/parity.json').write_text(json.dumps({'reference':PIN,'numpy':np.__version__,'cases':records,'blocker':blocker},indent=2)+'\n')
print(f'PASS parity: {len(records)} pinned-reference comparisons; original blocker distinguished')
