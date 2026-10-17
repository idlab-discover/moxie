"""Differential draws, fitted state, maps and actual kernel approximation."""
import json,subprocess
import numpy as np
from oracle import load,ROOT
oracle=load();count=0
worst={'32':0.,'64':0.}
conditioned_worst={'32':0.,'64':0.}
ill_conditioned=[]
kernel_errors=[]
def compare(kind,dtype,x,m=11,seed=42,value=1.,scale=False):
    global count
    x=np.asarray(x,dtype=dtype);n,p=x.shape
    bits='32' if dtype==np.float32 else '64'
    args=[ROOT/'builds/tests/kernel_approximation/fourier/evaluate',kind,bits,str(n),str(p),str(m),str(seed),str(value),'scale' if scale else 'numeric',*map(str,x.astype(float).ravel())]
    out=subprocess.check_output(args,text=True)
    got=np.array(out.splitlines(),dtype=float)
    rng=np.random.RandomState(seed)
    cls=oracle['RBFSampler' if kind=='rbf' else 'SkewedChi2Sampler']
    model=cls(**({'gamma':'scale' if scale else value} if kind=='rbf' else {'skewedness':value}),n_components=m,random_state=rng)
    model.fit(x)
    z=model.transform(x)
    want=np.concatenate([model.random_weights_.ravel().astype(float),model.random_offset_.astype(float),z.ravel().astype(float),[float(getattr(model,'_gamma',0)),rng.random_sample(),rng.standard_normal()]])
    tol=4e-5 if bits=='32' else 3e-8
    label=str((kind,bits,n,p,m,seed,value,scale))
    cut=p*m+m
    np.testing.assert_allclose(got[:cut],want[:cut],rtol=tol,atol=tol,err_msg=label)
    np.testing.assert_allclose(got[-3:],want[-3:],rtol=tol,atol=tol,err_msg=label)
    transformed=x if kind=='rbf' else np.log(x+value)
    # Cosine is 1-Lipschitz: propagate the standard dot-product rounding bound.
    phase_bound=4*p*np.finfo(dtype).eps*(abs(transformed.astype(float))@abs(model.random_weights_.astype(float))+abs(model.random_offset_))
    bound=tol+phase_bound*np.sqrt(2/m)
    assert np.all(abs(got[cut:-3].reshape(n,m)-z)<=bound), (label, np.max(abs(got[cut:-3].reshape(n,m)-z)),np.max(bound))
    assert got[-2]==want[-2], 'draw consumption'
    worst[bits]=max(worst[bits],float(np.nanmax(abs(got-want))))
    if np.max(bound)<.01:
        conditioned_worst[bits]=max(conditioned_worst[bits],float(np.nanmax(abs(got-want))))
    else:
        ill_conditioned.append({'case':label,'max_absolute_difference':float(np.max(abs(got[cut:-3].reshape(n,m)-z))),'max_rounding_bound':float(np.max(bound))})
    count+=1
    return got[p*m+m:-3].reshape(n,m)
for dtype in [np.float32,np.float64]:
    for kind in ['rbf','skew']:
        for seed in [0,1,42,4294967295]:
            for n,p,m in [(1,1,1),(2,1,3),(3,4,7),(5,3,100)]:
                x=np.arange(n*p).reshape(n,p)/10
                compare(kind,dtype,x,m,seed)
        for value in ([0.,.03,10.] if kind=='rbf' else [-2.,0.,.03,10.]):
            compare(kind,dtype,[[3,4],[5,6]],value=value)
    for x in [[[0],[1]],[[3,3],[3,3]],[[1e4,1e4+1],[1e4+2,1e4+3]],[[1e8,1e8+8],[1e8+16,1e8+24]],[[1e-12,2e-12],[3e-12,4e-12]]]:
        compare('rbf',dtype,x,scale=True)
    compare('skew',dtype,[[-.99,0],[1,3]],value=1.)
    compare('skew',dtype,[[-1+np.finfo(dtype).eps,0],[1e-30,1e30]],value=1.)
    x=np.random.RandomState(8).uniform(0,.2,(15,3)).astype(dtype)
    for kind in ['rbf','skew']:
        z=compare(kind,dtype,x,m=3000,value=.5)
        if kind=='rbf':exact=np.exp(-.5*((x[:,None]-x[None,:])**2).sum(axis=2))
        else:
            a=x[:,None]+.5;b=x[None,:]+.5
            exact=np.prod(2*np.sqrt(a*b)/(a+b),axis=2)
        error=float(np.max(abs(z@z.T-exact)))
        assert error<.08
        kernel_errors.append(dict(kind=kind,dtype=np.dtype(dtype).name,max_gram_error=error))
# Native negative capability checks complement original method-body comparisons.
for kind,bits,n,p,m,seed,value,mode,values in [
    ('rbf','64',1,1,0,42,'1','numeric',['0']),
    ('rbf','64',1,1,1,42,'-1','numeric',['0']),
    ('rbf','64',1,1,1,42,'nan','numeric',['0']),
    ('skew','64',1,1,1,42,'inf','numeric',['0']),
    ('skew','64',1,1,1,42,'1','numeric',['-1']),
    ('rbf','32',1,1,1,4294967296,'1','numeric',['0']),
    ('rbf','64',1,1,1,42,'1','numeric',['inf'])]:
    result=subprocess.run([ROOT/'builds/tests/kernel_approximation/fourier/evaluate',kind,bits,str(n),str(p),str(m),str(seed),value,mode,*values],capture_output=True,text=True)
    assert result.returncode and 'ValueError' in result.stderr+result.stdout, (kind,result.stdout,result.stderr)
report={'cases':count,'max_absolute_error':worst,'conditioned_max_absolute_error':conditioned_worst,'ill_conditioned_phase_diagnostics':ill_conditioned,'kernel_errors':kernel_errors,'oracle_revision':__import__('oracle').REV}
(ROOT/'docs/private/fourier/parity.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS',report)
