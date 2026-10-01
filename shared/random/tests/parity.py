"""Differential sequences and complete MT state, never distribution-only parity."""
from pathlib import Path
import json, subprocess
import numpy as np
from oracle import load, ROOT
sample,Dummy,check=load()
from our_rand_r_oracle import sequence as xorshift_sequence

# Compare every output AND full state; using the masked output as the next
# state would appear correct on short low-bit-only examples.
xorshift_seeds = sorted({0, 1, 42, 1273642419, 2**31-1, 2**31, 2**32-1,
                        *(1 << bit for bit in range(32)),
                        *(2**32-1 ^ (1 << bit) for bit in range(32))})
for seed in xorshift_seeds:
    count = 4096
    out = subprocess.check_output([ROOT/'builds/draws', 'our_rand_r', str(seed), str(count)], text=True)
    got = np.array(out.splitlines(), dtype=np.uint32).reshape(count, 2)
    np.testing.assert_array_equal(got, xorshift_sequence(seed, count), err_msg=f'our_rand_r seed={seed}')
print(f'PASS our_rand_r: {len(xorshift_seeds)} seeds x {count} exact output/state pairs against pinned sklearn')
CASES=0

def probe(mode, seed, count=100, n=0, k=1, method='auto', p=()):
    out=subprocess.check_output([ROOT/'builds/draws',mode,str(seed),str(count),str(n),str(k),method,*[str(int(round(x*1000000))) for x in p]],text=True)
    values, state=out.split('STATE\n')
    state=state.splitlines()
    dtype = float if mode in ['uniform','normal','mixed','gamma','standard_gamma','multivariate','multivariate_factor'] else np.int64
    return np.array(values.splitlines(), dtype=dtype),(int(state[0]),int(state[1]),float(state[2]),np.array(state[3:],dtype=np.uint32))
def compare(mode,seed,count=100,n=0,k=1,method='auto',p=()):
    global CASES
    r=np.random.RandomState(seed)
    if mode=='raw': want=r.randint(0,2**32,size=count,dtype=np.uint32)
    elif mode=='uniform': want=r.uniform(n,k,size=count)
    elif mode=='normal': want=r.normal(n,k,size=count)
    elif mode=='gamma': want=r.gamma(n/1000000,k/1000000,size=count)
    elif mode=='standard_gamma': want=r.standard_gamma(n/1000000,size=count)
    elif mode=='integer': want=r.randint(n,k,size=count)
    elif mode=='multivariate':
        mean,cov=multivariate_inputs(n,k)
        factor,_=probe('multivariate_factor',seed,0,n,k)
        factor=factor.reshape(n,n)
        np.testing.assert_allclose(factor.T@factor,cov,rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(np.linalg.svd(factor,compute_uv=False)**2,
                                   np.linalg.svd(cov,compute_uv=False),rtol=2e-12,atol=2e-12)
        want=(r.standard_normal((count,n))@factor+mean).ravel()
    elif mode=='poisson': want=r.poisson(n/k,size=count)
    elif mode=='binomial': want=r.binomial(n,k/1000000,size=count)
    elif mode in ['choice','choice_weighted']:
        want=r.choice(n,size=count,replace=bool(k),p=p if mode=='choice_weighted' else None)
    elif mode in ['shuffle','permutation_array']:
        want=np.arange(n if k==-1 else n*k,dtype=np.int64)*7+3
        want=want.reshape((n,) if k==-1 else (n,k))
        if mode=='shuffle': r.shuffle(want)
        else: want=r.permutation(want)
        want=want.ravel()
    elif mode=='categorical': want=r.choice(len(p),size=count,p=p)
    elif mode=='multinomial': want=r.multinomial(n,p,size=count).ravel()
    elif mode=='permutation': want=r.permutation(n)
    elif mode=='sample': want=sample(n,k,method=method,random_state=r)
    elif mode=='mixed':
        want=[]
        for _ in range(count): want.extend([r.standard_normal(),r.randint(-100,500),r.random_sample(),r.binomial(1000,.2),r.gamma(100,.01),r.gamma(.3,2),r.poisson(4.5),r.poisson(25)])
    got,state=probe(mode,seed,count,n,k,method,p)
    label=(mode,seed,count,n,k,method,p)
    if mode=='multivariate':
        np.testing.assert_allclose(got,want,rtol=2e-12,atol=2e-12,err_msg=str(label))
    elif mode in ['gamma','standard_gamma']:
        np.testing.assert_allclose(got,want,rtol=2e-14,atol=5e-324,err_msg=str(label))
    elif mode in ['normal','mixed']: np.testing.assert_allclose(got,want,rtol=2e-14,atol=2e-14,err_msg=str(label))
    else: np.testing.assert_array_equal(got,want,err_msg=str(label))
    oracle=r.get_state()
    assert state[:2]==oracle[2:4],(label,state[:2],oracle[2:4])
    np.testing.assert_allclose(state[2],oracle[4],rtol=2e-14,atol=2e-14,err_msg=str(label))
    np.testing.assert_array_equal(state[3],oracle[1],err_msg=str(label))
    CASES+=1

def multivariate_inputs(n,kind):
    mean=np.arange(n,dtype=float)-1.5
    if kind==0: cov=np.eye(n)
    elif kind==1: cov=np.diag(np.arange(1,n+1,dtype=float))
    elif kind==2: cov=np.ones((n,n))
    elif kind==3:
        cov=2*np.eye(n)
        for i in range(n-1): cov[i,i+1]=cov[i+1,i]=.5
    elif kind==4: cov=np.zeros((n,n))
    else: cov=np.diag(np.array([.0001**i for i in range(n)]))
    return mean,cov

for seed in [0,1,42,5489,4294967295,123456789]:
    compare('raw',seed,count=2000)
    for low,high in [(0,1),(2,2),(-3,9),(4,-2)]: compare('uniform',seed,n=low,k=high,count=513)
    for loc,scale in [(0,1),(-5,7),(3,0)]: compare('normal',seed,n=loc,k=scale,count=513)
    for low,high in [(0,1),(-7,19),(0,2**31-1),(0,2**32),(0,2**32+1),(-2**62,2**62),(0,2**63-1)]:
        compare('integer',seed,n=low,k=high,count=1001)
    for n in [0,1,10,60,61,100,1000,100000]:
        for p in [0,1,10000,200000,500000,800000,999999,1000000]:
            compare('binomial',seed,n=n,k=p,count=200)
    for p in [[],[1],[0,1],[1,0,0],[.2,.3,.5],[0,.1,.9],[.999999,.000001]]:
        if p: compare('categorical',seed,p=p)
        for n in [0,1,5,1000]: compare('multinomial',seed,n=n,p=p,count=101)
    for shape in [0,1,1000,100000,300000,999999,1000000,1000001,2000000,100000000,1000000000000]:
        compare('standard_gamma',seed,n=shape,count=513)
        for scale in [0,10000,1000000,3000000]:
            compare('gamma',seed,n=shape,k=scale,count=513)
    for n in [0,1,2,17,1000]:
        for count in sorted({0,n//2,n}):
            for replace in [0,1]: compare('choice',seed,n=n,k=replace,count=count)
        if n: compare('choice',seed,n=n,k=1,count=n+5)
        for k in [-1,0,1,3]:
            for mode in ['shuffle','permutation_array']: compare(mode,seed,n=n,k=k)
    for p in [[1.0],[0,1],[.2,.3,.5],[0,.1,.9],[.999999,.000001]]:
        for count in range(sum(x>0 for x in p)+1):
            compare('choice_weighted',seed,n=len(p),p=p,k=0,count=count)
        compare('choice_weighted',seed,n=len(p),p=p,k=1,count=513)
    for lam,denom in [(0,1),(1,1000000),(1,10),(1,1),(45,10),(9999999,1000000),
                      (10,1),(10000001,1000000),(25,1),(1000,1),(10**6,1),
                      (10**12,1),(10**18,1),(9223372006484770000,1)]:
        compare('poisson',seed,n=lam,k=denom,count=1001)
    for n in [0,1,3,5]:
        for kind in range(6):
            for count in [0,1,7]: compare('multivariate',seed,n=n,k=kind,count=count)
    compare('mixed',seed,count=501)
    for mode in ['raw','uniform','normal','integer','binomial','categorical','multinomial','mixed','gamma','standard_gamma','poisson']:
        compare(mode,seed,count=0,n=0,k=1,p=[.2,.3,.5])
    for n in [-10,-1,0,1,2,17,1000]: compare('permutation',seed,n=n)
    for n,k in [(0,0),(1,0),(1,1),(10,0),(100,1),(100,2),(100,19),(100,20),(100,98),(100,99),(100,100),(1001,10)]:
        for method in ['auto','tracking_selection','pool','reservoir_sampling']:
            compare('sample',seed,n=n,k=k,method=method)

# Actual pinned Dummy predict/proba bodies: shared mutable vs renewed integer seeds.
for strategy in ['uniform','stratified']:
    for seed in [0,42,4294967295]:
        for mutable in [False,True]:
            d=Dummy(); d.n_outputs_=1; d.n_classes_=3; d.classes_=np.arange(3)
            d.class_prior_=np.array([.2,.3,.5]); d.constant=None; d.sparse_output_=False; d._strategy=strategy
            d.random_state=np.random.RandomState(seed) if mutable else seed
            count=2 if mutable else 1
            if strategy=='uniform':
                want=np.concatenate([d.predict(np.zeros((41,2))) for _ in range(2)])
                got,state=probe('dummy_uniform',seed,count,41,p=[.000003])
            else:
                want=np.concatenate([d.predict_proba(np.zeros((41,2))).ravel() for _ in range(2)])
                got,state=probe('dummy_stratified',seed,count,41,p=[.2,.3,.5])
            if not mutable: got=np.tile(got,2)
            np.testing.assert_array_equal(got,want)
            if mutable:
                oracle=d.random_state.get_state()
                assert state[:2]==oracle[2:4]
                np.testing.assert_array_equal(state[3],oracle[1])
            CASES+=1
# Multioutput Dummy prediction consumes complete output columns in order.
for strategy in ['uniform','stratified']:
    for seed in [0,42]:
        d=Dummy(); d.n_outputs_=2; d.n_classes_=[3,2]
        d.classes_=[np.arange(3),np.arange(2)]
        d.class_prior_=[np.array([.2,.3,.5]),np.array([.7,.3])]
        d.constant=None; d.sparse_output_=False; d._strategy=strategy
        d.random_state=np.random.RandomState(seed)
        want=np.concatenate([d.predict(np.zeros((43,1))).T.ravel() for _ in range(2)])
        mode='dummy_uniform' if strategy=='uniform' else 'dummy_multi_stratified'
        probabilities=[.000003,.000002] if strategy=='uniform' else [.2,.3,.5]
        got,state=probe(mode,seed,2,43,p=probabilities)
        np.testing.assert_array_equal(got,want)
        oracle=d.random_state.get_state()
        assert state[:2]==oracle[2:4]
        np.testing.assert_array_equal(state[3],oracle[1])
        CASES+=1
# Explicitly record actual stdlib divergence, while requiring repeatability.
args=[ROOT/'builds/draws','stdlib','42','8','0','1','auto']
a=subprocess.check_output(args,text=True); b=subprocess.check_output(args,text=True)
assert a==b
native=np.array(a.splitlines(),float); numpy=np.random.RandomState(42).random_sample(8)
assert not np.array_equal(native,numpy)
(ROOT/'docs/private/random/stdlib-comparison.json').write_text(json.dumps({'stdlib':native.tolist(),'numpy':numpy.tolist()},indent=2))
for name in ['a','b']:
    actual=np.array(subprocess.check_output([ROOT/'tests/consumers'/name/'builds/probe'],text=True).splitlines(),float)
    np.testing.assert_array_equal(actual,np.random.RandomState(42).random_sample(2))
print(f'PASS {CASES} differential cases: values, all 624 state words, position and Gaussian cache; pinned sklearn Cython and Dummy bodies')

# Independent distribution check on a correlated, non-diagonal covariance.
mean,cov=multivariate_inputs(3,3)
samples,state=probe('multivariate',42,100000,3,3)
samples=samples.reshape(-1,3)
np.testing.assert_allclose(samples.mean(axis=0),mean,rtol=0,atol=.02)
np.testing.assert_allclose(np.cov(samples,rowvar=False),cov,rtol=0,atol=.035)
reference=np.random.RandomState(42)
reference.multivariate_normal(mean,cov,size=100000)
expected_state=reference.get_state()
assert state[:2]==expected_state[2:4]
np.testing.assert_array_equal(state[3],expected_state[1])
np.testing.assert_allclose(state[2],expected_state[4],rtol=2e-14,atol=2e-14)
print('PASS multivariate covariance reconstruction, normal-stream transform, empirical moments and NumPy draw consumption; SVD-dependent samples need not equal NumPy')

from normalization_parity import run as normalization_parity
normalization_parity(check)
