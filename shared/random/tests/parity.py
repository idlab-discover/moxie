"""Differential sequences and complete MT state, never distribution-only parity."""
from pathlib import Path
import json, subprocess
import numpy as np
from oracle import load, ROOT
sample,Dummy,check=load()
CASES=0

def probe(mode, seed, count=100, n=0, k=1, method='auto', p=()):
    out=subprocess.check_output([ROOT/'builds/draws',mode,str(seed),str(count),str(n),str(k),method,*[str(int(round(x*1000000))) for x in p]],text=True)
    values, state=out.split('STATE\n')
    state=state.splitlines()
    dtype = float if mode in ['uniform','normal','mixed','gamma','standard_gamma'] else np.int64
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
    elif mode=='binomial': want=r.binomial(n,k/1000000,size=count)
    elif mode=='categorical': want=r.choice(len(p),size=count,p=p)
    elif mode=='multinomial': want=r.multinomial(n,p,size=count).ravel()
    elif mode=='permutation': want=r.permutation(n)
    elif mode=='sample': want=sample(n,k,method=method,random_state=r)
    elif mode=='mixed':
        want=[]
        for _ in range(count): want.extend([r.standard_normal(),r.randint(-100,500),r.random_sample(),r.binomial(1000,.2),r.gamma(100,.01),r.gamma(.3,2)])
    got,state=probe(mode,seed,count,n,k,method,p)
    label=(mode,seed,count,n,k,method,p)
    if mode in ['gamma','standard_gamma']:
        np.testing.assert_allclose(got,want,rtol=2e-14,atol=5e-324,err_msg=str(label))
    elif mode in ['normal','mixed']: np.testing.assert_allclose(got,want,rtol=2e-14,atol=2e-14,err_msg=str(label))
    else: np.testing.assert_array_equal(got,want,err_msg=str(label))
    oracle=r.get_state()
    assert state[:2]==oracle[2:4],(label,state[:2],oracle[2:4])
    np.testing.assert_allclose(state[2],oracle[4],rtol=2e-14,atol=2e-14,err_msg=str(label))
    np.testing.assert_array_equal(state[3],oracle[1],err_msg=str(label))
    CASES+=1

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
    compare('mixed',seed,count=501)
    for mode in ['raw','uniform','normal','integer','binomial','categorical','multinomial','mixed','gamma','standard_gamma']:
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
