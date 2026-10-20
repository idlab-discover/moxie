"""Differential driver against the unmodified, revision-checked sklearn C++ engine."""
from pathlib import Path
import hashlib
import itertools
import json
import os
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT.parents[4] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
SOURCE = REF / 'sklearn/svm/src/liblinear'

def reference():
    assert subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'],text=True).strip() == REVISION
    paths = ['sklearn/svm/src/liblinear', 'sklearn/svm/src/newrand', 'sklearn/svm/_base.py', 'sklearn/svm/_liblinear.pyx']
    assert not subprocess.check_output(['git','-C',str(REF),'status','--porcelain','--',*paths],text=True)
    subprocess.run(['g++','-O3','-ffp-contract=off','-std=c++17','-I',str(SOURCE),str(ROOT/'tests/oracle.cpp'),str(SOURCE/'linear.cpp'),str(SOURCE/'tron.cpp'),'-o',str(ROOT/'builds/oracle')],check=True)

def write_case(path,x,y,sw,cw,solver,c=1.,epsilon=.1,tol=1e-8,fit=True,scale=1.,limit=10000,seed=42):
    args=[x.shape[0],x.shape[1],len(cw),solver,c,epsilon,tol,int(fit),scale,limit,seed]
    args.extend(x.flat);args.extend(y);args.extend(sw);args.extend(cw)
    path.write_text(' '.join(map(str,args))+'\n')

def invoke(binary,path):
    proc=subprocess.run([str(binary),str(path)],capture_output=True,text=True,timeout=60)
    if proc.returncode: raise AssertionError(f'{binary.name}: {proc.stderr}')
    lines=proc.stdout.splitlines(); r,p,status=map(int,lines[0].split())
    values=np.array([float(s) for s in lines[1:]])
    return values[:r*p].reshape(r,p),values[r*p:r*(p+1)],values[r*(p+1):].astype(int),status

def objective(w,b,x,y,sw,cw,solver,c,epsilon,scale,fit):
    augmented=np.concatenate([w,b[:,None]/scale],axis=1) if fit else w
    regularizer=np.sum(abs(augmented),axis=1) if solver in (5,6) else .5*np.sum(augmented*augmented,axis=1)
    if solver==4:
        scores=x@w.T+b
        loss=np.max(scores+1-np.eye(len(cw))[y.astype(int)],axis=1)-scores[np.arange(len(y)),y.astype(int)]
        costs=c*sw*cw[y.astype(int)]
        return np.array([regularizer.sum()+np.sum(costs[sw>0]*loss[sw>0])])
    result=[]
    for r in range(w.shape[0]):
        score=x@w[r]+b[r]
        costs=c*sw.copy()
        if solver<11:
            positive=1 if len(cw)==2 else r
            target=np.where(y==positive,1.,-1.)
            costs*=np.where(y==positive,cw[positive],cw[0] if len(cw)==2 else 1.)
            margin=target*score
            if solver in (0,6,7): loss=np.logaddexp(0,-margin)
            else:
                loss=np.maximum(1-margin,0)
                if solver in (1,2,5):loss=loss**2
        else:
            loss=np.maximum(np.abs(score-y)-epsilon,0)
            if solver in (11,12):loss=loss**2
        result.append(regularizer[r]+np.sum(costs[sw>0]*loss[sw>0]))
    return np.array(result)

def main():
    reference()
    output=ROOT/'builds/parity'; output.mkdir(exist_ok=True)
    rng=np.random.RandomState(13)
    tested=[]
    for solver,shape,fit,seed in itertools.product((0,1,2,3,4,5,6,7,11,12,13),((8,3),(5,9),(24,4)),(False,True),(0,42)):
        n,p=shape;x=rng.randn(n,p)
        k=0 if solver>=11 else (3 if n==24 else 2)
        y=rng.randn(n) if k==0 else (np.arange(n)%k).astype(float)
        sw=rng.uniform(.3,2,n);sw[-1]=0
        cw=np.linspace(.7,1.8,k)
        scale=2.3;c=.8;epsilon=.2;tol=1e-8
        name=f'{solver}-{n}-{p}-{int(fit)}-{seed}';path=output/(name+'.txt')
        write_case(path,x,y,sw,cw,solver,c,epsilon,tol,fit,scale,seed=seed)
        actual=invoke(ROOT/'builds/driver',path);expected=invoke(ROOT/'builds/oracle',path)
        np.testing.assert_allclose(actual[0],expected[0],rtol=2e-5,atol=2e-6,err_msg=name)
        np.testing.assert_allclose(actual[1],expected[1],rtol=2e-5,atol=2e-6,err_msg=name)
        ao=objective(*actual[:2],x,y,sw,cw,solver,c,epsilon,scale,fit)
        eo=objective(*expected[:2],x,y,sw,cw,solver,c,epsilon,scale,fit)
        np.testing.assert_allclose(ao,eo,rtol=1e-8,atol=1e-8,err_msg=name)
        assert np.all(ao <= objective(np.zeros_like(actual[0]),np.zeros_like(actual[1]),x,y,sw,cw,solver,c,epsilon,scale,fit)+1e-8),name
        if solver in (0,1,2,5,6,7,11,12):
            design=np.column_stack([x,np.full(n,scale)]) if fit else x
            weights=np.column_stack([actual[0],actual[1]/scale]) if fit else actual[0]
            for row,w in enumerate(weights):
                scores=design@w;costs=c*np.maximum(sw,0)
                if solver<11:
                    pos=1 if k==2 else row;sign=np.where(y==pos,1.,-1.)
                    costs*=np.where(y==pos,cw[pos],cw[0] if k==2 else 1.)
                    if solver in (0,6,7): mult=-costs*sign/(1+np.exp(sign*scores))
                    else: mult=-2*costs*sign*np.maximum(1-sign*scores,0)
                else:
                    residual=scores-y
                    mult=2*costs*np.sign(residual)*np.maximum(abs(residual)-epsilon,0)
                g=design.T@mult
                if solver in (5,6):
                    violation=np.where(w>0,abs(g+1),np.where(w<0,abs(g-1),np.maximum(abs(g)-1,0)))
                else: violation=abs(g+w)
                assert max(violation)<2e-5,(name,'KKT',violation)
        tested.append(dict(case=name,iterations=actual[2].tolist(),reference_iterations=expected[2].tolist(),max_coefficient_error=float(np.max(abs(actual[0]-expected[0])))))
    (output/'results.json').write_text(json.dumps(dict(revision=REVISION,cases=tested),indent=2)+'\n')
    # Capped trajectories, no intercept/bias scaling, zero/collinear designs,
    # signed sample weights and singleton regression exercise stopping edges.
    for solver, kind, limit in itertools.product((0,1,2,3,4,5,6,7,11,12,13), ('zero','collinear','weighted','tiny_c'), (0,1,3,10000)):
        n,p=6,2;k=0 if solver>=11 else 2
        x=np.zeros((n,p)) if kind=='zero' else np.arange(n*p,dtype=float).reshape(n,p)/7-1
        if kind=='collinear': x[:,1]=x[:,0]
        y=np.linspace(-1,1,n) if k==0 else (np.arange(n)%2).astype(float)
        sw=np.ones(n);cw=np.ones(k)
        if kind=='weighted': sw[-1]=-2;sw[-2]=0;cw[:]=[.2,3] if k else []
        c=1e-4 if kind=='tiny_c' else 2.
        scale=100. if kind=='weighted' else 1.
        fit=kind=='weighted'
        name=f'edge-{solver}-{kind}-{limit}';path=output/(name+'.txt')
        write_case(path,x,y,sw,cw,solver,c=c,epsilon=0,tol=1e-7,fit=fit,scale=scale,limit=limit)
        actual=invoke(ROOT/'builds/driver',path);expected=invoke(ROOT/'builds/oracle',path)
        np.testing.assert_allclose(actual[0],expected[0],rtol=5e-5,atol=3e-6,err_msg=name)
        np.testing.assert_allclose(actual[1],expected[1],rtol=5e-5,atol=3e-6,err_msg=name)
        if limit<=3:
            np.testing.assert_array_equal(actual[2],expected[2],err_msg=name)
        assert actual[3]==int(max(actual[2])>=limit),name
        tested.append(dict(case=name,iterations=actual[2].tolist(),reference_iterations=expected[2].tolist()))
    for solver,k,mode in itertools.product(range(8),(2,3),('zero_cost','drop_first','drop_last')):
        n,p=9,2;x=rng.randn(n,p);y=(np.arange(n)%k).astype(float)
        sw=np.ones(n);cw=np.ones(k)
        if mode=='zero_cost':cw[0]=0
        elif mode=='drop_first':sw[y==0]=0
        else:sw[y==k-1]=-1
        name=f'weights-{solver}-{k}-{mode}';path=output/(name+'.txt')
        write_case(path,x,y,sw,cw,solver,tol=1e-7)
        expected=invoke(ROOT/'builds/oracle',path)
        if not np.all(np.isfinite(expected[0])):
            proc=subprocess.run([str(ROOT/'builds/driver'),str(path)],capture_output=True,text=True,timeout=60)
            assert proc.returncode,'nonfinite oracle result must raise explicitly'
            tested.append(dict(case=name,numerical_error=True))
            continue
        actual=invoke(ROOT/'builds/driver',path)
        np.testing.assert_allclose(actual[0],expected[0],rtol=5e-5,atol=3e-6,err_msg=name)
        np.testing.assert_allclose(actual[1],expected[1],rtol=5e-5,atol=3e-6,err_msg=name)
        tested.append(dict(case=name,iterations=actual[2].tolist(),reference_iterations=expected[2].tolist()))
    subprocess.run(['g++','-O3','-std=c++17','-I',str(REF/'sklearn/svm/src/newrand'),str(ROOT/'tests/rng_oracle.cpp'),'-o',str(ROOT/'builds/rng_oracle')],check=True)
    expected=subprocess.check_output([str(ROOT/'builds/rng_oracle')],text=True)
    actual=subprocess.check_output([str(ROOT/'builds/rng_probe')],text=True)
    assert expected==actual,'bounded RNG values or draw consumption differ'
    (output/'results.json').write_text(json.dumps(dict(revision=REVISION,cases=tested,rng_words=15003),indent=2)+'\n')
    print(f'PASS {len(tested)} dense solver comparisons, all 11 IDs; 15003 RNG words')

if __name__=='__main__': main()
