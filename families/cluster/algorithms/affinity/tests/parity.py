"""Dense algorithm and wrapper differential cases against local source bodies."""
from pathlib import Path
from collections import defaultdict
import argparse,json,os,subprocess,warnings
import numpy as np
from oracle import load, verify, ROOT, REV

def cases():
    demo=np.array([[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]],dtype='d')
    yield 'demo',demo,'euclidean',None,5,.5,200,15,True,0
    for n in [1,2,3,8]:
        for p in [None,-1.,0.,1.,np.arange(n,dtype='d')]:
            yield f'equal-{n}-{str(p)}',np.zeros((n,n)),'precomputed',p,42,.5,200,15,True,0
    yield 'failure',np.array([[0,0],[1,1],[-2,-2.]]),'euclidean',-10.,82,.5,1,15,True,0
    yield 'nonconverged',np.array([[1,0,0,0,0,0],[0,1,1,1,0,0],[0,0,1,0,0,1.]]),'euclidean',None,34,.5,2,15,True,0
    yield 'duplicates',np.array([[1,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,1.]]),'precomputed',1.,0,.5,200,15,True,0
    r=np.random.RandomState(71)
    for n in [3,6,15,40]:
        x=r.normal(size=(n,4))
        s=-((x[:,None]-x[None,:])**2).sum(axis=2)
        for mode in ['euclidean','precomputed']:
            data=x if mode=='euclidean' else s
            for seed in [0,17,76]:
                for damping,maximum,window in [(.5,200,15),(.8,200,5),(.99,40,1),(.5,2,1)]:
                    yield f'random-{n}-{mode}-{seed}-{damping}-{maximum}',data,mode,None,seed,damping,maximum,window,True,0
        for p in [-100.,3.,np.linspace(-20,0,n),np.array([-3.])]:
            yield f'preference-{n}-{str(p)}',s,'precomputed',p,4,.7,300,10,False,0
        yield f'strides-{n}',s,'precomputed',None,9,.7,300,10,False,1
        yield f'asymmetric-{n}',r.normal(size=(n,n)),'precomputed',None,2,.65,200,15,True,0
    for pref in [np.array([np.nan,-1.,-2.]),np.array([np.inf,-1.,-2.]),np.array([-np.inf,-1.,-2.])]:
        yield 'nonfinite-vector-'+str(pref),np.array([[0.,-1.,-4.],[-1.,0.,-1.],[-4.,-1.,0.]]),'precomputed',pref,3,.6,20,5,True,0
    for scale in [1e-10,1e10]:
        yield f'scale-{scale}',demo*scale,'euclidean',None,3,.6,200,15,True,0

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver',type=Path,default=ROOT/'builds/driver')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'builds')
    options=parser.parse_args()
    options.output_dir.mkdir(parents=True,exist_ok=True)
    env=load(); results=[]
    for dtype in [np.float64,np.float32]:
        for name,raw,mode,p,seed,damping,maximum,window,copy,layout in cases():
            x=raw.astype(dtype)
            reference_x=x[::-1,::-1].copy() if layout else x.copy()
            model=env['AffinityPropagation'](affinity=mode,preference=p,random_state=seed,damping=damping,max_iter=maximum,convergence_iter=window,copy=copy)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit(reference_x)
            preferences=[] if p is None else np.atleast_1d(p).tolist()
            args=[str(options.driver.resolve()),'32' if dtype==np.float32 else '64',str(x.shape[0]),str(x.shape[1]),mode,str(seed),repr(damping),str(maximum),str(window),str(len(preferences)),str(int(copy)),str(layout),*map(repr,map(float,x.ravel())),*map(repr,map(float,preferences))]
            proc=subprocess.run(args,check=True,text=True,capture_output=True,env=os.environ)
            out=defaultdict(list)
            for line in proc.stdout.splitlines():
                key,*values=line.split();out[key].extend(values)
            def arr(k,dt=float):return np.asarray(out[k],dtype=dt)
            context=f'{name} / {dtype.__name__}'
            try:
                assert int(out['META'][0])==model.n_iter_,(out['META'],model.n_iter_)
                assert (out['META'][3]=='True') == (model.n_iter_==0)
                expected_converged=not any(isinstance(w.message,env['ConvergenceWarning']) for w in caught)
                assert (out['META'][2]=='True') == expected_converged
                actual_labels=arr('LABEL',int)
                actual_centers=arr('CENTER',int)
                tied=False
                try:
                    np.testing.assert_array_equal(actual_labels,model.labels_)
                    np.testing.assert_array_equal(actual_centers,model.cluster_centers_indices_)
                except AssertionError:
                    # Different dense matmul reduction orders can select different
                    # exemplars of a tied two-point cluster. Require the identical
                    # partition AND a roundoff-sized within-cluster score tie.
                    assert mode=='euclidean'
                    np.testing.assert_array_equal(actual_labels[:,None]==actual_labels[None,:],model.labels_[:,None]==model.labels_[None,:])
                    for k,chosen in enumerate(actual_centers):
                        members=np.flatnonzero(actual_labels==k)
                        original=model.cluster_centers_indices_[model.labels_[chosen]]
                        scores=model.affinity_matrix_[np.ix_(members,[chosen,original])].astype('d').sum(axis=0)
                        tolerance=(2e-6 if dtype==np.float32 else 2e-14)*max(1,abs(scores).max())
                        assert abs(scores[0]-scores[1])<=tolerance,(chosen,original,scores)
                    tied=True
                tol=2e-6 if dtype==np.float32 else 2e-14
                np.testing.assert_allclose(arr('S').reshape(model.affinity_matrix_.shape),model.affinity_matrix_,rtol=tol,atol=tol*np.max(np.abs(model.affinity_matrix_[np.isfinite(model.affinity_matrix_)]),initial=1e-300))
                np.testing.assert_allclose(arr('INPUT').reshape(reference_x.shape),reference_x,rtol=tol,atol=tol)
                if mode=='euclidean':
                    np.testing.assert_array_equal(arr('VALUE',dtype).reshape((len(actual_centers),x.shape[1])),reference_x[actual_centers])
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        if tied:model.cluster_centers_=reference_x[actual_centers]
                        expected=model.predict(reference_x)
                    np.testing.assert_array_equal(arr('PREDICT',int),expected)
                # Same unedited public helper's optional return branch.
                if mode=='precomputed':
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        helper=env['affinity_propagation'](x[::-1,::-1].copy() if layout else x.copy(),preference=p,random_state=seed,damping=damping,max_iter=maximum,convergence_iter=window,return_n_iter=True)
                    np.testing.assert_array_equal(helper[0],model.cluster_centers_indices_)
                    np.testing.assert_array_equal(helper[1],model.labels_)
                    assert helper[2]==model.n_iter_
            except Exception:
                (options.output_dir/'parity-failure.json').write_text(json.dumps({'case':context,'args':args,'stdout':proc.stdout,'reference_labels':model.labels_.tolist(),'reference_centers':np.asarray(model.cluster_centers_indices_).tolist(),'reference_iter':model.n_iter_},indent=2))
                print('FAIL',context,flush=True);raise
            results.append({'case':context,'iterations':model.n_iter_,'centers':len(model.cluster_centers_indices_),'tied_exemplar_roundoff':tied,'warnings':[str(w.message) for w in caught]})
    report={'revision':REV,'source_hashes':verify(),'driver':str(options.driver.resolve()),'count':len(results),'cases':results}
    (options.output_dir/'parity.json').write_text(json.dumps(report,indent=2))
    print(f'PASS {len(results)} dense dtype/algorithm/wrapper differential cases against local sklearn {REV}')
if __name__=='__main__':main()
