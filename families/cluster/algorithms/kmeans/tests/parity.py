"""Reproducible seeding and estimator comparisons with the local sklearn source."""
import json
import os
from pathlib import Path
import subprocess
import warnings
from collections import defaultdict
import numpy as np
from oracle import load, FAMILY, REV


def cases():
    r = np.random.RandomState(817)
    for n,d,k in [(1,1,1),(8,2,3),(31,7,5),(70,3,8),(257,2,4)]:
        x = r.normal(size=(n,d))
        for weights in [np.ones(n), r.uniform(.1,3,n), (np.arange(n)%3==0).astype(float)]:
            yield f'random-{n}-{d}-{k}',x,k,weights
    yield 'duplicates',np.array([[0.,0.],[0,0],[2,2],[2,2],[4,4],[4,4]]),5,np.ones(6)
    yield 'identical',np.ones((6,3)),4,np.array([0,1,2,0,0,3.])
    yield 'zero-weights',np.array([[0.,0.],[20,2],[2,1],[5,3],[7,9]]),3,np.array([0.,2,0,1,0])
    yield 'offset',r.normal(size=(30,4))+1e4,4,np.ones(30)
    yield 'small',r.normal(size=(10,2))*1e-10,3,np.ones(10)


def main():
    env = load()
    driver = FAMILY/'builds/kmeans_driver'
    reports=[]
    for dtype in [np.float32,np.float64]:
        for name,raw,k,raw_weights in cases():
            x=raw.astype(dtype);weights=raw_weights.astype(dtype)
            for seed in [0,42,117]:
                for mode in ['seed','lloyd','elkan']:
                    variants = [(t,300,1e-4,0,'k-means++',norm) for t in [0,1,5] for norm in [0,1,2]] if mode=='seed' else [(0,m,tol,runs,init,0) for m,tol,runs,init in [(300,1e-4,0,'k-means++'),(1,0,1,'k-means++'),(300,.1,4,'k-means++'),(300,0,0,'random'),(300,1e-4,3,'array')]]
                    for trials,maximum,tol,runs,init,supplied in variants:
                        if init=='random' and (weights>0).sum()<k:continue
                        rng=np.random.RandomState(seed)
                        norms=np.einsum('ij,ij->i',x,x)
                        if supplied == 2: norms = norms + dtype(1)
                        initial=x[:k].copy()
                        tail=norms if mode=='seed' and supplied else initial.ravel() if init=='array' else []
                        args=[str(driver), '32' if dtype==np.float32 else '64',mode,str(len(x)),str(x.shape[1]),str(k),str(seed),str(trials),str(maximum),repr(tol),str(runs),init,str(supplied),*map(str,map(float,x.ravel())),*map(str,map(float,weights)),*map(str,map(float,tail))]
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            if mode=='seed':
                                centers,indices=env['kmeans_plusplus'](x,k,sample_weight=weights,random_state=rng,n_local_trials=trials or None,x_squared_norms=norms if supplied else None)
                            else:
                                model=env['KMeans'](k,init=initial if init=='array' else init,n_init=runs or 'auto',max_iter=maximum,tol=tol,algorithm=mode,random_state=rng)
                                model.fit(x.copy(),sample_weight=weights)
                                centers=model.cluster_centers_
                        expected_rng=rng.randint(0,2**32,dtype=np.uint32)
                        proc=subprocess.run(args,capture_output=True,text=True,check=True)
                        out=defaultdict(list)
                        for line in proc.stdout.splitlines():
                            key,*values=line.split();out[key].extend(values)
                        def arr(key,dt=float):return np.array(out[key],dtype=dt)
                        context=f'{name}/{dtype.__name__}/{seed}/{mode}/{trials}/{maximum}/{tol}/{runs}/{init}/{supplied}'
                        try:
                            assert int(out['RNG'][0])==expected_rng,(out['RNG'],expected_rng)
                            np.testing.assert_allclose(arr('CENTER').reshape(centers.shape),centers,rtol=3e-5 if dtype==np.float32 else 2e-12,atol=3e-6 if dtype==np.float32 else 2e-13)
                            if mode=='seed':np.testing.assert_array_equal(arr('INDEX',int),indices)
                            else:
                                assert int(out['META'][0])==model.n_iter_,(out['META'],model.n_iter_)
                                np.testing.assert_array_equal(arr('LABEL',int),model.labels_)
                                predicted = arr('PREDICT',int)
                                reference_predicted = model.predict(x)
                                prediction_roundoff = not np.array_equal(predicted, reference_predicted)
                                if prediction_roundoff:
                                    # Large offsets can erase the norm/dot decision in Float32.
                                    # Check a forward-error bound, never an unrestricted label waiver.
                                    assert dtype == np.float32 and name == 'offset'
                                    exact = ((x.astype('d')[:,None,:] - centers.astype('d')[None,:,:])**2).sum(axis=2)
                                    scale = (x.astype('d')**2).sum(axis=1) + (centers.astype('d')**2).sum(axis=1).max()
                                    bound = 8 * x.shape[1] * np.finfo(dtype).eps * scale
                                    assert np.all(np.abs(exact[np.arange(len(x)),predicted] - exact[np.arange(len(x)),reference_predicted]) <= bound)
                                else:
                                    np.testing.assert_array_equal(predicted, reference_predicted)
                                threshold=4e-5 if dtype==np.float32 else 2e-12
                                np.testing.assert_allclose(float(out['META'][1]),model.inertia_,rtol=threshold,atol=1e-25)
                                expected_score = -env['_inertia_dense'](x, weights, centers, predicted.astype('int32'), 1) if prediction_roundoff else model.score(x,sample_weight=weights)
                                np.testing.assert_allclose(float(out['META'][2]),expected_score,rtol=threshold,atol=1e-25)
                                actual_distances = arr('TRANSFORM').reshape((len(x),k))
                                expected_distances = model.transform(x)
                                # Direct-distance accuracy is checked independently. The
                                # upstream norm identity loses precision near zero distances.
                                direct = np.sqrt(((x.astype('d')[:,None,:]-centers.astype('d')[None,:,:])**2).sum(axis=2))
                                np.testing.assert_allclose(actual_distances,direct,rtol=threshold,atol=3e-6 if dtype==np.float32 else 1e-10)
                                scale = (x.astype('d')**2).sum(axis=1)[:,None] + (centers.astype('d')**2).sum(axis=1)[None,:]
                                bound = 8*x.shape[1]*np.finfo(dtype).eps*scale
                                assert np.all(np.abs(actual_distances**2 - expected_distances**2) <= bound + threshold*direct**2)

                        except Exception:
                            (FAMILY/'builds/kmeans-failure.json').write_text(json.dumps({'case':context,'args':args,'stdout':proc.stdout,'expected_centers':centers.tolist()},indent=2))
                            print('FAIL',context,flush=True);raise
                        reports.append({'case':context,'prediction_roundoff':prediction_roundoff if mode!='seed' else False})
    (FAMILY/'builds/kmeans-parity.json').write_text(json.dumps({'revision':REV,'source_hash':env['source_hash'],'cases':reports,'count':len(reports),'numpy':np.__version__,'threads':1,'flags':['-O3','--fp-mode=contract=off']},indent=2))
    print('PASS',len(reports),'KMeans/seeding differential cases')

if __name__=='__main__':main()
