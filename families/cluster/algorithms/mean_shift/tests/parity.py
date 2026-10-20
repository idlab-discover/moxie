"""MeanShift differential coverage; no Python participates in the native runtime."""
from collections import defaultdict
import json
import platform
import subprocess
import sys
import warnings
import numpy as np
from oracle import load, FAMILY, REV

def run(dtype, mode, x, options, seeds=None):
    args=[FAMILY/'builds/mean_shift_driver',32 if dtype==np.float32 else 64,mode,*x.shape,*options]
    args.extend(map(float,x.ravel()))
    if seeds is not None: args.extend(map(float,seeds.ravel()))
    p=subprocess.run(list(map(str,args)),text=True,capture_output=True)
    assert p.returncode==0,(args[:10],p.stderr)
    result=defaultdict(list)
    for line in p.stdout.splitlines():
        key,*values=line.split(); result[key].extend(values)
    return {key:np.array(values,dtype=float if key in ['CENTER','BANDWIDTH'] else int) for key,values in result.items()}

def main():
    oracle,hashes=load(); rng=np.random.RandomState(72); reports=[]
    for dtype in [np.float32,np.float64]:
        tol=3e-5 if dtype==np.float32 else 1e-12
        x=np.vstack([rng.normal(loc=v,scale=.25,size=(21,3)) for v in [-3,0,3]]).astype(dtype)
        cases=[('blobs',x),('toy',np.array([[1,1],[2,1],[1,0],[4,7],[3,5],[3,6]],dtype=dtype)),
               ('constant',np.ones((10,1),dtype=dtype)),('duplicates',np.array([[1],[1],[1],[2],[2],[2],[3],[3]],dtype=dtype)),
               ('ties',np.array([[-2],[0],[2]],dtype=dtype)),('singleton',np.array([[7,3]],dtype=dtype)),
               ('wide',rng.normal(size=(23,17)).astype(dtype))]
        for name,data in cases:
            for bandwidth in [None,.5,2.]:
                for max_iter in [0,1,300]:
                    for cluster_all in [False,True]:
                        for binning in [False,True]:
                            config=dict(bandwidth=bandwidth,max_iter=max_iter,cluster_all=cluster_all,bin_seeding=binning)
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore'); expected=oracle['MeanShift'](**config).fit(data)
                            mode='fit' if cluster_all else 'orphans'
                            out=run(dtype,mode,data,[-1 if bandwidth is None else bandwidth,int(binning),1,max_iter,-1])
                            context=f'{dtype.__name__}/{name}/{config}'
                            np.testing.assert_array_equal(out['SHAPE'],expected.cluster_centers_.shape,err_msg=context)
                            np.testing.assert_allclose(out['CENTER'].reshape(expected.cluster_centers_.shape),expected.cluster_centers_,rtol=tol,atol=tol,err_msg=context)
                            np.testing.assert_array_equal(out['LABEL'],expected.labels_,err_msg=context)
                            np.testing.assert_array_equal(out['PREDICT'],expected.predict(data),err_msg=context)
                            assert out['ITER'][0]==expected.n_iter_,context
                            reports.append(context)
        for seed_data in [x[:3],np.array([[99,99,99],[-3,-3,-3]],dtype=dtype),np.vstack([x[0],x[0]])]:
            expected=oracle['MeanShift'](bandwidth=1,seeds=seed_data,cluster_all=False).fit(x)
            out=run(dtype,'orphans',x,[1,1,99,300,len(seed_data)],seed_data)
            np.testing.assert_allclose(out['CENTER'].reshape(expected.cluster_centers_.shape),expected.cluster_centers_,rtol=tol,atol=tol)
            np.testing.assert_array_equal(out['LABEL'],expected.labels_)
            assert out['ITER'][0]==expected.n_iter_
            reports.append(f'{dtype.__name__}/explicit-seeds/{len(seed_data)}')
        for quantile in [0,.3,.5,1]:
            for samples in [None,1,7,100]:
                expected=oracle['estimate_bandwidth'](x,quantile=quantile,n_samples=samples,random_state=17)
                out=run(dtype,'bandwidth',x,[quantile,samples or 0,17,0,0])
                np.testing.assert_allclose(out['BANDWIDTH'][0],expected,rtol=tol,atol=tol)
                reports.append(f'{dtype.__name__}/bandwidth/{quantile}/{samples}')
        large=rng.normal(size=(503,2)).astype(dtype)
        expected=oracle['estimate_bandwidth'](large,quantile=.03)
        out=run(dtype,'bandwidth',large,[.03,0,0,0,0])
        np.testing.assert_allclose(out['BANDWIDTH'][0],expected,rtol=tol,atol=tol)
        reports.append(f'{dtype.__name__}/bandwidth/batch-boundary')
        bins=np.array([[-2.5],[-1.5],[-.5],[.5],[1.5],[2.5],[.6],[.6]],dtype=dtype)
        for size in [0,.01,.3,1]:
            for freq in [1,2,99]:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore'); expected=oracle['get_bin_seeds'](bins,size,freq)
                out=run(dtype,'bins',bins,[size,freq,0,0,0])
                np.testing.assert_array_equal(out['SHAPE'], [len(expected), bins.shape[1]])
                np.testing.assert_allclose(out.get('CENTER',np.array([])),expected.ravel(),rtol=tol,atol=tol)
                reports.append(f'{dtype.__name__}/bins/{size}/{freq}')
    result=dict(revision=REV,source_hashes=hashes,cases=reports,numpy=np.__version__,python=sys.version,platform=platform.platform(),flags=['-O3','--fp-mode=contract=off'],seed=72,threads=1,tolerances='exact labels, shape and n_iter; centers/bandwidth Float32 3e-5, Float64 1e-12 (shared widened reduction)',oracle='Pinned sklearn MeanShift and helper bodies with independent SciPy cdist queries and serial task executor',assembly_receipt=json.loads((FAMILY/'builds/deps/ASSEMBLY.json').read_text()))
    (FAMILY/'builds/mean_shift-parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'PASS: {len(reports)} MeanShift differential configurations')

if __name__=='__main__': main()
