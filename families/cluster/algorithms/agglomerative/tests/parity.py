"""Exact tree/label comparisons and bounded numerical tolerances."""
from collections import defaultdict
import argparse
from pathlib import Path
import json
import platform
import subprocess
import sys
import warnings
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
from oracle import load,FAMILY,REV

DRIVER = FAMILY/'builds/agglomerative_driver'
RECEIPT = FAMILY/'builds/deps/ASSEMBLY.json'

def run(dtype,mode,x,linkage,metric,k=2,threshold=0,expect_error=False):
    args=[DRIVER,32 if dtype==np.float32 else 64,mode,*x.shape,linkage,metric,k or 0,threshold,*map(float,x.ravel())]
    p=subprocess.run(list(map(str,args)),text=True,capture_output=True)
    if expect_error:
        assert p.returncode!=0 and 'ValueError' in p.stdout+p.stderr,(p.stdout,p.stderr)
        return
    assert p.returncode==0,(args[:9],p.stderr)
    result=defaultdict(list)
    for line in p.stdout.splitlines():
        key,*values=line.split();result[key].extend(values)
    return {k:np.array(v,dtype=float if k in ['DIST','TRANSFORM','INVERSE'] else int) for k,v in result.items()}

def main():
    oracle,hashes=load();rng=np.random.RandomState(731);reports=[]
    for dtype in [np.float32,np.float64]:
        tol=2e-12 # all linkage math and mean pooling promote to double
        cases=[('random',rng.normal(size=(23,5)).astype(dtype)),
               ('wide',rng.normal(size=(7,19)).astype(dtype)),
               ('ties',np.array([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]],dtype=dtype)),
               ('identical',np.ones((7,3),dtype=dtype)),
               ('duplicates',np.array([[0,1],[0,1],[0,2],[0,2],[0,7],[0,7]],dtype=dtype)),
               ('two',np.array([[3,1],[5,6]],dtype=dtype))]
        for name,x in cases:
            for linkage in ['ward','complete','average','single']:
                for metric in (['euclidean','l2'] if linkage=='ward' else ['euclidean','l2','manhattan','l1','cityblock','cosine','chebyshev','precomputed']):
                    if metric=='cosine' and np.any(~np.any(x,axis=1)):continue
                    data=squareform(pdist(x)).astype(dtype) if metric=='precomputed' else x
                    builder=oracle['ward_tree'] if linkage=='ward' else oracle['linkage_tree']
                    kwargs={} if linkage=='ward' else dict(linkage=linkage,affinity=metric)
                    expected=builder(data,return_distance=True,**kwargs)
                    for k in sorted({1,2,len(data)//2,len(data)}):
                        context=f'{dtype.__name__}/{name}/{linkage}/{metric}/k={k}'
                        out=run(dtype,'fit',data,linkage,metric,k)
                        np.testing.assert_array_equal(out['CHILD'].reshape(-1,2),expected[0],err_msg=context)
                        np.testing.assert_allclose(out['DIST'],expected[-1],atol=tol,rtol=tol,err_msg=context)
                        labels=oracle['_hc_cut'](k,expected[0],len(data))
                        np.testing.assert_array_equal(out['LABEL'],labels,err_msg=context)
                        np.testing.assert_array_equal(out['COUNT'],[k,len(data),1,data.shape[1]],err_msg=context)
                        reports.append(context)
                    if metric=='euclidean':
                        for threshold in [0,expected[-1][0],expected[-1][-1],expected[-1][-1]+1]:
                            model=oracle['AgglomerativeClustering'](n_clusters=None,distance_threshold=threshold,linkage=linkage).fit(data)
                            out=run(dtype,'fit',data,linkage,metric,None,threshold)
                            np.testing.assert_array_equal(out['LABEL'],model.labels_)
                            reports.append(f'{dtype.__name__}/{name}/{linkage}/threshold={threshold}')
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore');tree=builder(data,n_clusters=2,return_distance=True,**kwargs)
                        out=run(dtype,'tree',data,linkage,metric)
                        np.testing.assert_array_equal(out['CHILD'].reshape(-1,2),tree[0])
                        np.testing.assert_allclose(out['DIST'],tree[-1],atol=tol,rtol=tol)
                        assert out['COUNT'].tolist()==[len(data),1,0]
                        reports.append(f'{dtype.__name__}/{name}/{linkage}/tree')
            for linkage in ['ward','complete','average','single']:
                metrics=['euclidean'] if linkage=='ward' else ['euclidean','manhattan','cosine','chebyshev','precomputed']
                for metric in metrics:
                    if metric=='cosine' and np.any(~np.any(x,axis=0)):continue
                    data=squareform(pdist(x.T)).astype(dtype) if metric=='precomputed' else x
                    for k in sorted({1,2,data.shape[1]})+[None]:
                        threshold=0 if k is None else None
                        expected=oracle['FeatureAgglomeration'](k,linkage=linkage,metric=metric,distance_threshold=threshold,compute_distances=True).fit(data)
                        out=run(dtype,'feature',data,linkage,metric,k,threshold or 0)
                        context=f'{dtype.__name__}/{name}/feature/{linkage}/{metric}/{k}'
                        np.testing.assert_array_equal(out['LABEL'],expected.labels_,err_msg=context)
                        np.testing.assert_array_equal(out['CHILD'].reshape(-1,2),expected.children_,err_msg=context)
                        np.testing.assert_allclose(out['DIST'],expected.distances_,atol=tol,rtol=tol,err_msg=context)
                        transformed=expected.transform(data)
                        assert transformed.dtype==np.float64
                        np.testing.assert_allclose(out['TRANSFORM'],transformed.ravel(),atol=tol,rtol=tol,err_msg=context)
                        np.testing.assert_allclose(out['INVERSE'],expected.inverse_transform(transformed).ravel(),atol=tol,rtol=tol,err_msg=context)
                        np.testing.assert_array_equal(out['COUNT'],[expected.n_clusters_,data.shape[1],1,data.shape[1]],err_msg=context)
                        reports.append(context)
        for linkage in ['single','complete','average']:
            data=np.array([[np.nan,2,7],[np.inf,np.nan,5],[-np.inf,99,np.nan]],dtype=dtype)
            expected=oracle['linkage_tree'](data,linkage=linkage,affinity='precomputed',return_distance=True)
            out=run(dtype,'tree',data,linkage,'precomputed')
            np.testing.assert_array_equal(out['CHILD'].reshape(-1,2),expected[0])
            np.testing.assert_allclose(out['DIST'],expected[-1],atol=tol,rtol=tol)
            reports.append(f'{dtype.__name__}/upper-triangle-only/{linkage}')
        for linkage in ['ward','complete','average','single']:
            data=np.array([[3],[1],[8],[4],[1]],dtype=dtype)
            expected=oracle['ward_tree'](data.ravel(),return_distance=True) if linkage=='ward' else oracle['linkage_tree'](data.ravel(),linkage=linkage,return_distance=True)
            out=run(dtype,'vector',data,linkage,'euclidean')
            np.testing.assert_array_equal(out['CHILD'].reshape(-1,2),expected[0])
            np.testing.assert_allclose(out['DIST'],expected[-1],atol=tol,rtol=tol)
            reports.append(f'{dtype.__name__}/vector/{linkage}')
    for magnitude in [1e200,1e-200]:
        x=np.array([[magnitude,magnitude],[magnitude,2*magnitude]])
        try:
            oracle['linkage_tree'](x,affinity='cosine')
        except ValueError:
            run(np.float64,'tree',x,'complete','cosine',expect_error=True)
        else:raise AssertionError('Expected nonfinite cosine distance rejection')
        reports.append(f'float64/cosine-magnitude-rejection/{magnitude}')
    report=dict(revision=REV,source_hashes=hashes,cases=reports,numpy=np.__version__,scipy=scipy.__version__,python=sys.version,platform=platform.platform(),seed=731,threads=1,flags=['-O3','--fp-mode=contract=off'],tolerance='2e-12 distances and Float64 mean outputs; exact children, labels, shapes and counts',oracle='Pinned sklearn Python and Cython MST/union-find/descendant bodies; Cython distance ABI adapted to SciPy cdist; original SciPy linkage for other paths',assembly_receipt=json.loads(RECEIPT.read_text()))
    (FAMILY/'builds/agglomerative-parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: {len(reports)} agglomerative differential configurations')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--driver',type=Path,default=DRIVER)
    parser.add_argument('--assembly-receipt',type=Path,default=RECEIPT)
    args=parser.parse_args()
    DRIVER=args.driver;RECEIPT=args.assembly_receipt
    main()
