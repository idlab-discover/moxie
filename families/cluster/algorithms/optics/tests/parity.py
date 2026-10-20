"""Exact graph/order/Xi hierarchy differential tests against pinned sklearn."""
from collections import defaultdict
import json
import platform
import subprocess
import sys
import warnings
import numpy as np
from scipy.spatial.distance import cdist
from oracle import load,FAMILY,REV


def run(dtype,x,config):
    args=[FAMILY/'builds/optics_driver',32 if dtype==np.float32 else 64,*x.shape,
          config.get('min_samples',5),config.get('max_eps',np.inf),config.get('metric','minkowski'),
          config.get('cluster_method','xi'),config.get('eps',-1),config.get('xi',.05),
          config.get('min_cluster_size',-1),int(config.get('predecessor_correction',True)),
          config.get('algorithm','auto'),config.get('p',2),*map(float,x.ravel())]
    p=subprocess.run(list(map(str,args)),capture_output=True,text=True)
    assert p.returncode==0,(args[:14],p.stderr)
    result=defaultdict(list)
    for line in p.stdout.splitlines():
        key,*values=line.split();result[key].extend(values)
    return {k:np.array(v,dtype=float if k in ['CORE','REACH'] else int) for k,v in result.items()}


def main():
    oracle,hashes=load();rng=np.random.RandomState(82);reports=[]
    cases=[('regression',np.array([1,2,3,1,8,8,7,100])[:,None]),
           ('reordered',np.array([1,2,3,8,8,7,100,1])[:,None]),
           ('duplicates',np.zeros((8,2))),('ties',np.arange(12)[:,None]),
           ('blobs',np.concatenate([rng.normal(v,.2,(13,3)) for v in [-3,0,3]])),
           ('nested',np.concatenate([rng.normal(0,.1,(30,2)),rng.normal(0,10,(25,2))])),
           ('wide',rng.normal(size=(17,19)))]
    for dtype in [np.float32,np.float64]:
        for name,raw in cases:
            data=raw.astype(dtype)
            configurations=[dict(min_samples=3,algorithm=a,metric=m) for a in ['auto','brute','kd_tree','ball_tree'] for m in ['minkowski','euclidean','manhattan','chebyshev']]
            configurations += [dict(min_samples=s,max_eps=e,xi=xi,predecessor_correction=pc,min_cluster_size=ms)
                               for s in [2,.2] for e in [0,.5,np.inf] for xi in [0,.05,.3,.99] for pc in [False,True] for ms in [2,.3,1.]]
            configurations += [dict(min_samples=3,cluster_method='dbscan',eps=e,max_eps=mx) for mx in [.5,np.inf] for e in [0,.1,.5]]
            configurations += [dict(min_samples=3,cluster_method='dbscan',eps=.5,min_cluster_size=99),dict(min_samples=3,cluster_method='dbscan'),dict(min_samples=3,metric='minkowski',p=3),dict(min_samples=3,metric='precomputed',algorithm='brute')]
            for config in configurations:
                x=cdist(data,data).astype(dtype) if config.get('metric')=='precomputed' else data
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');expected=oracle['OPTICS'](**config).fit(x)
                out=run(dtype,x,config)
                context=(dtype.__name__,name,config)
                for key,attr in [('LABEL','labels_'),('ORDER','ordering_'),('PRED','predecessor_')]:
                    np.testing.assert_array_equal(out[key],getattr(expected,attr),err_msg=str(context))
                for key,attr in [('CORE','core_distances_'),('REACH','reachability_')]:
                    np.testing.assert_allclose(out[key],getattr(expected,attr),rtol=2e-12,atol=2e-12,err_msg=str(context))
                if config.get('cluster_method','xi')=='xi':
                    h=expected.cluster_hierarchy_.reshape(-1,2)
                    np.testing.assert_array_equal(out['SHAPE'],h.shape,err_msg=str(context))
                    np.testing.assert_array_equal(out.get('HIER',[]),h.ravel(),err_msg=str(context))
                reports.append(dict(dtype=dtype.__name__,case=name,params=config))
    result=dict(revision=REV,source_hashes=hashes,cases=reports,python=sys.version,numpy=np.__version__,platform=platform.platform(),seed=82,threads=1,
        flags=['-O3','--fp-mode=contract=off'],tolerance='Exact labels, ordering, predecessor and hierarchy; Float64 distances rtol=atol=2e-12 for backend arithmetic and decimal rounding',
        oracle='Original pinned sklearn OPTICS/graph/extraction bodies with independent scipy.spatial.distance.cdist neighbor queries; estimator promotes both input dtypes to Float64',
        assembly_receipt=json.loads((FAMILY/'builds/deps/ASSEMBLY.json').read_text()))
    (FAMILY/'builds/optics-parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'PASS: {len(reports)} OPTICS differential configurations')

if __name__=='__main__':main()
