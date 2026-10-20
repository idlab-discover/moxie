"""Exact labels/core indices against pinned DBSCAN fit and Cython expansion."""
from collections import defaultdict
import json
import platform
import subprocess
import sys
import numpy as np
from scipy.spatial.distance import cdist
from oracle import load, FAMILY, REV


def main():
    reference, hashes = load()
    rng = np.random.RandomState(381)
    reports = []
    reduction_cases = 0
    for dtype in [np.float32, np.float64]:
        for n in [0, 1, 7, 8, 9, 127, 128, 129, 257, 8191, 8192, 8193, 16385]:
            for scale in [1., 1e8 if dtype == np.float32 else 1e16]:
                w = rng.choice([-scale, scale, -2., 1., 3.], size=n).astype(dtype)
                args = [FAMILY/'builds/dbscan_driver', 'sum32' if dtype == np.float32 else 'sum64', n, *map(float,w)]
                bits = int(subprocess.check_output(list(map(str,args)), text=True).strip())
                actual = np.array(bits,dtype=np.uint32 if dtype == np.float32 else np.uint64).view(dtype)[()]
                np.testing.assert_equal(actual, w.sum(), err_msg=f'{dtype}/{n}/{scale}')
                reduction_cases += 1
        cases = []
        def add(name, x, eps=1, minimum=3, weights=None, metric='euclidean', p=2):
            cases.append((name, np.asarray(x, dtype=dtype), eps, minimum,
                          None if weights is None else np.asarray(weights, dtype=dtype), metric, p))
        for minimum in [1, 2, 3, 4, 20]:
            add('sklearn-toy-'+str(minimum), [[0],[2],[3],[4],[6],[8],[10]], minimum=minimum)
        add('singleton', [[3, 4]], minimum=1)
        add('large-min-samples', [[0]], minimum=16777217, weights=[16777216])
        add('duplicates', [[0],[1],[1]], minimum=2)
        add('below-boundary', [[0],[1],[1]], eps=np.nextafter(1., 0.), minimum=2)
        add('above-boundary', [[0],[1],[1]], eps=np.nextafter(1., 2.), minimum=2)
        add('dtype-nextafter', [[0],[np.nextafter(dtype(1), dtype(2))]], minimum=2)
        for weights in [[5,5],[6,5],[6,6],[5,1],[5,0],[5.9,.1],[6,0],[6,-1],[-6,7],[0,0]]:
            add('signed-'+str(weights), [[0],[1]], eps=1.5, minimum=6, weights=weights)
        border = np.array([[0],[-1],[-1.5],[-1.5],[-1.5],[1],[1.5],[1.5],[1.5]])
        for order in [np.arange(9), np.arange(8,-1,-1), np.array([5,6,7,8,0,1,2,3,4])]:
            add('ambiguous-border-'+str(order.tolist()), border[order], minimum=4)
        for n in [7, 31, 129]:
            x = rng.normal(size=(n,3))
            for metric in ['euclidean','manhattan','chebyshev','minkowski']:
                add('random-'+str(n), x, eps=1.25, minimum=4, metric=metric, p=3)
                add('random-signed-'+str(n), x, eps=1.25, minimum=4, metric=metric, p=3, weights=rng.randint(-2,5,n))
        for n in [7, 8, 9, 16, 127, 128, 129, 257]:
            weights = np.resize(np.array([1e8,1,-1e8,1,1,1,1,1],dtype=dtype),n)
            add('cancellation-'+str(n), np.zeros((n,1)), minimum=4, weights=weights)
        for index in range(30):
            weights = rng.choice([1e8,-1e8,1,2,-1,-2],size=17).astype(dtype)
            add('cancellation-random-'+str(index), np.zeros((17,1)), minimum=2, weights=weights)
        x = rng.normal(size=(23, 4)).astype(dtype)
        add('precomputed', cdist(x,x).astype(dtype), eps=1.5, minimum=3, metric='precomputed')
        add('nonzero-diagonal', np.eye(10), eps=.5, minimum=5, metric='precomputed')
        add('asymmetric', [[2,0,2],[2,2,0],[2,2,2]], eps=.5, minimum=1, metric='precomputed')
        for name,x,eps,minimum,w,metric,p in cases:
            for algorithm in ['brute','kd_tree','ball_tree','auto']:
                if metric == 'precomputed' and algorithm not in ['brute','auto']: continue
                expected = reference(eps=eps,min_samples=minimum,metric=metric,p=p,algorithm=algorithm).fit(x,sample_weight=w)
                args = [FAMILY/'builds/dbscan_driver',32 if dtype==np.float32 else 64,len(x),x.shape[1],eps,minimum,algorithm,metric,p,'plain' if w is None else 'weighted',*map(float,x.ravel())]
                if w is not None: args.extend(map(float,w))
                proc = subprocess.run(list(map(str,args)), capture_output=True, text=True)
                context = f'{dtype.__name__}/{name}/{metric}/{algorithm}'
                assert proc.returncode == 0, (context,proc.stderr)
                out = defaultdict(list)
                for line in proc.stdout.splitlines():
                    fields=line.split(); out[fields[0]].extend(fields[1:])
                np.testing.assert_array_equal(np.array(out['LABEL'],dtype=int),expected.labels_,err_msg=context)
                np.testing.assert_array_equal(np.array(out['CORE'],dtype=int),expected.core_sample_indices_,err_msg=context)
                np.testing.assert_array_equal(np.array(out['SHAPE'],dtype=int),expected.components_.shape,err_msg=context)
                # Mojo's printed scalar representation is checked at dtype precision.
                np.testing.assert_array_equal(np.array(out['COMPONENT'],dtype=dtype).reshape(expected.components_.shape),expected.components_,err_msg=context)
                reports.append(context)
    result=dict(assembly_receipt=json.loads((FAMILY/'builds/deps/ASSEMBLY.json').read_text()), reduction_cases=reduction_cases, revision=REV,source_hashes=hashes,cases=reports,numpy=np.__version__,python=sys.version,platform=platform.platform(),flags=['-O3','--fp-mode=contract=off'],seed=381,threads=1,oracle='original DBSCAN.fit and compiled _dbscan_inner; independent SciPy cdist radius membership',tolerance='exact labels, core indices and components')
    (FAMILY/'builds/dbscan-parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'PASS: {len(reports)} DBSCAN differential configurations and {reduction_cases} signed reductions')

if __name__ == '__main__': main()
