"""Dense Birch tree, incremental state, label and transform comparisons."""
from collections import defaultdict
import json
import platform
import os
import subprocess
import warnings

import numpy as np
import scipy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from oracle import FAMILY, SOURCE, REV, load


def run(x, threshold, branching, k, batch=0, compute=True, final=-1, layout=0):
    args = [FAMILY/'builds/birch_driver', 32 if x.dtype == np.float32 else 64,
            *x.shape, threshold, branching, k or 0, batch, int(compute), final, layout,
            *map(float, x.ravel())]
    process = subprocess.run(list(map(str,args)),capture_output=True,text=True)
    if process.returncode:
        raise AssertionError(process.stderr + process.stdout)
    result = defaultdict(list)
    for line in process.stdout.splitlines():
        key, *values = line.split()
        result[key].extend(values)
    return {key: np.asarray(values, dtype=str if key == 'NAME' else float if key in {'CENTER','TRANSFORM','SS','LS'} else int)
            for key,values in result.items()}


def tree(node, result):
    result['NODE'].extend([int(node.is_leaf), len(node.subclusters_)])
    for cluster in node.subclusters_:
        result['SAMPLES'].append(cluster.n_samples_)
        result['SS'].append(cluster.squared_sum_)
        result['LS'].extend(cluster.linear_sum_)
        if cluster.child_ is not None:
            tree(cluster.child_, result)


def cases():
    rng = np.random.RandomState(401)
    yield 'random', rng.normal(size=(53, 4))
    yield 'wide', rng.normal(size=(19, 21))
    yield 'blobs', np.repeat(rng.normal(size=(5, 3))*6, 15, axis=0)+rng.normal(size=(75,3))*.05
    yield 'grid', np.array([[i,j] for i in range(5) for j in range(4)], dtype=float)
    yield 'duplicates', np.repeat(np.array([[0,0],[1,1],[4,4],[10,10.]]), 4, axis=0)
    yield 'identical', np.ones((9,3))
    yield 'singleton', np.array([[2.,3.]])
    yield 'threshold-equal', np.array([[0.],[1.],[2.],[3.]])
    # Upstream regression #23269: repeated Float32 centers during tiny-threshold splits.
    path = SOURCE/'sklearn/cluster/tests/test_birch.py'
    import ast
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n,ast.FunctionDef) and n.name == 'test_both_subclusters_updated')
    assignment = next(n for n in node.body if isinstance(n,ast.Assign))
    yield 'duplicate-split-regression', np.asarray(ast.literal_eval(assignment.value.args[0]))


def match_tree(out, reference, features, tolerance):
    """Allow sibling permutations only; preserve every parent/child relationship."""
    nodes = out['NODE'].reshape(-1, 2)
    counts = out['SAMPLES']
    linear = out['LS'].reshape(-1, features)
    squared = out['SS']
    node_index = 0
    cf_index = 0

    def read():
        nonlocal node_index, cf_index
        leaf, size = nodes[node_index]
        node_index += 1
        entries = []
        for _ in range(size):
            i = cf_index
            cf_index += 1
            child = None if leaf else read()
            entries.append((counts[i], linear[i], squared[i], child))
        return bool(leaf), entries

    def compare(actual, expected):
        leaf, entries = actual
        assert leaf == expected.is_leaf and len(entries) == len(expected.subclusters_)
        cost = cdist(np.array([e[1] for e in entries]),
                     np.array([e.linear_sum_ for e in expected.subclusters_]))
        rows, columns = linear_sum_assignment(cost)
        for i, j in zip(rows, columns):
            count, total, ss, child = entries[i]
            cluster = expected.subclusters_[j]
            assert count == cluster.n_samples_
            np.testing.assert_allclose(total, cluster.linear_sum_, rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(ss, cluster.squared_sum_, rtol=tolerance, atol=tolerance)
            if child is not None:
                compare(child, cluster.child_)
    compare(read(), reference)
    assert node_index == len(nodes) and cf_index == len(counts)


def main():
    env, hashes = load()
    reports = []
    ordering_cases = []
    split_sensitivity = []
    original_split = env['_split_node']

    def traced_split(node, threshold, branching_factor):
        centers = node.centroids_
        exact = env['euclidean_distances'](centers, Y_norm_squared=node.squared_norm_, squared=True)
        scalar = np.zeros_like(exact)
        for i in range(len(centers)):
            for j in range(len(centers)):
                if i == j:
                    continue
                dot = np.float64(0)
                norm_i = np.float64(0)
                norm_j = np.float64(0)
                for a, b in zip(centers[i], centers[j]):
                    dot += np.float64(a)*np.float64(b)
                    norm_i += np.float64(a)*np.float64(a)
                    norm_j += np.float64(b)*np.float64(b)
                if centers.dtype == np.float64:
                    norm_i = node.subclusters_[i].sq_norm_
                    norm_j = node.subclusters_[j].sq_norm_
                scalar[i,j] = max(0, (-2*dot + norm_i) + norm_j)
        if exact.argmax() != scalar.argmax():
            a, b = np.unravel_index(exact.argmax(), exact.shape)
            i, j = np.unravel_index(scalar.argmax(), scalar.shape)
            assert {a,b} == {i,j}, 'farthest pair changed, not just its orientation'
            bound = 32*np.finfo(centers.dtype).eps*(centers.shape[1]+2)*max(1,float(exact[a,b]))
            assert np.max(np.abs(exact-scalar)) <= bound
            split_sensitivity.append(float(np.max(np.abs(exact-scalar))))
        return original_split(node, threshold, branching_factor)
    env['_split_node'] = traced_split

    for dtype in (np.float32, np.float64):
        for name, raw in cases():
            x = raw.astype(dtype)
            for threshold, branching in ((.5,50), (.15,2), (1.1,3), (1e-5,5)):
                for k in (None,3):
                    for batch, compute, final in ((0,True,-1),(7,True,-1),(1,False,3)):
                        context = f'{dtype.__name__}/{name}/t={threshold}/b={branching}/k={k}/batch={batch}/labels={compute}/final={final}'
                        model = env['Birch'](threshold=threshold, branching_factor=branching, n_clusters=k, compute_labels=compute)
                        split_sensitivity.clear()
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            if not batch:
                                model.fit(x.copy())
                            else:
                                for start in range(0,len(x),batch): model.partial_fit(x[start:start+batch].copy())
                            if final >= 0:
                                model.n_clusters = final or None
                                model.partial_fit()
                        out = run(x,threshold,branching,k,batch,compute,final)
                        tol = 3e-5 if dtype == np.float32 else 2e-12
                        try:
                            count = len(model.subcluster_centers_)
                            np.testing.assert_array_equal(out['META'],[count,x.shape[1],int(compute),int(model.n_clusters is not None and count < model.n_clusters)])
                            native_centers = out['CENTER'].reshape(count,-1)
                            order_differs = not np.allclose(native_centers,model.subcluster_centers_,rtol=tol,atol=tol)
                            expected_tree = defaultdict(list)
                            tree(model.root_,expected_tree)
                            tree_differs = any(not np.array_equal(out[key],expected_tree[key]) for key in ['NODE','SAMPLES'])
                            if tree_differs or order_differs:
                                assert split_sensitivity, 'no reversed farthest-pair evidence'
                                match_tree(out,model.root_,x.shape[1],tol)
                            permutation = np.arange(count)
                            if order_differs:
                                assert split_sensitivity, 'no tied farthest-pair evidence'
                                rows, permutation = linear_sum_assignment(cdist(native_centers,model.subcluster_centers_))
                                np.testing.assert_array_equal(rows,np.arange(count))
                            np.testing.assert_allclose(native_centers,model.subcluster_centers_[permutation],rtol=tol,atol=tol)
                            label_map = {}
                            for actual, expected in zip(out['SUBLABEL'],model.subcluster_labels_[permutation]):
                                assert actual not in label_map or label_map[actual] == expected
                                label_map[actual] = expected
                            assert len(set(label_map.values())) == len(label_map)
                            def mapped(values):
                                return np.array([label_map[v] for v in values]) if order_differs else values
                            np.testing.assert_array_equal(mapped(out['SUBLABEL']),model.subcluster_labels_[permutation])
                            if compute: np.testing.assert_array_equal(mapped(out['LABEL']),model.labels_)
                            np.testing.assert_array_equal(mapped(out['PREDICT']),model.predict(x))
                            np.testing.assert_array_equal(mapped(out['QUERY']),model.predict(x+dtype(.137)))
                            actual_dist = out['TRANSFORM'].reshape(len(x),count)
                            reference_dist = model.transform(x)[:,permutation]
                            direct_dist = cdist(x.astype('d'), model.subcluster_centers_[permutation].astype('d'))
                            np.testing.assert_allclose(actual_dist,direct_dist,rtol=tol,atol=tol)
                            norm_scale = (np.square(x.astype('d')).sum(axis=1)[:,None]
                                          + np.square(model.subcluster_centers_[permutation].astype('d')).sum(axis=1)[None,:])
                            bound = tol + 32*np.finfo(dtype).eps*(x.shape[1]+2)*norm_scale
                            assert np.all(np.abs(actual_dist**2-reference_dist**2) <= bound)
                            np.testing.assert_array_equal(out['NAME'],[f'birch{i}' for i in range(count)])
                            if order_differs or tree_differs:
                                ordering_cases.append(dict(case=context, permutation=permutation.tolist(),
                                                           max_split_distance_roundoff=max(split_sensitivity), public_order_differs=order_differs))
                            else:
                                expected = defaultdict(list)
                                tree(model.root_,expected)
                                for key in ['NODE','SAMPLES']: np.testing.assert_array_equal(out[key],expected[key])
                                for key in ['SS','LS']: np.testing.assert_allclose(out[key],expected[key],rtol=tol,atol=tol)
                        except AssertionError as exc:
                            raise AssertionError(context) from exc
                        reports.append(context)
            if name == 'random':
                contiguous = run(x,.15,2,None)
                strided = run(x,.15,2,None,layout=1)
                for key in contiguous: np.testing.assert_array_equal(strided[key],contiguous[key])
    report = dict(reference=REV, source_hashes=hashes, python=platform.python_version(),
                  numpy=np.__version__, scipy=scipy.__version__, flags=['-O3','--fp-mode=contract=off'],
                  seed=401, machine=platform.machine(),
                  compiler=subprocess.check_output(['mojo','--version'],text=True).strip(),
                  threads={key:os.environ.get(key) for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']},
                  assembly=json.loads((FAMILY/'builds/deps/ASSEMBLY.json').read_text()),
                  tolerances={'float32':3e-5,'float64':2e-12}, cases=reports,
                  ordering_roundoff=ordering_cases,
                  scope='Original sklearn CF tree and global clustering; direct Float64 pairwise argmin adapter; dense only')
    (FAMILY/'builds/birch-parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: {len(reports)} Birch dense tree and estimator comparisons, plus strided inputs; {len(ordering_cases)} documented ordering-roundoff cases')


if __name__ == '__main__':
    main()
