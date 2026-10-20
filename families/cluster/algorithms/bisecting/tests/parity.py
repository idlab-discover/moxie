"""Split-level differential checks against original sklearn bodies and Cython."""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import warnings
import numpy as np
FAMILY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FAMILY/'algorithms/kmeans/tests'))
from oracle import load, SOURCE, REV


def oracle():
    env = load()
    path = SOURCE/'sklearn/cluster/_bisect_k_means.py'
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',str(path)],text=True)
    tree = ast.parse(path.read_text())
    classes = []
    for original in tree.body:
        if not isinstance(original, ast.ClassDef): continue
        methods = [n for n in original.body if isinstance(n, ast.FunctionDef)]
        for n in methods: n.decorator_list = []
        cls = ast.ClassDef(name=original.name, bases=[] if original.name=='_BisectingTree' else [ast.Name(id='_BaseKMeans',ctx=ast.Load())],keywords=[],body=methods,decorator_list=[])
        ast.copy_location(cls, original); classes.append(cls)
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])),str(path),'exec'),env)
    env['restart_trace'] = []
    def trace_single(original):
        def traced(x, w, centers, **kwargs):
            initial = centers.copy()
            result = original(x, w, centers, **kwargs)
            env['restart_trace'].append((x.copy(), initial, result[1]))
            return result
        return traced
    for name in ('_kmeans_single_lloyd', '_kmeans_single_elkan'):
        env[name] = trace_single(env[name])
    base = env['BisectingKMeans']
    class Traced(base):
        def _bisect(self, X, norms, weights, node):
            if not hasattr(self,'nodes'): self.nodes = [node]; self.stats = [(0,0)]
            super()._bisect(X,norms,weights,node)
            for child in (node.left,node.right):
                self.nodes.append(child)
                ix = child.indices
                labels = np.zeros(len(ix),dtype=np.int32)
                self.stats.append((weights[ix].sum(dtype=X.dtype),env['_inertia_dense'](X[ix],weights[ix],child.center.reshape(1,-1),labels,1)))
    env['Traced'] = Traced
    env['bisect_hash'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return env


def main():
    env=oracle(); reports=[]; error_cases=0
    r=np.random.RandomState(831)
    datasets=[('single',np.array([[1.,2],[10,2],[10,8]]),1,np.array([100.,2,1])),
              ('singleton',np.array([[2.]]),1,np.ones(1)),
              ('singleton-split',np.array([[0.],[0.],[1.]]),3,np.ones(3)),
              ('weighted-singleton-split',np.array([[0.],[0.],[.1]]),3,np.array([1.,1.,3.7])),
              ('all-zero-weights',np.arange(8.).reshape(-1,1),2,np.zeros(8)),
              ('one-positive-weight',np.arange(8.).reshape(-1,1),2,np.array([1.,0,0,0,0,0,0,0])),
              ('normal',r.normal(size=(31,4)),5,r.uniform(.1,4,31)),
              ('one-feature',r.normal(size=(29,1)),7,np.ones(29)),
              ('chunk',r.normal(size=(257,2)),4,np.ones(257)),
              ('ties',np.array([[1,1],[10,1],[3,1],[10,0],[2,1],[10,2],[10,8],[10,9],[10,10.]]),3,np.ones(9)),
              ('identical',np.ones((8,3)),4,np.ones(8)),
              ('duplicates',np.repeat(np.arange(4.),4).reshape(-1,1),5,np.ones(16)),
              ('zero-weights',r.normal(size=(17,3)),3,(np.arange(17)%3==0).astype(float))]
    for dtype in (np.float32,np.float64):
      for name,raw,k,raw_w in datasets:
       x=raw.astype(dtype); w=raw_w.astype(dtype)
       query=np.concatenate([x,r.normal(size=(11,x.shape[1]))*50]).astype(dtype)
       for seed in (0,42):
        for strategy in ('biggest_inertia','largest_cluster'):
         for algorithm in ('lloyd','elkan'):
          for init,runs,maximum,tol in [('random',1,300,1e-4),('k-means++',3,300,1e-4),('k-means++',4,1,0),('random',3,300,.5)]:
            context=f'{name}/{dtype.__name__}/{seed}/{strategy}/{algorithm}/{init}/{runs}/{maximum}/{tol}'
            rng=np.random.RandomState(seed)
            model=env['Traced'](k,init=init,n_init=runs,max_iter=maximum,tol=tol,algorithm=algorithm,bisecting_strategy=strategy,random_state=rng)
            args=[FAMILY/'builds/bisecting_driver',32 if dtype==np.float32 else 64,len(x),x.shape[1],len(query),k,seed,init,runs,maximum,tol,algorithm,strategy,*map(float,x.ravel()),*map(float,w),*map(float,query.ravel())]
            reference_error=None
            env['restart_trace'].clear()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore'); model.fit(x.copy(),sample_weight=w)
            except ValueError as exc: reference_error=exc
            proc=subprocess.run(list(map(str,args)),capture_output=True,text=True)
            if reference_error:
                assert proc.returncode > 0 and "Error" in (proc.stdout + proc.stderr),(context,proc.stderr)
                error_cases+=1;reports.append(dict(case=context,expected_error=str(reference_error)));continue
            assert proc.returncode==0,(context,proc.stderr)
            out=defaultdict(list)
            for line in proc.stdout.splitlines():
                key,*values=line.split();out[key].extend(values)
            def arr(key,dt=float):return np.array(out[key],dtype=dt)
            rt,at=(4e-5,4e-6) if dtype==np.float32 else (3e-12,3e-13)
            def close(a,b):np.testing.assert_allclose(a,b,rtol=rt,atol=at)
            try:
                assert arr('RNG',np.uint32)[0]==rng.randint(0,2**32,dtype=np.uint32)
                for key in ('LABEL','PREDICT'):
                    assert np.all((arr(key,int) >= 0) & (arr(key,int) < k))
                label_roundoff = not np.array_equal(arr('LABEL',int), model.labels_)
                mapping = np.arange(k)
                node_mapping = np.arange(2*k-1)
                tie_evidence = None
                if label_roundoff:
                    assert (name,dtype,seed,algorithm,init,runs,maximum,tol) == ('ties',np.float32,42,'lloyd','random',3,300,.5)
                    # The first restart of split two has exact distance ties.
                    # Bound the norm/dot discrepancy, and permit only this known
                    # sibling permutation, not arbitrary partition differences.
                    tx, initial, loss = env['restart_trace'][3]
                    scores = np.empty((len(tx),2),dtype=dtype)
                    for row in range(len(tx)):
                        for col in range(2):
                            norm = dtype(0); dot = dtype(0)
                            for f in range(tx.shape[1]):
                                norm += initial[col,f]*initial[col,f]
                                dot += tx[row,f]*initial[col,f]
                            scores[row,col] = norm - dtype(2)*dot
                    reference = env['_labels_inertia'](tx,np.ones(len(tx),dtype=dtype),initial,1)[0]
                    native = scores.argmin(axis=1)
                    different = native != reference
                    assert different.any()
                    direct = ((tx.astype('d')[:,None,:]-initial.astype('d')[None,:,:])**2).sum(axis=2)
                    np.testing.assert_array_equal(direct[different,0],direct[different,1])
                    bound=8*tx.shape[1]*np.finfo(dtype).eps*((tx.astype('d')**2).sum(axis=1)+(initial.astype('d')**2).sum(axis=1).max())
                    assert np.all(np.abs(scores[:,0]-scores[:,1])[different]<=bound[different])
                    assert loss == 4
                    mapping = np.array([1,0,2]); node_mapping = np.array([0,1,2,4,3])
                    tie_evidence = dict(rows=np.flatnonzero(different).tolist(),native_assignment=native.tolist(),reference_assignment=reference.tolist(),squared_distances=direct.tolist(),native_scores=scores.tolist())
                np.testing.assert_array_equal(mapping[arr('LABEL',int)],model.labels_)
                np.testing.assert_array_equal(mapping[arr('PREDICT',int)],model.predict(query))
                close(arr('CENTER').reshape(k,-1),model.cluster_centers_[mapping])
                close(arr('INERTIA'),model.inertia_)
                close(arr('SCORE'),model.score(x,sample_weight=w))
                direct=np.linalg.norm(query.astype('d')[:,None,:]-model.cluster_centers_.astype('d')[None,:,:],axis=2)
                close(arr('DISTANCE').reshape(len(query),k),direct[:,mapping])
                nodes=getattr(model,'nodes',[model._bisecting_tree]);stats=getattr(model,'stats',[(0,0)])
                node_values=arr('NODE').reshape(-1,6)
                assert len(node_values)==2*k-1
                close(arr('NODECENTER').reshape(-1,x.shape[1]),[nodes[i].center for i in node_mapping])
                for i in range(len(nodes)):
                    node = nodes[node_mapping[i]]
                    expected=[nodes.index(node.left) if node.left else -1,nodes.index(node.right) if node.right else -1,mapping[node.label] if hasattr(node,'label') else -1]
                    np.testing.assert_array_equal(node_values[i,:3],expected)
                    close(node_values[i,3:],[node.score,*stats[node_mapping[i]]])
            except Exception:
                (FAMILY/'builds/bisecting-failure.json').write_text(json.dumps(dict(case=context,args=list(map(str,args)),stdout=proc.stdout,expected_centers=model.cluster_centers_.tolist()),indent=2))
                print('FAIL',context,flush=True); raise
            reports.append(dict(case=context,split_sizes=[len(trace[0]) for trace in env["restart_trace"][::runs]],label_roundoff=label_roundoff,tie_evidence=tie_evidence,tree_differs_from_nearest=bool(np.any(model.predict(query)!=env['_labels_inertia'](query,np.ones(len(query),dtype=dtype),model.cluster_centers_,1)[0]))))
    assert any(case.get("tree_differs_from_nearest", False) for case in reports)
    assert any(case["case"].startswith("weighted-singleton-split/") and 1 in case.get("split_sizes", []) for case in reports)
    gaps = sum(case.get("label_roundoff", False) for case in reports)
    report=dict(label_roundoff_cases=gaps,exact_label_cases=len(reports)-error_cases-gaps,revision=REV,source_hash=env['bisect_hash'],count=len(reports),expected_errors=error_cases,cases=reports,numpy=np.__version__,python=sys.version,compiler=subprocess.check_output(['mojo','--version'],text=True).strip(),platform=platform.platform(),machine=platform.machine(),threads=1,flags=['-O3','--fp-mode=contract=off'])
    (FAMILY/'builds/bisecting-parity.json').write_text(json.dumps(report,indent=2))
    print('PASS',len(reports),'BisectingKMeans differential cases;',error_cases,'matching sampling errors;',gaps,'documented label-roundoff cases')

if __name__=='__main__':main()
