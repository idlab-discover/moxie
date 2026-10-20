"""Compare native minibatches with unedited local sklearn bodies and Cython updates."""
import ast
import contextlib
import types
import json
import platform
import sys
import subprocess
import warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
FAMILY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FAMILY/'algorithms/kmeans/tests'))
from oracle import load, SOURCE, REV


def oracle():
    env = load()
    env['_get_threadpool_controller'] = lambda: types.SimpleNamespace(limit=lambda **kw: contextlib.nullcontext())
    from sklearn.cluster._k_means_minibatch import _minibatch_update_dense
    env['_minibatch_update_dense'] = _minibatch_update_dense
    tree = ast.parse((SOURCE/'sklearn/cluster/_kmeans.py').read_text())
    step = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_mini_batch_step')
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MiniBatchKMeans')
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef)]
    for n in methods: n.decorator_list = []
    cls = ast.ClassDef(name='MiniBatchKMeans', bases=[ast.Name(id='_BaseKMeans', ctx=ast.Load())], keywords=[], body=methods, decorator_list=[])
    ast.copy_location(cls, original)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[step, cls], type_ignores=[])), str(SOURCE/'sklearn/cluster/_kmeans.py'), 'exec'), env)
    return env


def state(model, rng):
    return dict(META=[model.n_steps_, getattr(model,'n_iter_',-1),getattr(model,'inertia_',-1),getattr(model,'_no_improvement',0),model._n_since_last_reassign,getattr(model,'_ewa_inertia',None) or 0,model._init_size,model._batch_size], CENTER=model.cluster_centers_.ravel().copy(), COUNT=model._counts.copy(), LABEL=getattr(model,'labels_',np.array([])).copy(), RNGPOS=[rng.get_state()[2]], RNGGAUSS=list(rng.get_state()[3:]), RNGKEY=rng.get_state()[1].copy())


def main():
    env = oracle(); reports=[]
    r = np.random.RandomState(182)
    datasets = [(name, x, k, w) for name,x,k,w in [
        ('normal',r.normal(size=(31,4)),4,r.uniform(.1,4,31)),
        ('uniform',r.normal(size=(12,2)),3,np.ones(12)),
        ('duplicates',np.repeat(np.arange(6),4).reshape(12,2),8,np.ones(12)),
        ('identical',np.ones((9,3)),5,np.arange(9)),
        ('sparse-weights',r.normal(size=(17,3)),3,(np.arange(17)%3==0).astype(float)),
        ('many-centers',np.arange(70).reshape(-1,1),40,np.ones(70)),
    ]]
    roundoff_data = r.normal(size=(70,2))
    sort_cases = 0
    for dtype in [np.float32, np.float64]:
        for n in [1, 2, 4, 7, 16, 31, 64, 255, 256, 257, 511, 1025]:
            for values in [np.zeros(n),np.arange(n),np.arange(n)[::-1],r.randint(0,4,n),r.uniform(0,5,n)]:
                values = values.astype(dtype)
                args = [FAMILY/'builds/minibatch_driver',32 if dtype==np.float32 else 64,'sort',n,1,1,0,1,1,0,0,'array',0,10,.01,1,*np.zeros(n),*values]
                proc = subprocess.run(list(map(str,args)),capture_output=True,text=True,check=True)
                order = [int(line.split()[1]) for line in proc.stdout.splitlines()]
                np.testing.assert_array_equal(order,np.argsort(values),err_msg=f'count sort {dtype} {n}')
                sort_cases += 1
    seeding_roundoff = []
    for dtype in [np.float32,np.float64]:
      for name, raw,k,raw_w in datasets:
       x=raw.astype(dtype); w=raw_w.astype(dtype)
       for seed in [0,42]:
        variants=[('fit', 'k-means++', 5, 0, 0, 0., 10, .01, True),
                  ('fit', 'random', 3, 0, 0, 0., -1, .5, True),
                  ('fit', 'array', 1, 1, 4, 0., 0, 1., True),
                  ('fit', 'k-means++', 100, 2, 3, .01, -1, 0., False),
                  ('fit', 'array', 7, 1000, 3, 10., -1, 0., True),
                  ('partial','array',3,1,0,0.,10,1.,True),
                  ('partial','k-means++',5,0,0,0.,10,.01,False),
                  ('internal','random',3,0,0,0.,10,.5,True),
                  ('prefix','array',len(x),0,0,0.,-1,.5,True),
                  ('prefix','array',len(x),0,0,10.,-1,0.,True),
                  ('prefix','array',len(x),0,0,0.,0,0.,True)]
        for mode,init,batch,init_size,n_init,tol,patience,ratio,labels in variants:
            if name == 'many-centers' and init != 'array': continue
            if init=='random' and (w>0).sum()<k: continue
            maximum = 7 if mode == 'prefix' else 12
            initial=x[:k].copy()
            kwargs=dict(n_clusters=k, init=initial.copy() if init=='array' else init,batch_size=batch,max_iter=maximum,init_size=init_size or None,n_init=n_init or 'auto',tol=tol,max_no_improvement=None if patience<0 else patience,reassignment_ratio=ratio,compute_labels=labels)
            args=[str(FAMILY/'builds/minibatch_driver'),'32' if dtype==np.float32 else '64',mode,len(x),x.shape[1],k,seed,batch,maximum,tol,n_init,init,init_size,patience,ratio,int(labels),*x.ravel(),*w,*(initial.ravel() if init=='array' else [])]
            expected=[]
            oracle_error = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rng=np.random.RandomState(seed)
                    model=env['MiniBatchKMeans'](**kwargs,random_state=rng)
                    if mode=='fit':
                        model.fit(x,sample_weight=w); expected.append(state(model,rng))
                    elif mode=='prefix':
                        for step in range(1,maximum+1):
                            rng=np.random.RandomState(seed); model.random_state=rng; model.max_iter=step
                            model.fit(x,sample_weight=w); expected.append(state(model,rng))
                    else:
                        for step in range(maximum):
                            size=len(x) if step==0 else 1+(step*7)%len(x)
                            rows=(np.arange(size)+step*3)%len(x)
                            model.partial_fit(x[rows],sample_weight=w[rows]); expected.append(state(model,rng))
            except ValueError as exc:
                oracle_error = str(exc)
            proc=subprocess.run(list(map(str,args)),capture_output=True,text=True)
            context=(str(dtype),name,seed,mode,init,batch,init_size,n_init,tol,patience,ratio,labels)
            if oracle_error is not None:
                assert proc.returncode != 0 and 'ValueError' in proc.stderr + proc.stdout, (context, oracle_error, proc.returncode, proc.stdout, proc.stderr)
                reports.append(dict(case=context,steps=0,error=oracle_error))
                continue
            assert proc.returncode==0,(context,proc.stderr)
            actual=[]; current=defaultdict(list)
            for line in proc.stdout.splitlines():
                parts=line.split()
                if parts[0]=='END': actual.append(current); current=defaultdict(list)
                else: current[parts[0]].extend(map(float,parts[1:]))
            assert len(actual)==len(expected)
            for step,(got,want) in enumerate(zip(actual,expected)):
                try:
                    for key in ['RNGPOS','RNGKEY','RNGGAUSS','LABEL']:
                        np.testing.assert_array_equal(got[key],want[key],err_msg=key)
                    if mode in ('fit','prefix') or np.all(w == np.floor(w)):
                        np.testing.assert_array_equal(np.asarray(got['COUNT'],dtype=dtype),want['COUNT'],err_msg='COUNT')
                    else:
                        np.testing.assert_allclose(got['COUNT'],want['COUNT'],rtol=4*np.finfo(dtype).eps,atol=0,err_msg='fractional COUNT')
                    for key in ['CENTER']:
                        np.testing.assert_allclose(got[key],want[key],rtol=4e-5 if dtype==np.float32 else 2e-12,atol=2e-6 if dtype==np.float32 else 2e-13,err_msg=key)
                    for index in [0,1,3,4,6,7]: assert got['META'][index]==want['META'][index],('META',index,got['META'],want['META'])
                    for index in [2,5]:
                        if index==5 and got['META'][index]==-1: assert want['META'][index]==0
                        else: np.testing.assert_allclose(got['META'][index],want['META'][index],rtol=4e-5 if dtype==np.float32 else 2e-12,atol=2e-6 if dtype==np.float32 else 2e-13)
                except Exception as exc: raise AssertionError((context,step,str(exc))) from exc
            reports.append(dict(case=context,steps=len(expected)))
    x = roundoff_data.astype(np.float32)
    w = np.ones(len(x),np.float32)
    rng = np.random.RandomState(0)
    model = env['MiniBatchKMeans'](40,batch_size=5,random_state=rng)
    model._check_params_vs_input(x)
    rng.randint(0,len(x),model._init_size)
    distance_calls=[]
    candidate_rows=[]
    original = env['_euclidean_distances']
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        distance_calls.append(result.copy())
        candidate_rows.append(np.asarray(args[0]).copy())
        return result
    env['_euclidean_distances'] = capture
    expected = model._init_centroids(x,env['row_norms'](x,squared=True),'k-means++',rng,w,init_size=model._init_size)
    env['_euclidean_distances'] = original
    args = [FAMILY/'builds/minibatch_driver',32,'init',len(x),2,40,0,5,12,0,0,'k-means++',0,10,.01,1,*x.ravel(),*w]
    proc = subprocess.run(list(map(str,args)),capture_output=True,text=True,check=True)
    parsed=defaultdict(list)
    for line in proc.stdout.splitlines():
        key,*vals=line.split()
        parsed[key].extend(map(float,vals))
    np.testing.assert_array_equal(parsed['RNGKEY'],rng.get_state()[1])
    np.testing.assert_array_equal(parsed['RNGPOS'],[rng.get_state()[2]])
    native=np.array(parsed['CENTER']).reshape(expected.shape)
    divergence=np.flatnonzero(np.max(abs(native-expected),axis=1)>2e-6)
    if len(divergence):
        assert divergence[0] > 0
        evidenced = False
        # Establish that the first differing selection is a reduction-order tie.
        closest=distance_calls[0]
        for step,rows in enumerate(distance_calls[1:],1):
            candidates=np.minimum(closest,rows)
            blas=(candidates @ w[:,None]).ravel()
            sequential=np.array([np.add.accumulate(row*w)[-1] for row in candidates])
            if step==divergence[0]:
                assert blas.argmin()!=sequential.argmin()
                np.testing.assert_allclose(blas,sequential,rtol=8*np.finfo(np.float32).eps,atol=0)
                np.testing.assert_allclose(native[step],candidate_rows[step][sequential.argmin()],rtol=0,atol=2e-6)
                evidenced = True
                seeding_roundoff.append(dict(dtype='float32',first_center=int(step),blas_potentials=blas.tolist(),native_order_potentials=sequential.tolist(),rng_exact=True))
                break
            closest=candidates[blas.argmin()]
        assert evidenced
    (FAMILY/'builds/minibatch-parity.json').write_text(json.dumps(dict(revision=REV,source_hash=env['source_hash'],cases=reports,count=len(reports),numpy=np.__version__,platform=platform.platform(),mojo=subprocess.check_output(['mojo','--version'],text=True).strip(),flags=['-O3','--fp-mode=contract=off'],threads=1,sort_cases=sort_cases,seeding_roundoff=seeding_roundoff),indent=2))
    print('PASS',len(reports),'MiniBatchKMeans cases;',sum(r['steps'] for r in reports),'batch/prefix states;',sort_cases,'count sorts')

if __name__=='__main__': main()
