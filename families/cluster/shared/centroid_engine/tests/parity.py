"""Compare native kernels with unmodified, locally compiled sklearn Cython."""
import json, os, subprocess
from pathlib import Path
import numpy as np
from oracle import load
ROOT=Path(__file__).resolve().parents[1]
common,lloyd,elkan,mini=load()
records=[]
failures=[]
for dtype_name,dtype,tol in [('f32',np.float32,3e-5),('f64',np.float64,1e-12)]:
    for n,d,k in [(1,1,1),(9,3,1),(10,1,3),(19,7,3),(37,5,5),(65,17,4),(257,3,3)]:
      for mode in range(5):
       for layout in range(2):
        case=(dtype_name,n,d,k,mode,layout)
        raw=subprocess.check_output([str(ROOT/'builds/evaluate'),*map(str,case)],text=True)
        actual={}
        for line in raw.splitlines():
            parts=line.split()
            if parts[0]!='shape':
                name,index,value=parts
                assert int(index)==len(actual.setdefault(name,[]))
                actual[name].append(float(value))
        owner=(((np.arange(n)[:,None]*17+np.arange(d*2)[None,:]*7)%67-33).astype(dtype)/8)
        x=np.ascontiguousarray(owner[:,::2] if not layout else owner[:,::-2])
        if mode==2: x.fill(2)
        c=((np.arange(k)[:,None]*19+np.arange(d)[None,:]*11)%43-21).astype(dtype)/8
        if mode in (1,2): c[:]=x[0]
        w=((np.arange(n)*3)%7).astype(dtype)/2
        if mode==3: w.fill(0)
        if mode==4: w.fill(1)
        def check(name,value):
            value=np.asarray(value)
            a=np.asarray(actual[name]).reshape(value.shape)
            try:
                if 'labels' in name: np.testing.assert_array_equal(a,value)
                else: np.testing.assert_allclose(a,value,rtol=tol,atol=tol)
            except AssertionError as exc:
                raise AssertionError(f'{case}: {name}\n{exc}') from exc
        try:
            check('x',x);check('weights',w);check('initial',c)
            dist=np.empty((n,k),dtype=dtype)
            for i in range(n):
                for j in range(k): dist[i,j]=common._euclidean_dense_dense_wrapper(x[i],c[j],True)
            check('squared',dist);check('distance',np.sqrt(dist))
            ec=c.copy();bc=c.copy();bm=np.zeros(k,dtype=dtype)
            elabels=np.zeros(n,dtype=np.int32);upper=np.zeros(n,dtype=dtype);lower=np.zeros((n,k),dtype=dtype)
            def half_distances(centers):
                result=np.empty((k,k),dtype=dtype)
                for i in range(k):
                    for j in range(k): result[i,j]=common._euclidean_dense_dense_wrapper(centers[i],centers[j],False)*0.5
                return result
            half=half_distances(ec)
            elkan.init_bounds_dense(x,ec,half,elabels,upper,lower,1)
            check('upper_init',upper);check('lower_init',lower);check('elabels_init',elabels)
            for step in range(4):
                prefix=str(step)+'_'
                labels=np.zeros(n,dtype=np.int32);new=np.zeros_like(c);mass=np.zeros(k,dtype=dtype);shift=np.zeros(k,dtype=dtype)
                lloyd.lloyd_iter_chunked_dense(x,w,c,new,mass,labels,shift,1,False)
                check(prefix+'labels',labels)
                check(prefix+'inertia',common._inertia_dense(x,w,c,labels,1))
                check(prefix+'single',common._inertia_dense(x,w,c,labels,1,0))
                enew=np.zeros_like(ec);emass=np.zeros(k,dtype=dtype);eshift=np.zeros(k,dtype=dtype)
                next_center=np.partition(half,1,axis=0)[1] if k>1 else np.array([np.inf],dtype=dtype)
                elkan.elkan_iter_chunked_dense(x,w,ec,enew,emass,half,next_center,upper,lower,elabels,eshift,1,False)
                check(prefix+'elabels',elabels);check(prefix+'upper',upper);check(prefix+'lower',lower)
                elkan.elkan_iter_chunked_dense(x,w,ec,enew,emass,half,next_center,upper,lower,elabels,eshift,1,True)
                check(prefix+'ecenters',enew);check(prefix+'emasses',emass)
                check(prefix+'upper_advanced',upper);check(prefix+'lower_advanced',lower)
                ec=enew;half=half_distances(ec)
                lloyd.lloyd_iter_chunked_dense(x,w,c,new,mass,labels,shift,1,True)
                check(prefix+'centers',new);check(prefix+'masses',mass);check(prefix+'shift',shift)
                c=new
                blabels=np.zeros(n,dtype=np.int32)
                lloyd.lloyd_iter_chunked_dense(x,w,bc,np.zeros_like(bc),np.zeros_like(bm),blabels,np.zeros_like(bm),1,False)
                bnew=np.zeros_like(bc)
                mini._minibatch_update_dense(x,w,bc,bnew,bm,blabels,1)
                check(prefix+'batch_centers',bnew);check(prefix+'batch_masses',bm)
                bc=bnew
            records.append(case)
        except AssertionError as exc:
            failures.append(str(exc))
report={'passed':records,'failures':failures,'tolerances':{'float32':3e-5,'float64':1e-12},'threads':1}
(ROOT/'docs/private/centroid/parity.json').write_text(json.dumps(report,indent=2))
if failures:
    print('\n\n'.join(failures[:5]))
    raise AssertionError(f'{len(failures)} failing cases; {len(records)} passed; see docs/private/centroid/parity.json')
print(f'PASS {len(records)} dense kernel cases, 4 iterations each, local Cython oracle')

from selection_parity import main as selection_parity
selection_parity(common)
