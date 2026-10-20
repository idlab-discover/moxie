"""Raw residual/screening solver parity: Float64 1e-8, Float32 3e-4."""
from pathlib import Path
import sys, json, subprocess, warnings
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parents[1]/'tools'))
import parity_io as io
from pinned_coordinate import load
from childcheck import environment
reference,hashes=load(ROOT)
rng=np.random.default_rng(53)
cases=[]
for bits,dtype in ((64,np.float64),(32,np.float32)):
    for n,p in ((1,1),(7,12),(30,5)):
        for alpha,beta in ((0.,0.),(0.,.3),(.2,0.),(.2,.3),(100.,0.)):
            for limit in (1,1000):
                X=rng.normal(size=(n,p)).astype(dtype)
                y=rng.normal(size=n).astype(dtype)
                initial=rng.normal(size=p).astype(dtype)
                cases.append(dict(id=str(len(cases)),bits=bits,X=X,y=y,initial=initial,alpha=alpha,beta=beta,max_iter=limit,tol=1e-4,positive=0))
folder=ROOT/'builds/parity';folder.mkdir(parents=True,exist_ok=True)
fixtures=folder/'fixtures.txt'
fixtures.write_text('\n'.join(io.case_line(c['id'],c['bits'],{k:c[k] for k in ('alpha','beta','max_iter','tol','positive')},{k:c[k] for k in ('X','y','initial')}) for c in cases)+'\n')
out=subprocess.run([str(ROOT/'builds/parity_driver'),str(fixtures)],env=environment(ROOT),capture_output=True,text=True,check=True).stdout
(folder/'native.txt').write_text(out); results=io.parse_results(out)
assert set(results)=={c['id'] for c in cases}
report=[]
for c in cases:
    got=results[c['id']];assert 'error' not in got,(c,got)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected=reference.cd_fast.enet_coordinate_descent(c['initial'].copy(),c['alpha'],c['beta'],np.asfortranarray(c['X']),c['y'],c['max_iter'],c['tol'],np.random.RandomState(0),positive=bool(c['positive']))
    t=1e-8 if c['bits']==64 else 3e-4
    scale=max(1.,float(np.sum(c['y']**2)))
    coef=io.close(got['coef'],expected[0],t,t*scale,c['id']+' coef')
    gap=io.close(got['gap'],expected[1],t,t*scale,c['id']+' gap')
    if c['bits']==64:assert got['iterations']==expected[3],(c['id'],got['iterations'],expected[3])
    report.append(dict(case=c['id'],coef=coef,gap=gap))
(folder/'report.json').write_text(json.dumps(dict(revision=io.REVISION,sources=hashes,cases=len(cases),errors=report),indent=2)+'\n')
print(f'Coordinate engine parity passed: {len(cases)} pinned Cython cases')
