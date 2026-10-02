"""pyperf baseline: native timed regions and pinned reference, data setup excluded."""
import os, subprocess, sys, time, warnings
from pathlib import Path
import numpy as np
import pyperf
from oracle import update
ROOT=Path(__file__).resolve().parents[1]

def native_time(loops,dtype,mode,n,p):
    env=os.environ.copy();env.pop('MOXIE_STATISTICS_ALLOC_TRACE',None)
    output=subprocess.check_output([str(ROOT/'builds/memory_probe'),dtype,mode,str(n),str(p),'256','C',str(loops)],env=env,text=True)
    return float(next(line.split()[1] for line in output.splitlines() if line.startswith('elapsed ')))

def reference_time(loops,x,w,mode):
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        start=time.perf_counter()
        for _ in range(loops):
            if mode=='moments': update(x,0.,0.,0.,sample_weight=w)
            elif mode in ('precision0','precision1'):
                axis=0 if mode=='precision0' else 1
                np.nanmean(x,axis=axis);np.nanstd(x,axis=axis);np.nanmean(x,axis=axis)
            elif mode=='extrema':
                np.nanmin(x,axis=0);np.nanmax(x,axis=0);np.nanmax(np.abs(x),axis=0)
            else:
                np.nanmedian(x,axis=0)
                for j in range(x.shape[1]): np.nanpercentile(x[:,j],[25.,75.])
        return time.perf_counter()-start

if __name__=='__main__':
    destination=ROOT/'benchmarks/performance.json'
    if '--worker' not in sys.argv and destination.exists(): destination.unlink()
    runner=pyperf.Runner()
    runner.metadata.update(compiler='Mojo 1.0.0 / MAX 26.5.0',flags='-O3 --fp-mode=contract=off',
                           workers=1,notes='Concurrent runner peers may distort timings; no speedup claim. Native setup/startup excluded.')
    for dtype in ('f32','f64'):
        for n,p in ((8,3),(4096,64)):
            x=(((np.arange(n*p)*13)%101-50)/8).astype('float32' if dtype=='f32' else 'float64').reshape(n,p)
            x[::17,0]=np.nan
            w=np.where(np.arange(n)%5==0,-1,2).astype(x.dtype)
            for mode in ('moments','extrema','quantiles','precision0','precision1'):
                name=f'{dtype}-{n}x{p}-{mode}'
                runner.bench_time_func('native-'+name,native_time,dtype,mode,n,p)
                runner.bench_time_func('reference-'+name,reference_time,x,w,mode)
            runner.bench_time_func(f'loop-{dtype}-{n}x{p}',native_time,dtype,'loop',n,p)

            runner.bench_time_func(f'precision-loop-{dtype}-{n}x{p}',native_time,dtype,'precision_loop',n,p)
