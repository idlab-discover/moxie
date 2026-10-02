"""Executable allocation/RSS budgets plus an intentionally failing copy control."""
from pathlib import Path
import json, os, platform, re, subprocess, sys, shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check import environment
OUT=ROOT/'benchmarks/memory';OUT.mkdir(parents=True,exist_ok=True)
PROF=ROOT/'profiling';PROF.mkdir(exist_ok=True)
ENV=environment();ENV['MOXIE_STATISTICS_ALLOC_TRACE']='1'
RAM_BUDGET=512*1024**2
NOISE=8*1024**2
records=[]

def command(dtype,mode,n,p,tile,layout,reps=1):
    return [str(ROOT/'builds/memory_probe'),dtype,mode,str(n),str(p),str(tile),layout,str(reps)]

def expected_bytes(mode,n,p):
    return {'control':0,'precision0':0,'precision1':0,'moments':32*p,'extrema':24*p,'quantiles':24*p+8*n}[mode]

def native(dtype,mode,n,p,tile,layout):
    stem=f'{dtype}-{mode}-{n}-{p}-{tile}-{layout}'
    log=OUT/f'{stem}.rss'
    cmd=command(dtype,mode,n,p,tile,layout)
    run=subprocess.run([str(ROOT/'.pixi/envs/default/bin/time'),'-f','%M','-o',str(log),*cmd],env=ENV,check=True,text=True,capture_output=True)
    (OUT/f'{stem}.stdout').write_text(run.stdout)
    allocations=[int(line.split()[1]) for line in run.stdout.splitlines() if line.startswith('statistics_allocation ')]
    if mode!='forbidden':
        assert sum(allocations)==expected_bytes(mode,n,p),(cmd,allocations)
        expected_count={'control':0,'precision0':0,'precision1':0,'moments':4,'extrema':3,'quantiles':4}[mode]
        assert len(allocations)==expected_count,(cmd,allocations)
    rss=int(log.read_text().strip())*1024
    assert rss<RAM_BUDGET,(cmd,rss)
    rec=dict(command=cmd,rss=rss,allocations=allocations,stdout=run.stdout)
    records.append(rec)
    return rss

def allowed(mode,n,p): return expected_bytes(mode,n,p)+NOISE

# Full-table allocation must exceed the same acceptance threshold by a wide margin.
for dtype in ('f32','f64'):
    for n,p,layout in [(65536,64,'C'),(131072,64,'reverse'),(8192,1024,'C'),(5,3,'tiny')]:
        baseline=min(native(dtype,'control',n,p,256,layout) for _ in range(2))
        for mode in ('moments','extrema','precision0','precision1'):
            for tile in (1,257):
                observed=native(dtype,mode,n,p,tile,layout)-baseline
                assert observed<=allowed(mode,n,p),(dtype,mode,n,p,observed,allowed(mode,n,p))
        forbidden=native(dtype,'forbidden',n,p,256,layout)-baseline
        assert forbidden>allowed('moments',n,p),(dtype,n,p,'negative copy control not detected',forbidden)
    # Workspace growth: n sweep and p sweep. Only one column is ever allocated.
    for n,p,layout in [(8192,32,'C'),(65536,32,'reverse'),(65536,64,'C'),(5,3,'tiny')]:
        baseline=min(native(dtype,'control',n,p,256,layout) for _ in range(2))
        observed=native(dtype,'quantiles',n,p,256,layout)-baseline
        assert observed<=allowed('quantiles',n,p),(dtype,'quantiles',n,p,observed)
        if n*p>=2**21:
            forbidden=native(dtype,'forbidden',n,p,256,layout)-baseline
            assert forbidden>allowed('quantiles',n,p)

# Prevent silent bypasses of the traced allocation site. This source audit is
# paired with actual native allocator call counts below, not used alone.
source=(ROOT/'src/moxie/preprocessing/_moments.mojo').read_text()
assert source.count('zeros[DType.float64](')==1
for forbidden in ('.copy(','.astype[','.contiguous(','.flatten(','.reshape(','.T(', 'List[', 'alloc('):
    assert forbidden not in source,forbidden
assert source.count('_allocate_vector(features)')==10
assert source.count('_allocate_vector(x.shape[0])')==1
assert shutil.which('valgrind'), 'Valgrind required for actual NuMojo allocation evidence'

def allocation_calls(path):
    lines=path.read_text().splitlines();names={};counts=0;current=''
    for line in lines:
        m=re.match(r'(?:c?fn)=\((\d+)\) (.+)',line)
        if m:names[m[1]]=m[2]
    for line in lines:
        m=re.match(r'cfn=\((\d+)\)',line)
        if m:current=names.get(m[1],'')
        if line.startswith('calls=') and 'TCMallocGlobals::tc_new(' in current:
            counts+=int(line.split()[0][6:])
    assert counts>0,'profiler missed NuMojo allocator'
    return counts

calls=[]
for dtype in ('f32','f64'):
    dtype_calls=[]
    for n,p in ((32,4),(257,4),(32,1),(257,1),(16385,1)):
        totals={}
        for mode in ('control','moments','extrema','quantiles','precision0','precision1','forbidden'):
            path=PROF/f'alloc-{dtype}-{n}-{p}-{mode}.callgrind'
            cmd=['valgrind','--tool=callgrind',f'--callgrind-out-file={path}',*command(dtype,mode,n,p,7,'reverse')]
            run=subprocess.run(cmd,env=ENV,text=True,capture_output=True,check=True)
            (PROF/f'{path.stem}.log').write_text(run.stdout+run.stderr)
            totals[mode]=allocation_calls(path)
        delta={k:v-totals['control'] for k,v in totals.items()}
        calls.append(dict(dtype=dtype,n=n,p=p,totals=totals,delta=delta))
        dtype_calls.append(delta)
        for mode in ('moments','extrema','quantiles'):
            assert 0<delta[mode]<200,(dtype,n,mode,delta)
        assert delta['precision0']==delta['precision1']==0, (dtype,n,p,delta)
        assert delta['forbidden']>0, 'allocator evidence did not detect negative control'
    # No per-row/per-element allocator calls, including sorting and missing masks.
    assert all(delta == dtype_calls[0] for delta in dtype_calls), dtype_calls

tools={name:subprocess.check_output([str(executable),'--version'],env=ENV,text=True).splitlines()[0] for name,executable in [('valgrind','valgrind'),('time',ROOT/'.pixi/envs/default/bin/time'),('pixi','pixi')]}
report=dict(tool_versions=tools,cpu_info=Path('/proc/cpuinfo').read_text().split('\n\n')[0],thread_environment={k:ENV[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},ram_budget=RAM_BUDGET,rss_noise_allowance=NOISE,records=records,callgrind=calls,
            platform=platform.platform(),processor=platform.processor(),
            compiler=subprocess.check_output(['mojo','--version'],env=ENV,text=True),
            flags=['-O3','--fp-mode=contract=off'],workers=1,seed='deterministic modular arithmetic',
            scope='engine only; consumers must test their estimator/output paths')
(OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'PASS memory: {len(records)} isolated runs; traced bytes, native allocator calls, RSS and prohibited-copy controls',flush=True)
subprocess.run([sys.executable,str(ROOT/'tests/performance.py'),'--processes','1','--values','3','--warmups','1','--loops','3','-o',str(ROOT/'benchmarks/performance.json')],env=environment(),check=True)
