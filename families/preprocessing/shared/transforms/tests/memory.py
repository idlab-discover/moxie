"""Fail on excess live memory; supplement RSS with allocation-site and TCMalloc evidence."""
from pathlib import Path
import json, os, re, subprocess, sys, platform, hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/memory';OUT.mkdir(parents=True,exist_ok=True)
BIN=ROOT/'builds/workload'
# A restrictive allocation-site audit makes unmeasured hidden paths a hard failure.
# New allocating expressions require budget review, not simply a larger RSS allowance.
source=(ROOT/'src/moxie/preprocessing/_dense_transforms.mojo').read_text()
assert not re.search(r'\.(copy|astype|reshape|contiguous|T)\(',source)
sites=[l.strip() for l in source.splitlines() if '= NDArray[' in l]
assert sites==[
 'var result = NDArray[d](Shape(Int(x.shape[0]), Int(x.shape[1])))',
 'var result = NDArray[d](Shape(Int(x.shape[0]), Int(x.shape[1])))',
 'var tile = NDArray[d](Shape(1, width))',
 'var tile = NDArray[d](Shape(1, width) if axis == 1 else Shape(width, 1))',
 'self.values = NDArray[Self.d](Shape(features))'],sites
assert 'var width = min(p, tile_bytes //' in source
assert 'var width = min(Int(x.shape[axis]), tile_bytes //' in source
assert not re.search(r'\b(List|alloc|unsafe_alloc)\[',source)

records=[]
def measure(dtype,mode,n,p,layout,tile):
    key=f'{dtype}-{mode}-{n}-{p}-{layout}-{tile}'
    command=['time','-f','%M','-o',str(OUT/(key+'.rss')),str(BIN),dtype,mode,str(n),str(p),str(layout),str(tile),'1']
    result=subprocess.run(command,text=True,capture_output=True,check=True)
    (OUT/(key+'.log')).write_text(result.stdout+result.stderr)
    rss=int((OUT/(key+'.rss')).read_text())*1024
    largest=int(re.search(r'largest_tile (\d+)',result.stdout)[1])
    assert largest*(4 if dtype=='f32' else 8)<=tile
    record=dict(dtype=dtype,mode=mode,n=n,p=p,layout=layout,tile=tile,rss=rss,command=command)
    records.append(record);return rss

# Peak test RAM budget < 512 MiB per subprocess. Largest table is 64 MiB;
# startup/page/allocator noise allowance 12 MiB is smaller than an extra table.
noise=12*1024**2
for dtype in ('f32','f64'):
 for n,p,layout in [(32768,8,0),(8,32768,1),(8192,1024,0),(2,3,2),(512,513,3)]:
  table=n*p*(4 if dtype=='f32' else 8)
  for tile in (256,65536):
   control=measure(dtype,'control',n,p,layout,tile)
   control_into=measure(dtype,'control_into',n,p,layout,tile)
   for mode in ('copy','inplace','into','sink','norm_copy','norm_inplace','norm_into','norm_sink','features'):
    peak=measure(dtype,mode,n,p,layout,tile)
    base=control_into if mode.endswith('into') else control
    allowed=table if mode.endswith('copy') else (p*(4 if dtype=='f32' else 8) if mode=='features' else 0)
    assert peak-base<=allowed+tile+noise,(records[-1],base,allowed)
   if table>=32*1024**2:
    peak=measure(dtype,'negative',n,p,layout,tile)
    assert peak-control>tile+noise,('negative/control did not detect table',peak,control)

# Callgrind sees NuMojo's custom TCMalloc allocator (unlike default Massif/DHAT).
def calls(path):
 lines=path.read_text().splitlines();names={};count=0;current=''
 for line in lines:
  m=re.match(r'(?:c?fn)=\((\d+)\) (.+)',line)
  if m:names[m[1]]=m[2]
 for line in lines:
  m=re.match(r'cfn=\((\d+)\)',line)
  if m:current=names.get(m[1],'')
  if line.startswith('calls=') and 'TCMallocGlobals::tc_new(' in current:
   count+=int(line.split()[0][6:])
 return count
counts=[]
for dtype in ('f32','f64'):
 for mode in ('control','copy','inplace','into','sink','norm_copy','norm_inplace','norm_into','norm_sink','negative'):
  observed=[]
  for reps in (0,2):
   path=OUT/f'alloc-{dtype}-{mode}-{reps}.callgrind'
   command=['valgrind','--tool=callgrind',f'--callgrind-out-file={path}',str(BIN),dtype,mode,'16','8','1','65536',str(reps)]
   result=subprocess.run(command,text=True,capture_output=True,check=True)
   (path.with_suffix('.log')).write_text(result.stdout+result.stderr)
   observed.append(calls(path))
  delta=(observed[1]-observed[0])/2
  counts.append(dict(dtype=dtype,mode=mode,allocations_per_call=delta,counts=observed))
  # Array metadata and shared-view handles allocate too. No per-element allocations.
  limits={'control':0,'copy':8,'inplace':8,'into':0,'sink':120,'norm_copy':8,'norm_inplace':8,'norm_into':0,'norm_sink':120,'negative':16}
  assert delta<=limits[mode],counts[-1]
  if mode=='copy':assert delta>0,'allocator profiler missed NuMojo'
for dtype in ('f32','f64'):
 a={r['mode']:r['allocations_per_call'] for r in counts if r['dtype']==dtype}
 assert a['negative']>a['inplace'],'allocator failed deliberate temporary control'
(OUT/'results.json').write_text(json.dumps(dict(ram_budget_bytes=512*1024**2,noise_bytes=noise,records=records,allocator=counts,host=platform.platform(),cpu=Path('/proc/cpuinfo').read_text(), compiler=subprocess.check_output(['mojo','--version'],text=True), valgrind=subprocess.check_output(['valgrind','--version'],text=True), flags='-O3 --fp-mode=contract=off', seed='deterministic signed modular pattern', workers=1, threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')}, engine_sha256=hashlib.sha256(source.encode()).hexdigest(), binary_sha256=hashlib.sha256(BIN.read_bytes()).hexdigest()),indent=2)+'\n')
print(f'PASS memory: {len(records)} isolated RSS runs, {len(counts)} TCMalloc comparisons, bounded allocation sites, deliberate temporary rejected')
