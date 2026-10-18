"""pyperf whole-process baseline, with GNU time memory measurements."""
from pathlib import Path
import hashlib, json, os, platform, subprocess, sys, tempfile
from check import ROOT, run
FLAGS = ["-O3", "--fp-mode=contract=off"]

def main():
    (ROOT/'benchmarks').mkdir(exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='run-',dir=ROOT/'benchmarks'))
    cpu=min(os.sched_getaffinity(0)); commands=[]
    for bits in [32,64]:
        for rows,cols,k in [(2,4,2),(64,32,8),(2048,128,16)]:
            name=f'{bits}-{rows}-{cols}-{k}'
            cmd=[str(ROOT/'builds/tests/random_projection/benchmark'),str(bits),str(rows),str(cols),str(k)]
            measure=[sys.executable,'-m','pyperf','command','--processes','2','--values','3','--warmups','1','--loops','1','--affinity',str(cpu),'--copy-env','--name',name,'-o',str(out/(name+'.json')),*cmd]
            commands.append(measure); run(*measure)
            with (out/(name+'-rss.txt')).open('w') as f:
                run('time','-v',*cmd,stdout=subprocess.DEVNULL,stderr=f)
    metadata={'commands':commands,'platform':platform.platform(),'cpu':Path('/proc/cpuinfo').read_text(),'mojo':subprocess.check_output(['mojo','--version'],text=True),'flags':FLAGS,'python':sys.version,'pyperf':__import__('pyperf').__version__,'seed':42,'cpu_affinity':cpu,'threads':'OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1; serial projection loops','boundary':'whole process including startup, input generation, fit, transform, allocation, printing; compilation excluded','memory':'input O(n*d), components O(k*d), output O(n*k); no inverse in this workload','lock_sha256':hashlib.sha256((ROOT/'DEPENDENCIES.json').read_bytes()).hexdigest()}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(out)
if __name__=='__main__': main()
