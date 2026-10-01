"""Small pyperf whole-process integration baseline; includes startup and output."""
from pathlib import Path
import hashlib, json, os, platform, subprocess, sys, tempfile
from check import ROOT, CONSUMERS, environment

def main():
    out=Path(tempfile.mkdtemp(prefix='run-',dir=ROOT/'benchmarks'))
    commands=[]
    cpu=min(os.sched_getaffinity(0))
    for consumer in CONSUMERS:
        env=environment(consumer)
        for n in [16,10000]:
            command=[str(consumer/'builds/probe'),'42',str(n),'stratified','1','0','1']
            measure=[sys.executable,'-m','pyperf','command','--processes','2','--values','3',
                '--warmups','1','--loops','1','--affinity',str(cpu),'--copy-env',
                '--name',f'{consumer.name}-{n}','-o',str(out/f'{consumer.name}-{n}.json'),*command]
            commands.append(measure)
            subprocess.run(measure,check=True,cwd=consumer,env=env)
            with (out/f'{consumer.name}-{n}-rss.txt').open('w') as memory:
                subprocess.run(['time','-v',*command],check=True,cwd=consumer,env=env,
                    stdout=subprocess.DEVNULL,stderr=memory)
    metadata={'commands':commands,'platform':platform.platform(),'cpu':Path('/proc/cpuinfo').read_text(),
        'python':sys.version,'pyperf':__import__('pyperf').__version__,
        'mojo':subprocess.check_output(['mojo','--version'],text=True),'flags':'-O3 --fp-mode=contract=off',
        'threads':'serial RNG, OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1','seed':42,'affinity':cpu,
        'boundary':'whole process, native contracts, allocation, numerical work and stdout; compilation excluded',
        'array_memory':'counts 6*k Float64; draws n*k Int64; total O(n*k), k<=3; RNG 624 UInt32 words',
        'manifests':{str(c.relative_to(ROOT)):hashlib.sha256((c/'DEPENDENCIES.lock.json').read_bytes()).hexdigest() for c in CONSUMERS}}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(out)
if __name__=='__main__': main()
