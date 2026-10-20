"""GNU time whole-process peak RSS baseline; run after the required checks."""
from pathlib import Path
import json, os, platform, shutil, subprocess
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'benchmarks';out.mkdir(exist_ok=True)
    metadata={'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
              'compiler':subprocess.check_output(['mojo','--version'],text=True).strip(),
              'flags':'-O3 --fp-mode=contract=off','seed':42,'max_iter':40,'commands':[]}
    for dtype in ['32','64']:
        for n in [8,64,256,768]:
            command=[shutil.which('time'),'-v','-o',str(out/f'rss-{dtype}-{n}.txt'),'taskset','-c','0',str(ROOT/'builds/benchmark'),dtype,str(n),'1']
            p=subprocess.run(command,text=True,capture_output=True,check=True)
            metadata['commands'].append({'argv':command,'stdout':p.stdout,'stderr':p.stderr})
    (out/'memory.json').write_text(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
