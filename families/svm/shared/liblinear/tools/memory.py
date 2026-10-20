"""GNU time whole-process peak RSS, not an allocator or kernel-only measurement."""
from pathlib import Path
import subprocess
import shutil
ROOT=Path(__file__).resolve().parents[1]
output=ROOT/'profiling';output.mkdir(exist_ok=True)
for n,p in ((8,3),(128,16),(2048,64)):
    subprocess.run([shutil.which('time'),'-v','-o',str(output/f'memory-{n}-{p}.txt'),str(ROOT/'builds/benchmark'),str(n),str(p),'2','1'],check=True)
