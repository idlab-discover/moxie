"""Run Valgrind on the full native contract workload after pixi run build."""
from pathlib import Path
import os
import subprocess
ROOT=Path(__file__).resolve().parents[1]
output=ROOT/'profiling';output.mkdir(exist_ok=True)
env=os.environ.copy();env['XDG_CACHE_HOME']=str(ROOT/'.pixi/cache')
subprocess.run(['valgrind','--tool=memcheck','--track-origins=yes','--error-exitcode=99',f'--log-file={output}/memcheck-final.log',str(ROOT/'builds/test_engine')],cwd=ROOT,env=env,check=True)
