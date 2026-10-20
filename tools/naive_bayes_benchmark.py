"""Small baseline: native elapsed fitting time, excluding process startup."""
from pathlib import Path
import subprocess
import pyperf
ROOT=Path(__file__).resolve().parents[1]
def measure(loops, rows, features):
    values=subprocess.check_output([str(ROOT/'builds/tests/naive_bayes/counts/benchmark'),str(loops),str(rows),str(features)],text=True).split()
    elapsed,checksum=map(float,values)
    assert checksum==(rows+2)//3
    return elapsed
runner=pyperf.Runner()
runner.metadata.update({'compiler':'Mojo 1.0.0 -O3 --fp-mode=contract=off','storage':'NuMojo 0.10.0','seed':'none; ones and cyclic labels','timing':'native fit loop; allocations included; input/process setup excluded','threads':'affinity constrained','memory':'Float64 NF input copy; CN indicator; CF counts and transactional copies'})
for rows,features in [(3,2),(64,32),(256,128)]:
    runner.bench_time_func(f'MultinomialNB fit {rows}x{features}',measure,rows,features)
