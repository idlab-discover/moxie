from pathlib import Path
import sys
from check import CONSUMERS, build_consumer, run
c = Path.cwd().resolve()
assert c in CONSUMERS
build_consumer(c)
if sys.argv[1] == 'test': run(c/'builds/probe', c=c)
elif sys.argv[1] != 'build': raise ValueError(sys.argv[1])
