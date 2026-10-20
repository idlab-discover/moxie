from pathlib import Path
import sys
from check import compile_consumer, run, CONSUMERS
consumer = Path.cwd().resolve()
assert consumer in CONSUMERS
compile_consumer(consumer)
if sys.argv[1] == 'test':
    run(consumer/'builds/probe', consumer=consumer)
elif sys.argv[1] != 'build':
    raise ValueError(sys.argv[1])
