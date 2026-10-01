from pathlib import Path
import sys
from check import compile_consumer, run, CONSUMERS
consumer=Path.cwd().resolve()
assert consumer in CONSUMERS
compile_consumer(consumer)
if sys.argv[1]=='test': run(consumer/'builds/probe',42,10,'stratified',1,0,2,consumer=consumer)
elif sys.argv[1]=='parity':
    run(sys.executable,Path(__file__).resolve().parents[1]/'tests/parity.py','--consumer',consumer,consumer=consumer)
