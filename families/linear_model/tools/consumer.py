"""Run one isolated fixture with its own assembled namespace and output tree."""
from pathlib import Path
import sys
from check import ROOT, assemble, compile, run
consumer = Path.cwd().resolve()
assert consumer in [ROOT/'tests/consumers/a', ROOT/'tests/consumers/b']
assemble(consumer)
compile(consumer/'main.mojo', consumer/'builds/probe', consumer)
if sys.argv[1] == 'test': run([consumer/'builds/probe'], consumer)
else: assert sys.argv[1] == 'build'
