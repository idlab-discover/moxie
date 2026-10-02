"""Standalone consumer gate using the family-owned assembly configuration."""
from pathlib import Path
import sys
from check import CONSUMERS, assemble, compile_file, run
consumer=Path.cwd().resolve()
assert consumer in CONSUMERS
assemble(consumer)
run('mojo','precompile',consumer/'builds/deps/moxie','-I',consumer/'builds/deps','-o',consumer/'builds/moxie.mojoc',c=consumer)
compile_file(consumer,consumer/'main.mojo','probe')
if sys.argv[1]=='test': print(run(consumer/'builds/probe',c=consumer).stdout,end='')
elif sys.argv[1]!='build': raise ValueError(sys.argv[1])
