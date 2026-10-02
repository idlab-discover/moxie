from pathlib import Path
import sys
from check import assemble, compile_file, run
c=Path.cwd()
assemble(c)
compile_file(c/'main.mojo',c/'builds/probe',c)
if sys.argv[1]=='test': run(c/'builds/probe')
