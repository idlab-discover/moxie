from pathlib import Path
from check import assemble, compile_file, run
c=Path.cwd()
assemble(c)
compile_file(c,c/'main.mojo','probe')
print(run(c/'builds/probe',c=c).stdout)
