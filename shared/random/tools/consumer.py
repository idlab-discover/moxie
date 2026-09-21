"""Independent consumer checks using the same locked shared launcher."""
from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
consumer=Path.cwd().resolve()
assert consumer in [ROOT/'tests/consumers/a',ROOT/'tests/consumers/b']
subprocess.run([sys.executable,ROOT.parent/'assembly/tools/consume.py','--consumer',consumer,
    '--lock',consumer/'DEPENDENCIES.lock.json','--tool',ROOT.parent/'assembly/tools/assemble.py'],check=True)
subprocess.run(['mojo','build','-O3','--fp-mode=contract=off','-I',consumer/'builds/deps',consumer/'main.mojo','-o',consumer/'builds/probe'],check=True)
if sys.argv[1]=='test': subprocess.run([consumer/'builds/probe'],check=True)
