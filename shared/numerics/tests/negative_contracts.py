"""Expected candidate failure probes, distinct from approved-path tests."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'docs/private/numojo'
p=subprocess.run(['stdbuf','-o0',str(ROOT/'builds/edges'),'singleton'],capture_output=True,text=True)
(out/'singleton-reduction.txt').write_text(p.stdout+p.stderr)
assert p.returncode != 0 and 'index 0 is out of bounds' in p.stdout+p.stderr
p=subprocess.run(['mojo','build','-I',str(ROOT/'builds/deps'),str(ROOT/'tests/negative/rng_seed.mojo'),'-o',str(ROOT/'builds/rejected_rng')],capture_output=True,text=True)
(out/'rng-seed.txt').write_text(p.stdout+p.stderr)
assert p.returncode != 0 and 'seed' in p.stderr
print('PASS: singleton upstream sum rejected; caller-seeded RNG API unavailable')

# Exercise drift rejection without modifying the read-only source checkout.
import sys
sys.path.insert(0, str(ROOT / 'tools'))
import check
name = next(iter(check.LOCK['numojo']['files']))
check.LOCK['numojo']['files'][name] = '0' * 64
try:
    check.verify()
except RuntimeError as error:
    assert 'source drift' in str(error)
else:
    raise AssertionError('Dependency hash drift was accepted')
print('PASS: dependency hash drift fails closed')
