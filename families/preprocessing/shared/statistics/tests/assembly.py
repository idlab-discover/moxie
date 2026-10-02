"""Exact shared source ownership and missing-source rejection."""
import hashlib, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check import assembly_command, environment
reports=[]
for name in ('a','b'):
    c=ROOT/'tests/consumers'/name
    receipt=json.loads((c/'builds/deps/ASSEMBLY.json').read_text())
    assert receipt['modules']['moxie.preprocessing._moments']==['statistics']
    assert receipt['modules']['moxie.preprocessing._contracts']==['preprocessing']
    for module in ('_moments','_contracts','_metadata','_validation'):
        key=f'moxie/preprocessing/{module}.mojo'
        source=ROOT/f'src/{key}' if module=='_moments' else ROOT.parents[3]/f'src/{key}'
        assert receipt['files'][key]==hashlib.sha256(source.read_bytes()).hexdigest()
    reports.append(receipt)
assert reports[0]['files']==reports[1]['files']
assert reports[0]['modules']==reports[1]['modules']
c=ROOT/'tests/consumers/a'
lock=json.loads((c/'DEPENDENCIES.json').read_text())
next(project for project in lock['projects'] if project['id'] == 'statistics')['modules']['moxie.preprocessing._moments']['file']='src/moxie/absent.mojo'
bad=c/'builds/drift.json'
# Paths in the lock are relative to its location. Put the temporary alongside it.
bad=c/'drift.json'
try:
    bad.write_text(json.dumps(lock))
    p=subprocess.run(assembly_command(c, bad),env=environment(),capture_output=True,text=True)
    assert p.returncode and 'missing file' in p.stderr
    assert json.loads((c/'builds/deps/ASSEMBLY.json').read_text())==reports[0]
finally: bad.unlink(missing_ok=True)
print('PASS live shared ownership and missing-source rejection')

# Explicit private-module symbols, separate from empty package exports.
import re
source=(ROOT/'src/moxie/preprocessing/_moments.mojo').read_text()
actual=set(re.findall(r'^(?:def|struct|comptime) ([A-Za-z][A-Za-z_0-9]*)',source,re.M))
manifest=json.loads((ROOT/'MODULE_EXPORTS.json').read_text())
assert actual==set(manifest['modules']['moxie.preprocessing._moments'])
print('PASS exact module export inventory')
