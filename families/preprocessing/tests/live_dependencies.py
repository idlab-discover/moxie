"""Exercise live source updates and missing-source rejection without editing siblings."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from dependencies import assembly_command

with tempfile.TemporaryDirectory(dir=ROOT / 'builds', prefix='live-source-') as directory:
    consumer = Path(directory)
    source = consumer / 'source'
    source.mkdir()
    module = source / 'probe.mojo'
    module.write_text('comptime VALUE = 1\n')
    assembly = ROOT.parents[1] / 'shared/assembly'
    manifest = consumer / 'DEPENDENCIES.json'
    manifest.write_text(json.dumps({'schema': 1, 'overrides': [], 'projects': [
        {'id': 'assembly', 'path': str(assembly), 'modules': {}, 'assets': {}, 'exports': None},
        {'id': 'probe', 'path': str(source), 'modules': {'moxie.probe': {'file': 'probe.mojo'}},
         'assets': {}, 'exports': {'schema': 1, 'packages': {'moxie': {'VALUE': 'moxie.probe'}}},
         # Legacy source pins must not gate current working edits.
         'revision': '0' * 40, 'files': {'probe.mojo': '0' * 64}},
    ]}))
    command = assembly_command(consumer)
    assert Path(command[1]) == assembly / 'tools/assemble_family.py'
    for value in (1, 2):
        module.write_text(f'comptime VALUE = {value}\n')
        subprocess.run(command, check=True, capture_output=True, text=True)
        output = consumer / 'builds/deps/moxie/probe.mojo'
        assert output.read_bytes() == module.read_bytes()
        receipt = json.loads((consumer / 'builds/deps/ASSEMBLY.json').read_text())
        assert receipt['files']['moxie/probe.mojo'] == hashlib.sha256(module.read_bytes()).hexdigest()
    before = (consumer / 'builds/deps/ASSEMBLY.json').read_bytes()
    module.unlink()
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode and 'missing file' in failed.stderr
    assert (consumer / 'builds/deps/ASSEMBLY.json').read_bytes() == before
print('PASS live edits consumed, legacy pins ignored, missing sources rejected without fallback')
