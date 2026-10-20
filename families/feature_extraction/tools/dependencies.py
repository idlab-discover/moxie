"""Check the external NuMojo pin before assembling live family and Moxie sources."""
import hashlib
import json
from pathlib import Path
import subprocess


def verify_external_dependencies(manifest):
    manifest = Path(manifest)
    projects = json.loads(manifest.read_text())['projects']
    matches = [project for project in projects if project['id'] == 'numojo']
    if len(matches) != 1:
        raise ValueError('exactly one pinned NuMojo dependency required')
    project = matches[0]
    root = (manifest.parent / project['path']).resolve()
    revision, files = project.get('revision'), project.get('files')
    if not revision or not files:
        raise ValueError('missing NuMojo revision or file pins')
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise ValueError('NuMojo revision drift')
    if not set(project['assets'].values()).issubset(files):
        raise ValueError('missing NuMojo asset pins')
    for name, expected in files.items():
        source = root / name
        if not source.is_file():
            raise ValueError(f'missing NuMojo source: {name}')
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'NuMojo content drift: {name}')
