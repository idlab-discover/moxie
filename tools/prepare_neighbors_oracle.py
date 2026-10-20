"""Build the pinned local sklearn checkout for neighbors validation only."""
from pathlib import Path
import io
import subprocess
import tarfile
ROOT = Path(__file__).resolve().parents[1]
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
source = ROOT / 'builds/neighbors/reference'
source.mkdir(parents=True, exist_ok=True)
data = subprocess.check_output(['git', '-C', ROOT.parent / 'scikit-learn', 'archive', REVISION])
with tarfile.open(fileobj=io.BytesIO(data)) as archive:
    archive.extractall(source, filter='data')
environment = ROOT / 'builds/neighbors/oracle-env'
subprocess.run(['uv', 'venv', '--python', '3.12', environment], check=True)
subprocess.run(['uv', 'pip', 'install', '--python', environment / 'bin/python',
                '--target', ROOT / 'builds/neighbors/oracle-deps', source, 'pytest',
                '--config-settings=compile-args=-j2'], check=True)
