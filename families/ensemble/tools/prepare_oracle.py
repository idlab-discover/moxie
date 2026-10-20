"""Build the exact local sklearn revision in a family-local validation environment."""
from pathlib import Path
import hashlib, io, json, os, subprocess, sys, tarfile
ROOT = Path(__file__).resolve().parents[1]
REV = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'

def main():
    env = os.environ.copy()
    env.update(TMPDIR=str(ROOT/'builds/tmp'), PIP_CACHE_DIR=str(ROOT/'.pixi/pip-cache'), PYTHONDONTWRITEBYTECODE='1')
    (ROOT/'builds/tmp').mkdir(parents=True, exist_ok=True)
    source = ROOT/'builds/reference'; source.mkdir(parents=True, exist_ok=True)
    data = subprocess.check_output(['git', '-C', str(ROOT.parents[2]/'scikit-learn'), 'archive', REV])
    with tarfile.open(fileobj=io.BytesIO(data)) as archive: archive.extractall(source, filter='data')
    def run(*args): subprocess.run(list(map(str, args)), cwd=ROOT, env=env, check=True)
    oracle = ROOT/'builds/oracle-env'
    run(sys.executable, '-m', 'venv', '--clear', oracle)
    run(oracle/'bin/python', '-m', 'pip', 'install', '--quiet', source, '--config-settings=compile-args=-j4')
    dest = ROOT/'docs/private/forest_engine'; dest.mkdir(parents=True, exist_ok=True)
    (dest/'oracle-build.json').write_text(json.dumps({'revision': REV, 'archive_sha256': hashlib.sha256(data).hexdigest(), 'source': str(source)}, indent=2) + '\n')

if __name__ == '__main__': main()
