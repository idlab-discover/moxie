"""Build the exact local sklearn revision in a family-local validation environment."""
from pathlib import Path
import os,subprocess,tarfile,io,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
def main():
    env=os.environ.copy()
    env.update(UV_CACHE_DIR=str(ROOT/'.pixi/uv-cache'),TMPDIR=str(ROOT/'builds/tree/tmp'),PYTHONDONTWRITEBYTECODE='1')
    (ROOT/'builds/tree/tmp').mkdir(parents=True,exist_ok=True)
    source=ROOT/'builds/tree/reference';source.mkdir(parents=True,exist_ok=True)
    data=subprocess.check_output(['git','-C',str(ROOT.parent/'scikit-learn'),'archive',REV])
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:archive.extractall(source,filter='data')
    def run(*args):subprocess.run(list(map(str,args)),cwd=ROOT,env=env,check=True)
    run('uv','venv','--clear','--python',ROOT/'.pixi/envs/default/bin/python',ROOT/'builds/tree/oracle-env')
    run('uv','pip','install','--python',ROOT/'builds/tree/oracle-env/bin/python',source,'--config-settings=compile-args=-j2')
    # Treelite reference for the export_treelite byte oracle (validation only).
    run('uv','pip','install','--python',ROOT/'builds/tree/oracle-env/bin/python','treelite==4.6.1','pandas==3.0.6','matplotlib==3.11.2')
    dest=ROOT/'docs/private/tree/dense-trees';dest.mkdir(parents=True,exist_ok=True)
    (dest/'oracle-build.json').write_text(json.dumps({'revision':REV,'archive_sha256':hashlib.sha256(data).hexdigest(),'source':str(source)},indent=2)+'\n')
if __name__=='__main__':main()
