"""Execute verification notebooks headlessly and keep their outputs.

Usage: pixi run -e notebooks notebooks [paths or family names ...]

Each notebook runs in its own directory with a fresh Python kernel. Outputs are
written back in place so the side-by-side results can be read without rerunning.
A failing cell (including a parity assertion) marks the notebook as failed.
"""
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / 'notebooks'


def selected(arguments):
    if not arguments:
        return sorted(p for p in NOTEBOOKS.rglob('*.ipynb') if '_support' not in p.parts
                      and '.ipynb_checkpoints' not in p.parts)
    paths = []
    for argument in arguments:
        path = Path(argument)
        if not path.exists():
            path = NOTEBOOKS / argument
        path = path.resolve()
        paths += sorted(path.rglob('*.ipynb')) if path.is_dir() else [path]
    return paths


def main(arguments):
    failures = []
    for path in selected(arguments):
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=3600, kernel_name='python3',
                                resources={'metadata': {'path': str(path.parent)}})
        started = time.time()
        try:
            client.execute()
            status = 'ok'
        except CellExecutionError as error:
            status = 'FAILED'
            failures.append((path, str(error).splitlines()[-1] if str(error) else 'error'))
        nbformat.write(notebook, path)
        print(f'{status:6} {time.time() - started:7.1f}s {path.relative_to(ROOT)}', flush=True)
    for path, message in failures:
        print(f'FAILED {path.relative_to(ROOT)}: {message}', file=sys.stderr)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
