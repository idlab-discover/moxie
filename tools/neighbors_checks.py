"""Run the integrated neighbors gates with Moxie's locked environment."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]

def main(action):
    subprocess.run([sys.executable, ROOT / 'families/neighbors/tools/check.py', action], check=True)
    if action != 'example':
        subprocess.run([sys.executable, ROOT / 'families/neighbors/shared/query_engine/tools/check.py', action], check=True)

if __name__ == '__main__':
    main(sys.argv[1])
