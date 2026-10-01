"""Execute one isolated fixture project with the shared local assembler."""
import argparse
from pathlib import Path
import subprocess
from check import compile
from consume import consume

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--consumer', required=True, type=Path)
    p.add_argument('action', choices=['build', 'test'])
    args = p.parse_args()
    consumer = args.consumer.resolve()
    consume(consumer, consumer / 'DEPENDENCIES.json', ROOT / 'tools/assemble.py')
    compile(consumer / 'main.mojo', consumer / 'builds/probe', [consumer / 'builds/deps'])
    if args.action == 'test':
        subprocess.run([str(consumer / 'builds/probe')], check=True)
