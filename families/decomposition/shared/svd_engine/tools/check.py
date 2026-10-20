"""Build and validate the Moxie-owned dense decomposition engine."""
from pathlib import Path
import sys
import subprocess
ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT.parents[1]
sys.path.insert(0, str(FAMILY / "tools"))
import family
FLAGS = ["-O3", "--fp-mode=contract=off", "-I", ROOT / "builds/deps"]

def main(action):
    if action not in ("test-only", "parity-only"):
        family.assemble(ROOT)
        family.ownership(ROOT / "builds/deps")
        (ROOT / "builds").mkdir(exist_ok=True)
        for source in sorted((ROOT / "tests").glob("*.mojo")) + sorted((ROOT / "examples").glob("*.mojo")):
            family.run("mojo", "build", *FLAGS, source, "-o", ROOT / "builds" / source.stem)
    if action in ("test", "test-only"):
        for source in sorted((ROOT / "tests").glob("test_*.mojo")):
            family.run(ROOT / "builds" / source.stem)
        family.run(ROOT / "builds/randomized")
        result = subprocess.run(["mojo", "build", *map(str, FLAGS),
            str(ROOT / "tests/compile_fail/sparse.mojo"), "-o", str(ROOT / "builds/unsupported")],
            env=family.environment(), capture_output=True, text=True)
        assert result.returncode and "_sparse" in result.stderr, result.stderr
        print("PASS: SVD engine contracts, example and sparse exclusion")
    elif action in ("parity", "parity-only"):
        family.run(sys.executable, ROOT / "tests/parity.py")
    elif action != "build":
        raise ValueError(action)

if __name__ == "__main__":
    main(sys.argv[1])
