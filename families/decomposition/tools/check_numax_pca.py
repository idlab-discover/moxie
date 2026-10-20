"""Build the opt-in static Numax backend without changing shared dependencies.

Assembly verifies original sources. A recorded Mojo 1.1 syntax patch is applied
only to a separate generated copy of NuMojo; source checkouts stay read-only.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from family import ROOT, ASSEMBLY, run, working_lock


def build(*, pca_tests=True):
    assert subprocess.check_output(["mojo", "--version"], text=True).startswith("Mojo 1.1.0 ")
    metadata = list(Path(sys.prefix, "conda-meta").glob("max-core-*.json"))
    assert len(metadata) == 1 and json.loads(metadata[0].read_text())["version"] == "26.6.0"
    lock = working_lock(ROOT / "DEPENDENCIES.json")
    consumer = ROOT / "builds/numax-consumer"
    consumer.mkdir(parents=True, exist_ok=True)
    run(sys.executable, ASSEMBLY / "tools/assemble_family.py", "--consumer", consumer,
        "--lock", lock, "--output", consumer / "builds/deps")
    staging = ROOT / "builds/numax-deps"
    if staging.exists(): shutil.rmtree(staging)
    shutil.copytree(consumer / "builds/deps", staging)
    run("patch", "--batch", "--fuzz=0", "-p1", "-d", staging,
        "-i", ROOT / "tools/numojo_mojo11.patch")
    includes = ["-I", str(staging), "-I", str(ROOT.parents[2] / "numax"),
                "-I", str(ROOT / "backends/numax"), "-I", str(ROOT / "tests/pca"),
                "-I", str(ROOT / "tests/support")]
    run("mojo", "precompile", staging / "moxie", *includes,
        "-o", ROOT / "builds/moxie-numax-pca.mojoc")
    if not pca_tests: return
    for source in sorted((ROOT / "tests/numax_pca").glob("*.mojo")) + [ROOT / "examples/numax_pca.mojo"]:
        run("mojo", "build", "-O3", "--fp-mode=contract=off", *includes, source,
            "-o", ROOT / "builds" / ("numax_pca_" + source.stem))


if __name__ == "__main__":
    action = sys.argv[1]
    build()
    if action == "test": run(ROOT / "builds/numax_pca_static_contracts")
    elif action == "parity": run(sys.executable, ROOT / "tests/pca/parity.py", "--numax")
    elif action == "start": run(ROOT / "builds/numax_pca_numax_pca")
    elif action != "build": raise ValueError(action)
