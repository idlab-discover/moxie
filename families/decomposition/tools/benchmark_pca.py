"""A small repeatable native PCA baseline, with no speedup claim."""
from pathlib import Path
import hashlib
import platform
import subprocess
import pyperf

ROOT = Path(__file__).resolve().parents[1]


def timed(loops, dtype, m, n):
    result = subprocess.check_output([ROOT / "builds/pca_benchmark", dtype, str(m), str(n), str(loops)], text=True)
    return float(result.split()[0])


runner = pyperf.Runner()
runner.metadata.update({"compiler": subprocess.check_output(["mojo", "--version"], text=True).strip(),
    "flags": "-O3 --fp-mode=contract=off", "platform": platform.platform(),
    "source_sha256": hashlib.sha256((ROOT / "src/moxie/decomposition/_pca.mojo").read_bytes()).hexdigest(),
    "dependency_lock_sha256": hashlib.sha256((ROOT.parents[1] / "DEPENDENCIES.json").read_bytes()).hexdigest(),
    "solver": "full", "n_components": 3, "timing": "native fit region including allocations"})
for dtype in ("f32", "f64"):
    for shape in ((8, 3), (64, 12)):
        runner.bench_time_func(f"PCA_{dtype}_{shape[0]}x{shape[1]}", timed, dtype, *shape)
