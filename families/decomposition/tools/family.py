"""Assemble, build, test and validate the dense decomposition estimators.

Usage: python tools/family.py {build,test,parity,start}

Builds use the current Moxie dependency manifest and materialize its public
namespace in the ignored `builds/deps` directory.

Environment:
  MOXIE_ESTIMATORS   comma-separated subset of ESTIMATORS to build/check
                     (default: all). Useful while iterating on one estimator.
Builds run sequentially so concurrent agents do not oversubscribe the host.
"""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parents[1]
ASSEMBLY = CORE / "shared/assembly"
LOCK = CORE / "DEPENDENCIES.json"
INCLUDES = ["-I", str(ROOT / "builds/deps"), "-I", str(ROOT / "tests/support")]
FLAGS = ["-O3", "--fp-mode=contract=off", *INCLUDES]

# name -> (native sources to compile, binaries run by `test`, parity script)
ESTIMATORS = {
    "sparse_coding": (["tests/sparse_coding/contracts.mojo", "tests/sparse_coding/driver.mojo", "examples/sparse_coding.mojo"],
                      ["contracts", "example"], "tests/sparse_coding/parity.py"),
    "kernel_pca": (["tests/kernel_pca/contracts.mojo", "tests/kernel_pca/driver.mojo", "examples/kernel_pca.mojo"],
                   ["contracts", "example"], "tests/kernel_pca/parity.py"),
    "pca": (["tests/pca/contracts.mojo", "tests/pca/evaluate.mojo", "tests/pca/benchmark.mojo",
             "examples/pca.mojo"], ["contracts", "example"], "tests/pca/parity.py"),
    "incremental_pca": (["tests/incremental_pca/contracts.mojo", "tests/incremental_pca/driver.mojo",
                         "examples/incremental_pca.mojo"], ["contracts", "example"],
                        "tests/incremental_pca/parity.py"),
    "truncated_svd": (["tests/truncated_svd/contracts.mojo", "tests/truncated_svd/driver.mojo",
                       "examples/truncated_svd.mojo"], ["contracts", "example"], "tests/truncated_svd/parity.py"),
    "factor_analysis": (["tests/factor_analysis/contracts.mojo", "tests/factor_analysis/driver.mojo",
                         "examples/factor_analysis.mojo"], ["contracts", "example"], "tests/factor_analysis/parity.py"),
    "fastica": (["tests/fastica/contracts.mojo", "tests/fastica/driver.mojo", "examples/fastica.mojo"],
                ["contracts", "example"], "tests/fastica/parity.py"),
    "nmf": (["tests/nmf/contracts.mojo", "tests/nmf/driver.mojo", "examples/nmf.mojo"],
            ["contracts", "example"], "tests/nmf/parity.py"),
    "lda": (["tests/lda/contracts.mojo", "tests/lda/driver.mojo", "examples/lda.mojo"],
            ["contracts", "example"], "tests/lda/parity.py"),
}


def environment():
    env = os.environ.copy()
    for key, folder in {"XDG_CACHE_HOME": ".pixi/cache", "MODULAR_CACHE_DIR": ".pixi/modular-cache",
                        "TMPDIR": "builds/tmp"}.items():
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return env


def run(*args):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True, env=environment())


def selected():
    names = os.environ.get("MOXIE_ESTIMATORS")
    if not names:
        return list(ESTIMATORS)
    chosen = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in chosen if n not in ESTIMATORS]
    if unknown:
        raise SystemExit(f"unknown MOXIE_ESTIMATORS entries {unknown}; choose from {list(ESTIMATORS)}")
    return chosen


def binary(estimator, source):
    stem = Path(source).stem
    return ROOT / "builds" / (estimator + "_" + ("example" if source.startswith("examples/") else stem))


def working_lock(lock=LOCK):
    data = json.loads(lock.read_text())
    for entry in data["projects"]:
        entry["path"] = str((lock.parent / entry["path"]).resolve())
    destination = ROOT / "builds" / (lock.parent.name + "-" + lock.stem + ".resolved.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2) + "\n")
    return destination


def assemble(consumer=ROOT):
    run(sys.executable, ASSEMBLY / "tools/assemble_family.py", "--consumer", consumer,
        "--lock", working_lock(), "--output", consumer / "builds/deps")


def ownership(deps=ROOT / "builds/deps"):
    manifest = json.loads(LOCK.read_text())
    receipt = json.loads((deps / "ASSEMBLY.json").read_text())
    assert all("moxie-decomposition" not in p["path"] for p in manifest["projects"])
    for project in manifest["projects"]:
        if project["id"] not in ("decomposition", "svd_engine", "sparse_coding"):
            continue
        owner = CORE / project["path"]
        assert owner.resolve().is_relative_to(CORE)
        for module, spec in project["modules"].items():
            assert receipt["modules"][module] == [project["id"]], module
            actual = deps / (module.replace(".", "/") + ".mojo")
            assert actual.read_bytes() == (owner / spec["file"]).read_bytes(), module
        for target, source in project["assets"].items():
            assert (deps / target).read_bytes() == (owner / source).read_bytes()
    print("PASS: decomposition ownership, source bytes, licenses and internal dependency paths")


def build():
    version = subprocess.check_output(["mojo", "--version"], text=True, env=environment())
    assert version.startswith("Mojo 1.0.0 "), version
    metadata = list(Path(sys.prefix, "conda-meta").glob("max-core-*.json"))
    assert len(metadata) == 1 and json.loads(metadata[0].read_text())["version"] == "26.5.0"
    assemble()
    ownership()
    run("mojo", "precompile", ROOT / "builds/deps/moxie", *INCLUDES, "-o", ROOT / "builds/moxie-decomposition.mojoc")
    for estimator in selected():
        for source in ESTIMATORS[estimator][0]:
            run("mojo", "build", *FLAGS, ROOT / source, "-o", binary(estimator, source))
    run("mojo", "build", *FLAGS, ROOT / "tests/test_plumbing.mojo", "-o", ROOT / "builds/plumbing")
    run("mojo", "build", *FLAGS, ROOT / "examples/plumbing.mojo", "-o", ROOT / "builds/plumbing_example")
    run("mojo", "build", *FLAGS, ROOT / "tests/fluent.mojo", "-o", ROOT / "builds/fluent")
    if not os.environ.get("MOXIE_ESTIMATORS"):
        run("mojo", "build", *FLAGS, ROOT / "tests/support/benchmark_family.mojo", "-o", ROOT / "builds/benchmark_family")

    engine("build")


def engine(action):
    run(sys.executable, ROOT / "shared/svd_engine/tools/check.py", action)


def consumers():
    """Assemble two isolated namespaces and run the same probe in each."""
    copies = []
    for name in ("a", "b"):
        consumer = ROOT / "builds" / ("consumer-" + name)
        consumer.mkdir(parents=True, exist_ok=True)
        assemble(consumer)
        deps = consumer / "builds/deps"
        ownership(deps)
        run("mojo", "build", "-O3", "--fp-mode=contract=off", "-I", deps,
            ROOT / "tests/support/consumer_probe.mojo", "-o", consumer / "probe")
        run(consumer / "probe")
        copies.append(deps / "moxie/decomposition/_svd_engine.mojo")
    assert copies[0].stat().st_ino != copies[1].stat().st_ino
    assert copies[0].read_bytes() == copies[1].read_bytes()
    print("PASS: identical Moxie-owned engine in two isolated assembled consumers")


def test():
    ownership()
    for estimator in selected():
        for name in ESTIMATORS[estimator][1]:
            run(ROOT / "builds" / (estimator + "_" + name))
    run(ROOT / "builds/fluent")
    run(ROOT / "builds/plumbing")
    run(ROOT / "builds/plumbing_example")
    consumers()
    engine("test-only")


def parity():
    ownership()
    for estimator in selected():
        run(sys.executable, ROOT / ESTIMATORS[estimator][2])
    engine("parity-only")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "build"
    if action not in ("test-only", "parity-only"):
        build()
    if action in ("test", "test-only"): test()
    elif action in ("parity", "parity-only"): parity()
    elif action == "start": run(ROOT / "builds/pca_example")
    elif action != "build": raise ValueError(action)
