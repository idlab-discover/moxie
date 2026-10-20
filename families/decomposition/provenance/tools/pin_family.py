"""Explicit source snapshot for the dense decomposition estimators; never run during builds.

Starts from the SVD engine's reviewed lock (foundation, numerics, NuMojo, RNG,
assembler), then snapshots this family's estimator sources, the engine module
and the native numerical libraries. mojo-linalg is materialized by the shared
assembler into `builds/deps/mojo_linalg`, so one `-I builds/deps` import root
supplies `moxie`, `numojo` and `mojo_linalg`.

Environment:
  MOXIE_WORKSPACE     workspace root (default: parent of this repository)
  MOXIE_MOJO_LINALG   mojo-linalg checkout or worktree to pin
                      (default: $MOXIE_WORKSPACE/mojo-linalg)

Review `git diff FAMILY.dependencies.lock.json` after running.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "shared/svd_engine"
WORKSPACE = Path(os.environ.get("MOXIE_WORKSPACE", ROOT.parent)).resolve()
MOJO_LINALG = Path(os.environ.get("MOXIE_MOJO_LINALG", WORKSPACE / "mojo-linalg")).resolve()
NUMAX = WORKSPACE / "numax"
LOCK = ROOT / "FAMILY.dependencies.lock.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(name, root, modules, assets, exports=None, extra_files=()):
    files = set(assets.values()) | {v["file"] for v in modules.values()} | set(extra_files)
    return dict(id=name, path=os.path.relpath(root, ROOT),
                revision=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                files={f: sha(root / f) for f in sorted(files)},
                modules=modules, assets=assets, exports=exports)


def native_library(name, root, source):
    """Pin every Mojo source file of a sibling native library as assets."""
    sources = sorted((root / source).rglob("*.mojo"))
    assets = {name + "/" + str(p.relative_to(root / source)): str(p.relative_to(root)) for p in sources}
    if (root / "LICENSE").exists():
        assets["licenses/" + name + ".txt"] = "LICENSE"
    return assets


def main():
    lock = json.loads((ENGINE / "DEPENDENCIES.lock.json").read_text())
    # Preserve independently reviewed external kernel snapshots; do not follow drift.
    previous = json.loads(LOCK.read_text()) if LOCK.exists() else {"projects": []}
    projects = [p for p in previous["projects"] if p["id"] == "kernel-ridge-numerics"]
    for entry in lock["projects"]:
        if entry["id"] in {"decomposition", "svd_engine", "readiness"}:
            continue
        entry["path"] = os.path.relpath((ENGINE / entry["path"]).resolve(), ROOT)
        projects.append(entry)
    modules = {"moxie.decomposition." + p.stem: {"file": str(p.relative_to(ROOT))}
               for p in sorted((ROOT / "src/moxie/decomposition").glob("*.mojo")) if p.stem != "__init__"}
    extra = ["EXPORTS.json", "src/moxie/decomposition/__init__.mojo", "pixi.lock", "tools/numojo_mojo11.patch"]
    extra += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "backends/numax").rglob("*.mojo"))]
    projects.append(snapshot("decomposition", ROOT, modules, {},
                             json.loads((ROOT / "EXPORTS.json").read_text()), extra))
    projects.append(snapshot("svd_engine", ENGINE,
        {"moxie.decomposition._svd_engine": {"file": "src/moxie/decomposition/_svd_engine.mojo"}}, {},
        {"schema": 1, "packages": {"moxie": {}, "moxie.decomposition": {
            "randomized_svd": "moxie.decomposition._svd_engine"}}}))
    coding = ROOT / "shared/sparse_coding"
    projects.append(snapshot("sparse_coding", coding,
        {"moxie.decomposition._sparse_coding": {"file": "src/moxie/decomposition/_sparse_coding.mojo"}}, {},
        json.loads((coding / "EXPORTS.json").read_text()), ["EXPORTS.json"]))
    projects.append(snapshot("mojo_linalg", MOJO_LINALG, {}, native_library("mojo_linalg", MOJO_LINALG, "src/mojo_linalg")))
    # Numax is pinned for the separate opt-in Mojo 1.1 backend only. Its
    # namespace is not materialized: the default Mojo 1.0 build cannot use it.
    numax_assets = native_library("numax", NUMAX, "numax")
    numax = snapshot("numax", NUMAX, {}, numax_assets)
    numax["assets"] = {k: v for k, v in numax_assets.items() if k.startswith("licenses/")}
    projects.append(numax)
    LOCK.write_text(json.dumps(dict(schema=1, projects=projects, overrides=[]), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
