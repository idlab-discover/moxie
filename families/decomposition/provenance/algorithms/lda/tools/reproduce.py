"""Reproduce the shared Gamma API gap; success is diagnostic, not LDA parity."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    env = os.environ.copy()
    for key, relative in {
        "XDG_CACHE_HOME": ".pixi/cache",
        "MODULAR_CACHE_DIR": ".pixi/modular-cache",
        "PIXI_CACHE_DIR": ".pixi/pixi-cache",
        "TMPDIR": "builds/tmp",
    }.items():
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    lock = ROOT / "DEPENDENCIES.lock.json"
    data = json.loads(lock.read_text())
    pin = next(p for p in data["projects"] if p["id"] == "assembly")
    assembly = (ROOT / pin["path"]).resolve()
    launcher = assembly / "tools/consume_family.py"
    assert hashlib.sha256(launcher.read_bytes()).hexdigest() == pin["files"]["tools/consume_family.py"]
    subprocess.run([sys.executable, str(launcher), "--consumer", str(ROOT),
                    "--lock", str(lock), "--tool", str(assembly / "tools/assemble_family.py")],
                   env=env, check=True)
    reports = {}
    for name in ("random_control", "missing_gamma"):
        command = ["mojo", "build", "-O3", "--fp-mode=contract=off", "-I",
                   str(ROOT / "builds/deps"), str(ROOT / "tests" / (name + ".mojo")),
                   "-o", str(ROOT / "builds" / name)]
        result = subprocess.run(command, env=env, text=True, capture_output=True)
        reports[name] = dict(command=command, returncode=result.returncode,
                             stdout=result.stdout, stderr=result.stderr)
        if name == "random_control":
            assert result.returncode == 0, result.stderr
            subprocess.run([str(ROOT / "builds" / name)], env=env, check=True)
        else:
            assert result.returncode != 0 and "gamma" in result.stderr, result.stderr
    output = ROOT / "docs/private/lda/probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2) + "\n")
    print(reports["missing_gamma"]["stderr"])
    print("Confirmed missing shared Gamma API; this is not estimator validation.")


if __name__ == "__main__":
    main()
