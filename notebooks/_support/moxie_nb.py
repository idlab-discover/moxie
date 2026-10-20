"""Side-by-side scikit-learn / Moxie notebooks (validation tooling only).

First cell of every verification notebook::

    import sys
    from pathlib import Path
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                if (p / "notebooks/_support/moxie_nb.py").exists())
    sys.path.insert(0, str(ROOT / "notebooks/_support"))
    %load_ext moxie_nb
    nb = moxie_nb.Session("naive_bayes/plot_example")

Then share inputs with ``nb.share(X=X, y=y)`` and write a complete Mojo program
in a ``%%mojo <name>`` cell. The magic refreshes the Moxie assembly, compiles the
program against ``builds/deps`` with the repository's compiler limits, runs the
native binary and loads every array it saved into ``nb.results[<name>]``.
Learning and prediction happen only in that binary; Python prepares data,
runs the scikit-learn reference and compares results.

``%%mojo <name> --expect-fail`` documents an unsupported capability: the cell
passes only when compilation or execution fails, and the failure text is shown.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import html
import os
import shlex
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEPS = ROOT / "builds/deps"
SUPPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
from compiler_limits import compiler_command  # noqa: E402

_SESSION = None


@dataclass
class Check:
    name: str
    status: str
    detail: str
    sk_shape: str = ""
    mx_shape: str = ""
    max_abs: float = float("nan")
    max_rel: float = float("nan")
    tolerance: str = ""
    reason: str = ""


class Session:
    """Exchange directory, compiled programs and parity checks of one notebook."""

    def __init__(self, key):
        global _SESSION
        self.key = key
        self.dir = ROOT / "builds/notebooks" / key
        self.inputs = self.dir / "in"
        self.inputs.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.checks = []
        _SESSION = self

    # -- data exchange -------------------------------------------------
    def share(self, **arrays):
        """Save arrays as C-contiguous .npy files readable by moxie_nb_io.load."""
        for name, value in arrays.items():
            value = np.asarray(value)
            if value.dtype.kind in "OUS":
                value = value.astype(str)
            np.save(self.inputs / f"{name}.npy", np.ascontiguousarray(value))


    # -- native programs -----------------------------------------------
    def run_mojo(self, name, source, expect_fail=False, timeout=1800):
        _assemble()
        program = self.dir / f"{name}.mojo"
        binary = self.dir / f"{name}.bin"
        program.write_text(source)
        inputs = [source.encode(), (DEPS / "ASSEMBLY.json").read_bytes()]
        inputs += [p.read_bytes() for p in sorted(SUPPORT.glob("*.mojo"))]
        digest = hashlib.sha256(b"\0".join(inputs)).hexdigest()
        stamp = self.dir / f"{name}.sha256"
        outputs = self.dir / "out" / name
        outputs.mkdir(parents=True, exist_ok=True)
        for stale in outputs.glob("*.npy"):
            stale.unlink()
        failure = None
        if not (binary.exists() and stamp.exists() and stamp.read_text() == digest):
            build = subprocess.run(
                compiler_command(["mojo", "build", "-O3", "--fp-mode=contract=off",
                                  "-I", DEPS, "-I", SUPPORT, program, "-o", binary]),
                cwd=ROOT, capture_output=True, text=True)
            if build.returncode != 0:
                failure = "compilation failed\n" + build.stdout + build.stderr
            else:
                stamp.write_text(digest)
        output = ""
        if failure is None:
            env = dict(os.environ, MOXIE_NB_IN=str(self.inputs), MOXIE_NB_OUT=str(outputs),
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            run = subprocess.run([binary], cwd=self.dir, env=env, capture_output=True,
                                 text=True, timeout=timeout)
            output = run.stdout + run.stderr
            if run.returncode != 0:
                failure = f"exit status {run.returncode}\n" + output
            self.results[name] = {p.stem: np.load(p) for p in outputs.glob("*.npy")}
        if expect_fail:
            if failure is None:
                raise AssertionError(f"{name}: expected an unsupported-capability failure, "
                                     "but the program succeeded")
            print("Expected failure (unsupported in Moxie):\n" + failure)
            return
        if failure is not None:
            raise RuntimeError(f"{name}: {failure}")
        if output:
            print(output, end="")

    # -- comparison ----------------------------------------------------
    def compare(self, name, sklearn, moxie, rtol=1e-7, atol=0.0, exact=False, reason=""):
        """Record one parity check; tolerances must be justified in `reason`."""
        sk, mx = np.asarray(sklearn), np.asarray(moxie)
        check = Check(name, "PASS", "", str(sk.shape), str(mx.shape),
                      tolerance="exact" if exact else f"rtol={rtol:g}, atol={atol:g}",
                      reason=reason)
        if sk.shape != mx.shape:
            check.status, check.detail = "FAIL", "shape mismatch"
        elif sk.dtype.kind in "OUSb" or mx.dtype.kind in "OUSb" or exact:
            equal = np.array_equal(sk.astype(str) if sk.dtype.kind in "OUS" else sk,
                                   mx.astype(str) if sk.dtype.kind in "OUS" else mx)
            mismatches = int(np.sum(sk != mx)) if sk.size else 0
            check.detail = f"{mismatches} mismatches"
            check.status = "PASS" if equal else "FAIL"
        else:
            sk64, mx64 = sk.astype(np.float64), mx.astype(np.float64)
            diff = np.abs(sk64 - mx64)
            check.max_abs = float(np.nanmax(diff)) if diff.size else 0.0
            scale = np.maximum(np.abs(sk64), np.finfo(np.float64).tiny)
            check.max_rel = float(np.nanmax(diff / scale)) if diff.size else 0.0
            same_nan = np.array_equal(np.isnan(sk64), np.isnan(mx64))
            close = np.allclose(sk64, mx64, rtol=rtol, atol=atol, equal_nan=True)
            check.status = "PASS" if close and same_nan else "FAIL"
        self.checks.append(check)
        return check.status == "PASS"

    def note(self, name, status, detail):
        """Record a non-numerical outcome: GAP, ADAPTED or SKIP with a reason."""
        assert status in ("GAP", "ADAPTED", "SKIP", "PASS", "FAIL"), status
        self.checks.append(Check(name, status, detail))

    def report(self):
        """Render the parity table and return it as rows."""
        from IPython.display import HTML, display
        rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
                c.name, c.status, c.sk_shape, c.mx_shape, _fmt(c.max_abs), _fmt(c.max_rel),
                c.tolerance, c.detail, c.reason)) + "</tr>" for c in self.checks)
        head = "".join(f"<th>{h}</th>" for h in (
            "check", "status", "sklearn shape", "moxie shape", "max |diff|", "max rel",
            "tolerance", "detail", "tolerance reason"))
        display(HTML(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"))
        return self.checks

    def assert_parity(self):
        """Fail the notebook when any numerical check failed (gaps are reported, not failures)."""
        failed = [c.name for c in self.checks if c.status == "FAIL"]
        assert not failed, f"parity failures: {failed}"


def side_by_side(**frames):
    """Display labelled tables/arrays next to each other, e.g. sklearn=..., moxie=..."""
    from IPython.display import HTML, display
    cells = []
    for label, value in frames.items():
        body = value.to_html() if hasattr(value, "to_html") else (
            "<pre>" + html.escape(np.array2string(np.asarray(value), precision=6,
                                                  threshold=200)) + "</pre>")
        cells.append(f"<div style='flex:1;min-width:0;overflow:auto'><b>{html.escape(label)}"
                     f"</b>{body}</div>")
    display(HTML("<div style='display:flex;gap:1.5em'>" + "".join(cells) + "</div>"))


def load_ipython_extension(ipython):
    from IPython.core.magic import register_cell_magic

    @register_cell_magic
    def mojo(line, cell):
        args = shlex.split(line)
        if not args:
            raise ValueError("usage: %%mojo <name> [--expect-fail] [--timeout SECONDS]")
        if _SESSION is None:
            raise RuntimeError("create moxie_nb.Session(...) before a %%mojo cell")
        timeout = int(args[args.index("--timeout") + 1]) if "--timeout" in args else 1800
        _SESSION.run_mojo(args[0], cell, expect_fail="--expect-fail" in args,
                          timeout=timeout)

    ipython.user_ns.setdefault("moxie_nb", sys.modules[__name__])


def _assemble():
    subprocess.run([sys.executable, ROOT / "shared/assembly/tools/assemble_family.py",
                    "--consumer", ROOT, "--lock", ROOT / "DEPENDENCIES.json",
                    "--output", DEPS], cwd=ROOT, check=True, capture_output=True)


def _fmt(value):
    return "" if value != value else f"{value:.3g}"
