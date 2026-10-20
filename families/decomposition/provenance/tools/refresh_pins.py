"""Explicit, reviewed refresh of an existing dependency lock; never a build step.

For every project already listed in the lock, re-hash exactly the files it pins.
External repositories are moved to their current HEAD (the assembler requires an
exact HEAD match for them). Projects inside this family repository keep their
recorded ancestor revision, which the assembler accepts together with exact
file hashes. Neither the file lists nor module ownership are changed: adding a
new dependency remains the job of the owning pin tool.

Usage: python tools/refresh_pins.py LOCK [LOCK ...]
Inspect `git diff` of the rewritten locks before committing them.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

FAMILY = Path(__file__).resolve().parents[1]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def refresh(lock_path):
    lock_path = Path(lock_path).resolve()
    data = json.loads(lock_path.read_text())
    family_top = git(FAMILY, "rev-parse", "--show-toplevel")
    for project in data["projects"]:
        root = (lock_path.parent / project["path"]).resolve()
        if git(root, "rev-parse", "--show-toplevel") != family_top:
            project["revision"] = git(root, "rev-parse", "HEAD")
        project["files"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                            for name in sorted(project["files"])}
    lock_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for path in sys.argv[1:]:
        refresh(path)
