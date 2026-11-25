"""Flag new direct std.math exp, log, and pow imports in production Mojo."""

from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = {"exp", "log", "pow"}
IMPORT = re.compile(
    r"(?m)^[ \t]*from[ \t]+std\.math[ \t]+import[ \t]*"
    r"(\((?:[^)#]|#[^\n]*\n)*\)|[^\n;#]*)"
)

# Part 5 of tracking PR #10. Remove an entry when its owner migrates the
# operation and validates the affected numerical behavior.
EXISTING_IMPORTS = {
    "families/cluster/algorithms/kmeans/src/moxie/cluster/_kmeans.mojo": {"log"},
    "families/manifold/src/moxie/manifold/_classical_mds.mojo": {"pow"},
    "families/neural_network/algorithms/rbm/src/moxie/neural_network/_rbm.mojo": {"exp"},
    "families/svm/algorithms/c_svm/src/moxie/svm/_c_svm.mojo": {"log"},
    "families/svm/algorithms/nu_svm/src/moxie/svm/_nu_svm.mojo": {"log"},
    "families/svm/algorithms/one_class/src/moxie/svm/_one_class.mojo": {"log"},
    "families/svm/shared/liblinear/src/moxie/svm/_liblinear.mojo": {"exp", "log"},
    "families/svm/shared/libsvm/src/moxie/svm/_libsvm.mojo": {"exp", "log"},
    "src/moxie/linear_model/_bayes.mojo": {"log"},
    "src/moxie/linear_model/_linear_loss.mojo": {"exp"},
    "src/moxie/linear_model/_logistic.mojo": {"exp", "log"},
    "src/moxie/multioutput.mojo": {"log"},
    "src/moxie/preprocessing/_scalers.mojo": {"log"},
}


def imported_operations(source: str) -> set[str]:
    operations = set()
    for match in IMPORT.finditer(source):
        imports = re.sub(r"#[^\n]*", "", match[1]).strip("()")
        for item in imports.split(","):
            words = item.split()
            if words and (words[0] in OPERATIONS or words[0] == "*"):
                operations.add(words[0])
    return operations


def production_sources() -> list[str]:
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard",
         "--", "src", "families", "shared"],
        cwd=ROOT,
        text=True,
    )
    return sorted({
        name for name in files.split("\0")
        if name.endswith(".mojo")
        and "src" in PurePosixPath(name).parts[:-1]
        and (ROOT / name).is_file()
    })


def main() -> int:
    remaining = EXISTING_IMPORTS.copy()
    diagnostics = []
    for name in production_sources():
        found = imported_operations((ROOT / name).read_text(encoding="utf-8"))
        expected = remaining.pop(name, set())
        for operation in sorted(found - expected):
            diagnostics.append(f"{name}: new direct std.math import of {operation}")
        if missing := expected - found:
            remaining[name] = missing

    for name, operations in sorted(remaining.items()):
        diagnostics.append(f"{name}: remove baseline entries for {', '.join(sorted(operations))}")

    if diagnostics:
        print("\n".join(diagnostics), file=sys.stderr)
        return 1
    print(f"std.math import check passed ({len(EXISTING_IMPORTS)} baseline modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
