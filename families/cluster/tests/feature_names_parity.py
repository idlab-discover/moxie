"""Compare native centroid names with unedited local sklearn naming functions."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[2] / "scikit-learn"
REV = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"


def main():
    path = SOURCE / "sklearn/utils/validation.py"
    assert subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip() == REV
    assert not subprocess.check_output(
        ["git", "-C", str(SOURCE), "status", "--porcelain", "--", str(path)], text=True
    ).strip()
    names = {"_check_feature_names_in", "_generate_get_feature_names_out"}
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    env = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    generate = env["_generate_get_feature_names_out"]
    classes = {name.lower(): type(name, (), {})
               for name in ("KMeans", "MiniBatchKMeans", "BisectingKMeans")}
    output = subprocess.check_output(
        [str(ROOT / "builds/test_cluster_feature_names")], text=True
    )
    rows = output.splitlines()
    assert rows, "native test produced no feature names"
    offset = 0
    while offset < len(rows):
        prefix, count, _ = rows[offset].split()
        count = int(count)
        actual = []
        for line in rows[offset:offset + count]:
            row_prefix, row_count, name = line.split()
            assert row_prefix == prefix and int(row_count) == count
            actual.append(name)
        offset += count
        model = classes[prefix]()
        model.n_features_in_ = 2
        expected = generate(model, count)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(generate(model, count, ["a", "b"]), expected)
        for invalid in ([], ["a"], ["a", "b", "c"]):
            try:
                generate(model, count, invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("sklearn accepted invalid input feature count")
    report = {
        "reference": REV,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "native_name_checks": len(rows),
        "comparison": "exact strings, order and output count",
        "scope": "naming functions; native lifecycle tests cover fit, refit, clone and partial_fit",
    }
    (ROOT / "builds/feature-names-parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(rows)} exact centroid feature-name comparisons")


if __name__ == "__main__":
    main()
