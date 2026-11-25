"""Compare dense selector feature-name mapping with the pinned sklearn source."""
import argparse
import ast
import json
import subprocess
from types import SimpleNamespace

import numpy as np

from parity import ORACLE, REV, ROOT, oracle
from feature_selection_checks import run


def reference_names(namespace):
    validation = ORACLE / "sklearn/utils/validation.py"
    base = ORACLE / "sklearn/feature_selection/_base.py"
    for source in (validation, base):
        changes = subprocess.check_output(
            ["git", "-C", ORACLE, "status", "--porcelain", "--", source], text=True
        )
        assert not changes.strip(), source
    name_check = next(
        node for node in ast.parse(validation.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_check_feature_names_in"
    )
    selector_mixin = next(
        node for node in ast.parse(base.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "SelectorMixin"
    )
    names_out = next(
        node for node in selector_mixin.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_feature_names_out"
    )
    exec(compile(ast.Module([name_check], type_ignores=[]), str(validation), "exec"), namespace)
    exec(compile(ast.Module([names_out], type_ignores=[]), str(base), "exec"), namespace)
    return namespace["get_feature_names_out"]


def mojo_names(names):
    return f"List[String]({json.dumps(list(names))})" if len(names) else "List[String]()"


def compare_selection_state(names_out):
    source = [
        "from std.testing import assert_equal, assert_raises",
        "from moxie.feature_selection import SelectionState",
        "",
        "def main() raises:",
        "    var state = SelectionState()",
    ]
    comparisons = 0
    names = ["north", "west", "east", "south"]
    for mask in ([True, False, True, False], [False] * 4):
        native_mask = json.dumps(mask).replace("true", "True").replace("false", "False")
        ref = SimpleNamespace(n_features_in_=4, get_support=lambda: np.asarray(mask))
        source.append(f"    state.commit({native_mask})")
        for supplied in (None, names, []):
            argument = optional_names(supplied)
            try:
                expected = mojo_names(names_out(ref, supplied))
            except ValueError:
                source.extend([
                    '    with assert_raises(contains="length mismatch"):',
                    f"        _ = state.get_feature_names_out({argument})",
                ])
            else:
                source.append(f"    assert_equal(state.get_feature_names_out({argument}), {expected})")
            comparisons += 1

        ref.feature_names_in_ = np.asarray(names)
        source.append(f"    state.commit({native_mask}, {mojo_names(names)})")
        for supplied in (None, names, ["north", "wrong", "east", "south"]):
            argument = optional_names(supplied)
            try:
                expected = mojo_names(names_out(ref, supplied))
            except ValueError:
                source.extend([
                    '    with assert_raises(contains="differ from fit"):',
                    f"        _ = state.get_feature_names_out({argument})",
                ])
            else:
                source.append(f"    assert_equal(state.get_feature_names_out({argument}), {expected})")
            comparisons += 1

    compile_comparisons("selection-state-names-parity", source, comparisons)


def optional_names(names):
    return "None" if names is None else f"Optional[List[String]]({mojo_names(names)})"


def compare_univariate(namespace, names_out):
    variants = (
        ("SelectKBest", "k=2", "SelectKBest", {"k": 2}),
        ("SelectKBest", "k=0", "SelectKBest", {"k": 0}),
        ("SelectPercentile", "percentile=50", "SelectPercentile", {"percentile": 50}),
        ("SelectFpr", "alpha=0.1", "SelectFpr", {"alpha": 0.1}),
        ("SelectFdr", "alpha=0.1", "SelectFdr", {"alpha": 0.1}),
        ("SelectFwe", "alpha=0.1", "SelectFwe", {"alpha": 0.1}),
        ("GenericUnivariateSelect", 'mode="k_best", param=2', "SelectKBest", {"k": 2}),
    )
    names = ["north", "west", "east", "south"]
    # Both signal columns win; the nonadjacent positions expose mapping and order.
    x = [
        [0, 2, 7, 0], [0, 4, 6, 1], [1, 2, 5, 0],
        [8, 4, 1, 0], [9, 2, 0, 1], [9, 4, 0, 0],
    ]
    y = [0, 0, 0, 1, 1, 1]
    source = [
        "import numojo as nm",
        "from std.testing import assert_equal",
        "from moxie.feature_selection import (",
        "    BuiltinScorer, SelectKBest, SelectPercentile, SelectFpr,",
        "    SelectFdr, SelectFwe, GenericUnivariateSelect,",
        ")",
        "",
        "def main() raises:",
    ]
    comparisons = 0
    for dtype in (np.float32, np.float64):
        dt = np.dtype(dtype).name
        scores, pvalues = namespace["f_classif"](np.asarray(x, dtype=dtype), np.asarray(y, dtype=dtype))
        flat_x = np.asarray(x, dtype=float).ravel().tolist()
        source.extend([
            f"    var x_{dt} = nm.array[DType.{dt}]({flat_x}, [6, 4])",
            f"    var y_{dt} = nm.array[DType.{dt}]({[float(label) for label in y]}, [6])",
        ])
        for index, (kind, args, policy, params) in enumerate(variants):
            mask = namespace[policy](SimpleNamespace(scores_=scores, pvalues_=pvalues, **params))
            reference = SimpleNamespace(n_features_in_=4, get_support=lambda mask=mask: mask)
            selector = f"selector_{dt}_{index}"
            source.extend([
                f"    var {selector} = {kind}(BuiltinScorer[DType.{dt}](), {args})",
                f"    _ = {selector}.fit(x_{dt}, y_{dt}.copy())",
            ])
            for supplied in (None, names):
                argument = optional_names(supplied)
                expected = mojo_names(names_out(reference, supplied))
                source.append(f"    assert_equal({selector}.get_feature_names_out({argument}), {expected})")
                comparisons += 1
            reference.feature_names_in_ = np.asarray(names)
            expected = mojo_names(names_out(reference))
            source.extend([
                f"    {selector}.fit_named(x_{dt}, y_{dt}.copy(), {mojo_names(names)})",
                f"    assert_equal({selector}.get_feature_names_out(), {expected})",
            ])
            comparisons += 1

    compile_comparisons("univariate-names-parity", source, comparisons)


def compile_comparisons(name, source, comparisons):
    build = ROOT / "builds"
    program = build / f"{name}.mojo"
    binary = build / name
    program.write_text("\n".join(source) + "\n")
    run("mojo", "build", "-O3", "-I", build / "deps", program, "-o", binary)
    run(binary)
    print(f"PASS {comparisons} exact feature-name comparisons against sklearn {REV}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-state", action="store_true")
    args = parser.parse_args()
    namespace = oracle()
    names_out = reference_names(namespace)
    if args.selection_state:
        compare_selection_state(names_out)
    else:
        compare_univariate(namespace, names_out)


if __name__ == "__main__":
    main()
