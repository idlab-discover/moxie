"""Compare dense Mojo TF-IDF to the pinned sklearn method bodies."""

import ast
from itertools import product
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import sklearn.feature_extraction.text as sklearn_text
import sklearn.preprocessing._data as sklearn_data

TEXT = Path(__file__).resolve().parents[1]
MOXIE = TEXT.parents[3]
REFERENCE = MOXIE.parent / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
BUILD = TEXT / "builds"
sys.path.insert(0, str(MOXIE / "tools"))
from compiler_limits import compiler_command


def pinned_definitions(relative_path, names, namespace):
    source = REFERENCE / relative_path
    tree = ast.parse(source.read_text(), filename=str(source))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def reference_transformer():
    text = "sklearn/feature_extraction/text.py"
    data = "sklearn/preprocessing/_data.py"
    csr = "sklearn/utils/sparsefuncs_fast.pyx"
    git = ["git", "-C", str(REFERENCE)]
    assert subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() == REVISION
    assert not subprocess.check_output([*git, "status", "--porcelain", "--", text, data, csr])
    # The locked wheel supplies compiled CSR kernels and sklearn infrastructure.
    data_namespace = vars(sklearn_data).copy()
    data_namespace.setdefault("array_device", sklearn_data._array_api.device)
    normalize = pinned_definitions(data, {"normalize"}, data_namespace)["normalize"]
    namespace = vars(sklearn_text).copy()
    namespace["normalize"] = normalize
    return pinned_definitions(text, {"TfidfTransformer", "_document_frequency"}, namespace)[
        "TfidfTransformer"
    ]


MOJO = """from std.math import inf, isnan, nan
from std.testing import assert_equal, assert_true
from numojo import NDArray, Shape, zeros
from moxie.feature_extraction.text import TfidfTransformer

def array[dtype: DType](shape: Shape, values: List[Float64]) raises -> NDArray[dtype]:
    var result = zeros[dtype](shape)
    for i in range(len(values)):
        result.unsafe_set(i, values[i].cast[dtype]())
    return result^

def compare[dtype: DType](got: NDArray[dtype], shape: Shape, expected: List[Float64]) raises:
    assert_equal(got.shape, shape)
    var rtol = 2e-6 if dtype == DType.float32 else 3e-15
    var atol = 2e-7 if dtype == DType.float32 else 3e-16
    for i in range(len(expected)):
        var reference = expected[i]
        var actual = Float64(got.unsafe_get(i))
        if isnan(reference):
            assert_true(isnan(actual))
        elif reference == 0 or abs(reference) == inf[DType.float64]():
            assert_equal(actual, reference)
        else:
            assert_true(abs(actual - reference) <= atol + rtol * abs(reference))
"""


def numeric_literal(value):
    if np.isnan(value):
        return "nan[DType.float64]()"
    if np.isinf(value):
        return ("-" if value < 0 else "") + "inf[DType.float64]()"
    return repr(float(value))


def mojo_values(values):
    shape = ", ".join(map(str, values.shape))
    entries = ", ".join(numeric_literal(value) for value in values.flat)
    return f"Shape({shape}), [{entries}]"


def mojo_case(index, reference, X, query, **settings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fitted = reference(**settings)
        transformed = fitted.fit_transform(X).toarray()
        expected = fitted.transform(query, copy=False).toarray()
    input_dtype = f"DType.{X.dtype}"
    output_dtype = f"DType.{transformed.dtype}"
    norm = settings.get("norm", "l2")
    options = [
        f'norm=String("{norm}")' if norm is not None else "norm=None",
        *(f"{name}={value}" for name, value in settings.items() if name != "norm"),
    ]
    lines = [
        f"def case_{index}() raises:",
        f"    var model = TfidfTransformer[{input_dtype}]({', '.join(options)})",
        f"    var X = array[{input_dtype}]({mojo_values(X)})",
        f"    var query = array[{input_dtype}]({mojo_values(query)})",
        f"    compare[{output_dtype}](model.fit_transform(X), {mojo_values(transformed)})",
        f"    compare[{output_dtype}](model.transform(query, copy=False), {mojo_values(expected)})",
    ]
    if hasattr(fitted, "idf_"):
        lines.append(f"    compare[{output_dtype}](model.idf_(), {mojo_values(fitted.idf_)})")
    return "\n".join(lines)


def main():
    transformer = reference_transformer()
    cases = []
    for dtype, norm, use_idf, smooth_idf, sublinear_tf in product(
        [np.float32, np.float64, np.int64], ["l1", "l2", None], [True, False],
        [True, False], [True, False],
    ):
        train = np.array([[0, 2, 0, 7], [0, 0, 0, 0], [3, 3, 0, 1], [0, 4, 0, 2]], dtype=dtype)
        query = np.array([[5, 2, 0, 1], [0, 0, 0, 0], [0, 3, 0, 0]], dtype=dtype)
        cases.append(mojo_case(
            len(cases), transformer, train, query, norm=norm, use_idf=use_idf,
            smooth_idf=smooth_idf, sublinear_tf=sublinear_tf,
        ))
    for train, query, options in [
        ([[0, 1]], [[2, 0]], {"norm": None, "smooth_idf": False}),
        ([[1], [-3]], [[-2]], {"norm": "l1"}),
        ([[1], [2]], [[-2]], {"norm": None, "sublinear_tf": True}),
        ([[0.0, 1e-30], [2.0, 1e-30]], [[1e-30, 1e-30]], {"norm": "l2"}),
    ]:
        cases.append(mojo_case(
            len(cases), transformer, np.array(train, dtype=np.float32),
            np.array(query, dtype=np.float32), **options,
        ))
    for dtype in (np.uint8, np.bool_):
        matrix = np.array([[0, 1], [1, 1]], dtype=dtype)
        cases.append(mojo_case(len(cases), transformer, matrix, matrix))
    BUILD.mkdir(parents=True, exist_ok=True)
    source = BUILD / "tfidf_parity.mojo"
    output = BUILD / "tfidf_parity"
    calls = "\n".join(f"    case_{index}()" for index in range(len(cases)))
    source.write_text("\n\n".join([MOJO, *cases, f"def main() raises:\n{calls}\n"]))
    subprocess.run(compiler_command([
        "mojo", "build", "-O3", "--fp-mode=contract=off", "-I", BUILD / "deps",
        source, "-o", output,
    ]), check=True, cwd=MOXIE)
    subprocess.run([output], check=True)
    print(f"PASS: {len(cases)} dense TF-IDF differential cases against sklearn {REVISION}")


if __name__ == "__main__":
    main()
