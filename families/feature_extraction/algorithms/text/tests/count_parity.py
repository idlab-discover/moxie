"""Compare native count vectorization to the pinned sklearn text.py bodies."""

import json
import re
import subprocess
import warnings

import numpy as np
import sklearn.feature_extraction.text as sklearn_text

from parity import BUILD, MOXIE, REFERENCE, REVISION, compiler_command, pinned_definitions


HEADER = """from std.testing import assert_equal
from moxie.feature_extraction.text import CountVectorizer, strip_tags
from numojo import NDArray, Shape

def check[dtype: DType](actual: NDArray[dtype], shape: Shape, expected: List[Int]) raises:
    assert_equal(actual.shape, shape)
    for i in range(len(expected)):
        assert_equal(actual.unsafe_get(i), Int64(expected[i]).cast[dtype]())
"""


def reference_text():
    path = "sklearn/feature_extraction/text.py"
    git = ["git", "-C", str(REFERENCE)]
    assert subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() == REVISION
    assert not subprocess.check_output([*git, "status", "--porcelain", "--", path])
    return pinned_definitions(
        path,
        {"CountVectorizer", "_VectorizerMixin", "_preprocess", "_analyze", "_check_stop_list",
         "_document_frequency", "_make_int_array", "strip_tags"},
        vars(sklearn_text).copy(),
    )


def literal(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        bracket = ("[", "]") if isinstance(value, list) else ("(", ")")
        return bracket[0] + ", ".join(map(literal, value)) + bracket[1]
    if isinstance(value, dict):
        return "{" + ", ".join(f"{literal(key)}: {index}" for key, index in value.items()) + "}"
    if isinstance(value, bool):
        return str(value)
    return repr(value)


def matrix(value):
    return f"Shape({value.shape[0]}, {value.shape[1]}), {literal(value.astype(int).ravel().tolist())}"


def mojo_case(index, reference, train, query, inspect=None, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = reference(**options)
        X = model.fit_transform(train).toarray()
        Y = model.transform(query).toarray()
    dtype = f"DType.{X.dtype}"
    args = []
    declarations = []
    for name, value in options.items():
        if name == "dtype":
            continue
        if isinstance(value, list):
            declarations.append(f"    var {name}: List[String] = {literal(value)}")
            args.append(f"{name}={name}^")
        elif isinstance(value, dict):
            declarations.append(f"    var {name}: Dict[String, Int] = {literal(value)}")
            args.append(f"{name}={name}^")
        else:
            code = literal(value)
            if name in ("analyzer", "stop_words"):
                code = f"String({code})"
            if name in ("min_df", "max_df") and isinstance(value, float):
                code = f"Float64({code})"
            args.append(f"{name}={code}")
    forms = [
        "@no_inline",
        f"def case_{index}() raises:",
        *declarations,
        f"    var model = CountVectorizer[{dtype}]({', '.join(args)})",
        f"    var docs: List[String] = {literal(train)}",
        f"    var query: List[String] = {literal(query)}",
        "    var X = model.fit_transform(docs)",
        f"    check[{dtype}](X, {matrix(X)})",
        f"    check[{dtype}](model.transform(query), {matrix(Y)})",
        f"    check[{dtype}](model.transform(docs), {matrix(X)})",
        f"    assert_equal(model.get_feature_names_out(), {literal(model.get_feature_names_out().tolist())})",
        f"    assert_equal(model.fixed_vocabulary_(), {model.fixed_vocabulary_})",
        "    var inverse = model.inverse_transform(X)",
    ]
    for row, expected in enumerate(model.inverse_transform(X)):
        forms.append(f"    assert_equal(inverse[{row}], {literal(expected.tolist())})")
    if inspect is not None:
        for doc in inspect:
            forms.append(
                f"    assert_equal(model.build_analyzer()({literal(doc)}), "
                f"{literal(model.build_analyzer()(doc))})"
            )
    return "\n".join(forms)


def unicode_case(reference):
    documents = [
        "ÉTÉ Straße ΜΑΪΟΣ ΟΣ ΟΣΑ ΟΣ'Α Ῥόδος",
        "Σ ΣA AΣ AΣA AΣ. AΣ.A AΣ' AΣ'B AΣ‐B AΣ\u0345B",
        "İSTANBUL CAFE\u0301 ÉTE\u0301 𐐀𐐁 𐒠𐒡",
        "🐈‍⬛ hy-phen 日本語 ⅣⅥ ²½ ๑๒ _ __ _A x_y ሀሁ",
        "A\t\tB\tC  D\u00a0E\u2028\u2029F\u2003F",
        "a b c d x", "", " 	",
    ]
    reference_analyzer = reference().build_analyzer()
    return "\n".join([
        "def unicode_edges() raises:",
        "    var model = CountVectorizer()",
        "    var preprocess = model.build_preprocessor()",
        "    var tokenize = model.build_tokenizer()",
        *[line for doc in documents for line in [
            f"    assert_equal(preprocess({literal(doc)}), {literal(doc.lower())})",
            f"    assert_equal(tokenize({literal(doc)}), {literal(re.findall(reference().token_pattern, doc))})",
            f"    assert_equal(model.build_analyzer()({literal(doc)}), {literal(reference_analyzer(doc))})",
        ]],
    ])


def main():
    definitions = reference_text()
    reference = definitions["CountVectorizer"]
    train = ["Tea tea steep; Café", "Café COFFEE and the tea", "coffee tea latte tea", "the and"]
    query = ["TEA tea Latte latte", "unknown", "Café; and the"]
    settings = [
        {}, {"lowercase": False}, {"binary": True}, {"stop_words": "english"},
        {"stop_words": ["tea", "and"], "ngram_range": (1, 2)},
        {"ngram_range": (2, 3)}, {"min_df": 2}, {"min_df": 0.5},
        {"max_df": 2}, {"max_df": 0.5}, {"max_features": 2},
        {"vocabulary": ["tea", "Café", "absent"]},
        {"vocabulary": {"tea": 2, "café": 0, "absent": 1}}, {"dtype": np.float32},
        {"dtype": np.float64},
    ]
    cases = [mojo_case(i, reference, train, query, inspect=query, **options)
             for i, options in enumerate(settings)]
    char_train = ["A\tB\t\t猫 😺", "É Éİ 👩‍🔬", "中国 中国", "  X ", ""]
    char_query = ["x the	this", "猫 😺", ""]
    for analyzer in ("char", "char_wb"):
        for ngrams in ((1, 1), (1, 3), (2, 4), (4, 5)):
            cases.append(mojo_case(
                len(cases), reference, char_train, char_query, inspect=char_train + char_query,
                analyzer=analyzer, ngram_range=ngrams,
            ))
    for binary in (False, True):
        cases.append(mojo_case(
            len(cases), reference,
            ["zeta zeta zeta zeta alpha omega", "alpha beta", "alpha beta", "gamma"],
            ["alpha beta zeta"], max_features=2, binary=binary,
        ))
    BUILD.mkdir(parents=True, exist_ok=True)
    for encoding, data in [("utf-8", b"Coffee Caf\xc3\xa9\xe2\x82 tea"),
                           ("ascii", b"Coffee \xff Tea")]:
        fixture = BUILD / f"count-{encoding}.txt"
        fixture.write_bytes(data)
        for errors in ("ignore", "replace"):
            paths = [str(fixture)]
            cases.append(mojo_case(len(cases), reference, paths, paths, inspect=paths,
                                   input="filename", encoding=encoding, decode_error=errors))
    tags = ["a<a>é</a>ß", "<> <<x> <a", "<a\nλ>text", "α>", "<\nab> b <x<<>"]
    extra = [
        unicode_case(reference),
        "def tags() raises:\n" + "\n".join(
            f"    assert_equal(strip_tags({literal(s)}), {literal(definitions['strip_tags'](s))})"
            for s in tags
        ),
    ]
    source = BUILD / "count_parity.mojo"
    output = BUILD / "count_parity"
    calls = "\n".join(
        f'    print("count differential case {i}")\n    case_{i}()' for i in range(len(cases))
    )
    source.write_text("\n\n".join(
        [HEADER, *cases, *extra, f"def main() raises:\n{calls}\n    unicode_edges()\n    tags()\n"]
    ), encoding="utf-8")
    subprocess.run(compiler_command([
        "mojo", "build", "-O3", "--fp-mode=contract=off", "-I", BUILD / "deps",
        source, "-o", output,
    ]), check=True, cwd=MOXIE)
    result = subprocess.run([output], capture_output=True, text=True)
    if result.returncode:
        print(result.stdout, end="")
        print(result.stderr, end="")
        result.check_returncode()
    print(f"PASS: {len(cases)} dense CountVectorizer differential cases and Unicode/tag fixtures "
          f"against sklearn {REVISION}")


if __name__ == "__main__":
    main()
