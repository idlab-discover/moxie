"""Compare dense TF-IDF vectorization to the pinned sklearn text.py bodies."""

from itertools import product
import subprocess
import warnings

import numpy as np

from count_parity import literal, reference_text
from parity import (
    BUILD,
    MOJO,
    MOXIE,
    REVISION,
    compiler_command,
    mojo_values,
    pinned_definitions,
    reference_transformer,
)


def mojo_case(index, reference, train, query, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Unseen fixed terms without smoothing.
        warnings.simplefilter("ignore", UserWarning)  # Integer count dtype in sklearn.
        fitted = reference(**options)
        X = fitted.fit_transform(train)
        Y = fitted.transform(query)
        fit_only = reference(**options).fit(train)
    count_dtype = np.dtype(options.get("dtype", np.float64)).name
    result_dtype = X.dtype.name
    params = []
    for name, value in options.items():
        if name == "dtype":
            continue
        argument = literal(value)
        if name == "vocabulary":
            argument = f"List[String]({argument})"
        if name in ("analyzer", "stop_words"):
            argument = f"String({argument})"
        if name in ("min_df", "max_df") and isinstance(value, float):
            argument = f"Float64({argument})"
        params.append(f"{name}={argument}")
    lines = [
        "@no_inline",
        f"def case_{index}() raises:",
        f"    var model = TfidfVectorizer[DType.{count_dtype}]({', '.join(params)})",
        f"    var train: List[String] = {literal(train)}",
        f"    var query: List[String] = {literal(query)}",
        f"    var X: NDArray[DType.{result_dtype}] = model.fit_transform(train)",
        f"    compare(X, {mojo_values(X.toarray())})",
        f"    compare(model.transform(query), {mojo_values(Y.toarray())})",
        f"    assert_equal(model.get_feature_names_out(), {literal(fitted.get_feature_names_out().tolist())})",
        f"    assert_equal(model.fixed_vocabulary_(), {fitted.fixed_vocabulary_})",
        "    var inverse = model.inverse_transform(X)",
    ]
    for row, terms in enumerate(fitted.inverse_transform(X)):
        lines.append(f"    assert_equal(inverse[{row}], {literal(sorted(terms.tolist(), key=fitted.vocabulary_.get))})")
    if fitted.use_idf:
        lines.append(f"    compare(model.idf_(), {mojo_values(fitted.idf_)})")
    lines.extend([
        "    var cloned = clone(model)",
        "    _ = cloned.fit(train)",
        f"    compare(cloned.transform(query), {mojo_values(fit_only.transform(query).toarray())})",
    ])
    return "\n".join(lines)


def main():
    namespace = reference_text()
    namespace["TfidfTransformer"] = reference_transformer()
    reference = pinned_definitions(
        "sklearn/feature_extraction/text.py", {"TfidfVectorizer"}, namespace,
    )["TfidfVectorizer"]
    train = ["Tea tea steep; Café", "Café COFFEE and the tea", "coffee tea latte tea", "the and"]
    query = ["TEA tea Latte latte", "unknown", "Café; and the"]
    settings = [
        dict(norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf, dtype=dtype)
        for norm, use_idf, smooth_idf, sublinear_tf, dtype in product(
            (None, "l1", "l2"), (False, True), (False, True), (False, True), (np.float32, np.float64)
        )
    ]
    settings.extend([
        {"dtype": np.int64, "binary": True, "use_idf": False, "norm": None},
        {"stop_words": "english", "ngram_range": (1, 2), "min_df": 2},
        {"max_df": 0.8, "max_features": 2},
        {"analyzer": "char", "ngram_range": (2, 3), "lowercase": False},
        {"analyzer": "char_wb", "ngram_range": (2, 3), "min_df": 2},
        {"vocabulary": ["tea", "café", "absent"], "smooth_idf": False},
        {"strip_accents": "ascii", "lowercase": False},
        {"strip_accents": "unicode", "analyzer": "char_wb", "ngram_range": (1, 2)},
    ])
    cases = [mojo_case(i, reference, train, query, **params) for i, params in enumerate(settings)]
    calls = "\n".join(f"    case_{i}()" for i in range(len(cases)))
    source = BUILD / "tfidf_vectorizer_parity.mojo"
    output = BUILD / "tfidf_vectorizer_parity"
    header = MOJO.replace(
        "from moxie.feature_extraction.text import TfidfTransformer",
        "from moxie.base import clone\nfrom moxie.feature_extraction.text import TfidfVectorizer",
    )
    source.write_text("\n\n".join([header, *cases, f"def main() raises:\n{calls}\n"]), encoding="utf-8")
    subprocess.run(compiler_command([
        "mojo", "build", "-O3", "--fp-mode=contract=off", "-I", BUILD / "deps",
        source, "-o", output,
    ]), check=True, cwd=MOXIE)
    subprocess.run([output], check=True)
    print(f"PASS: {len(cases)} dense TfidfVectorizer differential cases against sklearn {REVISION}")


if __name__ == "__main__":
    main()
