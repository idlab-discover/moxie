# Dense text feature extraction

## TF-IDF weighting

```mojo
from moxie.feature_extraction.text import TfidfTransformer
from numojo import Shape, zeros

def main() raises:
    var counts = zeros[DType.int64](Shape(2, 3))
    counts.unsafe_set(0, 2)
    counts.unsafe_set(4, 1)
    var tfidf = TfidfTransformer[DType.int64]()
    var weighted = tfidf.fit_transform(counts)
    var idf = tfidf.idf_()
```

`TfidfTransformer[dtype]` takes a rank-two NuMojo `NDArray[dtype]` with
documents in rows and terms in columns. Float32 and Float64 retain their
dtype. Integer and Boolean counts produce Float64 weights. The input dtype is
fixed for each estimator instance and defaults to Float64. The result is an
owned dense matrix. `transform(X, copy=False)` also leaves the borrowed input
unchanged, as in scikit-learn's dense-input path. Input views may have positive
or negative strides.

The constructor defaults are `norm="l2"`, `use_idf=True`, `smooth_idf=True`
and `sublinear_tf=False`. `norm` accepts `"l1"`, `"l2"` or `None`. With
sublinear weighting, each nonzero count becomes `1 + log(count)`. Smoothing
adds one document to both the corpus size and each term's document frequency.
With smoothing disabled, an unseen feature's IDF is positive infinity.
Zero entries stay zero, even in those columns. As in scikit-learn, normalized
transforms reject nonfinite weighted values. With `norm=None`, those values
can propagate to the result.

`fit` and `set_params` return mutable borrows. `get_params` returns a
`TfidfTransformerParams` record. Pass a record or named keyword updates to
`set_params`. Clone copies configuration into an unfitted transformer.
Successful refits replace learned state, including clearing IDF when
`use_idf=False`. `idf_()` returns an independent rank-one array and raises
when the transformer has no learned IDF. `n_features_in_()` returns the fitted
width. NuMojo arrays carry no column labels. `get_feature_names_out()`
generates `x0`, `x1`, and so on, or returns the supplied `List[String]` after
checking its length.

The TF-IDF loop traverses NuMojo strides directly and allocates only the
result during transform. The weighting and row-norm accumulation run in one
pass, followed by row scaling when requested. Numax's array logarithms and
NuMojo's elementwise logarithms would add whole-matrix buffers and require
masking implicit zeros. Numax `core.libm.log` falls back to the standard
library for Float32. The implementation uses platform libm `log` and `logf`
for both dtypes. The standard dense normalizer suppresses tiny norms with an
epsilon guard. TF-IDF follows the reference CSR normalizer's exact zero check
and retains Float32 rounding of squared values before the Float64 sum.

### Validation

From the Moxie root with the locked Mojo and Python active:

```sh
python shared/assembly/tools/assemble_family.py \
  --consumer families/feature_extraction/algorithms/text \
  --lock DEPENDENCIES.json \
  --output families/feature_extraction/algorithms/text/builds/deps
python families/feature_extraction/algorithms/text/tests/parity.py
```

The oracle executes the local pinned scikit-learn TF-IDF and normalization
bodies. Python tooling uses the locked sklearn wheel's compiled CSR row
normalizers. The tests allow Float32 relative error of 2e-6 and Float64
relative error of 3e-15 for libm versus NumPy rounding and normalization.
Zero, infinity and NaN are checked separately. The family contract test
`tests/test_tfidf.mojo` covers lifecycle, input validation, array ownership,
strided input and `TransformerPipeline` composition.

## Document counts

```mojo
from moxie.feature_extraction.text import CountVectorizer, TfidfTransformer
from moxie.pipeline import TransformerPipeline

def main() raises:
    var documents: List[String] = ["Tea and coffee", "Coffee coffee"]
    var counter = CountVectorizer(stop_words=String("english"))
    var counts = counter.fit_transform(documents)
    var terms = counter.get_feature_names_out()  # ["coffee", "tea"]

    var pipeline = TransformerPipeline(counter^, TfidfTransformer[DType.int64]())
    var weighted = pipeline.fit_transform(documents)
```

`CountVectorizer[dtype, Y]` takes a `List[String]` and returns an owned dense
NuMojo matrix with documents in rows and terms in columns. The output dtype
defaults to Int64. Float32 and Float64 are also available. The optional fit
target is ignored. Its type `Y` defaults to the matching TF-IDF output array
type and can be set to the target type required by a typed pipeline.

With `input="content"` strings are documents. With `input="filename"` they
are paths read by Mojo. Use `decode(Span(bytes))` to decode byte sequences
before passing them as content. UTF-8, `utf8` and ASCII byte decoding support
`decode_error="strict"`, `"ignore"` or `"replace"`. Open file objects and other
codecs are outside the current API.

The default word analyzer uses Unicode 16.0.0 alphanumeric characters and
underscore with at least two code points per token. Lowercasing follows the
same Unicode version, including contextual Greek final sigma. The
`"char"` and `"char_wb"` analyzers operate on Unicode code points. Pass
a typed `String` for the analyzer name and for `stop_words=String("english")`,
or transfer a typed `List[String]` for custom stop words. Custom
`preprocessor`, `tokenizer` and `analyzer` functions take a `String`,
are declared `raises` and have no captured state. Preprocessors return
`String`. Tokenizers and analyzers return `List[String]`. A custom analyzer
receives the decoded document and replaces all other analysis steps.
`build_preprocessor()`, `build_tokenizer()` and `build_analyzer()` return
callable configuration snapshots. `get_stop_words()` returns a set, empty
by default. Custom regex patterns and built-in accent stripping raise.
`strip_tags(String)` is available independently.

`min_df` and `max_df` take `Int` document counts or `Float64` document
proportions. `max_features` enforces the feature limit after document-frequency
pruning. Equal term frequencies are resolved in lexical order for every
vocabulary size. The reference delegates ties to NumPy's default unstable
sort, so tied groups can produce different selections at the cut-off.
With `binary=True`, presence is used for both counts and feature ranking.
A fixed vocabulary can be transferred as a typed `List[String]` in column
order or `Dict[String, Int]` with unique, contiguous zero-based indices.
Fixed vocabularies can transform before fitting. Counts for unknown terms are
ignored. A learned vocabulary uses lexical term order.

`vocabulary_()` and `get_feature_names_out()` return independent snapshots.
`fixed_vocabulary_()` reports the active vocabulary's source.
`inverse_transform()` reads nonzero positions in column order and returns
`List[List[String]]`. It accepts strided matrices with the fitted width.
To change configuration, update the `CountVectorizerParams` record from
`get_params()` and pass it to `set_params`. Refitting replaces the vocabulary
after success. Clone copies configuration into an unfitted estimator.

Dense output uses memory proportional to document count times retained
vocabulary size. Use `min_df` or `max_features` to reduce the retained width.
`fit` avoids materializing a dense matrix. `fit_transform` counts each
document once and builds the matrix after pruning. NuMojo supplies the dense
allocation and strided reads. Numax matrix reductions would require a dense
matrix at the full pre-pruning width and cannot consume token strings or
vocabulary lookups.

### Validation

After assembling the text consumer as above, run:

```sh
python families/feature_extraction/algorithms/text/tests/count_parity.py
```

The comparator executes the pinned scikit-learn vectorizer and tokenizer method
bodies and compares dense counts, feature names, analyzer output and inverse
transforms exactly. The family contract test `tests/test_count_vectorizer.mojo`
checks refits, typed callbacks, invalid parameters, byte decoding, a strided
inverse transform and native TF-IDF composition.

The generated Unicode tables can be rebuilt with the locked Python from
`algorithms/text/tools/generate_unicode.py`. They derive from [Unicode
16.0.0](https://www.unicode.org/versions/Unicode16.0.0/) as exposed by CPython
3.14.7 and carry the accompanying Unicode license.
