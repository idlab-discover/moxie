# Adapted from scikit-learn sklearn/feature_extraction/text.py.
# Copyright (c) scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause

from std.builtin.sort import sort
from std.collections import Set
from std.math import isfinite
from std.utils import Variant

from numojo import NDArray, Shape, zeros
from moxie._feature_metadata import FeatureNamesOut
from moxie.feature_extraction._base import FeatureExtractor
from moxie.feature_extraction._text_analysis import (
    DEFAULT_TOKEN_PATTERN, DocumentAnalyzer, DocumentPreprocessor, DocumentTokenizer,
    TextPreprocessor, TextTokenizer, decode_bytes, decode_document, stop_word_set,
)
from moxie.feature_extraction._text_contracts import check_ngram_range

comptime DocumentFrequency = Variant[Int, Float64]
comptime TextStopWords = Variant[NoneType, String, List[String]]
comptime TextVocabulary = Variant[NoneType, List[String], Dict[String, Int]]
comptime TextAnalysis = Variant[String, TextTokenizer]


struct CountVectorizerParams(Copyable, Movable):
    """Constructor configuration for learning and applying a vocabulary."""

    var input: String
    var encoding: String
    var decode_error: String
    var strip_accents: Optional[String]
    var lowercase: Bool
    var preprocessor: Optional[TextPreprocessor]
    var tokenizer: Optional[TextTokenizer]
    var stop_words: TextStopWords
    var token_pattern: Optional[String]
    var ngram_range: Tuple[Int, Int]
    var analyzer: TextAnalysis
    var max_df: DocumentFrequency
    var min_df: DocumentFrequency
    var max_features: Optional[Int]
    var vocabulary: TextVocabulary
    var binary: Bool

    def __init__(
        out self, *,
        input: String = "content",
        encoding: String = "utf-8",
        decode_error: String = "strict",
        strip_accents: Optional[String] = None,
        lowercase: Bool = True,
        preprocessor: Optional[TextPreprocessor] = None,
        tokenizer: Optional[TextTokenizer] = None,
        stop_words: TextStopWords = NoneType(),
        token_pattern: Optional[String] = String(DEFAULT_TOKEN_PATTERN),
        ngram_range: Tuple[Int, Int] = (1, 1),
        analyzer: TextAnalysis = String("word"),
        max_df: DocumentFrequency = Float64(1),
        min_df: DocumentFrequency = 1,
        max_features: Optional[Int] = None,
        vocabulary: TextVocabulary = NoneType(),
        binary: Bool = False,
    ):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents.copy()
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.stop_words = stop_words.copy()
        self.token_pattern = token_pattern.copy()
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary = vocabulary.copy()
        self.binary = binary


@fieldwise_init
struct _Vocabulary(Copyable, Movable):
    var names: List[String]
    var columns: Dict[String, Int]
    var fixed: Bool


def _make_vocabulary(names: List[String], fixed: Bool) raises -> _Vocabulary:
    if len(names) == 0:
        raise Error("ValueError: empty vocabulary")
    var columns = Dict[String, Int](capacity=len(names))
    for i in range(len(names)):
        if names[i] in columns:
            raise Error(String("ValueError: duplicate term in vocabulary: ", names[i]))
        columns[names[i]] = i
    return _Vocabulary(names.copy(), columns^, fixed)


def _fixed_vocabulary(option: TextVocabulary) raises -> _Vocabulary:
    if option.isa[List[String]]():
        return _make_vocabulary(option[List[String]], True)
    ref columns = option[Dict[String, Int]]
    if len(columns) == 0:
        raise Error("ValueError: empty vocabulary")
    var names = List[String](length=len(columns), fill=String())
    var occupied = List[Bool](length=len(columns), fill=False)
    for entry in columns.items():
        var index = entry.value
        if index < 0 or index >= len(names) or occupied[index]:
            raise Error("ValueError: vocabulary indices must be unique and contiguous from zero")
        occupied[index] = True
        names[index] = entry.key
    return _Vocabulary(names^, columns.copy(), True)


def _df_limit(value: DocumentFrequency, documents: Int) raises -> Float64:
    if value.isa[Int]():
        var count = value[Int]
        if count < 1:
            raise Error("ValueError: document-frequency count must be positive")
        return Float64(count)
    var fraction = value[Float64]
    if not isfinite(fraction) or fraction < 0 or fraction > 1:
        raise Error("ValueError: document-frequency fraction must be in [0, 1]")
    return fraction * Float64(documents)


@fieldwise_init
struct _TermRank(Comparable, Copyable, Movable):
    var name: String
    var count: Int


    def __lt__(self, other: Self) -> Bool:
        return self.count > other.count or (
            self.count == other.count and self.name < other.name
        )

    def __eq__(self, other: Self) -> Bool:
        return self.count == other.count and self.name == other.name

    def __ne__(self, other: Self) -> Bool:
        return not self == other

    def __le__(self, other: Self) -> Bool:
        return not other < self

    def __gt__(self, other: Self) -> Bool:
        return other < self

    def __ge__(self, other: Self) -> Bool:
        return not self < other


def _row_counts(
    terms: List[String], binary: Bool, mut columns: Dict[String, Int]
) raises -> Dict[Int, Int]:
    var counts = Dict[Int, Int]()
    for term in terms:
        var column = columns.get(term, -1)
        if column < 0:
            column = len(columns)
            columns[term] = column
        counts[column] = 1 if binary else counts.get(column, 0) + 1
    return counts^


struct CountVectorizer[
    dtype: DType = DType.int64,
    Y: Movable = NDArray[DType.float32 if dtype == DType.float32 else DType.float64],
](FeatureExtractor, FeatureNamesOut, Copyable):
    """Convert a batch of strings to a dense document-by-term count matrix."""

    comptime Input = List[String]
    comptime Transformed = NDArray[Self.dtype]
    comptime Target = Self.Y
    comptime Params = CountVectorizerParams

    var _params: Self.Params
    var _vocabulary: Optional[_Vocabulary]

    def __init__(out self, params: Self.Params):
        comptime assert (
            Self.dtype == DType.int64 or Self.dtype == DType.float32 or Self.dtype == DType.float64
        ), "CountVectorizer supports Int64, Float32 and Float64 output"
        self._params = params.copy()
        self._vocabulary = None

    def __init__(
        out self, *,
        input: String = "content",
        encoding: String = "utf-8",
        decode_error: String = "strict",
        strip_accents: Optional[String] = None,
        lowercase: Bool = True,
        preprocessor: Optional[TextPreprocessor] = None,
        tokenizer: Optional[TextTokenizer] = None,
        stop_words: TextStopWords = NoneType(),
        token_pattern: Optional[String] = String(DEFAULT_TOKEN_PATTERN),
        ngram_range: Tuple[Int, Int] = (1, 1),
        analyzer: TextAnalysis = String("word"),
        max_df: DocumentFrequency = Float64(1),
        min_df: DocumentFrequency = 1,
        max_features: Optional[Int] = None,
        vocabulary: TextVocabulary = NoneType(),
        binary: Bool = False,
    ):
        self = Self(Self.Params(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor,
            tokenizer=tokenizer, stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, analyzer=analyzer, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
        ))

    def get_params(self, deep: Bool = True) -> Self.Params:
        return self._params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        self._params = params.copy()
        return self

    def __sklearn_clone__(self) -> Self:
        return Self(self._params)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._vocabulary)

    def decode(self, doc: String) raises -> String:
        return decode_document(doc, self._params.input, self._params.encoding, self._params.decode_error)

    def decode(self, doc: Span[UInt8, _]) raises -> String:
        return decode_bytes(doc, self._params.encoding, self._params.decode_error)

    def build_preprocessor(self) raises -> DocumentPreprocessor:
        if not self._params.preprocessor and self._params.strip_accents:
            raise Error("NotImplementedError: CountVectorizer built-in accent stripping is unavailable")
        return DocumentPreprocessor(self._params.lowercase, self._params.preprocessor)

    def build_tokenizer(self) raises -> DocumentTokenizer:
        if not self._params.tokenizer:
            if not self._params.token_pattern or self._params.token_pattern.value() != DEFAULT_TOKEN_PATTERN:
                raise Error("NotImplementedError: CountVectorizer requires the default pattern or a custom tokenizer")
        return DocumentTokenizer(self._params.tokenizer)

    def get_stop_words(self) raises -> Set[String]:
        if self._params.stop_words.isa[String]():
            return stop_word_set(self._params.stop_words[String], List[String]())
        if self._params.stop_words.isa[List[String]]():
            return stop_word_set(None, self._params.stop_words[List[String]])
        return Set[String]()

    def build_analyzer(self) raises -> DocumentAnalyzer:
        ref p = self._params
        if p.input != "content" and p.input != "filename":
            raise Error("NotImplementedError: CountVectorizer input must be content or filename")
        if p.analyzer.isa[TextTokenizer]():
            return DocumentAnalyzer(
                p.input, p.encoding, p.decode_error, String(), p.analyzer[TextTokenizer],
                DocumentPreprocessor(False, None), DocumentTokenizer(None),
                Set[String](), p.ngram_range,
            )
        check_ngram_range(p.ngram_range[0], p.ngram_range[1])
        var mode = p.analyzer[String]
        if mode != "word" and mode != "char" and mode != "char_wb":
            raise Error("ValueError: analyzer must be word, char, char_wb or a callable")
        var tokenizer = self.build_tokenizer() if mode == "word" else DocumentTokenizer(None)
        var stop_words = self.get_stop_words() if mode == "word" else Set[String]()
        return DocumentAnalyzer(
            p.input, p.encoding, p.decode_error, mode, None,
            self.build_preprocessor(), tokenizer^, stop_words^, p.ngram_range,
        )

    def _configured_vocabulary(self) raises -> _Vocabulary:
        if not self._params.vocabulary.isa[NoneType]():
            return _fixed_vocabulary(self._params.vocabulary)
        raise Error("NotFittedError: CountVectorizer vocabulary is unavailable")

    def _activate_fixed_vocabulary(mut self) raises:
        if not self._vocabulary:
            self._vocabulary = self._configured_vocabulary()

    def vocabulary_(self) raises -> Dict[String, Int]:
        if self._vocabulary:
            return self._vocabulary.value().columns.copy()
        return self._configured_vocabulary().columns.copy()

    def fixed_vocabulary_(self) raises -> Bool:
        if self._vocabulary:
            return self._vocabulary.value().fixed
        return self._configured_vocabulary().fixed

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        if self._vocabulary:
            return self._vocabulary.value().names.copy()
        return self._configured_vocabulary().names.copy()

    def _prune(
        self, columns: Dict[String, Int], frequencies: List[Int], totals: List[Int], documents: Int
    ) raises -> _Vocabulary:
        if len(columns) == 0:
            raise Error("ValueError: empty vocabulary; documents may contain only stop words")
        var low = _df_limit(self._params.min_df, documents)
        var high = _df_limit(self._params.max_df, documents)
        if high < low:
            raise Error("ValueError: max_df corresponds to fewer documents than min_df")
        var ranks = List[_TermRank](capacity=len(columns))
        for entry in columns.items():
            var frequency = Float64(frequencies[entry.value])
            if low <= frequency <= high:
                ranks.append(_TermRank(entry.key, totals[entry.value]))
        if len(ranks) == 0:
            raise Error("ValueError: after pruning, no terms remain")
        var limit = self._params.max_features.or_else(len(ranks))
        if limit < len(ranks):
            sort(Span(ranks))
        var names = List[String](capacity=min(limit, len(ranks)))
        for i in range(min(limit, len(ranks))):
            names.append(ranks[i].name)
        sort(Span(names))
        return _make_vocabulary(names, False)

    def _learn[return_counts: Bool](
        mut self, raw_documents: Self.Input,
    ) raises -> Optional[Self.Transformed]:
        var errors = self._params.decode_error
        if errors != "strict" and errors != "ignore" and errors != "replace":
            raise Error("ValueError: decode_error must be strict, ignore or replace")
        _ = _df_limit(self._params.min_df, len(raw_documents))
        _ = _df_limit(self._params.max_df, len(raw_documents))
        if self._params.max_features.or_else(1) < 1:
            raise Error("ValueError: max_features must be positive")
        check_ngram_range(self._params.ngram_range[0], self._params.ngram_range[1])
        var analyzer = self.build_analyzer()
        if not self._params.vocabulary.isa[NoneType]():
            var fixed = _fixed_vocabulary(self._params.vocabulary)
            var fixed_counts: Optional[Self.Transformed] = None
            comptime if return_counts:
                fixed_counts = self._transform(raw_documents, analyzer, fixed)
            else:
                for doc in raw_documents:
                    _ = analyzer(doc)
            self._vocabulary = fixed^
            return fixed_counts^

        var columns = Dict[String, Int]()
        var frequencies = List[Int]()
        var totals = List[Int]()
        var rows = List[Dict[Int, Int]](capacity=len(raw_documents) if return_counts else 0)
        for doc in raw_documents:
            var counts = _row_counts(analyzer(doc), self._params.binary, columns)
            while len(frequencies) < len(columns):
                frequencies.append(0)
                totals.append(0)
            for entry in counts.items():
                frequencies[entry.key] += 1
                totals[entry.key] += entry.value
            comptime if return_counts:
                rows.append(counts^)
        var vocabulary = self._prune(columns, frequencies, totals, len(raw_documents))
        var result: Optional[Self.Transformed] = None
        comptime if return_counts:
            var matrix = zeros[Self.dtype](Shape(len(rows), len(vocabulary.names)))
            var remap = List[Int](length=len(columns), fill=-1)
            for entry in vocabulary.columns.items():
                remap[columns[entry.key]] = entry.value
            for row in range(len(rows)):
                var offset = row * matrix.shape[1]
                for entry in rows[row].items():
                    var column = remap[entry.key]
                    if column >= 0:
                        matrix.unsafe_set(offset + column, Int64(entry.value).cast[Self.dtype]())
            result = matrix^
        self._vocabulary = vocabulary^
        return result^

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        _ = self._learn[False](X)
        return self

    def fit_transform(
        mut self, X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> Self.Transformed:
        var result = self._learn[True](X)
        return result.take()

    def _transform(
        self, documents: Self.Input, analyzer: DocumentAnalyzer, vocabulary: _Vocabulary
    ) raises -> Self.Transformed:
        var matrix = zeros[Self.dtype](Shape(len(documents), len(vocabulary.names)))
        for row in range(len(documents)):
            var offset = row * len(vocabulary.names)
            var counts = Dict[Int, Int]()
            for term in analyzer(documents[row]):
                var column = vocabulary.columns.get(term, -1)
                if column >= 0:
                    counts[column] = 1 if self._params.binary else counts.get(column, 0) + 1
            for entry in counts.items():
                matrix.unsafe_set(offset + entry.key, Int64(entry.value).cast[Self.dtype]())
        return matrix^

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        if self._vocabulary:
            return self._transform(X, self.build_analyzer(), self._vocabulary.value())
        var vocabulary = self._configured_vocabulary()
        return self._transform(X, self.build_analyzer(), vocabulary)

    def inverse_transform(self, X: Self.Transformed) raises -> List[List[String]]:
        if self._vocabulary:
            return self._inverse_transform(X, self._vocabulary.value().names)
        return self._inverse_transform(X, self._configured_vocabulary().names)

    def _inverse_transform[value_dtype: DType](self, X: NDArray[value_dtype], names: List[String]) raises -> List[List[String]]:
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != len(names):
            raise Error("ValueError: expected a nonempty matrix with fitted vocabulary width")
        var result = List[List[String]](capacity=X.shape[0])
        var row_stride = Int(X.strides[0])
        var column_stride = Int(X.strides[1])
        for row in range(X.shape[0]):
            var terms = List[String]()
            for column in range(X.shape[1]):
                var count = X.unsafe_get(row * row_stride + column * column_stride)
                if not isfinite(count.cast[DType.float64]()):
                    raise Error("ValueError: X contains nonfinite values")
                if count != 0:
                    terms.append(names[column])
            result.append(terms^)
        return result^
