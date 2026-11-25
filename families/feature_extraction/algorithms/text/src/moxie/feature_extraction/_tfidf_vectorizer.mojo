# Adapted from scikit-learn sklearn/feature_extraction/text.py.
# Copyright (c) scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense TF-IDF vectorization of text documents."""

from numojo import NDArray
from moxie._feature_metadata import FeatureNamesOut
from moxie.feature_extraction._base import FeatureExtractor
from moxie.feature_extraction._count_vectorizer import (
    CountVectorizer,
    CountVectorizerParams,
    DocumentFrequency,
    TextAnalysis,
    TextStopWords,
    TextVocabulary,
)
from moxie.feature_extraction._text_analysis import (
    DEFAULT_TOKEN_PATTERN,
    DocumentAnalyzer,
    DocumentPreprocessor,
    DocumentTokenizer,
    TextPreprocessor,
    TextTokenizer,
)
from moxie.feature_extraction._tfidf import TfidfTransformer, TfidfTransformerParams


@fieldwise_init
struct TfidfVectorizerParams(Copyable, Movable):
    """Configuration for document counting and TF-IDF weighting."""

    var count: CountVectorizerParams
    var weighting: TfidfTransformerParams


struct TfidfVectorizer[
    dtype: DType = DType.float64,
    Y: Movable & Deinitable = NDArray[DType.float32 if dtype == DType.float32 else DType.float64],
](FeatureExtractor, FeatureNamesOut, Copyable):
    """Learn a vocabulary and transform documents into a dense TF-IDF matrix."""

    comptime Counter = CountVectorizer[Self.dtype, Self.Y]
    comptime Weighting = TfidfTransformer[Self.dtype]
    comptime Input = List[String]
    comptime Target = Self.Y
    comptime Transformed = Self.Weighting.Transformed
    comptime Params = TfidfVectorizerParams

    var _counter: Self.Counter
    var _weighting_params: TfidfTransformerParams
    var _weighting: Self.Weighting

    def __init__(out self, params: Self.Params):
        self._counter = Self.Counter(params.count)
        self._weighting_params = params.weighting.copy()
        self._weighting = Self.Weighting()

    def __init__(
        out self,
        *,
        input: String = "content",
        encoding: String = "utf-8",
        decode_error: String = "strict",
        strip_accents: Optional[String] = None,
        lowercase: Bool = True,
        preprocessor: Optional[TextPreprocessor] = None,
        tokenizer: Optional[TextTokenizer] = None,
        analyzer: TextAnalysis = String("word"),
        stop_words: TextStopWords = NoneType(),
        token_pattern: Optional[String] = String(DEFAULT_TOKEN_PATTERN),
        ngram_range: Tuple[Int, Int] = (1, 1),
        max_df: DocumentFrequency = Float64(1),
        min_df: DocumentFrequency = 1,
        max_features: Optional[Int] = None,
        vocabulary: TextVocabulary = NoneType(),
        binary: Bool = False,
        norm: Optional[String] = String("l2"),
        use_idf: Bool = True,
        smooth_idf: Bool = True,
        sublinear_tf: Bool = False,
    ):
        self = Self(Self.Params(
            CountVectorizerParams(
                input=input, encoding=encoding, decode_error=decode_error,
                strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor,
                tokenizer=tokenizer, stop_words=stop_words, token_pattern=token_pattern,
                ngram_range=ngram_range, analyzer=analyzer, max_df=max_df, min_df=min_df,
                max_features=max_features, vocabulary=vocabulary, binary=binary,
            ),
            TfidfTransformerParams(norm.copy(), use_idf, smooth_idf, sublinear_tf),
        ))

    def get_params(self, deep: Bool = True) -> Self.Params:
        return Self.Params(self._counter.get_params(), self._weighting_params.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        _ = self._counter.set_params(params.count)
        self._weighting_params = params.weighting.copy()
        return self

    def __sklearn_clone__(self) -> Self:
        return Self(self.get_params())

    def __sklearn_is_fitted__(self) -> Bool:
        return self._weighting.__sklearn_is_fitted__()

    def decode(self, doc: String) raises -> String:
        return self._counter.decode(doc)

    def decode(self, doc: Span[UInt8, _]) raises -> String:
        return self._counter.decode(doc)

    def build_preprocessor(self) raises -> DocumentPreprocessor:
        return self._counter.build_preprocessor()

    def build_tokenizer(self) raises -> DocumentTokenizer:
        return self._counter.build_tokenizer()

    def build_analyzer(self) raises -> DocumentAnalyzer:
        return self._counter.build_analyzer()

    def get_stop_words(self) raises -> Set[String]:
        return self._counter.get_stop_words()

    def vocabulary_(self) raises -> Dict[String, Int]:
        return self._counter.vocabulary_()

    def fixed_vocabulary_(self) raises -> Bool:
        return self._counter.fixed_vocabulary_()

    def get_feature_names_out(
        self, input_features: Optional[List[String]] = None
    ) raises -> List[String]:
        return self._counter.get_feature_names_out(input_features)

    def idf_(self) raises -> Self.Transformed:
        self._check_fitted()
        return self._weighting.idf_()

    def set_idf_[origin: MutOrigin, values_origin: ImmOrigin](
        ref[origin] self, ref[values_origin] values: Self.Transformed
    ) raises -> ref[self] Self:
        """Set term weights for the learned vocabulary or a configured fixed vocabulary."""
        if not self._weighting_params.use_idf:
            raise Error("ValueError: idf_ requires use_idf=True")
        if values.ndim != 1 or values.shape[0] != len(self.get_feature_names_out()):
            raise Error("ValueError: idf_ must be rank one with one value per vocabulary term")
        if self.__sklearn_is_fitted__():
            self._weighting._set_idf(values)
        else:
            var weighting = Self.Weighting()
            _ = weighting.set_params(self._weighting_params)
            weighting._set_idf(values)
            self._counter._activate_fixed_vocabulary()
            self._weighting = weighting^
        return self

    def _check_fitted(self) raises:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: the TF-IDF vectorizer is not fitted")

    def _fit[return_weights: Bool](
        mut self, X: Self.Input
    ) raises -> Optional[Self.Transformed]:
        # Publish the vocabulary and weights together, after both stages succeed.
        var counter = self._counter.__sklearn_clone__()
        var weighting = Self.Weighting()
        _ = weighting.set_params(self._weighting_params)
        weighting._check_norm()
        var counts = counter.fit_transform(X)
        _ = weighting.fit(counts)
        var result: Optional[Self.Transformed] = None
        comptime if return_weights:
            result = weighting.transform(counts)
        self._counter = counter^
        self._weighting = weighting^
        return result^

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        _ = self._fit[False](X)
        return self

    def fit_transform(
        mut self, X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> Self.Transformed:
        var result = self._fit[True](X)
        return result.take()

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._check_fitted()
        return self._weighting.transform(self._counter.transform(X))

    def inverse_transform(self, X: Self.Transformed) raises -> List[List[String]]:
        return self._counter._inverse_transform(X, self.get_feature_names_out())
