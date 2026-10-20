# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Public scikit-learn API declarations; all new operations are deferred."""
from moxie.feature_extraction._api_types import APIValue

struct CountVectorizer(Movable):

    "API scaffold for sklearn.feature_extraction.text.CountVectorizer. Operations are deferred."

    var input: APIValue

    var encoding: APIValue

    var decode_error: APIValue

    var strip_accents: APIValue

    var lowercase: APIValue

    var preprocessor: APIValue

    var tokenizer: APIValue

    var stop_words: APIValue

    var token_pattern: APIValue

    var ngram_range: APIValue

    var analyzer: APIValue

    var max_df: APIValue

    var min_df: APIValue

    var max_features: APIValue

    var vocabulary: APIValue

    var binary: APIValue

    var dtype: APIValue

    var vocabulary_: APIValue

    var fixed_vocabulary_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: CountVectorizer.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: CountVectorizer.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: CountVectorizer.set_params is an API scaffold")

    def decode(mut self, doc: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decode(self, doc)."
        raise Error("NotImplementedError: CountVectorizer.decode is an API scaffold")

    def build_preprocessor(mut self) raises -> APIValue:
        "Scikit-learn signature: build_preprocessor(self)."
        raise Error("NotImplementedError: CountVectorizer.build_preprocessor is an API scaffold")

    def build_tokenizer(mut self) raises -> APIValue:
        "Scikit-learn signature: build_tokenizer(self)."
        raise Error("NotImplementedError: CountVectorizer.build_tokenizer is an API scaffold")

    def get_stop_words(mut self) raises -> APIValue:
        "Scikit-learn signature: get_stop_words(self)."
        raise Error("NotImplementedError: CountVectorizer.get_stop_words is an API scaffold")

    def build_analyzer(mut self) raises -> APIValue:
        "Scikit-learn signature: build_analyzer(self)."
        raise Error("NotImplementedError: CountVectorizer.build_analyzer is an API scaffold")

    def __init__(out self, *, input: Some[Movable] = "content", encoding: Some[Movable] = "utf-8", decode_error: Some[Movable] = "strict", strip_accents: Some[Movable] = None, lowercase: Some[Movable] = True, preprocessor: Some[Movable] = None, tokenizer: Some[Movable] = None, stop_words: Some[Movable] = None, token_pattern: Some[Movable] = "(?u)\\b\\w\\w+\\b", ngram_range: Some[Movable] = (1, 1), analyzer: Some[Movable] = "word", max_df: Some[Movable] = 1.0, min_df: Some[Movable] = 1, max_features: Some[Movable] = None, vocabulary: Some[Movable] = None, binary: Some[Movable] = False, dtype: Some[Movable] = DType.int64) raises:
        "Scikit-learn signature: __init__(self, *, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\\\\b\\\\w\\\\w+\\\\b', ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=np.int64)."
        raise Error("NotImplementedError: CountVectorizer.__init__ is an API scaffold")

    def fit(mut self, raw_documents: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, raw_documents, y=None)."
        raise Error("NotImplementedError: CountVectorizer.fit is an API scaffold")

    def fit_transform(mut self, raw_documents: Some[Movable], y: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, raw_documents, y=None)."
        raise Error("NotImplementedError: CountVectorizer.fit_transform is an API scaffold")

    def transform(mut self, raw_documents: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, raw_documents)."
        raise Error("NotImplementedError: CountVectorizer.transform is an API scaffold")

    def inverse_transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: inverse_transform(self, X)."
        raise Error("NotImplementedError: CountVectorizer.inverse_transform is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: CountVectorizer.get_feature_names_out is an API scaffold")

struct HashingVectorizer(Movable):

    "API scaffold for sklearn.feature_extraction.text.HashingVectorizer. Operations are deferred."

    var input: APIValue

    var encoding: APIValue

    var decode_error: APIValue

    var strip_accents: APIValue

    var lowercase: APIValue

    var preprocessor: APIValue

    var tokenizer: APIValue

    var stop_words: APIValue

    var token_pattern: APIValue

    var ngram_range: APIValue

    var analyzer: APIValue

    var n_features: APIValue

    var binary: APIValue

    var norm: APIValue

    var alternate_sign: APIValue

    var dtype: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: HashingVectorizer.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: HashingVectorizer.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: HashingVectorizer.set_params is an API scaffold")

    def decode(mut self, doc: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decode(self, doc)."
        raise Error("NotImplementedError: HashingVectorizer.decode is an API scaffold")

    def build_preprocessor(mut self) raises -> APIValue:
        "Scikit-learn signature: build_preprocessor(self)."
        raise Error("NotImplementedError: HashingVectorizer.build_preprocessor is an API scaffold")

    def build_tokenizer(mut self) raises -> APIValue:
        "Scikit-learn signature: build_tokenizer(self)."
        raise Error("NotImplementedError: HashingVectorizer.build_tokenizer is an API scaffold")

    def get_stop_words(mut self) raises -> APIValue:
        "Scikit-learn signature: get_stop_words(self)."
        raise Error("NotImplementedError: HashingVectorizer.get_stop_words is an API scaffold")

    def build_analyzer(mut self) raises -> APIValue:
        "Scikit-learn signature: build_analyzer(self)."
        raise Error("NotImplementedError: HashingVectorizer.build_analyzer is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: HashingVectorizer.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None)."
        raise Error("NotImplementedError: HashingVectorizer.fit_transform is an API scaffold")

    def __init__(out self, *, input: Some[Movable] = "content", encoding: Some[Movable] = "utf-8", decode_error: Some[Movable] = "strict", strip_accents: Some[Movable] = None, lowercase: Some[Movable] = True, preprocessor: Some[Movable] = None, tokenizer: Some[Movable] = None, stop_words: Some[Movable] = None, token_pattern: Some[Movable] = "(?u)\\b\\w\\w+\\b", ngram_range: Some[Movable] = (1, 1), analyzer: Some[Movable] = "word", n_features: Some[Movable] = 1048576, binary: Some[Movable] = False, norm: Some[Movable] = "l2", alternate_sign: Some[Movable] = True, dtype: Some[Movable] = DType.float64) raises:
        "Scikit-learn signature: __init__(self, *, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\\\\b\\\\w\\\\w+\\\\b', ngram_range=(1, 1), analyzer='word', n_features=2 ** 20, binary=False, norm='l2', alternate_sign=True, dtype=np.float64)."
        raise Error("NotImplementedError: HashingVectorizer.__init__ is an API scaffold")

    def partial_fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: partial_fit(self, X, y=None)."
        raise Error("NotImplementedError: HashingVectorizer.partial_fit is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: HashingVectorizer.fit is an API scaffold")

    def transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, X)."
        raise Error("NotImplementedError: HashingVectorizer.transform is an API scaffold")

struct TfidfTransformer(Movable):

    "API scaffold for sklearn.feature_extraction.text.TfidfTransformer. Operations are deferred."

    var norm: APIValue

    var use_idf: APIValue

    var smooth_idf: APIValue

    var sublinear_tf: APIValue

    var idf_: APIValue

    var n_features_in_: APIValue

    var feature_names_in_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: TfidfTransformer.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: TfidfTransformer.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: TfidfTransformer.set_params is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: TfidfTransformer.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None, fit_params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None, **fit_params)."
        raise Error("NotImplementedError: TfidfTransformer.fit_transform is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: TfidfTransformer.get_feature_names_out is an API scaffold")

    def __init__(out self, *, norm: Some[Movable] = "l2", use_idf: Some[Movable] = True, smooth_idf: Some[Movable] = True, sublinear_tf: Some[Movable] = False) raises:
        "Scikit-learn signature: __init__(self, *, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)."
        raise Error("NotImplementedError: TfidfTransformer.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X, y=None)."
        raise Error("NotImplementedError: TfidfTransformer.fit is an API scaffold")

    def transform(mut self, X: Some[Movable], copy: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: transform(self, X, copy=True)."
        raise Error("NotImplementedError: TfidfTransformer.transform is an API scaffold")

struct TfidfVectorizer(Movable):

    "API scaffold for sklearn.feature_extraction.text.TfidfVectorizer. Operations are deferred."

    var input: APIValue

    var encoding: APIValue

    var decode_error: APIValue

    var strip_accents: APIValue

    var lowercase: APIValue

    var preprocessor: APIValue

    var tokenizer: APIValue

    var analyzer: APIValue

    var stop_words: APIValue

    var token_pattern: APIValue

    var ngram_range: APIValue

    var max_df: APIValue

    var min_df: APIValue

    var max_features: APIValue

    var vocabulary: APIValue

    var binary: APIValue

    var dtype: APIValue

    var norm: APIValue

    var use_idf: APIValue

    var smooth_idf: APIValue

    var sublinear_tf: APIValue

    var vocabulary_: APIValue

    var fixed_vocabulary_: APIValue

    var idf_: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: TfidfVectorizer.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: TfidfVectorizer.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: TfidfVectorizer.set_params is an API scaffold")

    def decode(mut self, doc: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: decode(self, doc)."
        raise Error("NotImplementedError: TfidfVectorizer.decode is an API scaffold")

    def build_preprocessor(mut self) raises -> APIValue:
        "Scikit-learn signature: build_preprocessor(self)."
        raise Error("NotImplementedError: TfidfVectorizer.build_preprocessor is an API scaffold")

    def build_tokenizer(mut self) raises -> APIValue:
        "Scikit-learn signature: build_tokenizer(self)."
        raise Error("NotImplementedError: TfidfVectorizer.build_tokenizer is an API scaffold")

    def get_stop_words(mut self) raises -> APIValue:
        "Scikit-learn signature: get_stop_words(self)."
        raise Error("NotImplementedError: TfidfVectorizer.get_stop_words is an API scaffold")

    def build_analyzer(mut self) raises -> APIValue:
        "Scikit-learn signature: build_analyzer(self)."
        raise Error("NotImplementedError: TfidfVectorizer.build_analyzer is an API scaffold")

    def __init__(out self, *, input: Some[Movable] = "content", encoding: Some[Movable] = "utf-8", decode_error: Some[Movable] = "strict", strip_accents: Some[Movable] = None, lowercase: Some[Movable] = True, preprocessor: Some[Movable] = None, tokenizer: Some[Movable] = None, analyzer: Some[Movable] = "word", stop_words: Some[Movable] = None, token_pattern: Some[Movable] = "(?u)\\b\\w\\w+\\b", ngram_range: Some[Movable] = (1, 1), max_df: Some[Movable] = 1.0, min_df: Some[Movable] = 1, max_features: Some[Movable] = None, vocabulary: Some[Movable] = None, binary: Some[Movable] = False, dtype: Some[Movable] = DType.float64, norm: Some[Movable] = "l2", use_idf: Some[Movable] = True, smooth_idf: Some[Movable] = True, sublinear_tf: Some[Movable] = False) raises:
        "Scikit-learn signature: __init__(self, *, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, token_pattern='(?u)\\\\b\\\\w\\\\w+\\\\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)."
        raise Error("NotImplementedError: TfidfVectorizer.__init__ is an API scaffold")

    def fit(mut self, raw_documents: Some[Movable], y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, raw_documents, y=None)."
        raise Error("NotImplementedError: TfidfVectorizer.fit is an API scaffold")

    def fit_transform(mut self, raw_documents: Some[Movable], y: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, raw_documents, y=None)."
        raise Error("NotImplementedError: TfidfVectorizer.fit_transform is an API scaffold")

    def transform(mut self, raw_documents: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, raw_documents)."
        raise Error("NotImplementedError: TfidfVectorizer.transform is an API scaffold")

    def inverse_transform(mut self, X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: inverse_transform(self, X)."
        raise Error("NotImplementedError: TfidfVectorizer.inverse_transform is an API scaffold")

    def get_feature_names_out(mut self, input_features: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: get_feature_names_out(self, input_features=None)."
        raise Error("NotImplementedError: TfidfVectorizer.get_feature_names_out is an API scaffold")

def strip_accents_ascii(s: Some[Movable]) raises -> APIValue:
    "Scikit-learn signature: strip_accents_ascii(s)."
    raise Error("NotImplementedError: strip_accents_ascii is an API scaffold")

def strip_accents_unicode(s: Some[Movable]) raises -> APIValue:
    "Scikit-learn signature: strip_accents_unicode(s)."
    raise Error("NotImplementedError: strip_accents_unicode is an API scaffold")

def strip_tags(s: Some[Movable]) raises -> APIValue:
    "Scikit-learn signature: strip_tags(s)."
    raise Error("NotImplementedError: strip_tags is an API scaffold")

from ._stop_words import ENGLISH_STOP_WORDS
