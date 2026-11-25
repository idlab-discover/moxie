# Adapted from scikit-learn sklearn/feature_extraction/text.py.
# Copyright (c) scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause

from std.collections import Set

from moxie.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from moxie.feature_extraction._unicode_data import (
    CASED_RANGES, CASE_IGNORABLE_RANGES, LOWERCASE_PAIRS, WHITESPACE_RANGES, WORD_RANGES,
)

comptime DEFAULT_TOKEN_PATTERN = "(?u)\\b\\w\\w+\\b"
comptime TextPreprocessor = def(String) thin raises -> String
comptime TextTokenizer = def(String) thin raises -> List[String]


def _read_codepoint(table: Span[UInt8, _], index: Int) -> Int:
    var value = 0
    for i in range(6 * index, 6 * (index + 1)):
        var digit = Int(table[i])
        value = 16 * value + (digit - 48 if digit <= 57 else digit - 87)
    return value


def _find_pair(table: StaticString, cp: Int) -> Int:
    # Find the last pair whose first element is <= cp.
    var data = table.as_bytes()
    var left = 0
    var right = len(data) // 12
    while left < right:
        var mid = left + (right - left) // 2
        if _read_codepoint(data, 2 * mid) <= cp:
            left = mid + 1
        else:
            right = mid
    return left - 1


def _in_ranges(table: StaticString, cp: Int) -> Bool:
    var index = _find_pair(table, cp)
    return index >= 0 and cp <= _read_codepoint(table.as_bytes(), 2 * index + 1)


def _is_word(cp: Int) -> Bool:
    if cp < 128:
        return 48 <= cp <= 57 or 65 <= cp <= 90 or 97 <= cp <= 122 or cp == 95
    return _in_ranges(WORD_RANGES, cp)


def _is_whitespace(cp: Int) -> Bool:
    if cp < 128:
        return 9 <= cp <= 13 or 28 <= cp <= 32
    return _in_ranges(WHITESPACE_RANGES, cp)


def _utf8_width(cp: Int) -> Int:
    if cp < 0x80:
        return 1
    if cp < 0x800:
        return 2
    return 3 if cp < 0x10000 else 4


def _append_codepoint(mut bytes: List[UInt8], cp: Int):
    var width = _utf8_width(cp)
    if width == 1:
        bytes.append(UInt8(cp))
        return
    var prefix = 256 - (256 >> width)
    bytes.append(UInt8(prefix | (cp >> (6 * (width - 1)))))
    for shift in range(6 * (width - 2), -1, -6):
        bytes.append(UInt8(0x80 | ((cp >> shift) & 0x3F)))


def _lowercase(document: String) raises -> String:
    var output = List[UInt8](capacity=document.byte_length())
    var previous_cased = False
    var pending_sigma = -1
    for scalar in document.codepoints():
        var cp = Int(scalar)
        var ignorable = _in_ranges(CASE_IGNORABLE_RANGES, cp) if cp >= 128 else (
            cp == 39 or cp == 46 or cp == 58 or cp == 94 or cp == 96
        )
        if not ignorable:
            var cased = _in_ranges(CASED_RANGES, cp) if cp >= 128 else (
                65 <= cp <= 90 or 97 <= cp <= 122
            )
            if pending_sigma >= 0 and cased:
                # Restore ordinary sigma after seeing another cased character.
                output[pending_sigma] = 0x83
            pending_sigma = -1
            if cp == 0x03A3 and previous_cased:
                pending_sigma = len(output) + 1
                _append_codepoint(output, 0x03C2)
                previous_cased = cased
                continue
            previous_cased = cased
        if cp < 128:
            output.append(UInt8(cp + 32 if 65 <= cp <= 90 else cp))
        elif cp == 0x0130:
            _append_codepoint(output, 0x0069)
            _append_codepoint(output, 0x0307)
        else:
            var index = _find_pair(LOWERCASE_PAIRS, cp)
            var table = LOWERCASE_PAIRS.as_bytes()
            if index >= 0 and _read_codepoint(table, 2 * index) == cp:
                cp = _read_codepoint(table, 2 * index + 1)
            _append_codepoint(output, cp)
    return String(from_utf8=Span(output))


def _word_tokens(document: String) -> List[String]:
    var tokens = List[String]()
    var start = 0
    var offset = 0
    var length = 0
    for scalar in document.codepoints():
        var cp = Int(scalar)
        if _is_word(cp):
            if length == 0:
                start = offset
            length += 1
        else:
            if length >= 2:
                tokens.append(String(document[byte=start:offset]))
            length = 0
        offset += _utf8_width(cp)
    if length >= 2:
        tokens.append(String(document[byte=start:offset]))
    return tokens^


def _character_offsets(document: String) -> List[Int]:
    var offsets: List[Int] = [0]
    var offset = 0
    for cp in document.codepoints():
        offset += _utf8_width(Int(cp))
        offsets.append(offset)
    return offsets^


def _char_ngrams(
    document: String, min_n: Int, max_n: Int, padded_word: Bool, mut output: List[String]
):
    var offsets = _character_offsets(document)
    var length = len(offsets) - 1
    var first = min(min_n, length) if padded_word else min_n
    for n in range(first, min(max_n, length) + 1):
        for i in range(length - n + 1):
            output.append(String(document[byte=offsets[i]:offsets[i + n]]))


def _normalize_whitespace(document: String) -> String:
    var output = String()
    var offset = 0
    var start = 0
    var last_nonspace = 0
    var spaces = 0
    for scalar in document.codepoints():
        var cp = Int(scalar)
        if _is_whitespace(cp):
            if spaces == 0:
                last_nonspace = offset
            spaces += 1
        else:
            if spaces >= 2:
                output += document[byte=start:last_nonspace]
                output += " "
                start = offset
            spaces = 0
        offset += _utf8_width(cp)
    if spaces >= 2:
        output += document[byte=start:last_nonspace]
        output += " "
    else:
        output += document[byte=start:offset]
    return output^


def _word_char_ngrams(document: String, min_n: Int, max_n: Int) -> List[String]:
    var output = List[String]()
    var offset = 0
    var start = 0
    for scalar in document.codepoints():
        var cp = Int(scalar)
        var end = offset + _utf8_width(cp)
        if _is_whitespace(cp):
            if start < offset:
                var padded = String(" ", document[byte=start:offset], " ")
                _char_ngrams(padded, min_n, max_n, True, output)
            start = end
        offset = end
    if start < offset:
        var padded = String(" ", document[byte=start:offset], " ")
        _char_ngrams(padded, min_n, max_n, True, output)
    return output^


def strip_tags(s: String) -> String:
    """Replace each match of <([^>]+)> with one ASCII space."""
    var output = String()
    var bytes = s.as_bytes()
    var start = 0
    var opening = -1
    for i in range(len(bytes)):
        if bytes[i] == 60 and opening == -1:
            opening = i
        elif bytes[i] == 62 and opening >= 0:
            if i > opening + 1:
                output += s[byte=start:opening]
                output += " "
                start = i + 1
            opening = -1
    output += s[byte=start:len(bytes)]
    return output^


@fieldwise_init
struct DocumentPreprocessor(Copyable, Movable):
    var lowercase: Bool
    var callback: Optional[TextPreprocessor]

    def __call__(self, document: String) raises -> String:
        if self.callback:
            return self.callback.value()(document)
        if self.lowercase:
            return _lowercase(document)
        return document


@fieldwise_init
struct DocumentTokenizer(Copyable, Movable):
    var callback: Optional[TextTokenizer]

    def __call__(self, document: String) raises -> List[String]:
        if self.callback:
            return self.callback.value()(document)
        return _word_tokens(document)


@fieldwise_init
struct DocumentAnalyzer(Copyable, Movable):
    var input: String
    var encoding: String
    var decode_error: String
    var analyzer: String
    var callback: Optional[TextTokenizer]
    var preprocessor: DocumentPreprocessor
    var tokenizer: DocumentTokenizer
    var stop_words: Set[String]
    var ngram_range: Tuple[Int, Int]

    def __call__(self, document: String) raises -> List[String]:
        var decoded = decode_document(document, self.input, self.encoding, self.decode_error)
        if self.callback:
            return self.callback.value()(decoded)
        var text = self.preprocessor(decoded)
        var min_n = self.ngram_range[0]
        var max_n = self.ngram_range[1]
        if self.analyzer == "char_wb":
            return _word_char_ngrams(text, min_n, max_n)
        if self.analyzer == "char":
            var grams = List[String]()
            _char_ngrams(_normalize_whitespace(text), min_n, max_n, False, grams)
            return grams^
        var tokens = self.tokenizer(text)
        var filtered = List[String]()
        for token in tokens:
            if token not in self.stop_words:
                filtered.append(token)
        if min_n == 1 and max_n == 1:
            return filtered^
        var grams = List[String]()
        for n in range(min_n, min(max_n, len(filtered)) + 1):
            for i in range(len(filtered) - n + 1):
                grams.append(String(" ").join(filtered[i:i + n]))
        return grams^


def _utf8_prefix(bytes: Span[UInt8, _], offset: Int, ascii_only: Bool) -> Tuple[Int, Bool]:
    var first = Int(bytes[offset])
    if first < 128:
        return (1, True)
    if ascii_only or first < 0xC2 or first > 0xF4:
        return (1, False)
    var width = 2 if first < 0xE0 else (3 if first < 0xF0 else 4)
    for i in range(1, width):
        if offset + i >= len(bytes):
            return (i, False)
        var next_byte = Int(bytes[offset + i])
        var low = 0x80
        var high = 0xBF
        if i == 1:
            if first == 0xE0:
                low = 0xA0
            elif first == 0xED:
                high = 0x9F
            elif first == 0xF0:
                low = 0x90
            elif first == 0xF4:
                high = 0x8F
        if next_byte < low or next_byte > high:
            return (i, False)
    return (width, True)


def decode_bytes(bytes: Span[UInt8, _], encoding: String, errors: String) raises -> String:
    var ascii_only = encoding == "ascii"
    if not ascii_only and encoding != "utf-8" and encoding != "utf8":
        raise Error("NotImplementedError: CountVectorizer supports UTF-8 and ASCII byte decoding")
    if errors != "strict" and errors != "ignore" and errors != "replace":
        raise Error("ValueError: decode_error must be strict, ignore or replace")
    var output = List[UInt8]()
    var offset = 0
    var start = 0
    while offset < len(bytes):
        var prefix = _utf8_prefix(bytes, offset, ascii_only)
        if not prefix[1]:
            if errors == "strict":
                raise Error(String("UnicodeDecodeError: invalid ", encoding, " at byte ", offset))
            output.extend(bytes[start:offset])
            if errors == "replace":
                _append_codepoint(output, 0xFFFD)
            start = offset + prefix[0]
        offset += prefix[0]
    if start == 0:
        return String(from_utf8=bytes)
    output.extend(bytes[start:offset])
    return String(from_utf8=Span(output))


def decode_document(document: String, mode: String, encoding: String, errors: String) raises -> String:
    if mode == "content":
        return document
    if mode == "filename":
        with open(document, "r") as file:
            var contents = List[UInt8]()
            while True:
                var chunk = file.read_bytes(65536)
                if len(chunk) == 0:
                    break
                contents.extend(Span(chunk))
            return decode_bytes(Span(contents), encoding, errors)
    raise Error("NotImplementedError: CountVectorizer input must be content or filename")


def stop_word_set(option: Optional[String], custom: List[String]) raises -> Set[String]:
    var words = Set[String]()
    if option:
        if option.value() != "english" or len(custom) != 0:
            raise Error("ValueError: stop_words must be 'english' or a list of terms")
        comptime for i in range(len(ENGLISH_STOP_WORDS)):
            words.add(String(ENGLISH_STOP_WORDS[i]))
    else:
        for word in custom:
            words.add(word)
    return words^
