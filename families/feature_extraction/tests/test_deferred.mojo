from std.testing import assert_raises
from moxie.feature_extraction import FeatureHasher, image, text
from moxie.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer, ENGLISH_STOP_WORDS


def main() raises:
    with assert_raises():
        _ = FeatureHasher()
    _ = image.PatchExtractor()
    with assert_raises():
        _ = text.CountVectorizer()
    with assert_raises():
        _ = HashingVectorizer()
    with assert_raises():
        _ = TfidfTransformer()
    with assert_raises():
        _ = TfidfVectorizer()
    print("PASS: deferred feature-extraction constructors reject execution")
