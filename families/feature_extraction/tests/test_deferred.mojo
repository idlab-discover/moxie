from std.testing import assert_raises
from moxie.feature_extraction import FeatureHasher, image
from moxie.feature_extraction.text import HashingVectorizer, strip_accents_ascii, strip_accents_unicode


def main() raises:
    with assert_raises():
        _ = FeatureHasher()
    _ = image.PatchExtractor()
    with assert_raises():
        _ = HashingVectorizer()
    with assert_raises():
        _ = strip_accents_ascii("é")
    with assert_raises():
        _ = strip_accents_unicode("é")
    print("PASS: deferred feature-extraction APIs reject execution")
