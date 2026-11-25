from std.testing import assert_raises
from moxie.feature_extraction import FeatureHasher, image
from moxie.feature_extraction.text import HashingVectorizer, TfidfVectorizer


def main() raises:
    with assert_raises():
        _ = FeatureHasher()
    _ = image.PatchExtractor()
    with assert_raises():
        _ = HashingVectorizer()
    with assert_raises():
        _ = TfidfVectorizer()
    print("PASS: deferred feature-extraction constructors reject execution")
