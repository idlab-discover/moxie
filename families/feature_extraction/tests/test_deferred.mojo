from std.testing import assert_raises
from moxie.feature_extraction import FeatureHasher, image
from moxie.feature_extraction.text import HashingVectorizer


def main() raises:
    with assert_raises():
        _ = FeatureHasher()
    _ = image.PatchExtractor()
    with assert_raises():
        _ = HashingVectorizer()
    print("PASS: deferred feature-extraction APIs reject execution")
