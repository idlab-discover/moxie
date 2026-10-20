"""Encode a categorical city and numeric temperature into dense columns."""
from moxie.feature_extraction import DictVectorizer
from moxie.feature_extraction._dict_vectorizer import FeatureBatch, FeatureRow, FeatureEntry, FeatureKey, FeatureValue


def main() raises:
    var rows = FeatureBatch()
    var dubai = FeatureRow()
    dubai.append(FeatureEntry(FeatureKey("city"), FeatureValue(String("Dubai"))))
    dubai.append(FeatureEntry(FeatureKey("temperature"), FeatureValue(Float64(33))))
    rows.append(dubai^)
    var london = FeatureRow()
    london.append(FeatureEntry(FeatureKey("city"), FeatureValue(String("London"))))
    london.append(FeatureEntry(FeatureKey("temperature"), FeatureValue(Float64(12))))
    rows.append(london^)
    var vectorizer = DictVectorizer(sparse=False)
    var matrix = vectorizer.fit_transform(rows)
    for name in vectorizer.get_feature_names_out():
        print(name)
    print(matrix)
