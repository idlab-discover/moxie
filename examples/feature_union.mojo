"""Combine original and rescaled dense features, then use a real estimator."""
from numojo import zeros, Shape
from moxie.pipeline import FeatureUnion, UnionPair, UnionSentinel, Pipeline
from moxie.dummy import DummyRegressor
from moxie.base import clone


def main() raises:
    var X = zeros[DType.float64](Shape(2, 2))
    for i in range(4):
        X.itemset(i, Float64(i + 1))
    var y = zeros[DType.float64](Shape(2))
    y.itemset(0, 2)
    y.itemset(1, 4)
    var weights: Dict[String, Float64] = {"half": 0.5}
    var features = FeatureUnion(UnionPair(UnionSentinel("raw"), UnionSentinel("half")),
                                transformer_weights=weights^)
    var combined = features.fit_transform(X)
    print("Combined shape:", combined.shape[0], combined.shape[1])
    var names = features.get_feature_names_out()
    for i in range(combined.shape[1]):
        print(names[i], combined.item(0, i))
    # Replace DummyRegressor with any compatible supervised predictor.
    var model = Pipeline(features^, DummyRegressor())
    model.fit(X, y)
    print("Prediction:", model.predict(X).item(0))
    var fresh = clone(model)
    print("Clone fitted:", fresh.__sklearn_is_fitted__())
