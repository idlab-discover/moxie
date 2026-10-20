"""Classify a query using shrunken centroids and inspect its class probabilities.

Run `pixi run build`, then `builds/centroid_example` from the family root.
"""
import numojo as nm
from moxie.neighbors import NearestCentroid


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(6, 2))
    var y = nm.zeros[DType.float64](nm.Shape(6))
    for i in range(6):
        X.unsafe_set(i*2, Float64(i))
        X.unsafe_set(i*2+1, Float64(i % 3))
        y.unsafe_set(i, Float64(i // 3))
    var model = NearestCentroid(shrink_threshold=0.1, priors="empirical")
    model.fit(X, y)
    var query = nm.zeros[DType.float64](nm.Shape(1, 2))
    query.unsafe_set(0, 4.5)
    query.unsafe_set(1, 1)
    var prediction = model.predict(query)
    var probabilities = model.predict_proba(query)
    print("Predicted class:", prediction.unsafe_get(0))
    for c in range(model.classes_.shape[0]):
        print("Class", model.classes_.unsafe_get(c), "probability:", probabilities.unsafe_get(c))
