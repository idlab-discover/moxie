"""Cluster samples and pool correlated features using the same merge criterion."""
import numojo as nm
from moxie.cluster import AgglomerativeClustering, FeatureAgglomeration, ward_tree


def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(4, 2))
    x[nm.Item(2, 0)] = 8
    x[nm.Item(3, 0)] = 9
    var model = AgglomerativeClustering(2, compute_distances=True)
    var labels = model.fit_predict(x)
    print("Sample labels:", labels)
    var tree = ward_tree(x, return_distance=True)
    print("Merge children:", tree.children)
    var features = FeatureAgglomeration(1)
    var pooled = features.fit_transform(x)
    print("Mean of grouped features:", pooled)
    print("Reconstructed pooled features:", features.inverse_transform(pooled))
