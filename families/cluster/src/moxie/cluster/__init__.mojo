"""Contracts for the cluster family."""
from ._base import (
    Clusterer,
    PredictiveClusterer,
    ClusterDistanceTransformer,
    IncrementalClusterer,
    Biclusterer,
)

from ._affinity_propagation import AffinityPropagation, affinity_propagation

# Deferred public scikit-learn API declarations.
from ._api import (
    HDBSCAN,
    MiniBatchKMeans,
    SpectralBiclustering,
    SpectralClustering,
    SpectralCoclustering,
    spectral_clustering,
)

from ._kmeans import KMeans, k_means, kmeans_plusplus
from ._bisect_k_means import BisectingKMeans

from ._dbscan import DBSCAN, dbscan

from ._mean_shift import MeanShift, mean_shift, estimate_bandwidth, get_bin_seeds

from ._optics import OPTICS, compute_optics_graph, cluster_optics_dbscan, cluster_optics_xi

from ._agglomerative import AgglomerativeClustering, FeatureAgglomeration, linkage_tree, ward_tree

from ._birch import Birch
