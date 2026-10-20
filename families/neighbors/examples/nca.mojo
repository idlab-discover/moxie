"""Learn a supervised embedding, then use it with a nearest-neighbor estimator."""
import numojo as nm
from moxie.neighbors import NeighborhoodComponentsAnalysis


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(4, 2))
    X.unsafe_set(2, 0.2)
    X.unsafe_set(3, 1)
    X.unsafe_set(4, 2)
    X.unsafe_set(6, 2.4)
    X.unsafe_set(7, 1)
    var y = nm.zeros[DType.float64](nm.Shape(4))
    y.unsafe_set(2, 1)
    y.unsafe_set(3, 1)
    var nca = NeighborhoodComponentsAnalysis(n_components=1, random_state=42)
    var embedded = nca.fit_transform(X, y)
    print("Embedding shape:", embedded.shape)
    print("Output feature:", nca.get_feature_names_out()[0])
