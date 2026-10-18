"""Project dense data, then reconstruct it through the fitted pseudoinverse."""
from moxie.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
    ProjectionMatrix,
)


def main() raises:
    var X = ProjectionMatrix(
        3,
        4,
        List[Float64](
            [1.0, 2.0, 0.0, 4.0, 0.0, 3.0, 2.0, 0.0, 4.0, 0.0, 1.0, 2.0]
        ),
    )
    var gaussian = GaussianRandomProjection(
        n_components=2, random_state=42, compute_inverse_components=True
    )
    var projected = gaussian.fit_transform(X)
    print("Gaussian components:", gaussian.components_.data)
    print("Projected (3 x 2):", projected.data)
    print("Reconstructed (3 x 4):", gaussian.inverse_transform(projected).data)
    var sparse = SparseRandomProjection(
        n_components=2, density=0.5, random_state=42
    )
    var sparse_projected = sparse.fit_transform(X.to_csr())
    print("Sparse output:", sparse_projected.is_sparse)
    print("Sparse projected values:", sparse_projected.to_dense().data)
