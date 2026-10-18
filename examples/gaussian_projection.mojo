from moxie.random_projection import GaussianRandomProjection, ProjectionMatrix

def main() raises:
    var X = ProjectionMatrix(2, 3, List[Float64]([1., 2., 3., 4., 5., 6.]))
    var model = GaussianRandomProjection(2, random_state=42)
    var projected = model.fit_transform(X)
    print(projected.rows, projected.cols)
    print(model.inverse_transform(projected).data)
