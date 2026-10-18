from moxie.random_projection import GaussianRandomProjection, ProjectionMatrix

def main() raises:
    var model = GaussianRandomProjection[DType.float32](2)
    _ = model.fit(ProjectionMatrix[DType.float64](2, 4))
