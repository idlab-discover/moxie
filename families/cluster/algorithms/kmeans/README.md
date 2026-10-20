# KMeans

Pure-Mojo dense `kmeans_plusplus`, `KMeans` and `k_means`, integrated into the
family root build. See [API and numerical scope](docs/api.md).

From the family root, run `pixi run build`, `pixi run test`, `pixi run parity`.
This algorithm consumes the family centroid engine and shared Moxie numerics/RNG.
No Python or foreign numerical runtime is used by the implementation.
