# Native dictionary vectorization

Pure Mojo dense `DictVectorizer`, using the feature-extraction family's live
vocabulary contracts and NuMojo arrays. This work is not complete across every
dense reference dtype: complex and extended-precision branches remain open.

See [API and examples](docs/api.md), [scope](docs/scope.md), and
[dependency assembly](../../docs/plumbing.md).

```sh
PIXI_CACHE_DIR="$PWD/.pixi/pixi-cache" pixi install --locked
pixi run build
pixi run test
pixi run parity
pixi run python tools/performance.py
```

All dependency source is assembled from current owners and assembled under ignored `builds/deps`.
Python is used only for builds, reference comparisons and measurement.
Library learning and transformation execute in native Mojo.
