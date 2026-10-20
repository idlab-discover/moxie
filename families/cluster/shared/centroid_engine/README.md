# Dense centroid engine

Private Mojo kernels for the Moxie cluster family, using the family's live
NuMojo storage and current Moxie dependencies. See [API](docs/api.md) and
[dense scope](docs/scope.md).

The dense kernel and relocation parity suites pass against the local scikit-learn
reference. This is a private engine, not a KMeans or MiniBatchKMeans estimator.
Relocation ordering follows the NumPy four-lane selection backend documented
in the API guide; it is not a promise of identical ordering across NumPy backends.

```sh
PIXI_CACHE_DIR=$PWD/.pixi/pixi-cache pixi install --locked
PIXI_CACHE_DIR=$PWD/.pixi/pixi-cache pixi run --locked build
PIXI_CACHE_DIR=$PWD/.pixi/pixi-cache pixi run --locked test
PIXI_CACHE_DIR=$PWD/.pixi/pixi-cache pixi run --locked parity
```

Builds read live sources declared in `DEPENDENCIES.json`. No pinned launcher,
source hash gate or repinning step is used. Pixi toolchain locks remain in effect.
