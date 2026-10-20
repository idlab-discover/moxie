# Dense image feature extraction

Pure Mojo image patches, reconstruction, stateless `PatchExtractor`, and explicit
dense lattice graphs, consuming the live feature-extraction family and Moxie sources.
See [API and language adaptations](docs/api.md), [scope](docs/scope.md), and
[example](examples/patches.mojo).

```sh
PIXI_CACHE_DIR="$PWD/.pixi/pixi-cache" pixi install --locked
PIXI_CACHE_DIR="$PWD/.pixi/pixi-cache" pixi run build
PIXI_CACHE_DIR="$PWD/.pixi/pixi-cache" pixi run test
PIXI_CACHE_DIR="$PWD/.pixi/pixi-cache" pixi run parity
```

Builds validate module ownership and assemble a single namespace under ignored
`builds/deps`. The Pixi environment locks Mojo 1.0.0 and MAX 26.5.0.
Every gate assembles current sources from `DEPENDENCIES.json`, including current Moxie;
no source repinning is required.
