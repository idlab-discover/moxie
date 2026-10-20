# Isolated assembly consumer

This test project contributes one estimator-shaped ownership probe and consumes
both probe projects in a single assembled namespace. It implements no ML model.
From this directory run:

```sh
PIXI_CACHE_DIR="$PWD/.pixi/cache" pixi install --locked
PIXI_CACHE_DIR="$PWD/.pixi/cache" pixi run build
PIXI_CACHE_DIR="$PWD/.pixi/cache" pixi run test
```

The family test task also compiles and runs both consumers using its locked
compiler with separate import trees and consumer-local caches. See the family
`docs/plumbing.md` for ownership and dependency rules.
