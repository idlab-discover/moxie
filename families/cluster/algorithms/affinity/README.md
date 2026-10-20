# Affinity propagation in Mojo

Dense `moxie.cluster.AffinityPropagation` and `affinity_propagation`, using the
cluster family's live native storage and RNG dependencies from current Moxie.

Run `pixi run start` for the six-point clustering example.
Read [the API guide](docs/api.md) for typed parameters, mutable RNGs, copy modes
and convergence diagnostics, and [scope](docs/scope.md) for dense-only limits.

Validation: `pixi run build`, `pixi run test`, `pixi run parity`.
Use `PIXI_CACHE_DIR=$PWD/.pixi/pixi-cache` to keep Pixi caches local.
Live source paths are declared in `DEPENDENCIES.json`; there is no source
repinning step or hash gate. The reference is sklearn
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`; its license is retained in LICENSES.
