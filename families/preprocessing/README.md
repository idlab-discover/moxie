# Preprocessing

All 29 scikit-learn preprocessing public names have dense implementations in
Moxie. See the [API guide](docs/api.md) for supported inputs, native adaptations
and limitations. Sparse APIs remain deferred.

Public modules live in Moxie's `src/moxie/preprocessing`. This directory retains
family engines, contracts, parity tests, examples, licenses and locked standalone
tooling. Moxie's root build, test and parity commands include this family.
The original `moxie-preprocessing` checkout is not required.

From the Moxie repository root:

```sh
pixi run build
pixi run test
pixi run parity
pixi run python families/preprocessing/tools/check.py parity scalers
```

See [development](docs/development.md), [plumbing](docs/plumbing.md), and
[API surface validation](docs/api-surface.md). `SCAFFOLD_BASE` preserves original
scaffold provenance; the existing Pixi locks and oracle revision are retained.
