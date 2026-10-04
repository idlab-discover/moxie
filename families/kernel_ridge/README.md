# KernelRidge (integrated into Moxie)

Canonical implementation: `../../src/moxie/kernel_ridge.mojo` and
`../../src/moxie/_kernel_ridge/`. This directory owns its tests, examples,
parity tooling and [API documentation](docs/family.md).

From the Moxie root:

```sh
pixi run build
pixi run test
pixi run parity
# Focused family validation:
pixi run python families/kernel_ridge/tools/check.py test
pixi run python families/kernel_ridge/tools/check.py parity
pixi run python families/kernel_ridge/tools/check.py start
```

The root live dependency manifest assembles the family with Moxie's current
contracts and runtime dependencies. No sibling KernelRidge checkout is required.
The original repository and revision are recorded in `ORIGIN.json`.
Sparse input remains unsupported; this integration adds no algorithm capability.
