# moxie

A pure Mojo foundation for porting scikit-learn. This repository contains
estimator contracts and validation helpers; it does not implement ML algorithms.

```sh
pixi install --locked
pixi run build
pixi run test
pixi run start
```

The example demonstrates configuration, fit, prediction, and unfitted cloning.
It is explicitly a demonstration, not a statistical estimator.

- [Estimator interfaces and differences from Python](docs/interfaces.md)
- [Family repositories and scope](docs/families.md)
- [Contribution rules](AGENTS.md)

`src/moxie` is the importable Mojo package. `examples/` contains runnable usage
examples and `tests/` contains correctness checks. Build outputs live in
`builds/`. Benchmarks, profiling, research notes, and reference source checkouts
stay in ignored local directories.

The shared starting point is tagged `scaffold-v1.1`. Derivatives retain its Git
history and record the baseline commit in `SCAFFOLD_BASE`. Each family uses the
same `moxie` namespace, so modules can eventually be integrated without renaming.

Native model serialization is being developed on a feature branch. See the
[serialization guide](docs/serialization.md) for the shared engine, family-owned
schema contract, optional Snappy codec, and current limits.
