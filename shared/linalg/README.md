# Shared covariance solves

Pure Mojo covariance adapters using NuMojo storage and existing native numerical
kernels. Read [the API](docs/linalg.md) and [scope](docs/scope.md).

```sh
pixi install --locked
pixi run build
pixi run test
pixi run parity
```

Consume `moxie._linalg` through the shared verified assembler. Family adoption
is deliberate; this project does not change the frozen parent scaffold.
