# Shared labels

Pure Mojo class-domain encoding and dense target validation for Moxie families.
Read [the API and ownership contract](docs/labels.md) and [scope](docs/scope.md).

```sh
pixi install --locked
pixi run build
pixi run test
pixi run parity
```

Consumers import `moxie._labels` through the existing verified shared assembler.
This project provides private helpers; public preprocessing estimators and
family adoption remain separate work.
