# Foundation assembly

One verified dependency assembler for Moxie families and compact consumers.
Read [the consumer guide](docs/assembly.md) and [dense scope](docs/scope.md).

```sh
pixi install --locked
pixi run build
pixi run test
pixi run parity
```

The native runtime remains pure Mojo. Python implements only build, validation
and benchmark tooling. Inputs from Moxie, numerics, NuMojo and the dummy family
remain read-only; generated namespaces live under each consumer's ignored
`builds/deps`. The two fixture projects demonstrate shared local tool reuse.
