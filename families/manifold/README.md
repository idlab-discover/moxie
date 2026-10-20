# Manifold

Moxie owns the pure Mojo implementations of ClassicalMDS and trustworthiness,
the dense exchange helpers, tests and examples in this directory. Other public
manifold algorithms raise explicit unsupported errors. See [algorithm support](docs/algorithms.md)
and [scope](docs/scope.md) for supported metrics and native API differences.

From the Moxie root:

```sh
pixi run manifold-build
pixi run manifold-test
pixi run manifold-parity
pixi run manifold-start
pixi run manifold-independence
```

The test command includes the build, contract tests, two independently assembled
consumers, negative capability checks, and assembly collision/liveness checks.
The independence command compiles copied public consumers using only a copied
assembled namespace. No standalone manifold checkout supplies build inputs.

The core `DEPENDENCIES.json` owns public namespace assembly. The focused fixture
manifests consume current Moxie framework sources and the integrated decomposition
eigensolver. Compatible Pixi locks and the sklearn reference revision are preserved.
[PROVENANCE.json](PROVENANCE.json) and `SCAFFOLD_BASE` record the original source;
the retired sibling retains its Git history and ignored artifacts.

- [Family contracts](docs/family.md)
- [Public declarations and deferrals](docs/api-surface.md)
- [Dense exchange contracts](docs/plumbing.md)
- [Moxie interfaces](../../docs/interfaces.md)
- [Contribution rules](../../AGENTS.md)
