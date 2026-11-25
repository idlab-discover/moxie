# Manifold

Moxie owns exact TSNE, ClassicalMDS, trustworthiness and the dense manifold
contracts in this directory. See [exact t-SNE](docs/tsne.md),
[ClassicalMDS and trustworthiness](docs/algorithms.md), and [scope](docs/scope.md)
for supported metrics and native API differences.

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
and neighbors engines. [PROVENANCE.json](PROVENANCE.json) and `SCAFFOLD_BASE`
record the original source.

The notebook `notebooks/manifold/plot_lle_digits.ipynb` compares its Classical MDS
panel and trustworthiness scores with scikit-learn and checks the unsupported
manifold operations used by the example. Run it with
`pixi run -e notebooks notebooks notebooks/manifold`.

- [Family contracts](docs/family.md)
- [Public declarations and deferrals](docs/api-surface.md)
- [Dense exchange contracts](docs/plumbing.md)
- [Moxie interfaces](../../docs/interfaces.md)
- [Contribution rules](../../AGENTS.md)
