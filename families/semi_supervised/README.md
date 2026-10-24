# Semi-supervised learning

Moxie owns the pure Mojo dense implementations of `LabelPropagation`,
`LabelSpreading`, and `SelfTrainingClassifier` in this directory. The correctness
reference is scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

From the Moxie root:

```sh
pixi run semi-supervised-build
pixi run semi-supervised-test
pixi run semi-supervised-parity
pixi run semi-supervised-independence
```

The test command includes the build. It validates the full Moxie assembly and
two isolated family consumers. Parity executes the exact local sklearn algorithm
sources. The independence check copies the assembled namespace to a temporary
consumer and compiles and runs the public example there.

Graph estimators support RBF and typed dense callable affinities. Self-training
supports threshold and k-best selection, finite or unlimited iteration counts,
and native probabilistic estimator adapters. `GaussianNBAdapter` consumes
Moxie's Naive Bayes implementation. Python is used only for validation.

Side-by-side notebooks in `notebooks/semi_supervised/` replay
`plot_label_propagation_digits`, `plot_self_training_varying_threshold` and
`plot_semi_supervised_versus_svm_iris` against sklearn. Run them with
`pixi run -e notebooks notebooks notebooks/semi_supervised`.

See [the API guide](docs/api-surface.md), [supported scope](docs/scope.md), and
[extension contracts](docs/plumbing.md). Sparse inputs and KNN graphs remain
excluded. This is dense estimator support, not full sklearn parity.

[Provenance](PROVENANCE.json) records the original repository revision.
`SCAFFOLD_BASE`, the compatible toolchain lock and license assets are retained.
The retired checkout preserves its Git history and ignored artifacts; builds
read no files from it. Source dependencies follow current local working trees.
