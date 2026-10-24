# Neighbors in Moxie

Moxie owns the dense neighbors implementations, internal query and spatial-tree
engines, tests, examples and license assets in this directory. The standalone
`moxie-neighbors` checkout is no longer a source dependency.

Implemented exports include NearestNeighbors, KNeighborsClassifier/Regressor,
RadiusNeighborsClassifier/Regressor, LocalOutlierFactor, KernelDensity,
NearestCentroid, NeighborhoodComponentsAnalysis, KDTree, BallTree and VALID_METRICS.
The supported contract remains partial. Sparse storage and graph APIs, Python
reflection and metadata routing remain unsupported; see [scope](docs/scope.md),
[estimator APIs](docs/estimators.md), and [NCA](docs/nca.md).

From the Moxie root:

```sh
pixi run neighbors-build
pixi run neighbors-test
pixi run neighbors-parity
pixi run neighbors-start
pixi run python families/neighbors/shared/query_engine/tools/check.py test
pixi run python families/neighbors/shared/query_engine/tools/check.py parity
pixi run python tools/neighbors_independence.py
```

Parity uses the exact local scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Prepare its compiled validation-only
oracle with `pixi run python tools/prepare_neighbors_oracle.py`. The family runner
uses its Python 3.12 environment and packages under `builds/neighbors/`; `SKLEARN_ORACLE_PYTHON` and `PYTHONPATH` can
select another compiled instance of that same reference. Python is never used
by library learning or inference.

The parity task includes `tests/parity_end_to_end.py`, which runs 872 seeded
configurations through the native public estimators and compiled reference.
It covers Float32/Float64 queries, KNN/radius classification and regression,
LOF outlier/novelty detection, KDE and NearestCentroid. It checks the loaded
reference estimator sources against the pinned revision. Radius comparisons
preserve index/distance pairing while allowing documented tie-order differences;
the classifier fixture uses an explicit resolved outlier label to avoid the
pinned reference's `most_frequent` fast-path inconsistency. This suite supplements
the existing lifecycle, invalid-input, label and dtype contracts.

[Provenance](PROVENANCE.json) records the original repository and revision.
`SCAFFOLD_BASE`, toolchain locks and license notices retain their provenance.
The historical `_api` and `_trees_api` declarations are excluded from assembly.
Current shared contracts are in [Moxie's interface guide](../../docs/interfaces.md).
