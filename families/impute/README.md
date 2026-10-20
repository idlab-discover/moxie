# Imputation in Moxie

Moxie owns the implementations, helpers, tests and examples in this directory.
Import the public estimators from `moxie.impute` in Moxie's assembled
`builds/deps` namespace. The standalone checkout is no longer a dependency.

Implemented contracts include dense floating SimpleImputer, integer and string
specializations, numeric and string MissingIndicator, and dense KNNImputer.
KNN uses the [integrated neighbors engine](../neighbors/README.md).
IterativeImputer remains a declaration whose constructor raises
`NotImplementedError`. Sparse and heterogeneous categorical behavior remains
unsupported. See [the API](docs/api-surface.md) for exact limits and adaptations.

From the Moxie root:

```sh
pixi run impute-build
pixi run impute-test
pixi run impute-parity
pixi run impute-start
pixi run impute-independence
```

The test command builds all family modules, examples, native estimator contracts
and two isolated consumer namespaces. Parity compares floating, integer, string
and KNN behavior with the pinned local scikit-learn source. The independence
command assembles the complete Moxie namespace and runs isolated public consumers.
These checks do not establish full scikit-learn parity.

Family and simple-algorithm Pixi environments retain their compatible locks and
local build/test/parity commands. The root runner uses Moxie's locked environment.
Generated binaries and raw oracle evidence remain ignored.

- [Simple and indicator API](algorithms/simple/docs/api.md)
- [KNN API](algorithms/knn/docs/api.md)
- [Dense scope](docs/scope.md)
- [Shared plumbing](docs/plumbing.md)
- [Family contract](docs/family.md)
- [Development policy](../../docs/family-development.md)

`PROVENANCE.json` records the source revision and sklearn reference revision.
`SCAFFOLD_BASE` retains the original scaffold provenance. Historical Git history
and ignored local artifacts remain in `../../../retired-moxie-impute`.
